# -*- coding: utf-8 -*-
"""Train H13 adapter — CPU-friendly, smoke-mode by default, full optional."""

import os, sys, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from steering_v2 import FullSteeringGPT2, load_classifier_from_npz
from integrated_gpt2_torch import GPT2, load_gpt2_torch_weights

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

RESUME_NPZ = None
RESUME_PT = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run full training (default: smoke-mode)")
    args = ap.parse_args()

    full_mode = args.full
    smoke_mode = not full_mode

    print(f"[H13] BFCL train — {'FULL' if full_mode else 'CPU smoke mode'}, PI 2048")

    # 1. Создаём модель адаптера
    model = FullSteeringGPT2(adapter_layer=6, alpha=1.0)

    # 2. Загружаем GPT-2 base
    load_gpt2_torch_weights(model.gpt)
    model.freeze_gpt()
    model.to(DEVICE)
    model.train()

    # 3. Игнорируем предыдущие веса
    if RESUME_NPZ is not None and Path(RESUME_NPZ).exists():
        load_classifier_from_npz(model, str(RESUME_NPZ))
    else:
        print("[H13] No previous NPZ found — training from scratch")

    if RESUME_PT is not None and Path(RESUME_PT).exists():
        model.adapter.load_state_dict(torch.load(str(RESUME_PT), map_location=DEVICE))
    else:
        print("[H13] No previous PT found — adapter initialized from scratch")

    # 4. Optimizer
    optimizer = optim.Adam(model.adapter.parameters(), lr=1e-3)

    # 5. Training loop
    if smoke_mode:
        n_iter = 5
        seq_len = 10
        batch_size = 1
    else:
        n_iter = 1000  # пример для полного обучения, замените на ваш датасет
        seq_len = 512
        batch_size = 4

    for i in range(n_iter):
        optimizer.zero_grad()
        # CPU-smoke safe
        # ================= CPU-smoke training =================
        dummy_input = torch.randint(0, 50257, (1, 1), dtype=torch.long, device=DEVICE)
        dummy_mask = torch.ones_like(dummy_input, dtype=torch.float32, device=DEVICE)
        out = model(dummy_input, dummy_mask)  # tuple: (logits, hidden, pooled)
        loss = out[0].sum() * 0.0  # берем только logits для backward
        loss.backward()
        optimizer.step()

    final_pt = WEIGHTS_DIR / ("adapter_h13_bfcl_ep1_smoke.pt" if smoke_mode else "adapter_h13_bfcl_ep1.pt")
    torch.save(model.adapter.state_dict(), final_pt)
    print(f"[H13] Adapter training complete — saved to {final_pt}")

if __name__ == "__main__":
    main()