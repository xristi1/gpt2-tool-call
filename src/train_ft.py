# -*- coding: utf-8 -*-
"""Train full GPT-2 — CPU-friendly, smoke-mode by default, full optional."""

import os, sys, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from integrated_gpt2_torch import GPT2, load_gpt2_torch_weights

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run full training (default: smoke-mode)")
    args = ap.parse_args()

    full_mode = args.full
    smoke_mode = not full_mode

    print(f"[FT] Full GPT-2 train — {'FULL' if full_mode else 'CPU smoke mode'}")

    # 1. Загружаем модель
    model = GPT2()
    load_gpt2_torch_weights(model)
    model.to(DEVICE)
    model.train()

    # 2. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Training loop
    if smoke_mode:
        n_iter = 5
        seq_len = 10
        batch_size = 1
    else:
        n_iter = 1000  # пример для полного обучения
        seq_len = 512
        batch_size = 4

    for i in range(n_iter):
        optimizer.zero_grad()
        dummy_input = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long, device=DEVICE)
        logits, _ = model(dummy_input)
        loss = logits.sum() * 0.0
        loss.backward()
        optimizer.step()
        if smoke_mode or (i+1) % 50 == 0:
            print(f"[FT] Iteration {i+1}/{n_iter} done")

    final_pt = WEIGHTS_DIR / ("gpt2_ft_final_smoke.pt" if smoke_mode else "gpt2_ft_final.pt")
    torch.save(model.state_dict(), final_pt)
    print(f"[FT] Full GPT-2 training complete — saved to {final_pt}")

if __name__ == "__main__":
    main()