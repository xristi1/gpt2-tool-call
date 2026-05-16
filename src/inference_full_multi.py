# -*- coding: utf-8 -*-
"""Inference example — multi-prompt, first prompt JSON-like, others smoke <functioncall>."""

import torch
import json
from pathlib import Path
import sys

# ---------------- Fix imports ----------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.integrated_gpt2_torch import GPT2, encode, decode
from src.steering_v2 import FullSteeringGPT2, load_classifier_from_npz

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

# ---------------- Paths ----------------
BASE_DIR = project_root / "weights"
FT_PT = BASE_DIR / "gpt2_ft_final_smoke.pt"
ADAPTER_PT = BASE_DIR / "adapter_h13_bfcl_ep1_smoke.pt"
ADAPTER_NPZ = BASE_DIR / "adapter_torch_EN_BFCL.npz"

# ---------------- Load model ----------------
m = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
m.gpt.load_state_dict(torch.load(FT_PT, map_location=DEVICE))
if ADAPTER_NPZ.exists():
    load_classifier_from_npz(m, str(ADAPTER_NPZ))
m.adapter.load_state_dict(torch.load(ADAPTER_PT, map_location=DEVICE))
m.freeze_gpt()
m.to(DEVICE)
m.eval()
print("[INFO] Model ready for inference.")

# ---------------- Prompts ----------------
prompts = [
    ("get_weather", "What's the weather in Paris?"),
    ("get_time", "What time is it in New York?"),
    ("get_temperature", "Tell me the current temperature in Tokyo.")
]

# ---------------- Inference ----------------
for i, (func_name, user_query) in enumerate(prompts):
    spec = {
        "name": func_name,
        "description": f"Function {func_name}",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
    prompt = (
        f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        f"{json.dumps(spec, indent=2)}\n\n\n"
        f"USER: {user_query}\n\n\n"
        f"ASSISTANT: <functioncall> "
    )

    ids = encode(prompt)
    with torch.no_grad():
        for _ in range(40):
            ids_tensor = torch.tensor([ids], device=DEVICE)
            mask = torch.ones_like(ids_tensor, dtype=torch.float32, device=DEVICE)
            logits, *_ = m(ids_tensor, mask)
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            if decode([nxt]) in ["}", "\n"]:
                break

    raw_result = decode(ids[len(encode(prompt)):])
    # ---------------- Post-processing ----------------
    if i == 0:
        last_close = raw_result.rfind("}")
        cleaned_result = raw_result[:last_close + 1] if last_close != -1 else raw_result
    else:
        cleaned_result = "<functioncall>" if raw_result == "<functioncall>" else raw_result
        if "}" in raw_result:
            cleaned_result = raw_result[:raw_result.find("}") + 1]

    # ---------------- Print output ----------------
    print("\n[USER QUERY]")
    print(user_query)
    print("\n[RAW OUTPUT]")
    print(raw_result)
    print("\n[CLEANED JSON OUTPUT]")
    print(cleaned_result)
    print("\n" + "="*50)