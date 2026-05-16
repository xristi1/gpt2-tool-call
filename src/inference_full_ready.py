# -*- coding: utf-8 -*-
"""Inference example — full FT GPT-2 + H13 adapter, ready to generate JSON."""

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

# ---------------- Paths to weights ----------------
BASE_DIR = project_root / "weights"
FT_PT = BASE_DIR / "gpt2_ft_final.pt"
ADAPTER_PT = BASE_DIR / "adapter_h13_bfcl_ep1.pt"
ADAPTER_NPZ = BASE_DIR / "adapter_torch_EN_BFCL.npz"

# ---------------- Load model ----------------
print("[INFO] Loading full FT GPT-2 + adapter...")
m = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
m.gpt.load_state_dict(torch.load(FT_PT, map_location=DEVICE))
if ADAPTER_NPZ.exists():
    load_classifier_from_npz(m, str(ADAPTER_NPZ))
m.adapter.load_state_dict(torch.load(ADAPTER_PT, map_location=DEVICE))
m.freeze_gpt()
m.to(DEVICE)
m.eval()
print("[INFO] Model ready for inference.")

# ---------------- Prepare prompt ----------------
spec = {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
}
prompt = (
    f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
    f"{json.dumps(spec, indent=2)}\n\n\n"
    f"USER: What's the weather in Paris?\n\n\n"
    f"ASSISTANT: <functioncall> "
)

# ---------------- Inference ----------------
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

# ---------------- Post-processing ----------------
raw_result = decode(ids[len(encode(prompt)):])
if "}" in raw_result:
    cleaned_result = raw_result[:raw_result.find("}") + 1]
else:
    cleaned_result = raw_result

print("\n[RAW OUTPUT]")
print(raw_result)
print("\n[CLEANED JSON OUTPUT]")
print(cleaned_result)