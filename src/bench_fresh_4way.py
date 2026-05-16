# -*- coding: utf-8 -*-
"""Fresh bench — 4 models on 690 fresh prompts (no leakage).

Variants:
  1. plain      — pure GPT-2 124M (HF base, no FT, no adapter)
  2. adapter    — frozen GPT-2 + h13_ep1 adapter (250K trainable)
  3. ft         — full FT GPT-2 (124M trainable, no adapter)
  4. ft_adapter — full FT GPT-2 + h13_ep1 adapter on top (combo)

Smoke mode: runs only 20+5+5+5 items per model (~seconds on CPU)
"""
import os, sys, json, re, time, argparse
from pathlib import Path
import torch
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, str(Path(__file__).resolve().parent))

from integrated_gpt2_torch import GPT2, load_gpt2_torch_weights, encode, decode
from steering_v2 import FullSteeringGPT2, load_classifier_from_npz
from long_context_pi_chunk import interpolate_wpe

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

# ================= DATA FILES =================
FRESH_SIMPLE = [
    str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_opus.json"),
    str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_bio.json"),
    str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_industrial.json"),
    str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_culture.json"),
    str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_materials.json"),
    str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_nichetech.json"),
]
FRESH_MULTIPLE = str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_multiple.json")
FRESH_PARALLEL = str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_parallel.json")
FRESH_IRREL = str(Path(__file__).resolve().parent.parent / "bench" / "fresh_bench_irrelevance.json")

def load_all():
    simple = []
    for f in FRESH_SIMPLE:
        if Path(f).exists():
            with open(f, encoding='utf-8') as fh: simple += json.load(fh)
    mult = []
    if Path(FRESH_MULTIPLE).exists():
        with open(FRESH_MULTIPLE, encoding='utf-8') as fh: mult = json.load(fh)
    par = []
    if Path(FRESH_PARALLEL).exists():
        with open(FRESH_PARALLEL, encoding='utf-8') as fh: par = json.load(fh)
    irr = []
    if Path(FRESH_IRREL).exists():
        with open(FRESH_IRREL, encoding='utf-8') as fh: irr = json.load(fh)
    return simple, mult, par, irr

# ================= PROMPT BUILDERS =================
def build_prompt_simple(item):
    fn_json = json.dumps(item["function"], indent=2)[:600]
    return f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{fn_json}\n\n\nUSER: {item['prompt'][:400]}\n\n\nASSISTANT: <functioncall> "

def build_prompt_multi(item):
    funcs = item["function"]
    if isinstance(funcs, dict): funcs = [funcs]
    fn_json = json.dumps(funcs, indent=2)[:1200]
    return f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{fn_json}\n\n\nUSER: {item['prompt'][:400]}\n\n\nASSISTANT: <functioncall> "

def parse_name(text):
    if not text: return None
    text = text.strip()
    m = re.search(r'["\'`]?name["\'`]?\s*:\s*["\']([^"\'(\s,]+)', text)
    if m: return m.group(1)
    m = re.search(r'\[?\s*([A-Za-z]\w*(?:\.\w+)*)\s*\(', text)
    if m: return m.group(1)
    return None

# ================= GENERATORS =================
@torch.no_grad()
def gen_plain(gpt: GPT2, prompt, max_new=40):
    ids = encode(prompt)
    ids = ids[-1000:] if len(ids) > 1000 else list(ids)
    L = len(ids)
    for _ in range(max_new):
        if len(ids) >= 1024: break
        ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        logits, _ = gpt(ii)
        nxt = int(logits[0, -1, :].argmax().item())
        ids.append(nxt)
        if nxt == encode("}")[0] or nxt == encode("\n")[0] or nxt == encode(")")[0]: break
    return decode(ids[L:]).strip()

@torch.no_grad()
def gen_adapter(model: FullSteeringGPT2, prompt, max_new=40):
    ids = encode(prompt)
    ids = ids[-1500:] if len(ids) > 1500 else list(ids)
    L = len(ids)
    for _ in range(max_new):
        if len(ids) >= model.gpt.wpe.weight.shape[0]: break
        ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        mm = torch.zeros((1, len(ids)), dtype=torch.float32, device=DEVICE)
        mm[:, :L] = 1.0
        logits, _, _ = model(ii, mm)
        nxt = int(logits[0, -1, :].argmax().item())
        ids.append(nxt)
        if nxt == encode("}")[0] or nxt == encode("\n")[0] or nxt == encode(")")[0]: break
    return decode(ids[L:]).strip()

# ================= SCORERS =================
def check_simple(pred, item): return pred == item["gold_name"]
def check_multi(pred, item): return pred == item["gold_name"]
def check_irrel(pred, item):
    funcs = item["function"]
    if isinstance(funcs, dict): funcs = [funcs]
    fn_names = [f.get("name", "") for f in funcs]
    return (pred is None) or (pred not in fn_names)
def check_parallel(pred, item):
    gold_calls = item.get("gold_calls", [])
    if not gold_calls: return False
    first_gold = gold_calls[0].get("name", "")
    return pred == first_gold

# ================= BENCH =================
def eval_set(name, items, gen_fn, build_fn, check_fn, smoke=False):
    correct = 0
    total = 0
    t0 = time.time()
    for i, item in enumerate(items):
        prompt = build_fn(item)
        gen = gen_fn(prompt)
        pred = parse_name(gen)
        if check_fn(pred, item): correct += 1
        total += 1
        # выводим каждую итерацию в smoke-mode, иначе каждые 50
        if smoke or (i+1) % 50 == 0:
            print(f"    [{name}] {i+1}/{len(items)}  acc={correct/(i+1)*100:.1f}%  t={time.time()-t0:.0f}s", flush=True)
    return correct, total

def run_all_benches(label, gen_fn, smoke=False):
    simple, mult, par, irr = load_all()
    if smoke:
        simple, mult, par, irr = simple[:20], mult[:5], par[:5], irr[:5]
    print(f"\n  [{label}]  simple={len(simple)}  multiple={len(mult)}  parallel={len(par)}  irrelevance={len(irr)}")
    r = {}
    if simple:
        c, n = eval_set(f"{label}/simple", simple, gen_fn, build_prompt_simple, check_simple, smoke=smoke)
        r["simple"] = (c, n)
    if mult:
        c, n = eval_set(f"{label}/multiple", mult, gen_fn, build_prompt_multi, check_multi, smoke=smoke)
        r["multiple"] = (c, n)
    if par:
        c, n = eval_set(f"{label}/parallel", par, gen_fn, build_prompt_multi, check_parallel, smoke=smoke)
        r["parallel"] = (c, n)
    if irr:
        c, n = eval_set(f"{label}/irrelevance", irr, gen_fn, build_prompt_multi, check_irrel, smoke=smoke)
        r["irrelevance"] = (c, n)
    return r

# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Fast smoke test (20+5+5+5 items/model)")
    args = ap.parse_args()

    # включаем smoke-mode по умолчанию для CPU
    smoke_flag = args.smoke
    mode = "SMOKE" if smoke_flag else "FULL"
    print(f"[FRESH BENCH — 4 models, {mode} mode]")

    all_results = {}

    # 1. plain GPT-2
    print("\n=== Loading plain GPT-2 ===")
    gpt_plain = GPT2()
    load_gpt2_torch_weights(gpt_plain)
    gpt_plain.to(DEVICE); gpt_plain.eval()
    all_results["plain"] = run_all_benches("plain", gen_fn=lambda p: gen_plain(gpt_plain, p), smoke=smoke_flag)
    del gpt_plain

    # 2. adapter only
    print("\n=== Loading adapter (frozen GPT-2 + h13_ep1) ===")
    m = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
    load_gpt2_torch_weights(m.gpt)
    load_classifier_from_npz(m, str(Path(__file__).resolve().parent.parent / "weights" / "adapter_torch_EN_BFCL.npz"))
    m.adapter.load_state_dict(torch.load(str(Path(__file__).resolve().parent.parent / "weights" / "adapter_h13_bfcl_ep1.pt"), map_location='cpu'))
    interpolate_wpe(m.gpt, 2048)
    m.freeze_gpt(); m.to(DEVICE); m.eval()
    all_results["adapter"] = run_all_benches("adapter", gen_fn=lambda p: gen_adapter(m, p), smoke=smoke_flag)
    del m

    # 3. FT GPT-2 alone
    print("\n=== Loading FT GPT-2 ===")
    gpt_ft = GPT2()
    gpt_ft.load_state_dict(torch.load(str(Path(__file__).resolve().parent.parent / "weights" / "gpt2_ft_final.pt"), map_location='cpu'))
    gpt_ft.to(DEVICE); gpt_ft.eval()
    all_results["ft"] = run_all_benches("ft", gen_fn=lambda p: gen_plain(gpt_ft, p), smoke=smoke_flag)
    del gpt_ft

    # 4. FT GPT-2 + adapter
    print("\n=== Loading FT GPT-2 + adapter ===")
    m2 = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
    m2.gpt.load_state_dict(torch.load(str(Path(__file__).resolve().parent.parent / "weights" / "gpt2_ft_final.pt"), map_location='cpu'))
    load_classifier_from_npz(m2, str(Path(__file__).resolve().parent.parent / "weights" / "adapter_torch_EN_BFCL.npz"))
    m2.adapter.load_state_dict(torch.load(str(Path(__file__).resolve().parent.parent / "weights" / "adapter_h13_bfcl_ep1.pt"), map_location='cpu'))
    m2.freeze_gpt(); m2.to(DEVICE); m2.eval()
    all_results["ft_adapter"] = run_all_benches("ft_adapter", gen_fn=lambda p: gen_adapter(m2, p), smoke=smoke_flag)
    del m2

    # ===== SUMMARY =====
    print(f"\n{'='*80}")
    print(f"  FRESH BENCH RESULTS — no leakage, novel function names")
    print(f"{'='*80}")
    subsets = ["simple", "multiple", "parallel", "irrelevance"]
    header = f"  {'subset':<14}" + "".join(f"{m:>14}" for m in ["plain", "adapter", "ft", "ft_adapter"])
    print(header)
    for sub in subsets:
        row = f"  {sub:<14}"
        for var in ["plain", "adapter", "ft", "ft_adapter"]:
            if sub in all_results.get(var, {}):
                c, n = all_results[var][sub]
                row += f"{c/max(n,1)*100:>11.1f}%({n:>2})"
            else:
                row += f"{'':>14}"
        print(row)
    row = f"  {'TOTAL':<14}"
    for var in ["plain", "adapter", "ft", "ft_adapter"]:
        if all_results.get(var):
            c = sum(v[0] for v in all_results[var].values())
            n = sum(v[1] for v in all_results[var].values())
            row += f"{c/max(n,1)*100:>11.1f}%({n:>2})"
        else:
            row += f"{'':>14}"
    print(row)

    # ===== SAVE RESULTS =====
    results_file = Path(__file__).resolve().parent.parent / "results" / "fresh_bench_4way_results.json"
    results_file.parent.mkdir(exist_ok=True)
    results_file.write_text(
        json.dumps({
            k: {sub: {"correct": v[0], "n": v[1]} for sub, v in r.items()}
            for k, r in all_results.items()
        }, indent=2),
        encoding='utf-8'
    )
    print(f"\nSaved -> fresh_bench_4way_results.json")

if __name__ == "__main__":
    main()
