"""Full fine-tune GPT-2 124M on tool-call data (no frozen base, no adapter).

Data: BFCL train + Glaive + xLAM held-out slice mixed.
Loss: causal LM on assistant response (mask system+user tokens with -100).
LR: 1e-5, AdamW. PAD=512. batch=1, grad_accum=4. 1 epoch over ~1500 samples.

Save every 500 steps to gpt2_ft_step{N}.pt.
"""
import os, sys, json, time, random, re, glob
from pathlib import Path
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, ".")

from integrated_gpt2_torch import GPT2, load_gpt2_torch_weights, encode, decode

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

BFCL_DATA = Path(os.environ.get("BFCL_DATA", "./data/bfcl_v4"))
GLAIVE_FILE = Path(os.environ.get("GLAIVE_FILE", "./data/glaive/glaive-function-calling-v2.json"))
XLAM_FILE = os.environ.get("XLAM_FILE", "./data/xlam/train.parquet")

OUT_DIR = Path(__file__).resolve().parent.parent / "weights"
OUT_DIR.mkdir(exist_ok=True)

PAD = 512
LR = 1e-5
BATCH = 1
GRAD_ACCUM = 4
N_SAMPLES = 1500
SAVE_EVERY = 500


def build_bfcl_pairs(n_total=500):
    pairs = []
    subsets = ["BFCL_v4_simple_python.json", "BFCL_v4_multiple.json", "BFCL_v4_parallel.json",
               "BFCL_v4_irrelevance.json"]
    per_subset = n_total // len(subsets)
    for fname in subsets:
        q_path = BFCL_DATA / fname
        a_path = BFCL_DATA / "possible_answer" / fname
        if not q_path.exists(): continue
        is_irrel = "irrelevance" in fname
        with open(q_path, encoding='utf-8') as f:
            questions = [json.loads(l) for l in f if l.strip()][:per_subset]
        answers = {}
        if a_path.exists():
            with open(a_path, encoding='utf-8') as f:
                answers = {json.loads(l)["id"]: json.loads(l).get("ground_truth", []) for l in f if l.strip()}
        for q in questions:
            qid = q["id"]
            qq = q["question"][0] if isinstance(q["question"], list) and q["question"] else q["question"]
            content = qq[0].get("content", "") if isinstance(qq, list) and qq else str(qq)
            func_specs = q.get("function", [])
            if isinstance(func_specs, dict): func_specs = [func_specs]
            if not content or not func_specs: continue
            fn_json = json.dumps(func_specs[0] if func_specs else {}, indent=2)[:500]
            prompt = (f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
                      f"{fn_json}\n\n\nUSER: {content[:300]}\n\n\nASSISTANT: <functioncall> ")
            if is_irrel:
                gold = '{"name":"none","arguments":{}}'
            else:
                if qid not in answers or not answers[qid]: continue
                g = answers[qid][0]
                if not isinstance(g, dict): continue
                gn = list(g.keys())[0]
                gargs = {}
                for ar, opts in g[gn].items():
                    if not opts: continue
                    v = next((vv for vv in opts if vv != ""), opts[0])
                    if v in (None, ""): continue
                    gargs[ar] = v
                gold = json.dumps({"name": gn, "arguments": gargs}, separators=(",", ":"))
            pairs.append({"prompt": prompt, "gold": gold})
    return pairs


def build_glaive_pairs(n=500):
    with open(GLAIVE_FILE, encoding='utf-8') as f:
        glaive = json.load(f)
    pairs = []
    for ex in glaive[100:5000]:
        if len(pairs) >= n: break
        st = ex.get("system", "")
        chat = ex.get("chat", "")
        um = re.search(r"USER:\s*(.+?)(?:\n\n\nASSISTANT:|$)", chat, re.DOTALL)
        fm = re.search(r"<functioncall>\s*(\{.+?\})\s*<\|endoftext\|>", chat, re.DOTALL)
        if not (um and fm and st): continue
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', fm.group(1))
        if not nm: continue
        gold = fm.group(1).replace("'", '"')[:200]
        # ensure valid
        gn = nm.group(1)
        gold_str = json.dumps({"name": gn, "arguments": {}}, separators=(",", ":"))
        prompt = st.strip() + f"\n\nUSER: {um.group(1).strip()[:300]}\n\n\nASSISTANT: <functioncall> "
        pairs.append({"prompt": prompt, "gold": gold_str})
    return pairs


def build_xlam_pairs(n=500):
    import pandas as pd
    df = pd.read_parquet(XLAM_FILE)
    pairs = []
    # use middle slice to avoid both train-anchor (start) and held-out eval (tail)
    for i in range(20000, 30000):
        if len(pairs) >= n: break
        msgs = df.iloc[i]['messages']
        sys_t = ""; user_t = ""; gold_name = ""
        for m in msgs:
            if m['role'] == 'system': sys_t = m['content']
            elif m['role'] == 'user': user_t = m['content']
            elif m['role'] == 'assistant':
                mm = re.search(r"<tool_call>\s*\{['\"]tool_name['\"]\s*:\s*['\"]([^'\"]+)['\"]", m['content'])
                if mm: gold_name = mm.group(1)
                break
        if not gold_name or not user_t: continue
        prompt = (f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
                  f"{sys_t[:500]}\n\n\nUSER: {user_t[:300]}\n\n\nASSISTANT: <functioncall> ")
        gold_str = json.dumps({"name": gold_name, "arguments": {}}, separators=(",", ":"))
        pairs.append({"prompt": prompt, "gold": gold_str})
    return pairs


def make_batch(samples, indices):
    B = len(indices)
    input_ids = torch.zeros((B, PAD), dtype=torch.long)
    labels = torch.full((B, PAD), -100, dtype=torch.long)
    for i, idx in enumerate(indices):
        s = samples[idx]
        prompt_ids = encode(s["prompt"])
        gold_ids = encode(s["gold"])[:60]
        max_prompt = PAD - len(gold_ids)
        prompt_ids = prompt_ids[-max_prompt:]
        seq = prompt_ids + gold_ids
        T = len(seq)
        v_start = T - len(gold_ids)
        input_ids[i, :T] = torch.tensor(seq, dtype=torch.long)
        labels[i, v_start:T] = torch.tensor(gold_ids, dtype=torch.long)
    return input_ids.to(DEVICE), labels.to(DEVICE)


def main():
    print("[FULL FT GPT-2 124M] no freeze, no adapter")
    print("Loading base GPT-2 124M...")
    model = GPT2()
    load_gpt2_torch_weights(model)
    model.to(DEVICE); model.train()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_p:,}  (all trainable)")

    print("Building train mix...")
    bfcl = build_bfcl_pairs(500)
    print(f"  BFCL: {len(bfcl)}")
    glaive = build_glaive_pairs(500)
    print(f"  Glaive: {len(glaive)}")
    xlam = build_xlam_pairs(500)
    print(f"  xLAM: {len(xlam)}")
    train = bfcl + glaive + xlam
    rng = random.Random(42); rng.shuffle(train)
    train = train[:N_SAMPLES]
    print(f"  TOTAL: {len(train)} samples")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"\n=== Training ===  PAD={PAD} batch={BATCH}x{GRAD_ACCUM}  LR={LR}  steps={len(train)//BATCH}")
    indices = list(range(len(train)))
    rng.shuffle(indices)
    t0 = time.time()
    step = 0
    accum = 0
    running_loss = 0.0
    opt.zero_grad()
    for i in range(0, len(indices), BATCH):
        bi = indices[i:i+BATCH]
        ii, lb = make_batch(train, bi)
        logits, _ = model(ii)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = lb[:, 1:].contiguous()
        valid = (shift_labels != -100)
        if not valid.any(): continue
        _, _, V = shift_logits.shape
        loss = F.cross_entropy(shift_logits.reshape(-1, V), shift_labels.reshape(-1), ignore_index=-100)
        loss_normalized = loss / GRAD_ACCUM
        loss_normalized.backward()
        running_loss += loss.item()
        accum += 1
        if accum == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 20 == 0:
                avg = running_loss / (accum * 20)
                print(f"  step {step}  loss={loss.item():.3f}  avg20={avg:.3f}  t={time.time()-t0:.0f}s", flush=True)
                running_loss = 0.0
            if step % SAVE_EVERY == 0:
                out = OUT_DIR / f"gpt2_ft_step{step}.pt"
                torch.save(model.state_dict(), out)
                print(f"  [SAVED -> {out.name}]")
            accum = 0

    # Final
    out = OUT_DIR / f"gpt2_ft_final.pt"
    torch.save(model.state_dict(), out)
    print(f"\nDONE. Final saved -> {out.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
