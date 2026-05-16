"""H13: train adapter на BFCL training data — закрытие industry namespace gap.

BFCL имена (`math.factorial`, `uber.ride`, `github_star`) не в Glaive/xLAM.
Adapter их не знает. Решение: тренировать на BFCL pairs.

Resume h12_ep5 (multi-turn agent on PI 2K). Train ~600 BFCL prompts.
W_steer-only, lr=3e-5, 4 epochs save every.

Glaive anchor 50% — preserve multi-turn + glaive (tradition).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, json, re, time, random
from pathlib import Path
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, "code")

from integrated_gpt2_torch import load_gpt2_torch_weights, encode, decode
from steering_v2 import FullSteeringGPT2, load_classifier_from_npz
from long_context_pi_chunk import interpolate_wpe

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

CKPT_DIR = Path(__file__).resolve().parent.parent / "weights"
RESUME_PT = CKPT_DIR / "adapter_h12_mt_anchor_ep5.pt"
RESUME_NPZ = CKPT_DIR / "adapter_torch_EN_BFCL.npz"
BFCL_DATA = Path(os.environ.get("BFCL_DATA", "./data/bfcl_v4"))

PI_TARGET = 2048
PAD = 2048
BATCH = 2
EPOCHS = 4
LR = 3e-5


def build_bfcl_training_pairs():
    """Build (prompt, gold) pairs from BFCL simple_python + live_simple."""
    pairs = []
    for fname in ["BFCL_v4_simple_python.json", "BFCL_v4_live_simple.json"]:
        questions_path = BFCL_DATA / fname
        answers_path = BFCL_DATA / "possible_answer" / fname
        if not (questions_path.exists() and answers_path.exists()):
            continue
        with open(questions_path, encoding='utf-8') as f:
            questions = [json.loads(l) for l in f if l.strip()]
        with open(answers_path, encoding='utf-8') as f:
            answers = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in f if l.strip()}

        for q in questions:
            qid = q["id"]
            if qid not in answers: continue
            gold = answers[qid]
            if not isinstance(gold, list) or not gold: continue
            gold_dict = gold[0]
            gold_name = list(gold_dict.keys())[0]
            gold_args_options = gold_dict[gold_name]

            # Extract user content
            qq = q["question"][0]
            content = qq[0].get("content", "") if isinstance(qq, list) and qq else str(qq)
            if not content: continue

            # Function specs
            func_specs = q.get("function", [])
            if isinstance(func_specs, dict): func_specs = [func_specs]
            if not func_specs: continue

            # Build Glaive-style prompt
            fn_json = json.dumps(func_specs[0], indent=2)[:600]
            prompt = (
                f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
                f"{fn_json}\n\n\n"
                f"USER: {content}\n\n\n"
                f"ASSISTANT: <functioncall> "
            )

            # Build gold call (берём первое не-empty значение из options для каждого arg)
            gold_args = {}
            for arg, options in gold_args_options.items():
                if not options: continue
                # First non-empty
                val = next((v for v in options if v != ""), options[0] if options else None)
                if val is None or val == "": continue
                gold_args[arg] = val

            gold_call = json.dumps({"name": gold_name, "arguments": gold_args}, separators=(",", ":"))
            gold_str = gold_call

            pairs.append({"prompt": prompt, "gold": gold_str})
    return pairs


def load_glaive_anchor(rng, n_total=200):
    """Anchor data: Glaive in same format."""
    out = []
    gp = Path(os.environ.get("GLAIVE_FILE", "./data/glaive/glaive-function-calling-v2.json"))
    with open(gp, encoding='utf-8') as f:
        glaive = json.load(f)
    for ex in glaive:
        if len(out) >= n_total: break
        chat = ex.get("chat", "")
        sys_match = re.search(r"SYSTEM:.*?(?=USER:)", chat, re.DOTALL)
        if not sys_match: continue
        um = re.search(r"USER:\s*(.+?)(?:\n\n\nASSISTANT:|$)", chat, re.DOTALL)
        if not um: continue
        prompt_user = um.group(1).strip()[:300]
        fm = re.search(r"<functioncall>\s*(\{.+?\})\s*<\|endoftext\|>", chat, re.DOTALL)
        if not fm: continue
        try: d = json.loads(fm.group(1).replace("'", '"'))
        except: continue
        name = d.get("name", "")
        if not name: continue
        # Mimic prompt format
        prompt = sys_match.group(0).strip() + f"\n\nUSER: {prompt_user}\n\n\nASSISTANT: <functioncall> "
        gold_str = json.dumps({"name": name, "arguments": d.get("arguments", {})}, separators=(",", ":"))
        out.append({"prompt": prompt, "gold": gold_str})
    return out


def make_batch(samples, indices, pad=PAD):
    B = len(indices)
    input_ids = torch.zeros((B, pad), dtype=torch.long)
    mask_pool = torch.zeros((B, pad), dtype=torch.float32)
    labels = torch.full((B, pad), -100, dtype=torch.long)
    for i, idx in enumerate(indices):
        s = samples[idx]
        prompt_ids = encode(s["prompt"])
        gold_ids = encode(s["gold"])[:60]   # full JSON call may be long
        max_prompt = pad - len(gold_ids)
        prompt_ids = prompt_ids[-max_prompt:]
        seq = prompt_ids + gold_ids
        T = len(seq)
        v_len = len(gold_ids)
        v_start = T - v_len
        input_ids[i, :T] = torch.tensor(seq, dtype=torch.long)
        mask_pool[i, :v_start] = 1.0
        labels[i, v_start:T] = torch.tensor(gold_ids, dtype=torch.long)
    return input_ids.to(DEVICE), mask_pool.to(DEVICE), labels.to(DEVICE)


@torch.no_grad()
def eval_bfcl(model, samples, n=30):
    """Quick eval — proper BFCL prompt + extract from generation."""
    model.eval()
    name_correct = 0; full_correct = 0; tok_total = 0; tok_ok = 0
    for s in samples[:n]:
        prompt_ids = encode(s["prompt"])
        gold_ids = encode(s["gold"])[:30]
        max_prompt = PAD - len(gold_ids) - 5
        prompt_ids = prompt_ids[-max_prompt:]
        cur = list(prompt_ids); gen = []
        for _ in range(min(len(gold_ids), 25)):
            ii = torch.tensor([cur], dtype=torch.long, device=DEVICE)
            mm = torch.zeros((1, len(cur)), dtype=torch.float32, device=DEVICE)
            mm[:, :len(prompt_ids)] = 1.0
            logits, _, _ = model(ii, mm)
            nxt = int(logits[0, -1, :].argmax().item())
            gen.append(nxt); cur.append(nxt)
        if gen and gold_ids and gen[0] == gold_ids[0]: name_correct += 1
        # Full match: first 10 tokens (включая имя)
        if len(gen) >= 10 and len(gold_ids) >= 10 and all(gen[k] == gold_ids[k] for k in range(10)):
            full_correct += 1
        for g, p in zip(gold_ids, gen):
            tok_total += 1
            if g == p: tok_ok += 1
    return name_correct/n, full_correct/n, tok_ok/max(tok_total,1)


def main():
    print(f"[H13] BFCL train, resume h12_ep5, PI {PI_TARGET}")
    model = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
    load_gpt2_torch_weights(model.gpt)
    load_classifier_from_npz(model, str(RESUME_NPZ))
    model.adapter.load_state_dict(torch.load(str(RESUME_PT), map_location='cpu'))
    print(f"  Resumed h12_ep5")
    interpolate_wpe(model.gpt, PI_TARGET)
    model.freeze_gpt(); model.to(DEVICE)
    for name, p in model.adapter.named_parameters():
        p.requires_grad = ("W_steer" in name)

    print("Building BFCL training pairs...")
    rng = random.Random(42)
    bfcl_pairs = build_bfcl_training_pairs()
    print(f"  BFCL pairs: {len(bfcl_pairs)}")
    glaive_anchor = load_glaive_anchor(rng, n_total=300)
    print(f"  Glaive anchor: {len(glaive_anchor)}")
    train = bfcl_pairs + glaive_anchor
    rng.shuffle(train)
    eval_s = bfcl_pairs[-50:] if len(bfcl_pairs) > 50 else bfcl_pairs

    # Show example
    print(f"\n  Example training pair:")
    print(f"  prompt: {train[0]['prompt'][:200]}...")
    print(f"  gold: {train[0]['gold'][:120]}")

    # Baseline
    n1, f3, tok = eval_bfcl(model, eval_s)
    print(f"\n  BASELINE: name@1={n1*100:.1f}%  full10={f3*100:.1f}%  tok={tok*100:.1f}%")
    torch.save(model.adapter.state_dict(), CKPT_DIR / "adapter_h13_bfcl_ep0.pt")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    rng2 = random.Random(123); indices = list(range(len(train)))

    print(f"\n=== Training {EPOCHS} epochs ===")
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); rng2.shuffle(indices)
        ep_loss = 0.0; n_b = 0
        for start in range(0, len(indices), BATCH):
            bi = indices[start:start+BATCH]
            ii, mm, lb = make_batch(train, bi)
            opt.zero_grad()
            logits, _, _ = model(ii, mm)
            shift_logits = logits[:, :-1, :].contiguous(); shift_labels = lb[:, 1:].contiguous()
            valid = (shift_labels != -100)
            if not valid.any(): continue
            _, _, V_ = shift_logits.shape
            loss = F.cross_entropy(shift_logits.reshape(-1, V_), shift_labels.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1
            if n_b % 50 == 0:
                print(f"  ep{ep} batch {n_b}/{len(indices)//BATCH+1} loss={loss.item():.3f} t={time.time()-t0:.0f}s", flush=True)
        ep_path = CKPT_DIR / f"adapter_h13_bfcl_ep{ep}.pt"
        torch.save(model.adapter.state_dict(), ep_path)
        n1, f3, tok = eval_bfcl(model, eval_s)
        elapsed = time.time() - t0
        print(f"\nep{ep}: avg_loss={ep_loss/max(n_b,1):.3f}  name@1={n1*100:.1f}%  full10={f3*100:.1f}%  tok={tok*100:.1f}%  ({elapsed:.0f}s) [SAVED → {ep_path.name}]", flush=True)

    print(f"\n[H13] DONE.")


if __name__ == "__main__":
    main()
