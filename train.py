"""Overnight trainer: mixed stories / python / chat data, time-budgeted cosine schedule, resumable.

usage: train.py --hours 8 [--dim 512 --layers 12 --heads 8] [--resume] [--bench]
Stage 1 (first 80% of the budget): broad mix.  Stage 2 (last 20%): mostly chat + code Q&A, LR decaying to ~0.
"""
import argparse, json, math, os, time
import numpy as np, torch
from tokenizers import Tokenizer
from model import GPT, Config, END, EOT

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=8)
p.add_argument("--dim", type=int, default=512)
p.add_argument("--layers", type=int, default=12)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--ctx", type=int, default=512)
p.add_argument("--micro", type=int, default=16, help="sequences per micro-batch")
p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dtype", default="fp32", choices=["bf16", "fp32"])
p.add_argument("--compile", action="store_true")
p.add_argument("--resume", action="store_true")
p.add_argument("--bench", type=int, default=0, help="just time N steps and exit")
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--eval_s", type=int, default=300)
p.add_argument("--ck_s", type=int, default=600)
p.add_argument("--sample_s", type=int, default=1200)
p.add_argument("--out", default="run")
a = p.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)
tok = Tokenizer.from_file("tokenizer.json")
cfg = Config(vocab=tok.get_vocab_size(), ctx=a.ctx, layers=a.layers, heads=a.heads, dim=a.dim, dropout=a.dropout)
BUDGET, STAGE2_AT, WARMUP = a.hours * 3600, 0.8, 150
os.makedirs(a.out, exist_ok=True)

data = {f"{n}_{s}": np.memmap(f"data/{n}_{s}.bin", dtype=np.uint16, mode="r")
        for n in ("stories", "code", "chat", "codeqa") for s in ("train", "val")}
# chat = prose conversation (the priority); codeqa = "write a function..." chats
MIX1 = {"stories": 0.25, "code": 0.15, "codeqa": 0.10, "chat": 0.50}
MIX2 = {"stories": 0.05, "code": 0.05, "codeqa": 0.15, "chat": 0.75}

model = GPT(cfg).to(dev)
n_params = sum(p_.numel() for p_ in model.parameters())
print(f"{n_params/1e6:.1f}M params, {dev}, {a.dtype}, batch {a.micro*a.accum*a.ctx/1e3:.0f}k tok/step, "
      f"budget {a.hours}h", flush=True)
decay = [p_ for n, p_ in model.named_parameters() if p_.ndim >= 2]
nodecay = [p_ for n, p_ in model.named_parameters() if p_.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=(dev == "mps"))
fwd = torch.compile(model) if a.compile else model
amp = (lambda: torch.autocast("mps", dtype=torch.bfloat16)) if a.dtype == "bf16" else (lambda: torch.autocast("mps", enabled=False))

state = {"step": 0, "elapsed": 0.0, "tokens": 0, "best": 9e9}
CK = f"{a.out}/last.pt"
if a.resume and os.path.exists(CK):
    ck = torch.load(CK, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); state = ck["state"]
    print(f"resumed at step {state['step']}, {state['elapsed']/60:.0f} min", flush=True)


def get_batch(split, mix, n):
    names = list(mix)
    src = rng.choice(len(names), size=n, p=[mix[k] for k in names])
    rows = []
    for i in range(len(names)):
        d = data[f"{names[i]}_{split}"]
        starts = rng.integers(0, len(d) - a.ctx - 1, size=int((src == i).sum()))
        rows += [np.asarray(d[s:s + a.ctx + 1], dtype=np.int64) for s in starts]
    x = torch.from_numpy(np.stack(rows))
    return x[:, :-1].to(dev), x[:, 1:].to(dev)


@torch.no_grad()
def evaluate(n=24):
    model.eval()
    out = {}
    for name in ("stories", "code", "chat", "codeqa"):
        ls = []
        for _ in range(n):
            x, y = get_batch("val", {name: 1.0}, 8)
            with amp():
                ls.append(model(x, y)[1].item())
        out[name] = sum(ls) / n
    model.train()
    return out


PROMPTS = [("chat", "<|user|>Hi! How are you doing today?<|end|><|bot|>"),
           ("chat", "<|user|>Write a Python function that reverses a string.<|end|><|bot|>"),
           ("chat", "<|user|>What should I cook for dinner tonight?<|end|><|bot|>"),
           ("story", "Once upon a time, there was a little dog named Max."),
           ("code", "def fibonacci(n):\n")]


def samples():
    out = []
    for kind, pr in PROMPTS:
        ids = torch.tensor([tok.encode(pr).ids], device=dev)
        with amp():
            g = model.generate(ids, max_new=120, stop=(END, EOT))
        out.append(f"--- {kind}: {pr!r}\n{tok.decode(g[0, ids.shape[1]:].tolist())}")
    return "\n".join(out)


def save(path, extra=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "cfg": cfg, **(extra or {})}, path + ".tmp")
    os.replace(path + ".tmp", path)


def lr_at(frac, step):
    return a.lr * min(1.0, (step + 1) / WARMUP) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


model.train()
t_start = time.time() - state["elapsed"]
last_eval = last_ck = last_sample = time.time()
bench_t0, skipped = None, 0
EVAL_S, CK_S, SAMPLE_S = a.eval_s, a.ck_s, a.sample_s
metrics = open(f"{a.out}/metrics.jsonl", "a")
while True:
    state["elapsed"] = time.time() - t_start
    frac = state["elapsed"] / BUDGET
    if frac >= 1 or (a.bench and state["step"] >= a.bench):
        break
    mix = MIX1 if frac < STAGE2_AT else MIX2
    lr = lr_at(frac, state["step"])
    for g in opt.param_groups:
        g["lr"] = lr
    tot = 0.0
    for _ in range(a.accum):
        x, y = get_batch("train", mix, a.micro)
        with amp():
            loss = fwd(x, y)[1] / a.accum
        loss.backward()
        tot += loss.item()
    if not math.isfinite(tot):  # skip diverged batch
        opt.zero_grad(set_to_none=True); skipped += 1
        if skipped > 50:
            raise SystemExit("too many non-finite losses")
        continue
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    state["step"] += 1; state["tokens"] += a.micro * a.accum * a.ctx
    if a.bench and state["step"] == 3:
        torch.mps.synchronize(); bench_t0 = time.time()
    if state["step"] % 20 == 0:
        print(f"[{state['elapsed']/3600:5.2f}h] step {state['step']} loss {tot:.3f} lr {lr:.2e} "
              f"{state['tokens']/1e6:.0f}M tok", flush=True)
    if time.time() - last_eval > EVAL_S and not a.bench:
        last_eval = time.time()
        v = evaluate()
        score = v["chat"]
        print(f"   EVAL chat {v['chat']:.3f} stories {v['stories']:.3f} code {v['code']:.3f} codeqa {v['codeqa']:.3f}", flush=True)
        metrics.write(json.dumps({"t": state["elapsed"], "step": state["step"], "train": tot, **v}) + "\n"); metrics.flush()
        if score < state["best"]:
            state["best"] = score
            save(f"{a.out}/best.pt")
    if time.time() - last_sample > SAMPLE_S and not a.bench:
        last_sample = time.time()
        txt = samples()
        print(txt, flush=True)
        open(f"{a.out}/samples.txt", "a").write(f"\n===== step {state['step']} ({state['elapsed']/3600:.2f}h)\n{txt}\n")
    if time.time() - last_ck > CK_S and not a.bench:
        last_ck = time.time(); save(CK)

if a.bench:
    torch.mps.synchronize()
    dt = (time.time() - bench_t0) / (state["step"] - 3)
    print(f"BENCH {n_params/1e6:.1f}M params: {dt:.2f}s/step, {a.micro*a.accum*a.ctx/dt/1e3:.1f}k tok/s", flush=True)
    raise SystemExit
v = evaluate(48)
print("FINAL EVAL", v, flush=True)
save(f"{a.out}/final.pt")
if v["chat"] < state["best"]:
    save(f"{a.out}/best.pt")
open(f"{a.out}/DONE", "w").write(json.dumps(v))
print("DONE", flush=True)
