"""Chat with Kadmos-1.   usage: chat.py [checkpoint.pt]   (default run/best.pt)
Commands: reset | quit | /temp 0.7 | /len 200
"""
import sys, torch
from tokenizers import Tokenizer
from model import GPT, Config, END, EOT

dev = "mps" if torch.backends.mps.is_available() else "cpu"
path = sys.argv[1] if len(sys.argv) > 1 else "run/best.pt"
ck = torch.load(path, map_location=dev, weights_only=False)
cfg = Config(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
model = GPT(cfg).to(dev)
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer.json")
temp, max_new, turns = 0.7, 200, []  # turns: list of (user_text, bot_text)

print(f"Kadmos-1 loaded from {path}. "
      "'reset' clears memory, '/temp X' changes randomness, 'quit' exits.")
while True:
    try:
        msg = input("you: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if msg == "quit": break
    if msg == "reset": turns = []; continue
    if msg.startswith("/temp"): temp = float(msg.split()[1]); continue
    if msg.startswith("/len"): max_new = int(msg.split()[1]); continue
    if not msg: continue
    # build the prompt from the most recent turns that fit in the context window
    ids = []
    for u, b in reversed(turns + [(msg, None)]):
        t = tok.encode(f"<|user|>{u}<|end|><|bot|>" + (f"{b}<|end|>" if b is not None else "")).ids
        if len(ids) + len(t) > model.c.ctx - max_new - 8 and ids: break
        ids = t + ids
    x = torch.tensor([ids], device=dev)
    out = model.generate(x, max_new=max_new, temp=temp)
    reply = tok.decode([i for i in out[0, x.shape[1]:].tolist() if i not in (EOT, END)]).strip()
    print(f"bot: {reply}")
    turns.append((msg, reply))
