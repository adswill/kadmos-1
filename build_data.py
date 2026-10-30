"""Build the overnight training data (everything is tokenized by our own tokenizer, trained here).

Sources
  stories : TinyStories subset                        -> grammar / coherence
  code    : Python files from codeparrot-clean         -> raw code modelling
  chat    : DailyDialog, EmpatheticDialogues, OpenAssistant (short EN threads), Everyday Conversations,
            plus code Q&A (Python code instructions, CodeAlpaca, python-codes-25k)
Chat format: <|user|>text<|end|><|bot|>text<|end|>...<|endoftext|>   (text may contain newlines, so code fits)
Outputs: tokenizer.json, data/{stories,code,chat}_{train,val}.bin (uint16)
"""
import re, random, collections, numpy as np
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOT, USER, BOT, END = "<|endoftext|>", "<|user|>", "<|bot|>", "<|end|>"
SPECIAL = [EOT, USER, BOT, END]
N_STORIES, N_CODE, N_SODA, VOCAB = 600_000, 80_000, 200_000, 16384
random.seed(0)
Path("data").mkdir(exist_ok=True)


def clean(s):
    """Normalize prose (single line)."""
    s = s.replace("_comma_", ",").replace("\n", " ")
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"\s*[’‘]\s*", "'", s)
    s = re.sub(r"\s+([,.!?;:%])", r"\1", s)
    s = re.sub(r"\s+'\s*(s|t|m|d|re|ve|ll)\b", r"'\1", s)
    s = re.sub(r"(\w)\s+n't", r"\1n't", s)
    s = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", s)
    return re.sub(r"\s+", " ", s).strip()


def ok(s, max_len=100_000):
    return bool(s) and len(s) <= max_len and sum(ord(c) > 127 for c in s) <= 0.05 * len(s)


def fmt(turns):
    """turns alternate user/bot, already cleaned."""
    if len(turns) < 2 or not all(ok(t) for t in turns):
        return None
    return "".join(f"{USER if i % 2 == 0 else BOT}{t}{END}" for i, t in enumerate(turns)) + EOT


chat = {"train": [], "val": []}

# ---------- everyday chat ----------
dd = load_dataset("roskoN/dailydialog", revision="refs/convert/parquet")
for split, key in (("train", "train"), ("validation", "val")):
    chat[key] += [fmt([clean(u) for u in r["utterances"]]) for r in dd[split]]

ed = load_dataset("facebook/empathetic_dialogues", revision="refs/convert/parquet")
for split, key in (("train", "train"), ("validation", "val")):
    convs = collections.defaultdict(list)
    for r in ed[split]:
        convs[r["conv_id"]].append((r["utterance_idx"], r["utterance"]))
    chat[key] += [fmt([clean(u) for _, u in sorted(v)]) for v in convs.values()]

ev = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")
for split, key in (("train_sft", "train"), ("test_sft", "val")):
    chat[key] += [fmt([clean(m["content"]) for m in r["messages"] if m["role"] != "system"]) for r in ev[split]]

oa = load_dataset("OpenAssistant/oasst1")
for split, key in (("train", "train"), ("validation", "val")):
    msgs = {r["message_id"]: r for r in oa[split] if r["lang"] == "en"}
    has_child = {r["parent_id"] for r in msgs.values()}
    for mid, r in msgs.items():
        if mid in has_child or r["role"] != "assistant":
            continue  # one thread per leaf
        chain, cur = [], r
        while cur is not None:
            chain.append(cur)
            cur = msgs.get(cur["parent_id"]) if cur["parent_id"] else None
        texts = [clean(m["text"]) for m in reversed(chain)]
        if chain[-1]["parent_id"] is None and len(texts) % 2 == 0 and all(len(t) <= 500 for t in texts) \
                and sum(map(len, texts)) <= 1500:
            chat[key].append(fmt(texts))
# SODA: synthetic social conversations (streamed; first N only)
n = 0
for r in load_dataset("allenai/soda", split="train", streaming=True):
    t = fmt([clean(u) for u in r["dialogue"][:14]])
    if t and len(t) < 3500:
        chat["train" if n % 100 else "val"].append(t)
        n += 1
        if n >= N_SODA:
            break

# Anthropic helpful-* : human/assistant chats
for cfgname in ("helpful-base", "helpful-online", "helpful-rejection-sampled"):
    hh = load_dataset("Anthropic/hh-rlhf", data_dir=cfgname)
    for split, key in (("train", "train"), ("test", "val")):
        for r in hh[split]:
            parts = [x.strip() for x in re.split(r"\n\n(?:Human|Assistant):", r["chosen"]) if x.strip()]
            if len(parts) % 2 == 0 and r["chosen"].lstrip().startswith("Human:") \
                    and all(len(x) <= 600 for x in parts) and sum(map(len, parts)) <= 1800:
                chat[key].append(fmt([clean(x) for x in parts]))
print("chat (prose):", {k: len([t for t in v if t]) for k, v in chat.items()}, flush=True)

# ---------- code Q&A as chat ----------
FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def code_turn(instr, inp, out):
    instr = (instr or "").strip() + (("\n" + inp.strip()) if inp and inp.strip() else "")
    out = FENCE.sub("", (out or "").strip()).strip()
    if not instr or not out or len(instr) > 600 or len(out) > 1800 or not ok(out) or not ok(instr):
        return None
    return fmt([instr, f"```python\n{out}\n```"])


code_qa = []
for name, split in (("iamtarun/python_code_instructions_18k_alpaca", "train"), ("sahil2801/CodeAlpaca-20k", "train"),
                    ("flytech/python-codes-25k", "train")):
    for r in load_dataset(name, split=split):
        t = code_turn(r["instruction"], r.get("input"), r["output"])
        if t:
            code_qa.append(t)
code_qa = list(dict.fromkeys(code_qa))  # dedupe (python-codes-25k has repeats)
random.shuffle(code_qa)
codeqa = {"val": code_qa[:500], "train": code_qa[500:]}
print("code Q&A:", len(code_qa), flush=True)

chat = {k: [t for t in dict.fromkeys(v) if t] for k, v in chat.items()}
for v in chat.values():
    random.shuffle(v)
print({k: len(v) for k, v in chat.items()}, "chats total")
print(chat["train"][0], "\n---\n", chat["train"][1], "\n---", flush=True)

# ---------- stories ----------
ts = load_dataset("roneneldan/TinyStories")
stories = {"train": [clean(t) + EOT for t in ts["train"].select(range(N_STORIES))["text"]],
           "val": [clean(t) + EOT for t in ts["validation"].select(range(2000))["text"]]}

# ---------- python files (streamed, so only the shards we need are downloaded) ----------
files = []
for r in load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True):
    c = r["content"].replace("\t", "    ")
    if 300 <= len(c) <= 8000 and ok(c):
        files.append(c + EOT)
        if len(files) >= N_CODE + 1500:
            break
code = {"train": files[:N_CODE], "val": files[N_CODE:]}
print("python files:", {k: len(v) for k, v in code.items()}, flush=True)

# ---------- tokenizer ----------
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
sample = random.sample(chat["train"], min(len(chat["train"]), 120_000)) + codeqa["train"] \
    + random.sample(stories["train"], 60_000) + random.sample(code["train"], 40_000)
tok.train_from_iterator(sample, trainers.BpeTrainer(
    vocab_size=VOCAB, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.save("tokenizer.json")
assert [tok.token_to_id(s) for s in SPECIAL] == [0, 1, 2, 3]


def dump(texts, path):
    n = 0
    with open(path, "wb") as f:
        for i in range(0, len(texts), 20_000):
            ids = np.fromiter((t for e in tok.encode_batch(texts[i:i + 20_000]) for t in e.ids), dtype=np.uint16)
            ids.tofile(f); n += len(ids)
    print(f"{path}: {n/1e6:.1f}M tokens", flush=True)


for name, d in (("stories", stories), ("code", code), ("chat", chat), ("codeqa", codeqa)):
    dump(d["train"], f"data/{name}_train.bin")
    dump(d["val"], f"data/{name}_val.bin")
print("DONE", flush=True)
