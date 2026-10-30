"""Export a training checkpoint as a small fp16 inference file.
usage (from the repository root): python tools/export.py run/best.pt kadmos-1.pt
"""
import sys, os, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, model

sys.modules.setdefault("model2", model)  # checkpoints written while the module was called model2

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
out = {"model": {k: v.half() for k, v in ck["model"].items()},
       "cfg": dataclasses.asdict(ck["cfg"])}
torch.save(out, sys.argv[2])
print("wrote", sys.argv[2], f"({os.path.getsize(sys.argv[2])/1e6:.0f} MB)")
