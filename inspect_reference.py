"""Record how transformers' online NF4 quantization lays out every tensor (reference for the offline converter)."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch, bitsandbytes as bnb
from sn_loop_common import load

tok, model = load("nf4", compile_sinkhorn=False)
def h(t): return hashlib.sha1(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
ref = {"quantized": {}, "other": {}}
for name, mod in model.named_modules():
    if isinstance(mod, bnb.nn.Linear4bit):
        w = mod.weight; qs = w.quant_state
        ref["quantized"][name] = dict(shape=list(qs.shape), blocksize=qs.blocksize, quant_type=qs.quant_type,
            dtype=str(qs.dtype), compress=w.compress_statistics, nested=qs.nested, data_dtype=str(w.data.dtype),
            data_shape=list(w.data.shape), absmax_dtype=str(qs.absmax.dtype), bias=mod.bias is not None,
            sha_data=h(w.data), sha_absmax=h(qs.absmax), device=str(w.device))
qnames = {n + ".weight" for n in ref["quantized"]}
for name, p in list(model.named_parameters()) + list(model.named_buffers()):
    if name in qnames: continue
    ref["other"][name] = dict(shape=list(p.shape), dtype=str(p.dtype), device=str(p.device))
json.dump(ref, open("nf4_reference.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("量化层", len(ref["quantized"]), "其他张量", len(ref["other"]))
q = next(iter(ref["quantized"].values())); print("示例量化层属性", {k: v for k, v in q.items() if not k.startswith("sha")})
import collections
c = collections.Counter((v["dtype"], v["device"]) for v in ref["other"].values()); print("其他张量 (dtype, device):", dict(c))
print("量化层前缀:", sorted({n.split(".layers.")[0] + (".layers.*." + n.split(".layers.")[1].split(".", 1)[1] if ".layers." in n else "") for n in ref["quantized"]})[:30])
print("未量化的非 visual 张量示例:", [n for n in ref["other"] if "visual" not in n][:40])
