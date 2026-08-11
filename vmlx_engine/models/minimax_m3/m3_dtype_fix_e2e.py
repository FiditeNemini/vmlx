"""PROVE the fix end-to-end: convert all quantized scales/biases float16->bfloat16 on the loaded
real M3 model and re-measure full decode tok/s. Expect ~10.3 -> ~20 tok/s."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from runtime import load_minimax_m3, _sample

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
templated = tok.apply_chat_template([{"role":"user","content":"Write a palindrome checker in Python."}],
                                    add_generation_prompt=True, tokenize=False)
ids = tok.encode(templated, add_special_tokens=False)

def decode_speed(ndec=48):
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache); mx.eval(logits)
    out=[]
    for _ in range(4):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt)
        logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits)
    ts=[]
    for _ in range(ndec):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt)
        t0=time.perf_counter(); logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits)
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts), tok.decode(out[:30]).replace(chr(10)," ")[:120]

base, txt = decode_speed()
print("BEFORE (float16 scales): %.1f ms/tok  %.1f tok/s" % (base, 1000/base))

# convert ALL quantized scales/biases float16 -> bfloat16 in place
import collections
cnt = collections.Counter()
def convert(module):
    for name, child in module.children().items():
        if isinstance(child, list):
            for c in child:
                if isinstance(c, nn.Module): convert(c)
            continue
        if isinstance(child, nn.Module):
            if hasattr(child, "scales") and child.scales is not None and child.scales.dtype == mx.float16:
                child.scales = child.scales.astype(mx.bfloat16); cnt["scales"] += 1
                if getattr(child, "biases", None) is not None and child.biases.dtype == mx.float16:
                    child.biases = child.biases.astype(mx.bfloat16); cnt["biases"] += 1
            convert(child)
convert(model)
mx.eval(model.parameters())
print("converted scales=%d biases=%d float16->bfloat16, resident %.1fGB" % (
    cnt["scales"], cnt["biases"], mx.get_active_memory()/1e9))

fix, txt2 = decode_speed()
print("AFTER  (bfloat16 scales): %.1f ms/tok  %.1f tok/s  (%.2fx)" % (fix, 1000/fix, base/fix))
print("\nsample BEFORE:", txt)
print("sample AFTER :", txt2)
print("coherent + faster?" , "YES" if (base/fix > 1.3 and len(txt2.strip())>10) else "CHECK")
