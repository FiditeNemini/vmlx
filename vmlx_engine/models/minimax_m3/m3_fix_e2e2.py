"""Definitive e2e: explicitly convert switch_mlp (+shared+attn+head) scales/biases fp16->bf16
across ALL layers, ASSERT the dtype changed, then measure full decode. Expect big speedup."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from runtime import load_minimax_m3, _sample

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
ids = tok.encode(tok.apply_chat_template([{"role":"user","content":"Write a palindrome checker in Python."}],
                                         add_generation_prompt=True, tokenize=False), add_special_tokens=False)

def decode_speed(ndec=48):
    cache = model.make_cache(); logits = model(mx.array([ids]), cache=cache); mx.eval(logits)
    out=[]
    for _ in range(4):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt)
        logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits)
    ts=[]
    for _ in range(ndec):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt)
        t0=time.perf_counter(); logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits)
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts), tok.decode(out[:28]).replace(chr(10)," ")[:110]

base,txt = decode_speed()
print("BEFORE: %.1f ms/tok  %.1f tok/s" % (base, 1000/base))

n=0
def cvt(m):
    global n
    for attr in ("gate_proj","up_proj","down_proj","q_proj","k_proj","v_proj","o_proj"):
        sub = getattr(m, attr, None)
        if sub is not None and getattr(sub,"scales",None) is not None and sub.scales.dtype==mx.float16:
            sub.scales = sub.scales.astype(mx.bfloat16)
            if getattr(sub,"biases",None) is not None and sub.biases.dtype==mx.float16:
                sub.biases = sub.biases.astype(mx.bfloat16)
            n+=1
for L in model.model.layers:
    cvt(L.self_attn)
    mlp = L.mlp
    if hasattr(mlp,"switch_mlp"):
        cvt(mlp.switch_mlp); cvt(mlp.shared_experts)
    else:
        cvt(mlp)
mx.eval(model.parameters())
gp = model.model.layers[10].mlp.switch_mlp.gate_proj
assert gp.scales.dtype==mx.bfloat16, "switch_mlp NOT converted!"
print("converted %d projections; layers[10].switch_mlp.gate_proj.scales=%s, resident %.1fGB" % (
    n, gp.scales.dtype, mx.get_active_memory()/1e9))

fix,txt2 = decode_speed()
print("AFTER : %.1f ms/tok  %.1f tok/s  (%.2fx)" % (fix, 1000/fix, base/fix))
print("sample:", txt2)
