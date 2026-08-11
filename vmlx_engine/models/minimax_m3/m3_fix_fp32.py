"""Test FLOAT32 (not bf16) switch_mlp scales in the full pipelined decode — the one untested
combo. Synthetic (fp32 scales, pipelined) was fast; real (fp16) is slow; bf16 e2e didn't help."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from runtime import load_minimax_m3, _sample

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
ids = tok.encode(tok.apply_chat_template([{"role":"user","content":"Write a palindrome checker in Python."}],
                                         add_generation_prompt=True, tokenize=False), add_special_tokens=False)
def dspeed(ndec=48):
    cache=model.make_cache(); logits=model(mx.array([ids]),cache=cache); mx.eval(logits); out=[]
    for _ in range(4):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt); logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits)
    ts=[]
    for _ in range(ndec):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt)
        t0=time.perf_counter(); logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts), tok.decode(out[:26]).replace(chr(10)," ")[:100]

base,_=dspeed(); print("BEFORE (fp16 scales)        : %.1f ms  %.1f tok/s" % (base,1000/base))

def conv(dt):
    n=0
    for L in model.model.layers:
        mods=[L.self_attn.q_proj,L.self_attn.k_proj,L.self_attn.v_proj,L.self_attn.o_proj]
        mlp=L.mlp
        if hasattr(mlp,"switch_mlp"):
            mods+=[mlp.switch_mlp.gate_proj,mlp.switch_mlp.up_proj,mlp.switch_mlp.down_proj,
                   mlp.shared_experts.gate_proj,mlp.shared_experts.up_proj,mlp.shared_experts.down_proj]
        else:
            mods+=[mlp.gate_proj,mlp.up_proj,mlp.down_proj]
        for sub in mods:
            if getattr(sub,"scales",None) is not None and sub.scales.dtype!=dt:
                sub.scales=sub.scales.astype(dt)
                if getattr(sub,"biases",None) is not None: sub.biases=sub.biases.astype(dt)
                n+=1
    mx.eval(model.parameters()); return n

n=conv(mx.float32)
gp=model.model.layers[10].mlp.switch_mlp.gate_proj
print("converted %d -> float32 (verify %s), resident %.1fGB" % (n, gp.scales.dtype, mx.get_active_memory()/1e9))
fix,txt=dspeed(); print("AFTER  (float32 scales)     : %.1f ms  %.1f tok/s  (%.2fx)" % (fix,1000/fix,base/fix))
print("sample:", txt)
