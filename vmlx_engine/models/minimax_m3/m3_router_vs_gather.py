"""Split the MoE 73% into ROUTER (gate+group_expert_select top-4 of 100) vs GATHER (switch_mlp).
Whichever skip speeds decode most is the REAL bottleneck."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import minimax_m3 as M
from runtime import load_minimax_m3, _sample

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
ids = tok.encode(tok.apply_chat_template([{"role":"user","content":"Write a palindrome checker."}],
                                         add_generation_prompt=True, tokenize=False), add_special_tokens=False)
def dspeed(ndec=40):
    cache = model.make_cache(); logits = model(mx.array([ids]), cache=cache); mx.eval(logits); out=[]
    for _ in range(4):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt); logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits)
    ts=[]
    for _ in range(ndec):
        nxt=_sample(logits[0,-1],0.0,1.0,ids+out,1.0); out.append(nxt)
        t0=time.perf_counter(); logits=model(mx.array([[nxt]]),cache=cache); mx.eval(logits); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

orig = M.SparseMoeBlock.__call__
base = dspeed(); print("baseline                       : %.1f ms  %.1f tok/s" % (base, 1000/base))

# fixed top-4 inds (skip group_expert_select compute), KEEP gather
FIX = mx.array([[[0,1,2,3]]], dtype=mx.uint32)
def only_gather(self, x):
    B,L,_ = x.shape
    inds = mx.broadcast_to(FIX, (B,L,4))
    y = self.switch_mlp(x, inds).sum(axis=-2)
    return y + self.shared_experts(x)
M.SparseMoeBlock.__call__ = only_gather
g = dspeed(); print("skip ROUTER (fixed inds, gather):  %.1f ms  (router cost=%.1f = %.0f%%)" % (g, base-g, 100*(base-g)/base))

# run router/topk, SKIP gather (no switch_mlp)
def only_router(self, x):
    inds, scores = M.group_expert_select(self.gate(x), self.e_score_correction_bias, self.top_k, 1, 1,
                                         self.routed_scaling_factor, self.norm_topk_prob)
    return scores.sum(axis=-1, keepdims=True) * x * 0.0 + self.shared_experts(x)
M.SparseMoeBlock.__call__ = only_router
r = dspeed(); print("skip GATHER (router only)       :  %.1f ms  (gather cost=%.1f = %.0f%%)" % (r, base-r, 100*(base-r)/base))

# skip both (shared only)
M.SparseMoeBlock.__call__ = lambda self, x: self.shared_experts(x)
sh = dspeed(); print("skip BOTH (shared only)         :  %.1f ms  (both=%.1f = %.0f%%)" % (sh, base-sh, 100*(base-sh)/base))
M.SparseMoeBlock.__call__ = orig
print("\n=> router cost vs gather cost reveals the real lever (routing top-k vs the matmul gather).")
