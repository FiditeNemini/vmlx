"""Clean force-eval breakdown of ONE real MoE block: gate, routing, gather (real-inds vs
const-inds), shared, full — at fp16 then bf16 scales. Kills the DCE artifact."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import minimax_m3 as M
from runtime import load_minimax_m3

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
blk = model.model.layers[10].mlp           # real SparseMoeBlock
x = mx.random.normal((1,1,6144)).astype(mx.bfloat16); mx.eval(x)
inds_r, sc = M.group_expert_select(blk.gate(x), blk.e_score_correction_bias, blk.top_k,1,1,
                                   blk.routed_scaling_factor, blk.norm_topk_prob); mx.eval(inds_r, sc)
inds_c = mx.array([[[0,1,2,3]]], dtype=mx.uint32)
print("real inds:", inds_r.tolist())

def b(fn, n=400):
    for _ in range(20): mx.eval(fn())
    t0=time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter()-t0)/n*1000

print("\n=== fp16 scales (as loaded) ===")
print("gate(x)                     : %.4f" % b(lambda: blk.gate(x)))
print("group_expert_select         : %.4f" % b(lambda: M.group_expert_select(blk.gate(x), blk.e_score_correction_bias, blk.top_k,1,1, blk.routed_scaling_factor, blk.norm_topk_prob)))
print("switch_mlp(x, REAL inds)    : %.4f" % b(lambda: blk.switch_mlp(x, inds_r)))
print("switch_mlp(x, CONST inds)   : %.4f" % b(lambda: blk.switch_mlp(x, inds_c)))
print("shared_experts(x)           : %.4f" % b(lambda: blk.shared_experts(x)))
print("full block(x)               : %.4f" % b(lambda: blk(x)))

# convert switch_mlp scales -> bf16
for p in ("gate_proj","up_proj","down_proj"):
    sub = getattr(blk.switch_mlp, p)
    sub.scales = sub.scales.astype(mx.bfloat16)
    if sub.biases is not None: sub.biases = sub.biases.astype(mx.bfloat16)
mx.eval(blk.parameters())
print("\n=== bf16 switch_mlp scales ===")
print("switch_mlp(x, REAL inds)    : %.4f" % b(lambda: blk.switch_mlp(x, inds_r)))
print("switch_mlp(x, CONST inds)   : %.4f" % b(lambda: blk.switch_mlp(x, inds_c)))
print("full block(x)               : %.4f" % b(lambda: blk(x)))
