"""Resolve the real cause on the ACTUAL switch path (QuantizedSwitchLinear.__call__ w/ mode +
sorted_indices), testing BOTH hypotheses: (1) float16-scales slow path, (2) unsorted/scattered
gather indices. Answers 'requantize?' with data. No rebuild."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from runtime import load_minimax_m3

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
gp = model.model.layers[10].mlp.switch_mlp.gate_proj
x = mx.random.normal((1,1,6144)).astype(mx.bfloat16)
inds_u = mx.array([[[88,3,61,27]]], dtype=mx.uint32)     # realistic UNSORTED top-4
inds_s = mx.sort(inds_u, axis=-1)                          # sorted
print("gp.mode=%s scales=%s bits=%s gs=%s" % (
    getattr(gp,"mode","?"), gp.scales.dtype, gp.bits, gp.group_size))

def bench(fn, n=300):
    for _ in range(15): mx.eval(fn())
    t0=time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter()-t0)/n*1000

print("\n=== REAL switch path: gp(x, indices, sorted_indices) ===")
print("fp16 scales, UNsorted inds        : %.4f ms" % bench(lambda: gp(x, inds_u)))
print("fp16 scales, sorted inds (sorted=T): %.4f ms" % bench(lambda: gp(x, inds_s, sorted_indices=True)))

# convert scales/biases -> bfloat16 in place, verify
gp.scales = gp.scales.astype(mx.bfloat16)
if gp.biases is not None: gp.biases = gp.biases.astype(mx.bfloat16)
mx.eval(gp.parameters())
assert gp.scales.dtype == mx.bfloat16, "convert failed"
print("\n-- after scales->bfloat16 (verified %s) --" % gp.scales.dtype)
print("bf16 scales, UNsorted inds        : %.4f ms" % bench(lambda: gp(x, inds_u)))
print("bf16 scales, sorted inds (sorted=T): %.4f ms" % bench(lambda: gp(x, inds_s, sorted_indices=True)))

# also raw gather_qmm WITH the real mode + sorted_indices to match the microbench discrepancy
gp2 = model.model.layers[11].mlp.switch_mlp.gate_proj   # fresh fp16 one
def raw(w,s,b,inds,sort,mode,n=300):
    for _ in range(15): mx.eval(mx.gather_qmm(x,w,s,b,rhs_indices=inds,transpose=True,group_size=128,bits=2,mode=mode,sorted_indices=sort))
    t0=time.perf_counter()
    for _ in range(n): mx.eval(mx.gather_qmm(x,w,s,b,rhs_indices=inds,transpose=True,group_size=128,bits=2,mode=mode,sorted_indices=sort))
    return (time.perf_counter()-t0)/n*1000
m = getattr(gp2, "mode", None)
print("\n=== raw gather_qmm w/ real mode=%s ===" % m)
print("fp16, unsorted, mode set : %.4f ms" % raw(gp2.weight, gp2.scales, gp2.biases, inds_u, False, m))
print("fp16, no-mode (default)  : %.4f ms" % bench(lambda: mx.gather_qmm(x,gp2.weight,gp2.scales,gp2.biases,rhs_indices=inds_u,transpose=True,group_size=128,bits=2)))
print("\n=> compare: which combo is fast tells us scales-dtype vs sort vs mode is the lever.")
