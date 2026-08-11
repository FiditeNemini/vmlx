"""Pin WHY real expert gather (0.59ms) is 3x slower than fresh (0.19ms), same shape.
Suspect: scales/biases dtype or weight property triggers a slow gather_qmm path. FIND THE FIX."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchLinear
from runtime import load_minimax_m3

BUNDLE = sys.argv[1]
model, tok, eos = load_minimax_m3(BUNDLE)
gp = model.model.layers[10].mlp.switch_mlp.gate_proj
x = mx.random.normal((1,1,6144)).astype(mx.bfloat16)
inds = mx.array([[[3,27,61,88]]], dtype=mx.uint32)

print("REAL : weight", gp.weight.dtype, gp.weight.shape, "| scales", gp.scales.dtype, gp.scales.shape,
      "| biases", (gp.biases.dtype if gp.biases is not None else None))
q = SwitchLinear(6144,3072,100).to_quantized(group_size=128, bits=2); mx.eval(q.parameters())
print("FRESH: weight", q.weight.dtype, q.weight.shape, "| scales", q.scales.dtype, q.scales.shape,
      "| biases", (q.biases.dtype if q.biases is not None else None))

def bench(w,s,b,n=200):
    mx.eval(w,s,b)
    for _ in range(10): mx.eval(mx.gather_qmm(x,w,s,b,rhs_indices=inds,transpose=True,group_size=128,bits=2))
    t0=time.perf_counter()
    for _ in range(n): mx.eval(mx.gather_qmm(x,w,s,b,rhs_indices=inds,transpose=True,group_size=128,bits=2))
    return (time.perf_counter()-t0)/n*1000

print("\n=== bench (ms) ===")
print("real as-is                 : %.4f" % bench(gp.weight, gp.scales, gp.biases))
print("fresh as-is                : %.4f" % bench(q.weight, q.scales, q.biases))
# cast real scales/biases to fresh's dtype
fs = q.scales.dtype
print("real + scales/biases->%-6s: %.4f" % (str(fs).split('.')[-1],
      bench(gp.weight, gp.scales.astype(fs), gp.biases.astype(fs) if gp.biases is not None else None)))
for dt in (mx.float16, mx.bfloat16, mx.float32):
    print("real + scales/biases->%-9s: %.4f" % (str(dt).split('.')[-1],
          bench(gp.weight, gp.scales.astype(dt), gp.biases.astype(dt) if gp.biases is not None else None)))
# fresh weight but real-dtype scales (isolate weight vs scales)
print("fresh w + fresh-scales->realdt: %.4f" % bench(q.weight, q.scales.astype(gp.scales.dtype),
      q.biases.astype(gp.biases.dtype) if (q.biases is not None and gp.biases is not None) else q.biases))
# real weight forced contiguous + astype roundtrip
wc = mx.contiguous(gp.weight); mx.eval(wc)
print("real contiguous(weight)    : %.4f" % bench(wc, gp.scales, gp.biases))
