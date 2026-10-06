"""Dense JANGH (jangtq2) projection: y = x W^T with W in the JANGH codebook format, stored as ONE-expert banks
`<path>.tq2_packed` (1, N, K*bits/32) uint32 + `<path>.tq2_scales` (1, N) float16, every token routed to expert 0.
Reuses the routed kernels unchanged (decode gather_qmv with in-kernel H32; prefill host H32 rows + gather_qmm_sorted),
the Raptor-0.6 dense-JANGH design (docs: jang/docs/runtime/raptor-jangh-2026-09-29). Added 2026-10-05 for
K2-Horizon / Diplodocus mixed bundles (gate/up affine, down JANGH). Config entry: {"mode": "jangtq2", "bits": b,
"rotation": "hadamard32"|"none"} on a NON-routed path (e.g. model.layers.N.mlp.down_proj); the routed installer ignores
such entries only when the model builds a TQLinear for them (payload validation fails closed otherwise)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from . import kernels as K
from .format import codebook

# Rows at which a dense JANGH projection switches from the per-row gather QMV
# (weights re-read for EVERY row: Qwen3.8-27B JANGH2 forward +15 ms per extra
# row) to the one-pass sorted QMM (~38 ms fixed, then nearly flat).  Measured
# on M5 Max, 27B JANGH2 full forward: rows 1/2/4/5/8/16 = 37/53/83/97/150/264 ms
# per-row vs 36/75/77/78/88/101 ms sorted -> crossover between 3 and 4 rows.
# Verify windows (DFlash2 blocks of 5-8 rows, MTP depth+1) sit above it.
SORT_THRESHOLD = 4
ROTATIONS = ("none", "hadamard32")


class TQLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, bits: int, rotation: str = "hadamard32"):
        super().__init__()
        if rotation not in ROTATIONS or input_dims % 32:
            raise ValueError("jangtq2 dense: unsupported rotation or input width")
        self.bits, self.rotation, self.input_dims, self.output_dims = bits, rotation, input_dims, output_dims
        self.num_experts = 1
        self.tq2_packed = mx.zeros((1, output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.tq2_scales = mx.zeros((1, output_dims), dtype=mx.float16)
        self._cb = mx.array(codebook(bits))
        self.is_jangtq2_dense = True

    def to_quantized(self, **kwargs):
        if kwargs.get("mode", "jangtq2") != "jangtq2" or kwargs.get("bits", self.bits) != self.bits:
            raise ValueError("jangtq2 dense: quantization entry differs from installed module")
        return self

    def __call__(self, x: mx.array) -> mx.array:
        lead, Kd = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, Kd); T = x2.shape[0]
        idx = mx.zeros((T,), dtype=mx.uint32)
        rot = self.rotation == "hadamard32"
        if T < SORT_THRESHOLD:
            y = K.gather_qmv(x2, self.tq2_packed, self.tq2_scales, self._cb, idx, self.bits, x_per_dispatch=True, rotate=rot)
        else:
            xr = K.h32_rows(x2, x2.dtype) if rot else x2
            y = K.gather_qmm_sorted(xr, self.tq2_packed, self.tq2_scales, self._cb, idx, self.bits)
        return y.astype(x.dtype).reshape(*lead, self.output_dims)


def dense_entries(config: dict) -> dict[str, dict]:
    q = config.get("quantization") or {}
    return {p: e for p, e in q.items() if isinstance(e, dict) and e.get("mode") == "jangtq2" and ".switch_mlp." not in p}


def install_jangh_dense(model: nn.Module, config: dict) -> int:
    """Replace every module whose path has a dense jangtq2 entry with a TQLinear BEFORE nn.quantize. Paths are the
    model's own module paths (bundle namespace). Fails closed on an entry that matches no Linear."""
    ents = dense_entries(config)
    if not ents:
        return 0
    mods = dict(model.named_modules()); done = set()
    for path, e in ents.items():
        parent_path, _, leaf = path.rpartition(".")
        parent = mods.get(parent_path); child = getattr(parent, leaf, None) if parent is not None else None
        if child is not None and hasattr(child, "tq2_packed"):
            continue          # the model class already built a JANGH module here (e.g. k2_horizon / kolibri1 __init__)
        if child is None or not hasattr(child, "weight") or child.weight.ndim != 2:
            raise ValueError(f"jangtq2 dense: entry {path} matches no Linear")
        N, Kd = child.weight.shape
        setattr(parent, leaf, TQLinear(Kd, N, int(e["bits"]), e.get("rotation", "hadamard32")))
        done.add(path)
    return len(done)
