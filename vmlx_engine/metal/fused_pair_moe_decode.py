"""Fused pair-SwiGLU routed-MoE decode kernels for DSV4 JANG bundles.

Replaces the stock single-token routed-expert chain (two gather_qmm dispatches
for gate/up + compiled fp32 SwiGLU/score elementwise chain + one gather_qmm for
down) in ``MoE._weighted_routed_experts`` with two custom simdgroup kernels:

* pair kernel — one simdgroup per (row m<2048, slot k<6): lane-strided uint32
  word reads over the gate AND up 2-bit gs64 rows in a single pass, fp32
  ``scale*sum(x*q) + bias*sum(x)`` per-word refactor, ``simd_sum``, round to
  the activation dtype (gather_qmm output boundary), fp32 limited SwiGLU
  (up clip +-limit, gate clip max-only), fp32 route score, round back.
* down kernel — one simdgroup per (slot k<6, row d<4096): 2-bit gs32 dot over
  the activated row, per-expert round to activation dtype. The cross-expert
  fp32 sum stays in stock ``_dsv4_accumulate_moe`` so that boundary is
  untouched.

Numerics: residual vs MLX's qmv is last-ulp reassociation only (same
acceptance class as MoE prefix cold/warm); the installer runs a live
self-test against the stock method on the first validated module and refuses
to patch when the relative error exceeds a conservative bound.

Prefill and any non-(1,1) shapes always take the stock path. Modules whose
layout differs from the validated DSV4 shape (gate/up b2gs64 (E,2048,256),
down b2gs32 (E,4096,128), k=6, swiglu_limit=10) are left stock.

Env: ``VMLX_DSV4_FUSED_MOE_PAIR`` — default on; ``0``/``off``/``false``
disables.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import mlx.core as mx

_log = logging.getLogger(__name__)

_D = 4096
_M = 2048
_K = 6
_LIMIT = 10.0
_W_GU = _D // 16
_W_DN = _M // 16
_G_GU = 64
_G_DN = 64
_GS_GU = 64
_GS_DN = 32
_SELF_TEST_MAX_REL = 2.5e-2

_ORIGINAL_ATTR = "_vmlx_dsv4_pair_original_weighted_call"
_MODULE_OK_ATTR = "_vmlx_pair_fused_ok"

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_PAIR_SRC = f"""
    uint tid = thread_position_in_grid.x;
    uint sgid = tid / 32;
    uint lane = thread_index_in_simdgroup;
    uint k = sgid / {_M};
    uint m = sgid % {_M};
    if (k >= {_K}) return;
    uint e = (uint)inds[k];
    size_t rowoff = ((size_t)e * {_M} + m);
    const device uint32_t* grow = gw + rowoff * {_W_GU};
    const device uint32_t* urow = uw + rowoff * {_W_GU};
    size_t soff = rowoff * {_G_GU};
    float gacc = 0.0f;
    float uacc = 0.0f;
    for (uint i = 0; i < {_W_GU // 32}; i++) {{
        uint w_idx = lane + 32u * i;
        uint grp = w_idx >> 2;              // gs64 -> 4 words per group
        float gsc = (float)gs[soff + grp];
        float gbi = (float)gb[soff + grp];
        float usc = (float)us[soff + grp];
        float ubi = (float)ub[soff + grp];
        uint32_t gwrd = grow[w_idx];
        uint32_t uwrd = urow[w_idx];
        uint xbase = w_idx * 16u;
        float qsg = 0.0f;
        float qsu = 0.0f;
        float xs = 0.0f;
        for (uint j = 0; j < 16; j++) {{
            float xv = (float)x[xbase + j];
            xs += xv;
            qsg += xv * (float)((gwrd >> (2u * j)) & 3u);
            qsu += xv * (float)((uwrd >> (2u * j)) & 3u);
        }}
        gacc += gsc * qsg + gbi * xs;
        uacc += usc * qsu + ubi * xs;
    }}
    gacc = simd_sum(gacc);
    uacc = simd_sum(uacc);
    if (lane == 0) {{
        float g = (float)((T)gacc);         // gather_qmm output rounding
        float u = (float)((T)uacc);
        u = clamp(u, -{_LIMIT}f, {_LIMIT}f);
        g = min(g, {_LIMIT}f);
        float a = g / (1.0f + metal::exp(-g)) * u;   // silu(g) * u in fp32
        a *= scores[k];
        act[(size_t)k * {_M} + m] = (T)a;
    }}
"""

_DOWN_SRC = f"""
    uint tid = thread_position_in_grid.x;
    uint sgid = tid / 32;
    uint lane = thread_index_in_simdgroup;
    uint k = sgid / {_D};
    uint d = sgid % {_D};
    if (k >= {_K}) return;
    uint e = (uint)inds[k];
    size_t rowoff = ((size_t)e * {_D} + d);
    const device uint32_t* row = dw + rowoff * {_W_DN};
    size_t soff = rowoff * {_G_DN};
    float acc = 0.0f;
    for (uint i = 0; i < {_W_DN // 32}; i++) {{
        uint w_idx = lane + 32u * i;
        uint grp = w_idx >> 1;              // gs32 -> 2 words per group
        float sc = (float)ds[soff + grp];
        float bi = (float)db[soff + grp];
        uint32_t wrd = row[w_idx];
        uint abase = (uint)k * {_M} + w_idx * 16u;
        float qs = 0.0f;
        float xs = 0.0f;
        for (uint j = 0; j < 16; j++) {{
            float av = (float)act[abase + j];
            xs += av;
            qs += av * (float)((wrd >> (2u * j)) & 3u);
        }}
        acc += sc * qs + bi * xs;
    }}
    acc = simd_sum(acc);
    if (lane == 0) {{
        out[(size_t)k * {_D} + d] = (T)acc;  // per-expert round; fp32 sum stays in stock accumulate
    }}
"""

_KERNELS: Optional[tuple[Any, Any]] = None
_INSTALLED_CLASSES: set[type] = set()
_LAST_STATUS: dict[str, Any] = {"installed": 0, "reason": None}


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_FUSED_MOE_PAIR", "1").strip().lower()
    return value not in {"0", "off", "false", "no"}


def _get_kernels() -> tuple[Any, Any]:
    global _KERNELS
    if _KERNELS is None:
        pair = mx.fast.metal_kernel(
            name="vmlx_dsv4_pair_swiglu",
            input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "inds", "scores"],
            output_names=["act"],
            header=_HEADER,
            source=_PAIR_SRC,
        )
        down = mx.fast.metal_kernel(
            name="vmlx_dsv4_down6",
            input_names=["act", "dw", "ds", "db", "inds"],
            output_names=["out"],
            header=_HEADER,
            source=_DOWN_SRC,
        )
        _KERNELS = (pair, down)
    return _KERNELS


def _fused_routed(switch_mlp: Any, x_flat: mx.array, inds_flat: mx.array,
                  scores_flat: mx.array, dtype: mx.Dtype) -> mx.array:
    pair, down = _get_kernels()
    g = switch_mlp.gate_proj
    u = switch_mlp.up_proj
    d = switch_mlp.down_proj
    act = pair(
        inputs=[x_flat, g.weight, g.scales, g.biases,
                u.weight, u.scales, u.biases, inds_flat, scores_flat],
        template=[("T", dtype)],
        grid=(32 * _M * _K, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(_K, _M)],
        output_dtypes=[dtype],
    )[0]
    out = down(
        inputs=[act, d.weight, d.scales, d.biases, inds_flat],
        template=[("T", dtype)],
        grid=(32 * _K * _D, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(_K, _D)],
        output_dtypes=[dtype],
    )[0]
    return out


def _validate_module(mlp: Any) -> Optional[str]:
    switch_mlp = getattr(mlp, "switch_mlp", None)
    if switch_mlp is None:
        return "no switch_mlp"
    args = getattr(mlp, "args", None)
    if getattr(args, "num_experts_per_tok", None) != _K:
        return "num_experts_per_tok != 6"
    if float(getattr(args, "swiglu_limit", _LIMIT)) != _LIMIT:
        return "swiglu_limit != 10"
    for name, gs_expect, shape_expect in (
        ("gate_proj", _GS_GU, (_M, _W_GU)),
        ("up_proj", _GS_GU, (_M, _W_GU)),
        ("down_proj", _GS_DN, (_D, _W_DN)),
    ):
        proj = getattr(switch_mlp, name, None)
        if proj is None:
            return f"missing {name}"
        if getattr(proj, "bits", None) != 2:
            return f"{name} bits != 2"
        if getattr(proj, "group_size", None) != gs_expect:
            return f"{name} group_size != {gs_expect}"
        if getattr(proj, "mode", "affine") != "affine":
            return f"{name} mode != affine"
        weight = getattr(proj, "weight", None)
        scales = getattr(proj, "scales", None)
        biases = getattr(proj, "biases", None)
        if weight is None or scales is None or biases is None:
            return f"{name} missing quant tensors"
        if weight.dtype != mx.uint32:
            return f"{name} weight dtype {weight.dtype}"
        if tuple(weight.shape[1:]) != shape_expect:
            return f"{name} shape {tuple(weight.shape)}"
    return None


def _self_test(mlp: Any, original: Any) -> Optional[str]:
    dtype = mlp.switch_mlp.gate_proj.scales.dtype
    x = (mx.random.normal((1, 1, _D)) * 0.5).astype(dtype)
    n_experts = mlp.switch_mlp.gate_proj.weight.shape[0]
    step = max(1, n_experts // _K)
    inds = mx.array([(i * step) % n_experts for i in range(_K)],
                    dtype=mx.uint32).reshape(1, 1, _K)
    scores = mx.random.uniform(shape=(1, 1, _K)).astype(mx.float32)
    scores = scores / scores.sum()
    try:
        ref = original(mlp, x, inds, scores).astype(mx.float32)
        got = _fused_routed(
            mlp.switch_mlp, x.reshape(-1), inds.reshape(-1),
            scores.reshape(-1), x.dtype,
        ).reshape(1, 1, _K, _D).astype(mx.float32)
        mx.eval(ref, got)
    except Exception as err:  # kernel compile / dispatch failure
        return f"self-test execution failed: {err}"
    denom = max(float(mx.abs(ref).max()), 1e-6)
    rel = float(mx.abs(got - ref).max()) / denom
    if rel > _SELF_TEST_MAX_REL:
        return f"self-test rel diff {rel:.3e} > {_SELF_TEST_MAX_REL:.1e}"
    _LAST_STATUS["self_test_rel"] = rel
    return None


def install_dsv4_fused_pair_moe(model: Any) -> int:
    """Validate + patch MoE routed decode with the fused pair kernels.

    Returns the number of modules the fused path covers (0 = stock)."""
    if not _enabled():
        _LAST_STATUS.update(installed=0, reason="disabled via env")
        return 0
    modules = []
    moe_cls = None
    for _name, module in model.named_modules():
        if hasattr(module, "switch_mlp") and hasattr(module, "_weighted_routed_experts"):
            modules.append(module)
            moe_cls = type(module)
    if not modules:
        _LAST_STATUS.update(installed=0, reason="no MoE modules found")
        return 0

    valid = []
    for module in modules:
        reason = _validate_module(module)
        if reason is None:
            valid.append(module)
        else:
            setattr(module, _MODULE_OK_ATTR, False)
    if not valid:
        _LAST_STATUS.update(installed=0, reason="no modules with validated layout")
        return 0

    original = getattr(moe_cls, _ORIGINAL_ATTR, None)
    if original is None:
        original = moe_cls._weighted_routed_experts

    fail = _self_test(valid[0], original)
    if fail is not None:
        _LAST_STATUS.update(installed=0, reason=fail)
        _log.warning("DSV4 fused pair MoE decode refused: %s", fail)
        return 0

    for module in valid:
        setattr(module, _MODULE_OK_ATTR, True)

    if moe_cls not in _INSTALLED_CLASSES:
        setattr(moe_cls, _ORIGINAL_ATTR, original)

        def _pair_weighted(self, x, inds, scores):
            if (
                x.shape[0] == 1
                and x.shape[1] == 1
                and getattr(self, _MODULE_OK_ATTR, False)
            ):
                out = _fused_routed(
                    self.switch_mlp,
                    x.reshape(-1),
                    inds.reshape(-1),
                    scores.reshape(-1).astype(mx.float32),
                    x.dtype,
                )
                return out.reshape(1, 1, _K, _D)
            return original(self, x, inds, scores)

        moe_cls._weighted_routed_experts = _pair_weighted
        _INSTALLED_CLASSES.add(moe_cls)

    _LAST_STATUS.update(installed=len(valid), reason=None)
    return len(valid)


def dsv4_fused_pair_moe_status() -> dict[str, Any]:
    return dict(_LAST_STATUS)
