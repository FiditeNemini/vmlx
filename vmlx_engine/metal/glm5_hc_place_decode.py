"""Single-dispatch GLM-5.3 hyper-connection placement for decode."""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

_OBSERVED = False


def fused_glm5_hc_place_requested() -> bool:
    value = os.environ.get("VMLX_GLM5_FUSED_HC_PLACE", "0").strip().lower()
    return value not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=8)
def _kernel(streams: int, hidden: int):
    source = f"""
        uint index = thread_position_in_grid.x;
        if (index >= {streams * hidden}u) return;
        uint target = index / {hidden}u;
        uint dim = index % {hidden}u;

        float value = float(T(post[target])) * float(block_out[dim]);
        for (uint source = 0u; source < {streams}u; ++source) {{
            value += float(T(comb[source * {streams}u + target])) *
                float(residual[source * {hidden}u + dim]);
        }}
        output[index] = T(value);
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_glm5_hc_place_h{streams}_d{hidden}",
        input_names=["post", "comb", "block_out", "residual"],
        output_names=["output"],
        header="#include <metal_stdlib>\nusing namespace metal;\n",
        source=source,
    )


def glm5_hc_place_decode(
    post: mx.array,
    comb: mx.array,
    block_out: mx.array,
    residual: mx.array,
    *,
    enabled: bool,
) -> mx.array | None:
    """Return placed streams for the exact single-token shape or ``None``."""

    if not enabled or residual.ndim != 4:
        return None
    batch, rows, streams, hidden = (int(value) for value in residual.shape)
    if batch != 1 or rows != 1 or streams != 4 or hidden <= 0:
        return None
    if tuple(post.shape) != (1, 1, streams):
        return None
    if tuple(comb.shape) != (1, 1, streams, streams):
        return None
    if tuple(block_out.shape) != (1, 1, hidden):
        return None
    supported = (mx.float16, mx.bfloat16, mx.float32)
    if residual.dtype not in supported or block_out.dtype != residual.dtype:
        return None
    if post.dtype != mx.float32 or comb.dtype != mx.float32:
        return None

    output = _kernel(streams, hidden)(
        inputs=[post, comb, block_out, residual],
        template=[("T", residual.dtype)],
        grid=(streams * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(residual.shape)],
        output_dtypes=[residual.dtype],
    )[0]
    global _OBSERVED
    if not _OBSERVED:
        _OBSERVED = True
    return output


def glm5_hc_place_status() -> dict[str, object]:
    return {
        "installed": _OBSERVED,
        "observed_calls": int(_OBSERVED),
        "reason": None,
    }


__all__ = [
    "fused_glm5_hc_place_requested",
    "glm5_hc_place_decode",
    "glm5_hc_place_status",
]
