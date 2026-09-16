"""Opt-in Flash post-RMS sigmoid/product with stock RMS rounding preserved.

Only productive AR at the qualified FP16 shape/runtime is admitted. Quantized
weight names/config dtypes are not activation evidence. No persistent tensors,
extra evaluation, or host synchronization are introduced here.
"""

from functools import lru_cache
import importlib.metadata

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active


@lru_cache(maxsize=1)
def _compatible_runtime() -> bool:
    # The stock sigmoid/compiler arithmetic is version dependent. Fail closed
    # outside the actual numerical qualification, even with explicit opt-in.
    try:
        return (
            importlib.metadata.version("mlx") == "0.32.2"
            and mx.device_info().get("device_name") == "Apple M5 Max"
        )
    except (importlib.metadata.PackageNotFoundError, RuntimeError, OSError):
        return False


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="vmlx_qwen4_precise_sigmoid_product_v1",
        input_names=["normed", "gate"],
        output_names=["output"],
        source=r'''
      {
      #pragma clang fp contract(off)
      #pragma clang fp reassociate(off)
      uint i = thread_position_in_grid.x;
      if (i >= N) return;
      float g = float(gate[i]);
      float e = metal::precise::exp(metal::abs(g));
      float y = 1.0f / (1.0f + e);
      float s = (g < 0.0f) ? y : (1.0f - y);
      output[i] = T(float(normed[i]) * s);
      }
    ''',
    )


def precise_sigmoid_epilogue(normed, gate, *, output_dtype, enabled: bool):
    """Return the qualified graph, or None before creating any custom kernel.

    The caller owns stock RMS and route admission. Unsupported inputs retain
    stock arithmetic; admitted execution errors propagate normally, including
    deferred GPU failures. Never retry an owning forward after cache advance.
    """
    if (
        not enabled
        or not affine_moe_ar_scope_active()
        or normed.shape != (1, 1, 48, 128)
        or gate.shape != normed.shape
        or normed.dtype != mx.float16
        or gate.dtype != mx.float16
        or output_dtype != mx.float16
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or not _compatible_runtime()
    ):
        return None
    return _kernel()(
        inputs=[normed, gate],
        template=[("T", normed.dtype), ("N", normed.size)],
        grid=(normed.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[normed.shape],
        output_dtypes=[normed.dtype],
    )[0]
