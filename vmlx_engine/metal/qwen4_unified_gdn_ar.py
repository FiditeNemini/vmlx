"""Default-off single-token GDN recurrence experiment for Flash-Next.

Reuse the qualified recurrence arithmetic without changing verify admission.
Projections, mixed quantization, PLE, QSA and cache ownership stay with the
caller. Numerical component evidence is not whole-model speed admission.
"""

from functools import cache
import os

import mlx.core as mx

from .qwen4_unified_gdn_verify import _coefficient_layout_supported, _kernel


def unified_gdn_ar_requested() -> bool:
    return os.environ.get("VMLX_QWEN4_UNIFIED_GDN_AR", "0") == "1"


@cache
def _hardware_supported() -> bool:
    return (mx.__version__ == "0.32.2"
            and mx.device_info().get("architecture") == "applegpu_g17s")


def qwen4_unified_gdn_ar(
    qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
    recurrent_state, norm_weight, norm_eps, *, enabled=False,
):
    """Return output/conv/state, or None before any unsupported-path mutation.

    The caller must also exclude masks, per-sequence lengths, verification,
    training and checkpoint segmentation. Initial state creation stays stock.
    No dtype conversion is used to force admission.
    """
    if (not enabled or qkv is None or tuple(qkv.shape) != (1, 1, 10240)
            or qkv.dtype != mx.float16 or norm_eps != 1e-6):
        return None
    expected = (
        (z, (1, 1, 6144), mx.float16),
        (a, (1, 1, 48), mx.float16),
        (b, (1, 1, 48), mx.float16),
        (conv_state, (1, 3, 10240), mx.float16),
        (conv_weight, (10240, 4, 1), mx.float16),
        (recurrent_state, (1, 48, 128, 128), mx.float32),
        (norm_weight, (128,), mx.float16),
    )
    if (not _coefficient_layout_supported(A_log, dt_bias)
            or any(v is None or tuple(v.shape) != shape or v.dtype != dtype
                   for v, shape, dtype in expected)
            or mx.default_device() != mx.gpu or not _hardware_supported()):
        return None
    outputs = _kernel()(
        inputs=[qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias,
                recurrent_state, norm_weight, float(norm_eps)],
        template=[("T", mx.float16), ("GT", mx.result_type(a.dtype, dt_bias.dtype)),
                  ("HK", 16), ("HV", 48), ("DK", 128), ("DV", 128),
                  ("K", 4), ("S", 1), ("TY", 32), ("RATIO", 3)],
        grid=(32, 32, 48), threadgroup=(32, 32, 1),
        output_shapes=[(1, 1, 6144), (1, 3, 10240), (1, 48, 128, 128),
                       (1, 0, 48, 128, 128), (1, 0, 3, 10240)],
        output_dtypes=[mx.float16, mx.float16, mx.float32, mx.float32, mx.float16],
    )
    # S=1 writes no intermediate snapshots. Empty outputs retain the shared
    # kernel ABI; the three AR arrays go through the caller's existing setters.
    return tuple(outputs[:3])
