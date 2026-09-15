# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM AR selected-latent decode with a bounded compiled core.

Gather the growing latent history outside compilation. The compiled graph
sees only 2051 selected slots, with the causal position and validity as data.
Preserve the native FP32 SDPA and BF16 return; no online-softmax rewrite,
changed selection order, repacking or retained model/cache arrays.
"""

from functools import lru_cache
import importlib.metadata
import logging

import mlx.core as mx

from .affine_moe_pair_decode import affine_moe_ar_scope_active

logger = logging.getLogger(__name__)
_FAILED = False
_OBSERVED = False
_CALL_COUNT = 0
_TRACE_COUNT = 0


@lru_cache(maxsize=1)
def _compatible_runtime() -> bool:
    try:
        return (importlib.metadata.version("mlx") == "0.32.2"
                and "Apple M5" in mx.device_info().get("device_name", ""))
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        return False


def _core(query, keys, indices, valid, past):
    global _TRACE_COUNT
    _TRACE_COUNT += 1  # Python trace count only; never a GPU data readback.
    allowed = valid & (indices <= past)
    bias = mx.where(allowed, mx.array(0.0, mx.float32),
                    mx.array(-mx.inf, mx.float32))[:, :, None, :]
    result = mx.fast.scaled_dot_product_attention(
        query.transpose(0, 2, 1, 3).astype(mx.float32),
        keys.astype(mx.float32), keys.astype(mx.float32),
        scale=0.0625, mask=bias,
    ).astype(query.dtype)
    return result.transpose(0, 2, 1, 3)


@lru_cache(maxsize=1)
def _compiled_core():
    return mx.compile(_core)


def _eligible(query, latent, indices, valid, past, scale) -> bool:
    return (
        query.shape == (1, 64, 1, 512)
        and query.dtype == mx.bfloat16
        and latent.ndim == 4 and latent.shape[:2] == (1, 1)
        and latent.shape[3] == 512 and latent.dtype == mx.bfloat16
        and indices.shape == (1, 1, 2051) and indices.dtype == mx.int32
        and valid.shape == indices.shape and valid.dtype == mx.bool_
        and isinstance(past, int) and 2051 <= past < 2**31
        and latent.shape[2] == past + 1 and scale == 0.0625
    )


def glm5_compiled_dsa_output(query, latent, indices, valid, *, past, scale, enabled):
    """Return None for unqualified layouts, prefill/non-AR or compiler failure."""
    global _FAILED, _OBSERVED, _CALL_COUNT
    if not enabled or _FAILED or not affine_moe_ar_scope_active():
        return None
    if not _eligible(query, latent, indices, valid, past, scale) or not _compatible_runtime():
        return None
    try:
        safe = mx.where(valid, indices, mx.zeros_like(indices))
        selected = mx.take(latent[0, 0], safe.reshape(-1), axis=0)
        result = _compiled_core()(
            query, selected.reshape(1, 1, 2051, 512), indices, valid,
            mx.array(past, dtype=mx.int32),
        )
        if not _OBSERVED:
            # First-use failure must be observed before announcing engagement.
            mx.eval(result)
            logger.info("GLM compiled DSA decode observed: batch=1 heads=64 rank=512 "
                        "selected=2051 fp32_sdpa=true traces=%d", _TRACE_COUNT)
            _OBSERVED = True
    except (ValueError, RuntimeError) as exc:
        _FAILED = True
        logger.warning("GLM compiled DSA decode unavailable; retaining stock attention: %s", exc)
        return None
    _CALL_COUNT += 1
    return result
