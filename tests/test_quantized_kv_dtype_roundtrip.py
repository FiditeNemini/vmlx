# SPDX-License-Identifier: Apache-2.0
"""Quantized-KV scales/biases must survive an L2 round-trip unchanged.

Legacy NumPy-backed safetensors writes widened BF16 to FP32 and needed
``__orig_dtypes__`` to restore the scales. Current writes preserve raw BF16 bits
in U16 carriers with per-tensor declarations. These cases retain the historical
FP32 compatibility contract; actual current disk I/O is covered separately.

The packed data itself is integer and is deliberately left alone.
"""

from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")

from vmlx_engine.block_disk_store import _deserialize_block, _serialize_block


def _block():
    packed = mx.zeros((2, 4), dtype=mx.uint32)
    scales = mx.ones((2, 2), dtype=mx.bfloat16)
    zeros = mx.zeros((2, 2), dtype=mx.bfloat16)
    return [
        (
            "quantized_kv",
            (packed, scales, zeros),
            (packed, scales, zeros),
            {"group_size": 64, "bits": 4},
        )
    ]


def _cast_like_safetensors(tensors):
    # Simulate an actual legacy record, before per-tensor dtype metadata.
    meta = json.loads(bytes(tensors["__vmlx_block_meta__"].tolist()).decode())
    meta.pop("__tensor_dtypes__", None)
    result = {
        k: (v.astype(mx.float32) if "bfloat16" in str(getattr(v, "dtype", "")) else v)
        for k, v in tensors.items()
    }
    result["__vmlx_block_meta__"] = mx.array(list(json.dumps(meta).encode()), dtype=mx.uint8)
    return result


def test_scale_and_bias_dtypes_are_recorded():
    tensors, _dtype, _n = _serialize_block(_block())
    meta = json.loads(bytes(tensors["__vmlx_block_meta__"].tolist()).decode())
    recorded = meta.get("__orig_dtypes__", {})
    for key in (
        "0_q_keys_scales",
        "0_q_keys_zeros",
        "0_q_values_scales",
        "0_q_values_zeros",
    ):
        assert key in recorded, f"{key} not recorded; its dtype cannot be restored"
        assert "bfloat16" in recorded[key]


def test_bf16_survives_the_disk_round_trip():
    tensors, dtype, _n = _serialize_block(_block())
    restored = _deserialize_block(_cast_like_safetensors(tensors), dtype)
    _tag, keys_tuple, values_tuple, _meta = restored[0]
    for label, tup in (("keys", keys_tuple), ("values", values_tuple)):
        for idx, part in ((1, "scales"), (2, "zeros")):
            assert str(tup[idx].dtype).endswith("bfloat16"), (
                f"{label} {part} came back as {tup[idx].dtype}; a cache HIT now "
                f"differs numerically from a recompute"
            )


def test_packed_integer_payload_is_untouched():
    tensors, dtype, _n = _serialize_block(_block())
    restored = _deserialize_block(_cast_like_safetensors(tensors), dtype)
    _tag, keys_tuple, _values, _meta = restored[0]
    assert str(keys_tuple[0].dtype).endswith("uint32")
