"""Metadata-only TQ decode limits: no MLX import, tensor allocation or codec."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def validator():
    # Load the pure validator without the engine package's runtime imports.
    path = Path(__file__).parents[1] / "vmlx_engine/cache_record_validator.py"
    spec = importlib.util.spec_from_file_location("tq_budget_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(validator, surface, key_dtype, value_dtype):
    shape = (1, 1, 8, 4)  # 32 decoded elements per lane; no actual array.
    tensor = SimpleNamespace(shape=(1,), itemsize=1, nbytes=1)
    keys = SimpleNamespace(
        shape=shape, index_bits=3, indices_packed=tensor, qjl_packed=tensor,
        residual_norms=tensor, vector_norms=tensor,
    )
    values = SimpleNamespace(
        shape=shape, index_bits=3, indices_packed=tensor, vector_norms=tensor,
    )
    config = dict(
        key_bits=3, value_bits=3, key_dtype=key_dtype, value_dtype=value_dtype,
        key_dim=4, value_dim=4, offset=8, seed=0,
    )
    if surface == "paged":
        return validator.validate_cache_record(
            [("turboquant_kv", keys, values, config)], expected_num_layers=1,
        )[:2]
    tensors = {
        f"tq_0_{prefix}_{field}": tensor
        for prefix, fields in (
            ("ck", ("indices_packed", "qjl_packed", "residual_norms", "vector_norms")),
            ("cv", ("indices_packed", "vector_norms")),
        )
        for field in fields
    }
    metadata = {
        "__num_layers__": "1", "__layer_0_class__": "TurboQuantKVCache",
        "__tq_0_ck_shape__": json.dumps(shape),
        "__tq_0_cv_shape__": json.dumps(shape),
        **{f"__tq_0_{key}__": str(value) for key, value in config.items()},
    }
    return validator.validate_tq_native_metadata(
        tensors, metadata, expected_num_layers=1,
    )


@pytest.mark.parametrize("surface", ["paged", "native"])
@pytest.mark.parametrize("lane", ["key", "value"])
def test_decoded_tensor_budget_uses_declared_dtype(validator, monkeypatch, surface, lane):
    monkeypatch.setattr(validator, "MAX_TENSOR_BYTES", 64)
    for dtype in ("float16", "bfloat16"):
        assert _validate(validator, surface, dtype, dtype)[0]
    dtypes = {"key": "float16", "value": "float16", lane: "float32"}
    ok, reason = _validate(validator, surface, dtypes["key"], dtypes["value"])
    assert not ok, "128 decoded FP32 bytes must not fit the 64-byte tensor cap"
    assert "128 bytes" in reason
    # The same FP32 metadata is valid at its exact byte boundary.
    monkeypatch.setattr(validator, "MAX_TENSOR_BYTES", 128)
    assert _validate(validator, surface, dtypes["key"], dtypes["value"])[0]


@pytest.mark.parametrize("lane", ["key", "value"])
def test_native_decoded_total_counts_each_lane_dtype(validator, monkeypatch, lane):
    monkeypatch.setattr(validator, "MAX_TENSOR_BYTES", 128)
    monkeypatch.setattr(validator, "MAX_TOTAL_RECORD_BYTES", 160)
    assert _validate(validator, "native", "float16", "bfloat16")[0]
    dtypes = {"key": "float16", "value": "float16", lane: "float32"}
    ok, reason = _validate(validator, "native", dtypes["key"], dtypes["value"])
    assert not ok, "128 + 64 decoded bytes must not fit the 160-byte record cap"
    assert "decoded total 192 bytes" in reason
    monkeypatch.setattr(validator, "MAX_TOTAL_RECORD_BYTES", 192)
    assert _validate(validator, "native", dtypes["key"], dtypes["value"])[0]
