"""Bounded parameter loading: exact values and pending-work cleanup."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


@pytest.mark.parametrize("pipeline", [False, True])
def test_parameter_loading_preserves_values_and_drains(monkeypatch, pipeline):
    from vmlx_engine.utils import jang_loader

    weights = {str(i): (mx.arange(32) + i).astype(mx.bfloat16) for i in range(53)}
    model = SimpleNamespace(parameters=lambda: weights)
    submitted = []
    waits = []
    original_async = mx.async_eval
    original_sync = mx.synchronize

    def submit(*arrays):
        submitted.extend(arrays)
        return original_async(*arrays)

    def synchronize():
        waits.append(len(submitted))
        return original_sync()

    monkeypatch.setattr(mx, "async_eval", submit)
    monkeypatch.setattr(mx, "synchronize", synchronize)
    jang_loader._chunked_eval_params(model, chunk_size=25, pipeline_gpu=pipeline)
    assert len(submitted) == (3 if pipeline else 0)
    assert waits == ([3] if pipeline else [])
    for i, value in weights.items():
        assert value.dtype == mx.bfloat16
        assert value.tolist() == [float(j + int(i)) for j in range(32)]


def test_failed_weight_read_drains_prior_submission(monkeypatch):
    from vmlx_engine.utils import jang_loader

    model = SimpleNamespace(parameters=lambda: {str(i): mx.array(i) for i in range(3)})
    original_eval = mx.eval
    original_sync = mx.synchronize
    reads = []
    waits = []

    def evaluate(*arrays):
        reads.append(len(arrays))
        if len(reads) == 2:
            raise OSError("weight read failed")
        return original_eval(*arrays)

    def synchronize():
        waits.append(True)
        return original_sync()

    monkeypatch.setattr(mx, "eval", evaluate)
    monkeypatch.setattr(mx, "synchronize", synchronize)
    with pytest.raises(OSError, match="weight read failed"):
        jang_loader._chunked_eval_params(model, chunk_size=1, pipeline_gpu=True)
    assert reads == [1, 1]
    assert waits == [True]
