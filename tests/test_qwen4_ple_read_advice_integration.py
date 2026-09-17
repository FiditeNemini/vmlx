"""Optional file-page hints cannot change PLE bytes, ordering or I/O lifetime."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from tests.test_qwen4_ple_host_gather import _bits, _table
from vmlx_engine.models.qwen4_exp import table_reader


def test_default_does_not_construct_advisor(tmp_path, monkeypatch):
    monkeypatch.delenv("VMLX_QWEN4_PLE_READ_ADVICE", raising=False)
    def forbidden():
        raise AssertionError("Default must not inspect residency")
    monkeypatch.setattr(table_reader.SelectedPageReadAdvisor, "create", forbidden)
    table = _table(tmp_path, [(2, 32), (6, 32)], mx.float16, 160)
    try:
        assert table._read_advisor is None
        assert table.read_advice_stats["calls"] == 0
    finally:
        table.close()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("status", ["advised", "resident", "error", "partial", "limit"])
def test_exact_mixed_layout_and_failure_fallback(tmp_path, monkeypatch, caplog, dtype, status):
    monkeypatch.setenv("VMLX_QWEN4_PLE_READ_ADVICE", "1")
    calls = []
    def advise(selections):
        calls.append([(reader, local.copy()) for reader, local in selections])
        return {"status": status, "hinted": int(status in {"advised", "partial"}),
                "bytes": 16384 if status in {"advised", "partial"} else 0}
    advisor = SimpleNamespace(advise=advise)
    monkeypatch.setattr(table_reader.SelectedPageReadAdvisor, "create", lambda: advisor)
    table = _table(tmp_path, [(1, 32), (2, 64), (6, 32), (8, 128)], dtype)
    table._host_assembly = True
    rows = np.array([25, 0, 8, 14, 25, 7, 0], dtype=np.int64)
    caplog.set_level("INFO")
    try:
        table._read_advisor = None
        expected = _bits(table.gather_mlx(rows)).copy()
        table._read_advisor = advisor
        actual = table.gather_mlx(rows)
        assert actual.dtype == dtype
        np.testing.assert_array_equal(_bits(actual), expected)
        assert len(calls) == 1 and len(calls[0]) == 12
        for reader, local in calls[0]:
            assert np.all(local >= 0) and np.all(local < reader.shape[0])
        assert table.read_advice_stats["calls"] == 1
        assert table.read_advice_stats["fallbacks"] == int(status in {"error", "partial", "limit"})
        if status != "resident":
            assert "reader=unchanged_mmap" in caplog.text
    finally:
        table.close()


def test_ineligible_and_pread_do_not_hint(tmp_path, monkeypatch):
    table = _table(tmp_path, [(2, 32), (6, 32)], mx.float16, 160)
    def forbidden(_):
        raise AssertionError("Hinted ineligible path")
    table._read_advisor = SimpleNamespace(advise=forbidden)
    try:
        rows = np.array([0, 1, 7], dtype=np.int64)
        table._read_host_assembled(rows, use_pread=True)
        monkeypatch.setattr(table_reader, "_READ_ADVICE_MAX_ROWS", 2)
        table._read_host_assembled(rows)
        assert table.read_advice_stats["calls"] == 0
    finally:
        table.close()


def test_real_reader_error_not_hidden_by_failed_hint(tmp_path, monkeypatch):
    table = _table(tmp_path, [(2, 32), (6, 32)], mx.float16, 160)
    table._read_advisor = SimpleNamespace(advise=lambda _: {"status": "error", "hinted": 0, "bytes": 0})
    error = OSError("real tensor read failure")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(table.shards[0], "read_rows", fail)
    try:
        with pytest.raises(OSError) as caught:
            table._read_host_assembled(np.array([0], dtype=np.int64))
        assert caught.value is error
    finally:
        table.close()
