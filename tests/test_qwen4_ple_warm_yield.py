"""PLE page-cache warm yields to foreground n-gram lookups (audit 2026-10-07).

A first request sent while the 57.6 GB Allosaurus warm ran waited behind it (TTFT 15.9 s vs 0.74 s after the warm).
The warm must pause while lookups are active, resume afterwards, and still read every byte."""
import threading
import time
from types import SimpleNamespace

from vmlx_engine.models.qwen4_exp.table_reader import FileBackedQuantizedNGramTable


def _fake_table(tmp_path, size):
    path = tmp_path / "table.safetensors"
    path.write_bytes(b"\x01" * size)
    reader = SimpleNamespace(path=path, data_offset=0, data_nbytes=size)
    table = object.__new__(FileBackedQuantizedNGramTable)
    table.shards = [SimpleNamespace(weight=reader, scales=reader, biases=reader)]
    return table


def test_warm_pauses_while_lookups_run_and_reads_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PLE_WARM_YIELD_MS", "100")
    table = _fake_table(tmp_path, 3 << 20)
    stop = time.monotonic() + 0.4

    def lookups():
        while time.monotonic() < stop:
            table._mark_foreground()
            time.sleep(0.01)

    worker = threading.Thread(target=lookups)
    worker.start()
    time.sleep(0.02)
    thread = table.start_page_cache_warm()
    thread.join(5)
    worker.join()
    assert not thread.is_alive()
    # three spans of the same file (weight/scales/biases), all read
    assert table.page_cache_warm["bytes"] == 3 * (3 << 20)
    assert table.page_cache_warm["yielded_s"] >= 0.3


def test_warm_without_lookups_does_not_yield(tmp_path, monkeypatch):
    monkeypatch.setenv("VMLX_QWEN4_PLE_WARM_YIELD_MS", "250")
    import vmlx_engine.models.qwen4_exp.table_reader as tr
    tr._FOREGROUND_AT[0] = 0.0
    table = _fake_table(tmp_path, 1 << 20)
    thread = table.start_page_cache_warm()
    thread.join(5)
    assert table.page_cache_warm["bytes"] == 3 * (1 << 20)
    assert table.page_cache_warm["yielded_s"] == 0.0


def test_model_forward_marks_foreground():
    import inspect
    import vmlx_engine.models.qwen4_exp.language as lang
    src = inspect.getsource(lang.Qwen4ExpTextModel.__call__)
    assert "_ple_mark_foreground()" in src


def test_warm_never_runs_while_a_read_is_in_flight(tmp_path, monkeypatch):
    import vmlx_engine.models.qwen4_exp.table_reader as tr
    monkeypatch.setenv("VMLX_QWEN4_PLE_WARM_YIELD_MS", "50")
    tr._FOREGROUND_AT[0] = 0.0
    table = _fake_table(tmp_path, 1 << 20)
    with tr._ForegroundRead():
        thread = table.start_page_cache_warm()
        time.sleep(0.4)  # far past the 50 ms window: still blocked because a read is in flight
        assert table.page_cache_warm.get("running") is True
    thread.join(5)
    assert table.page_cache_warm["bytes"] == 3 * (1 << 20)
    assert table.page_cache_warm["yielded_s"] >= 0.3
