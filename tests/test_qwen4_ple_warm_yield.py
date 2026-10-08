"""PLE page-cache warm yields to foreground n-gram lookups (audit 2026-10-07).

A first request sent while the 57.6 GB Allosaurus warm ran waited behind it (2k TTFT 15.4 s vs 5.4 s with the pause).
The warm must pause while a PLE read is in flight, resume afterwards, never pause for forwards, and still read every
byte."""
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


def test_model_forward_does_not_pause_the_warm():
    """1.6.76 stamped every forward, so the warm starved while requests ran and cold 8k prefill fell 32-46 %."""
    import inspect
    import vmlx_engine.models.qwen4_exp.language as lang
    src = inspect.getsource(lang.Qwen4ExpTextModel.__call__)
    assert "mark_foreground" not in src


def test_default_grace_is_short_and_keyed_to_reads(tmp_path, monkeypatch):
    """Default 50 ms grace after the last READ: a read that just ended pauses the warm briefly, never for seconds."""
    import vmlx_engine.models.qwen4_exp.table_reader as tr
    monkeypatch.delenv("VMLX_QWEN4_PLE_WARM_YIELD_MS", raising=False)
    monkeypatch.delenv("VMLX_QWEN4_PLE_WARM_YIELD", raising=False)
    tr.mark_foreground()  # a read just finished
    table = _fake_table(tmp_path, 1 << 20)
    thread = table.start_page_cache_warm()
    thread.join(5)
    assert table.page_cache_warm["bytes"] == 3 * (1 << 20)
    assert table.page_cache_warm["yielded_s"] < 0.5


def test_yield_off_never_pauses(tmp_path, monkeypatch):
    import vmlx_engine.models.qwen4_exp.table_reader as tr
    monkeypatch.setenv("VMLX_QWEN4_PLE_WARM_YIELD", "off")
    table = _fake_table(tmp_path, 1 << 20)
    with tr._ForegroundRead():
        thread = table.start_page_cache_warm()
        thread.join(5)
    assert table.page_cache_warm["bytes"] == 3 * (1 << 20)
    assert table.page_cache_warm["yielded_s"] == 0.0


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
