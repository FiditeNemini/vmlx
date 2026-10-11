"""Native GLM cache administration must not address the Qwen SSD namespace."""
import asyncio
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

pytest.importorskip('mlx.core')
from vmlx_engine import glm_dflash2 as runtime


def test_stats_are_native_metadata_and_not_a_shared_budget(tmp_path, monkeypatch):
    disk = runtime.PrefixDiskTier(tmp_path, 1000)
    try:
        disk.index['snapshot'] = {'n': 3, 'bytes': 123, 'used': 1.0}
        disk.writes = 2
        disk.hits = 1
        store = SimpleNamespace(entries=OrderedDict([((1, 2, 3), object())]), max_entries=12, disk=disk)
        monkeypatch.setattr(runtime, '_PREFIX_STORES', {1: store})
        monkeypatch.setattr(runtime, '_PREFIX_DISK', {'root': str(tmp_path), 'max_bytes': 1000})
        stats = runtime.session_cache_stats()
        assert stats['ram']['tokens'] == 3
        assert stats['ram']['entries'] == 1
        assert stats['ssd']['writes'] == 2
        assert stats['ssd']['hits'] == 1
        assert stats['ssd']['bundle_budget']['bytes_after'] == 123
        assert stats['ssd']['bundle_budget']['root'] == str(tmp_path)
        assert stats['ssd']['bundle_budget']['compliant'] is True
        assert stats['ssd']['shared_pool'] is False
        assert 'global_budget' not in stats['ssd']
    finally:
        disk.close()


def test_clear_drains_writer_and_only_removes_owned_snapshots(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def write(self, name, key, snap):
        entered.set()
        assert release.wait(3)
        (self.dir / f'{name}.safetensors').write_bytes(b'owned')
        with self._lock:
            self.index[name] = {'n': len(key), 'bytes': 5, 'used': 1}
            self._save_index()
        return {'durable': True}
    monkeypatch.setattr(runtime.PrefixDiskTier, '_write', write)
    disk = runtime.PrefixDiskTier(tmp_path, 1000)
    (tmp_path / 'NOTES.md').write_text('preserve')
    receipt = disk.put((1, 2), None)
    assert entered.wait(1)
    result = []
    thread = threading.Thread(target=lambda: result.append(disk.clear()))
    thread.start()
    try:
        thread.join(.05)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(3)
        disk.close()
    assert receipt.result()['durable']
    assert result == [1]
    assert disk.index == {}
    assert not list(tmp_path.glob('*.safetensors'))
    assert (tmp_path / 'NOTES.md').read_text() == 'preserve'


def test_ram_clear_preserves_disk(monkeypatch):
    disk = SimpleNamespace(clear=lambda: pytest.fail('RAM clear touched SSD'))
    store = SimpleNamespace(entries={1: object()}, disk=disk)
    monkeypatch.setattr(runtime, '_PREFIX_STORES', {1: store})
    assert runtime.clear_sessions(False) == 0
    assert store.entries == {}


def test_loaded_bundle_selects_native_cache_owner(monkeypatch):
    from vmlx_engine import server, dflash2_runtime
    for family in ('glm5_next', 'glm5_next_text'):
        monkeypatch.setattr(server, '_current_model_config', lambda: SimpleNamespace(family_name=family))
        assert server._active_dflash2_cache_runtime() is runtime
    monkeypatch.setattr(server, '_current_model_config', lambda: SimpleNamespace(family_name='qwen3_5'))
    assert server._active_dflash2_cache_runtime() is dflash2_runtime


@pytest.mark.asyncio
async def test_native_clear_excludes_generation_and_does_not_clear_busy_engine(monkeypatch):
    from vmlx_engine import server
    lock = asyncio.Lock()
    calls = []
    monkeypatch.setattr(runtime, 'clear_sessions', lambda include_ssd: calls.append(include_ssd))
    async def worker(fn):
        assert lock.locked()
        fn()
    monkeypatch.setattr(server, '_engine', SimpleNamespace(_generation_lock=lock, _run_model_call=worker))
    async with lock:
        assert await server._clear_glm_dflash2_cache(True) is False
    assert calls == []
    assert await server._clear_glm_dflash2_cache(True) is True
    assert calls == [True]
    assert not lock.locked()
