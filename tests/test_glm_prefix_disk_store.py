"""Filesystem ownership and aggregate quota; no model or Metal work."""
import threading

from vmlx_engine.utils.glm_prefix_disk_store import GlmPrefixDiskStore


def test_two_namespaces_share_quota_and_invalidate_discovery(tmp_path):
    a = GlmPrefixDiskStore(tmp_path, "model-a", 65536)
    b = GlmPrefixDiskStore(tmp_path, "model-b", 65536)
    try:
        a.save("a" * 40, 2048, lambda p: p.write_bytes(b"a" * 40000))
        assert "a" * 40 in a.records()
        b.save("b" * 40, 4096, lambda p: p.write_bytes(b"b" * 40000))
        assert not a.contains("a" * 40, 2048)
        assert "a" * 40 not in a.records()
        assert b.load("b" * 40, 4096, lambda p: p.read_bytes()) == b"b" * 40000
        result = b.budget.refresh_health()
        assert result.accounted and result.compliant
        assert result.bytes_after <= result.max_size_bytes
        assert result.capacity_evicted_entries_total >= 1
    finally:
        a.close(); b.close()


def test_second_opener_cannot_delete_an_active_publication(tmp_path):
    a = GlmPrefixDiskStore(tmp_path, "same-model", 65536)
    entered, release = threading.Event(), threading.Event()
    errors, opened, pending = [], [], []

    def writer(path):
        path.write_bytes(b"snapshot")
        pending.append(path)
        entered.set()
        assert release.wait(5)

    def save():
        try:
            a.save("a" * 40, 2048, writer)
        except Exception as error:
            errors.append(error)

    def open_second():
        try:
            opened.append(GlmPrefixDiskStore(tmp_path, "same-model", 65536))
        except Exception as error:
            errors.append(error)

    write_thread = threading.Thread(target=save)
    open_thread = threading.Thread(target=open_second)
    write_thread.start()
    try:
        assert entered.wait(2)
        open_thread.start()
        open_thread.join(.1)
        assert open_thread.is_alive()
        assert pending[0].is_file()
    finally:
        release.set()
        write_thread.join(5)
        if open_thread.ident is not None:
            open_thread.join(5)
    try:
        assert not errors
        assert len(opened) == 1
        assert opened[0].load("a" * 40, 2048, lambda p: p.read_bytes()) == b"snapshot"
    finally:
        a.close()
        for store in opened:
            store.close()


def test_eviction_waits_until_reader_materialization_finishes(tmp_path):
    a = GlmPrefixDiskStore(tmp_path, "model-a", 65536)
    b = GlmPrefixDiskStore(tmp_path, "model-b", 65536)
    entered, release, wrote = threading.Event(), threading.Event(), threading.Event()
    errors, values = [], []
    a.save("a" * 40, 2048, lambda p: p.write_bytes(b"a" * 40000))

    def reader(path):
        entered.set()
        assert release.wait(5)
        return path.read_bytes()

    def read():
        try:
            values.append(a.load("a" * 40, 2048, reader))
        except Exception as error:
            errors.append(error)

    def write():
        try:
            b.save("b" * 40, 4096, lambda p: p.write_bytes(b"b" * 40000))
            wrote.set()
        except Exception as error:
            errors.append(error)

    rt, wt = threading.Thread(target=read), threading.Thread(target=write)
    rt.start()
    try:
        assert entered.wait(2)
        wt.start()
        wt.join(.1)
        assert not wrote.is_set()
    finally:
        release.set()
        rt.join(5)
        if wt.ident is not None:
            wt.join(5)
    try:
        assert not errors and wrote.is_set()
        assert values == [b"a" * 40000]
    finally:
        a.close(); b.close()


def test_clear_preserves_other_namespace_and_notes(tmp_path):
    a = GlmPrefixDiskStore(tmp_path, "model-a", 65536)
    b = GlmPrefixDiskStore(tmp_path, "model-b", 65536)
    try:
        a.save("a" * 40, 2048, lambda p: p.write_bytes(b"a"))
        b.save("b" * 40, 2048, lambda p: p.write_bytes(b"b"))
        note = a.directory / "NOTES.md"
        note.write_text("preserve")
        assert a.clear() == 1
        assert a.records() == {}
        assert note.read_text() == "preserve"
        assert b.load("b" * 40, 2048, lambda p: p.read_bytes()) == b"b"
    finally:
        a.close(); b.close()


def test_corrupt_record_cleanup_preserves_concurrent_replacement(tmp_path):
    store = GlmPrefixDiskStore(tmp_path, "model", 65536)
    try:
        path = store.save("a" * 40, 2048, lambda p: p.write_bytes(b"old"))
        old_identity = store.file_identity(path)
        store.save("a" * 40, 2048, lambda p: p.write_bytes(b"replacement"))
        assert store.remove("a" * 40, 2048, old_identity) is False
        assert store.load("a" * 40, 2048, lambda p: p.read_bytes()) == b"replacement"
        assert store.remove("a" * 40, 2048, store.file_identity(path)) is True
        assert not store.contains("a" * 40, 2048)
    finally:
        store.close()
