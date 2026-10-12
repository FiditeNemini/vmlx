"""Block group files (2026-10-12): consecutive SSD-only blocks share one hard-linked file; reads are bit-identical.

A long store wrote one safetensors file per 64-token block and a restore paid a file open + header check + load +
deserialize + validation per block (80k Naive layout: 1,252 reads, 2.0-2.5 s). The writer now packs up to
VMLX_BLOCK_GROUP_FILES consecutive eligible blocks of one batch into one file hard-linked under every member's own
path; reconstruct reads a run of members as one span. These tests pin: identical reconstruction with groups on/off
(incl. a prefix ending inside a group and a CacheList + pending-rotating mixed layout), that groups really form and
share inodes, that a member's cleanup leaves the others readable, that nothing staged is left behind, and that the
budget charges a group's bytes once."""
import os

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402

import vmlx_engine.block_disk_store as bds  # noqa: E402
import vmlx_engine.paged_cache as pgc  # noqa: E402
import vmlx_engine.prefix_cache as pc  # noqa: E402


def _layout(T, n_full=3, n_swa=5, seed=7):
    """Naive-like: CacheList(KVCache attn, KVCache indexer) full layers + RotatingKVCache(64) SWA layers."""
    mx.random.seed(seed)
    full = [(KVCache(), KVCache()) for _ in range(n_full)]
    swa = [RotatingKVCache(max_size=64, keep=0) for _ in range(n_swa)]
    fed = 0
    while fed < T:
        n = min(512, T - fed)
        for attn, idx in full:
            attn.update_and_fetch(mx.random.normal((1, 2, n, 16)).astype(mx.bfloat16),
                                  mx.random.normal((1, 2, n, 16)).astype(mx.bfloat16))
            idx.update_and_fetch(mx.random.normal((1, 1, n, 8)).astype(mx.bfloat16),
                                 mx.zeros((1, 1, n, 0), dtype=mx.bfloat16))
        for c in swa:
            c.update_and_fetch(mx.random.normal((1, 2, n, 16)).astype(mx.bfloat16),
                               mx.random.normal((1, 2, n, 16)).astype(mx.bfloat16))
        fed += n
    mx.eval([a.state for a, _ in full], [i.state for _, i in full], [c.state for c in swa])
    data = []
    for li in range(n_full + n_swa):
        if li < n_full:
            attn, idx = full[li]
            data.append({"class_name": "CacheList", "state": None, "meta_state": None,
                         "sub_caches": [{"class_name": "KVCache", "state": attn.state, "meta_state": attn.meta_state},
                                        {"class_name": "KVCache", "state": idx.state, "meta_state": idx.meta_state}]})
        else:
            c = swa[li - n_full]
            data.append({"state": c.state, "meta_state": c.meta_state, "class_name": "RotatingKVCache"})
    return data


def _store(tmp, data, T, group):
    os.environ["VMLX_BLOCK_GROUP_FILES"] = str(group)
    store = bds.BlockDiskStore(str(tmp), max_size_gb=4, allow_tq_native=False)
    mgr = pgc.PagedCacheManager(block_size=64, max_blocks=4096, disk_store=store, disk_only=True)
    cache = pc.BlockAwarePrefixCache(model=None, paged_cache_manager=mgr)
    cache.store_cache("cold", list(range(T)), data)
    cache.settle_write_behind(wait=True)
    return store, cache


def _restore(cache, tokens):
    table, _rest = cache.fetch_cache("w", tokens)
    assert table is not None and table.block_ids
    restored = cache.reconstruct_cache(table)
    assert restored is not None
    mx.eval([c.state for c in restored])
    states = []
    for c in restored:
        sub = getattr(c, "caches", None)
        for s in (sub if sub else [c]):
            states.append([a for a in (s.state if isinstance(s.state, (list, tuple)) else [s.state]) if a is not None])
    cache.release_cache("w")
    return table.num_tokens, states


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VMLX_SSD_WRITE_BEHIND", "0")
    yield
    os.environ.pop("VMLX_BLOCK_GROUP_FILES", None)


@pytest.mark.parametrize("T", [2000, 1347])          # 1347: not a multiple of the 64-token block
@pytest.mark.parametrize("n_swa", [5, 0])
def test_group_files_reconstruct_bit_identical_to_one_file_per_block(tmp_path, T, n_swa):
    """n_swa=5: Naive-like mixed SWA (restorable only at the exact terminal, so only the full prefix);
    n_swa=0: CacheList-only, where a prefix ending INSIDE a group is restorable too."""
    data = _layout(T, n_swa=n_swa)
    s0, c0 = _store(tmp_path / "plain", data, T, 0)
    s1, c1 = _store(tmp_path / "group", data, T, 16)
    try:
        assert s0.group_files_written == 0 and s1.group_files_written > 0
        probes = [list(range(T)) + [1, 2, 3]]
        if n_swa == 0:
            probes.append(list(range(T // 2 + 37)) + [9])          # ends inside a group
        for tokens in probes:
            n0, a = _restore(c0, tokens)
            n1, b = _restore(c1, tokens)
            assert n0 == n1 > 0
            assert len(a) == len(b)
            for xa, xb in zip(a, b):
                assert len(xa) == len(xb)
                for p, q in zip(xa, xb):
                    assert p.shape == q.shape and mx.array_equal(p, q).item()
        assert getattr(s1, "group_span_reads", 0) > 0
        assert not s1._group_cache                       # released after every reconstruct
    finally:
        s0.shutdown(); s1.shutdown()


def test_group_members_share_one_inode_and_nothing_staged_remains(tmp_path):
    T = 2000
    store, _cache = _store(tmp_path, _layout(T), T, 16)
    try:
        files = [os.path.join(r, f) for r, _, fs in os.walk(tmp_path) for f in fs if f.endswith(".safetensors")]
        assert not [f for f in files if ".tmp." in f]
        inodes = {os.stat(f).st_ino for f in files}
        linked = [f for f in files if os.stat(f).st_nlink > 1]
        assert linked and len(inodes) < len(files)
        rows = store._read_conn.execute("SELECT group_size, file_size, file_name FROM blocks").fetchall()
        grouped = [r for r in rows if r[0]]
        assert grouped
        # a row charges its share: the shares of one group file sum to the file's size
        by_inode = {}
        for _gs, fsize, name in grouped:
            st = os.stat(tmp_path / name)
            by_inode.setdefault(st.st_ino, [st.st_size, 0])[1] += fsize
        assert all(size == charged for size, charged in by_inode.values())
    finally:
        store.shutdown()


def test_cleaning_one_member_keeps_the_rest_of_its_group_readable(tmp_path):
    T = 2000
    data = _layout(T)
    store, cache = _store(tmp_path, data, T, 16)
    try:
        table, _ = cache.fetch_cache("w", list(range(T)) + [1])
        hashes = [cache.paged_cache.allocated_blocks[b].block_hash for b in table.block_ids]
        cache.release_cache("w")
        rows = {r[0]: r for r in store._read_conn.execute("SELECT block_hash, group_size FROM blocks").fetchall()}
        member = next(i for i, h in enumerate(hashes) if rows.get(h.hex(), (None, None))[1] and i > 2)
        conn = store._read_conn
        import sqlite3
        wconn = sqlite3.connect(str(store._db_path))
        store._cleanup_entry(wconn, hashes[member].hex())       # removes it and its descendants
        wconn.commit(); wconn.close()
        remaining = {r[0] for r in conn.execute("SELECT block_hash FROM blocks").fetchall()}
        assert hashes[member].hex() not in remaining and hashes[member - 1].hex() in remaining
        # the earlier members of the same group file still read back
        assert store.read_block(hashes[member - 1]) is not None
    finally:
        store.shutdown()


def test_group_span_falls_back_when_a_member_row_is_gone(tmp_path):
    T = 2000
    store, cache = _store(tmp_path, _layout(T), T, 16)
    try:
        table, _ = cache.fetch_cache("w", list(range(T)) + [1])
        hashes = [cache.paged_cache.allocated_blocks[b].block_hash for b in table.block_ids]
        cache.release_cache("w")
        start = next(i for i, h in enumerate(hashes)
                     if (store._query_index_row("SELECT group_size FROM blocks WHERE block_hash = ?", (h.hex(),))[0] or (0,))[0])
        full = store._read_group_span_guarded(hashes[start:])
        assert full is not None and full[1] >= 2
        import sqlite3
        wconn = sqlite3.connect(str(store._db_path))
        wconn.execute("DELETE FROM blocks WHERE block_hash = ?", (hashes[start + 1].hex(),))
        wconn.commit(); wconn.close()
        short = store._read_group_span_guarded(hashes[start:])
        assert short is None or short[1] == 1 or short[1] < full[1]
        store.release_group_cache()
    finally:
        store.shutdown()


def test_ineligible_payloads_are_never_grouped():
    import numpy as np
    meta = lambda types: np.frombuffer(__import__("json").dumps({"__layer_types__": types}).encode(), dtype=np.uint8)
    assert bds._numpy_block_group_eligible({"__vmlx_block_meta__": meta({"0": "kv"}), "layer_0_keys": np.zeros((1, 1, 2, 2))})
    assert not bds._numpy_block_group_eligible({"__vmlx_block_meta__": meta({"0": "rotating_kv"})})
    assert not bds._numpy_block_group_eligible({"__vmlx_block_meta__": meta({"0": "cumulative"})})
    assert not bds._numpy_block_group_eligible({"__vmlx_block_meta__": meta({"0": "kv"}), "layer_0_keys_scales": np.zeros(2)})
    assert not bds._numpy_block_group_eligible({"layer_0_keys": np.zeros(2)})          # no metadata


def test_apportioned_size_charges_a_linked_file_once(tmp_path):
    f = tmp_path / "g.safetensors"
    f.write_bytes(b"x" * 1000)
    for i in range(3):
        os.link(f, tmp_path / f"m{i}.safetensors")
    sizes = [bds._apportioned_size(os.stat(p)) for p in [f] + [tmp_path / f"m{i}.safetensors" for i in range(3)]]
    assert sum(sizes) == 1000
