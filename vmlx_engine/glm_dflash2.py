"""DFlash2 speculative decoding for GLM-5.3-Flash (glm5_next) targets (2026-10-09).

Drafter: incoai/GLM-5.3-Flash-DFlash2 (z-lab DFlash2: 5 sliding-window qwen3 layers, block 8, two-tap grouped dynamic
conv, candidate selector top-16), loaded with ``dflash.model_mlx.load_draft`` from its ORIGINAL BF16 files and
optionally affine-8 quantized at load (``quantize_draft``); it shares the target's embedding and lm_head (``bind``).

GLM target contract (SGLang PR 36708 ``_prepare_aux_hidden_state`` + its unit test): the drafter's context taps are the
COMPLETED outputs of target layers ``dflash_config.target_layer_ids`` = [5, 14, 24, 33, 42], each contracted over the
4 mHC streams by an unweighted mean (the same contraction GLM applies before its final norm).

Verify / rollback: the verify forward runs [last committed token] + drafts with ``n_confirmed=1``, so every KDA layer
records the exact recurrent + conv state after each position (``Glm5KDACache.set_speculative_states``) and a partial
rejection restores the state after the accepted prefix (``rollback_speculative(rejected)``); MLA/DSA caches trim
(an incomplete indexer pool is dropped and recomputed from the packed history on the next append).
Cache invariant (as upstream dflash): the last emitted token is sampled but never forwarded.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn


class _MeanStreamTap:
    """Wraps one decoder layer; when armed, stores mean over the mHC stream axis of its output: (B, T, 4, D) -> (B, T, D)."""

    def __init__(self, layer, slot: int, store: list, armed: dict):
        self._layer, self._slot, self._store, self._armed = layer, slot, store, armed

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        if self._armed["on"]:
            self._store[self._slot] = mx.mean(out, axis=2)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


class GLMDFlashTarget:
    """Speculative-decoding view of a loaded glm5_next model (VLM wrapper or text Model)."""

    def __init__(self, model, layer_ids):
        lm = model.language_model if hasattr(model, "language_model") else model
        self.lm, self.inner = lm, lm.model
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self._store: list = [None] * len(self.layer_ids)
        self._armed = {"on": False}
        layers = self.inner.layers
        for slot, lid in enumerate(self.layer_ids):
            cur = layers[lid]
            if isinstance(cur, _MeanStreamTap):        # re-wrap safely (adapter rebuilt for the same model)
                cur = cur._layer
            layers[lid] = _MeanStreamTap(cur, slot, self._store, self._armed)

    def close(self):
        layers = self.inner.layers
        for lid in self.layer_ids:
            if isinstance(layers[lid], _MeanStreamTap):
                layers[lid] = layers[lid]._layer

    # dflash.bind() looks for .model.embed_tokens and .lm_head
    @property
    def model(self):
        return self.inner

    @property
    def lm_head(self):
        return self.lm.lm_head

    def make_cache(self):
        return self.lm.make_cache()

    def forward(self, ids: mx.array, cache, verify: bool, taps: bool = True, last_row: bool = False):
        self._armed["on"] = taps
        try:
            h = self.inner(ids, cache=cache, n_confirmed=1 if (verify and ids.shape[1] > 1) else 0, return_pre_norm=True)
        finally:
            self._armed["on"] = False
        if last_row:
            h = h[:, -1:]
        return self.lm.lm_head(self.inner.norm(h))

    def hidden(self) -> mx.array:
        return mx.concatenate(self._store, axis=-1)

    @staticmethod
    def rollback(cache, rejected: int) -> None:
        from vmlx_engine.models.glm5_next.glm5_next import Glm5KDACache

        for c in cache:
            if isinstance(c, Glm5KDACache):
                if rejected > 0:
                    if not c.rollback_speculative(rejected):
                        raise RuntimeError("glm dflash2: KDA speculative rollback failed (no recorded states)")
                else:
                    c.commit_speculative()
            elif rejected > 0:
                c.trim(rejected)


def load_glm_drafter(path: str, quantize_bits: int = 8, group_size: int = 64):
    """incoai DFlash2 drafter from its original files; affine `quantize_bits` at load (0 = keep BF16)."""
    from pathlib import Path

    import dflash.model_mlx as runtime

    original = runtime.snapshot_download
    if Path(path).expanduser().is_dir():                   # local files: never resolve as a Hub repo id
        runtime.snapshot_download = lambda model_id, **_kw: model_id
    try:
        draft = runtime.load_draft(str(Path(path).expanduser()))
    finally:
        runtime.snapshot_download = original
    if quantize_bits:
        quantize_draft(draft, quantize_bits, group_size)
    return draft


def quantize_draft(draft, bits: int = 8, group_size: int = 64) -> int:
    """Affine-quantize the drafter's Linear layers (attention, MLP, fc, selector projection) in place; the selector
    codebooks (embeddings) and the dynamic-conv parameters stay BF16. Returns the number of quantized modules."""
    count = {"n": 0}

    def pred(path, m):
        ok = isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0
        count["n"] += int(ok)
        return ok
    nn.quantize(draft, group_size=group_size, bits=bits, class_predicate=pred)
    mx.eval(draft.parameters())
    return count["n"]


@dataclass
class SpecStats:
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    tokens: int = 0
    decode_s: float = 0.0
    rounds: int = 0
    accepted: list = field(default_factory=list)
    ar_steps: int = 0                       # adaptive mode: tokens produced by plain target steps
    cached_tokens: int = 0                  # prompt tokens restored from the lane's prefix store
    cache_source: str = ""                  # "ram" | "disk" for a restored prefix (usage cache_detail)
    blocks: list = field(default_factory=list)   # block size used by each verify round

    @property
    def tok_per_round(self) -> float:
        return (sum(self.accepted) + len(self.accepted)) / max(1, len(self.accepted))

    @property
    def decode_tps(self) -> float:
        return self.tokens / max(self.decode_s, 1e-9)


def _prefill(target: GLMDFlashTarget, prompt: mx.array, cache, step: int, hidden_limit: int | None, taps: bool = True):
    chunks, start, logits = [], 0, None
    while start < prompt.size:
        rem = prompt.size - start
        end = start + (1 if rem == 1 else min(step, rem - 1)) if taps else start + min(step, rem)
        logits = target.forward(prompt[None, start:end], cache, verify=False, taps=taps, last_row=True)
        if taps:
            h = target.hidden()
            if hidden_limit is not None:
                if chunks:
                    h = mx.concatenate((chunks[0], h), axis=1)
                chunks = [h[:, -hidden_limit:]]
            else:
                chunks.append(h)
        mx.eval([c.state for c in cache], *(chunks[-1:] if chunks else []))
        start = end
    hidden = None
    if taps:
        hidden = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
    return logits, hidden


def generate_ar(target: GLMDFlashTarget, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                top_p: float = 1.0, prefill_step: int = 2048, stats: SpecStats | None = None) -> Iterator[int]:
    """Plain autoregressive control on the SAME forward path (no taps), for in-process A/B."""
    from dflash.model_mlx import make_sampler

    sampler = make_sampler(temperature, top_p, 0)
    prompt = mx.array(prompt_ids); cache = target.make_cache(); st = stats or SpecStats()
    t0 = time.perf_counter()
    logits, _ = _prefill(target, prompt, cache, prefill_step, None, taps=False)
    tok = sampler(logits[:, -1:]); mx.eval(tok)
    st.prompt_tokens, st.prefill_s = prompt.size, time.perf_counter() - t0
    t1 = time.perf_counter(); n = 0
    from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
    # pipelined like the served loop: the next step's graph is built (inside the productive-AR scope, which selects the
    # AR-only kernels) and queued BEFORE the current token is read back
    while True:
        if n + 1 < max_tokens:
            with affine_moe_ar_scope():
                logits = target.forward(tok.reshape(1, 1), cache, verify=False, taps=False)
            nxt = sampler(logits[:, -1:]); mx.async_eval(nxt)
        t = int(tok.item()); n += 1; yield t
        if t in eos or n >= max_tokens:
            break
        tok = nxt
    st.tokens, st.decode_s = n, time.perf_counter() - t1


class _PrefixSnapshot:
    __slots__ = ("cache", "hidden", "logits")

    def __init__(self, cache, hidden, logits):
        self.cache, self.hidden, self.logits = cache, hidden, logits


def _detached_copy(v):
    return mx.logical_and(v, True) if v.dtype == mx.bool_ else v * 1      # a new buffer, logical size only


def _clone_caches(cache, detach_mla: bool = False):
    """Isolated GLM native-state copies (the prefix caches' clone contract).

    detach_mla=True (snapshots taken DURING a live prefill): MLA state is really copied at its logical size. Sharing
    storage would mark the live cache's preallocated capacity buffers copy-on-write, and its next append then copies the
    whole buffer: boundary snapshots every 2048 tokens made an 8k cold prefill 15.7 -> 27.7 s (2026-10-10). KDA state is
    replaced functionally on every step, so it is shared safely either way."""
    from vmlx_engine.models.glm5_next.glm5_next import Glm5MLACache, clone_glm5_next_layer_cache
    out = []
    for c in cache:
        if detach_mla and isinstance(c, Glm5MLACache):
            out.append(clone_glm5_next_layer_cache(c, copy_fn=_detached_copy, copy_is_detached=True))
        else:
            out.append(clone_glm5_next_layer_cache(c, copy_fn=lambda v: v))
    return out


def _cache_nbytes(cache) -> int:
    return sum(int(v.nbytes) for c in cache for v in list(c.state) if v is not None)


class PrefixStore:
    """In-RAM prompt-prefix snapshots for the GLM DFlash2 lane (2026-10-10).

    The lane rebuilt the whole history every turn (served multi-turn: 18 s TTFT at 8k tokens, while the batched AR
    lane restored its prefix cache in 0.3 s). After each prompt prefill this keeps the target's KDA/MLA caches, the
    drafter's context rows (its sliding window of tapped hidden states) and the last logits, keyed by the prompt token
    ids. A request whose prompt starts with a stored prompt restores that state and prefills only the suffix. The
    drafter cache itself is rebuilt from the context rows exactly as after a cold prefill, so no drafter state is kept.
    Also snapshots every PREFIX_SNAPSHOT_EVERY (2048) absolute prompt tokens during prefill, for mid-history
    divergence. LRU over `max_entries` snapshots (VMLX_GLM_DFLASH2_PREFIX_ENTRIES, default 12) and a total cap
    (VMLX_GLM_DFLASH2_PREFIX_MAX_GB, default 3; shared KDA arrays are counted per snapshot, so this over-counts)."""

    def __init__(self, max_entries: int = 4, max_bytes: float = 3 * 2 ** 30, disk: "PrefixDiskTier | None" = None):
        from collections import OrderedDict
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = float(max_bytes)
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.disk = disk
        self.last_persistence = None
        self.last_source = ""                   # "ram" | "disk" | "" (miss) for the latest lookup

    def lookup(self, ids):
        best = None
        for key in self.entries:
            n = len(key)
            if n <= len(ids) and (best is None or n > len(best)) and tuple(ids[:n]) == key:
                best = key
        if self.disk is not None:
            # a longer prefix on SSD (written by an earlier server process, or evicted from RAM) beats a RAM hit
            found = self.disk.lookup(ids, min_len=len(best) if best is not None else 0)
            if found is not None:
                n, snap = found
                key = tuple(int(t) for t in ids[:n])
                self.entries[key] = snap
                self._evict()
                if key in self.entries:
                    self.entries.move_to_end(key)
                self.hits += 1
                self.last_source = "disk"
                return n, snap
        if best is None:
            self.misses += 1
            self.last_source = ""
            return None
        self.entries.move_to_end(best); self.hits += 1
        self.last_source = "ram"
        return len(best), self.entries[best]

    def put(self, ids, cache, hidden, logits):
        key = tuple(int(t) for t in ids)
        snap = _PrefixSnapshot(_clone_caches(cache, detach_mla=True), hidden, logits)
        mx.eval([c.state for c in snap.cache])
        self.entries[key] = snap
        self.entries.move_to_end(key)
        self._evict()
        if self.disk is not None:
            self.last_persistence = self.disk.put(key, snap)

    def _evict(self):
        while len(self.entries) > self.max_entries or (len(self.entries) > 1 and self.nbytes() > self.max_bytes):
            self.entries.popitem(last=False)
            self.evictions += 1

    def nbytes(self) -> int:
        return sum(_cache_nbytes(e.cache) for e in self.entries.values())


class PrefixDiskTier:
    """SSD tier of the GLM DFlash2 prefix store (2026-10-10): every snapshot the RAM store keeps is also written to
    `root` by a background thread (one safetensors file per snapshot: each layer's native cache state + typed
    meta_state, the drafter context rows, the last logits and the prompt ids), so a restarted server restores long
    histories instead of re-prefilling them. A RAM miss looks here for the longest stored prefix; the file's ids are
    compared token by token before use. `root` is per bundle (see configure_prefix_disk). LRU by last use over
    `max_bytes`; a file that fails to load is deleted and treated as a miss."""

    FORMAT = "glm-dflash2-prefix-v1"

    def __init__(self, root, max_bytes: float):
        import queue
        import threading
        from pathlib import Path
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = float(max_bytes)
        self.hits = self.misses = self.writes = self.evictions = self.errors = 0
        self._lock = threading.Lock()
        self.index: dict[str, dict] = {}
        self._load_index()
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="glm-dflash2-prefix-disk", daemon=True)
        self._thread.start()

    @staticmethod
    def digest(ids) -> str:
        import hashlib

        import numpy as np
        return hashlib.sha256(np.asarray(list(ids), dtype=np.int64).tobytes()).hexdigest()[:40]

    # index: <dir>/index.json {name: {"n", "bytes", "used"}}; files written as <name>.part.safetensors, then renamed
    def _load_index(self):
        import json
        p = self.dir / "index.json"
        try:
            raw = json.loads(p.read_text()) if p.exists() else {}
        except Exception:
            raw = {}
        for name, e in raw.items():
            if (self.dir / f"{name}.safetensors").exists():
                self.index[name] = {"n": int(e["n"]), "bytes": int(e["bytes"]), "used": float(e["used"])}
        for f in self.dir.iterdir():                       # unfinished writes (<name>.part) and unindexed files
            if f.suffix == ".safetensors" and f.stem not in self.index:
                f.unlink(missing_ok=True)

    def _save_index(self):
        import json
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(self.index))
        os.replace(tmp, self.dir / "index.json")

    def nbytes(self) -> int:
        with self._lock:
            return sum(e["bytes"] for e in self.index.values())

    def put(self, key: tuple, snap: "_PrefixSnapshot"):
        from concurrent.futures import Future

        receipt = Future()
        name = self.digest(key)
        with self._lock:
            if name in self.index:
                self.index[name]["used"] = time.time()
                receipt.set_result({"outcome": "already_durable", "durable": True,
                                    "retained_tokens": len(key), "detail": "glm-dflash2-prefix-disk"})
                return receipt
        self._queue.put((name, key, snap, receipt))
        return receipt

    def flush(self) -> None:
        self._queue.join()

    def clear(self) -> int:
        """Clear this bundle's indexed snapshots while serving is excluded."""
        self.flush()  # A queued write must not recreate an entry after Clear.
        with self._lock:
            names = list(self.index)
            for name in names:
                (self.dir / f"{name}.safetensors").unlink(missing_ok=True)
                self.index.pop(name, None)
            self._save_index()
        return len(names)

    def close(self) -> None:
        """Drain queued snapshots and release the writer at model teardown.

        The serving lifecycle has already stopped generation before calling
        this method. Normal generation continues to enqueue without waiting.
        """
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _writer(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                name, key, snap, receipt = item
                receipt.set_result(self._write(name, key, snap))
            except Exception as e:                         # never fail a generation over the SSD tier
                self.errors += 1
                import logging
                logging.getLogger(__name__).warning("glm dflash2 prefix disk write failed: %s", e)
                if not receipt.done():
                    receipt.set_result({"outcome": "failed", "durable": False,
                                        "retained_tokens": 0, "detail": str(e)[:200]})
            finally:
                self._queue.task_done()

    def _write(self, name, key, snap):
        import json
        arrays = {"ids": mx.array(list(key), dtype=mx.int32), "hidden": snap.hidden, "logits": snap.logits}
        layers = []
        for i, c in enumerate(snap.cache):
            state = list(c.state)
            layers.append([type(c).__name__, [str(m) for m in c.meta_state], [v is not None for v in state]])
            for j, v in enumerate(state):
                if v is not None:
                    arrays[f"c{i}.{j}"] = v
        tmp = self.dir / f"{name}.part.safetensors"
        mx.save_safetensors(str(tmp), arrays, metadata={"format": self.FORMAT, "layers": json.dumps(layers)})
        os.replace(tmp, self.dir / f"{name}.safetensors")
        size = (self.dir / f"{name}.safetensors").stat().st_size
        with self._lock:
            self.index[name] = {"n": len(key), "bytes": int(size), "used": time.time()}
            self.writes += 1
            while len(self.index) > 1 and sum(e["bytes"] for e in self.index.values()) > self.max_bytes:
                old = min((k for k in self.index if k != name), key=lambda k: self.index[k]["used"])
                self.index.pop(old)
                (self.dir / f"{old}.safetensors").unlink(missing_ok=True)
                self.evictions += 1
            if sum(e["bytes"] for e in self.index.values()) > self.max_bytes:     # this snapshot alone exceeds it
                self.index.pop(name)
                (self.dir / f"{name}.safetensors").unlink(missing_ok=True)
                self.evictions += 1
            self._save_index()
            retained = name in self.index
            return {"outcome": "stored" if retained else "refused", "durable": retained,
                    "retained_tokens": len(key) if retained else 0,
                    "detail": "glm-dflash2-prefix-disk" if retained else "snapshot exceeds disk budget"}

    def lookup(self, ids, min_len: int = 0):
        with self._lock:
            cands = sorted({e["n"] for e in self.index.values() if min_len < e["n"] <= len(ids)}, reverse=True)
            by_name = dict(self.index)
        for n in cands:
            name = self.digest(ids[:n])
            if name not in by_name:
                continue
            snap = self._read(name, ids[:n])
            if snap is not None:
                with self._lock:
                    if name in self.index:
                        self.index[name]["used"] = time.time()
                self.hits += 1
                return n, snap
        self.misses += 1
        return None

    def _read(self, name, ids):
        import json

        from vmlx_engine.models.glm5_next import glm5_next as G
        path = self.dir / f"{name}.safetensors"
        try:
            arrays, meta = mx.load(str(path), return_metadata=True)
            if meta.get("format") != self.FORMAT or arrays["ids"].tolist() != [int(t) for t in ids]:
                raise ValueError("prefix snapshot does not match")
            cache = []
            for i, (cls_name, meta_state, present) in enumerate(json.loads(meta["layers"])):
                cls = {"Glm5KDACache": G.Glm5KDACache, "Glm5MLACache": G.Glm5MLACache}[cls_name]
                state = [arrays[f"c{i}.{j}"] if p else None for j, p in enumerate(present)]
                cache.append(cls.from_state(state, tuple(meta_state)))
            return _PrefixSnapshot(cache, arrays["hidden"], arrays["logits"])
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("glm dflash2 prefix snapshot %s unusable (%s); deleted", name, e)
            self.errors += 1
            with self._lock:
                self.index.pop(name, None)
                path.unlink(missing_ok=True)
                self._save_index()
            return None


_PREFIX_STORES: dict = {}
PREFIX_SNAPSHOT_EVERY = int(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_EVERY", "2048"))
_PREFIX_DISK = {"root": None, "max_bytes": 0.0}


def session_cache_stats() -> dict:
    """Metadata-only view of the native store; never query the Qwen store.

    The native disk quota is per bundle, not the generic shared SSD pool.
    Keep that scope explicit until the stores share a budget coordinator.
    """
    stores = list(_PREFIX_STORES.values())
    keys = [key for store in stores for key in list(store.entries)]
    result = {"runtime": "glm-dflash2", "ram": {
        "entries": len(keys),
        "max_entries": sum(store.max_entries for store in stores),
        "tokens": sum(len(key) for key in keys),
        "tokens_note": "sum_across_entries; prefixes_may_overlap",
    }, "ssd": None}
    disks = list({id(store.disk): store.disk for store in stores if store.disk is not None}.values())
    if not _PREFIX_DISK["root"] and not disks:
        return result
    size = sum(disk.nbytes() for disk in disks)
    cap = _PREFIX_DISK["max_bytes"]
    import math
    limit = int(cap) if math.isfinite(cap) else 0
    result["ssd"] = {
        "hits": sum(d.hits for d in disks), "misses": sum(d.misses for d in disks),
        "writes": sum(d.writes for d in disks), "write_errors": sum(d.errors for d in disks),
        "pending_writes": sum(d._queue.unfinished_tasks for d in disks),
        "budget_scope": "bundle", "shared_pool": False,
        "bundle_budget": {
            "root": _PREFIX_DISK["root"], "scope": "bundle",
            "bytes_after": size, "max_size_bytes": limit,
            "accounted": bool(disks), "telemetry_stale": False,
            "compliant": bool(disks) and (not limit or size <= limit),
            "evicted_entries_total": sum(d.evictions for d in disks),
        },
    }
    return result


def clear_sessions(include_ssd: bool) -> int:
    """Called on the model worker with the engine generation lock held."""
    removed = 0
    for store in list(_PREFIX_STORES.values()):
        if include_ssd and store.disk is not None:
            removed += store.disk.clear()
        store.entries.clear()
    return removed


def _bundle_fingerprint(model_path) -> str:
    """Bundle identity for the SSD tier: path + the small metadata files + shard names/sizes/mtimes."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256(PrefixDiskTier.FORMAT.encode())
    p = Path(model_path).expanduser().resolve()
    h.update(str(p).encode())
    for name in ("config.json", "jang_config.json", "model.safetensors.index.json"):
        f = p / name
        if f.exists():
            h.update(f.read_bytes())
    for f in sorted(p.glob("*.safetensors")):
        s = f.stat()
        h.update(f"{f.name}:{s.st_size}:{int(s.st_mtime)}".encode())
    return h.hexdigest()[:16]


def configure_prefix_disk(enabled: bool, block_cache_dir, max_gb, model_path) -> None:
    """CLI hook (non-batched launches): the DFlash2 lane's SSD tier follows the server's block disk cache flags
    (on unless --disable-block-disk-cache; --disable-prefix-cache governs the batched scheduler only; --block-disk-cache-dir;
    --block-disk-cache-max-gb,
    where 0 = unlimited as for the block store). Files live in the sibling directory
    '<block cache dir>-glm-dflash2/<bundle fingerprint>', outside the block store's own budget scan.
    VMLX_GLM_DFLASH2_DISK_DIR (a directory, or 0 = off) and VMLX_GLM_DFLASH2_DISK_MAX_GB override."""
    from pathlib import Path
    env_dir = os.environ.get("VMLX_GLM_DFLASH2_DISK_DIR")
    root = None
    if env_dir is not None:
        root = None if env_dir in ("", "0") else Path(env_dir).expanduser()
    elif enabled:
        base = Path(block_cache_dir or Path.home() / ".cache" / "vmlx-engine" / "block-cache").expanduser()
        root = Path(str(base) + "-glm-dflash2")
    env_gb = os.environ.get("VMLX_GLM_DFLASH2_DISK_MAX_GB")
    gb = float(env_gb) if env_gb else float(max_gb if max_gb is not None else 10.0)
    if root is not None and model_path:
        root = root / _bundle_fingerprint(model_path)
    _PREFIX_DISK.update(root=str(root) if root is not None else None,
                        max_bytes=gb * 2 ** 30 if gb > 0 else float("inf"))
    close_prefix_stores()


def close_prefix_stores() -> None:
    """Called only between model lifetimes, never on a generation path."""
    for store in list(_PREFIX_STORES.values()):
        if store.disk is not None:
            store.disk.close()
    _PREFIX_STORES.clear()


def shutdown_runtime() -> None:
    """Finish SSD writes and remove adapters retaining an unloaded model."""
    close_prefix_stores()
    for target in list(_TARGETS.values()):
        target.close()
    _TARGETS.clear()


def prefix_store_for(target) -> "PrefixStore":
    key = id(target)
    st = _PREFIX_STORES.get(key)
    if st is None:
        disk = PrefixDiskTier(_PREFIX_DISK["root"], _PREFIX_DISK["max_bytes"]) if _PREFIX_DISK["root"] else None
        st = _PREFIX_STORES[key] = PrefixStore(int(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_ENTRIES", "12")),
                                                float(os.environ.get("VMLX_GLM_DFLASH2_PREFIX_MAX_GB", "3")) * 2 ** 30,
                                                disk=disk)
    return st


class AdaptiveSpec:
    """Online speculation policy (2026-10-10): draft only when it pays, at the block size that pays most.

    Per draft position i (1-based) it keeps decayed counts of how often position i was reached (all earlier drafts
    accepted) and accepted, i.e. the conditional acceptance p_i. Expected tokens of a b-token verify round:
    E(b) = 1 + sum_{i<b} prod_{j<=i} p_j. Costs are measured: an EMA of the wall time of a round per block size (draft +
    verify, synchronized by the acceptance read-back) and of one plain target step. Each decision picks
    argmax_b E(b)/cost(b) if that beats 1/cost(AR) by `margin`, else an AR burst; after `probe_every` AR tokens one
    round is forced at the best estimated block so p_i keeps tracking the text (code drafts well, prose does not).
    Unmeasured block costs are interpolated linearly from the measured ones (prior: AR x (1 + slope (b-1)); slope 0.22
    = measured GLM verify cost per extra row after the 2026-10-10 verify kernels: 2 rows 1.18x, 4 rows 1.61x, 8 rows
    2.55x an AR step). Every `explore_every` rounds the next round tries a neighbour of the best block (b+1, then
    b-1) so the cost/acceptance tables are filled by measurement, not by the prior (with a 0.35 prior the policy
    locked onto blocks 2-3 while block 4 was 6-12% faster).
    """

    def __init__(self, bmax: int, bmin: int = 2, decay: float = 0.97, prior_p: float = 0.6, prior_n: float = 2.0,
                 probe_every: int = 24, margin: float = 0.05, ar_burst: int = 8, slope: float = 0.22,
                 explore_every: int = 6):
        self.bmax, self.bmin = int(bmax), int(bmin)
        self.decay, self.probe_every, self.margin, self.ar_burst = decay, probe_every, margin, ar_burst
        self.n_try = [prior_n] * (self.bmax + 1)
        self.n_acc = [prior_n * prior_p] * (self.bmax + 1)
        self.t_ar = None
        self.t_round: dict[int, float] = {}
        self.ar_since_probe = 0
        self.slope, self.explore_every = slope, explore_every
        self.rounds_since_explore = 0
        self._explore_flip = False

    def p(self, i: int) -> float:
        return self.n_acc[i] / max(self.n_try[i], 1e-9)

    def expected(self, b: int) -> float:
        e, c = 1.0, 1.0
        for i in range(1, b):
            c *= self.p(i); e += c
        return e

    def cost(self, b: int) -> float:
        if b in self.t_round:
            return self.t_round[b]
        known = sorted(self.t_round)
        if len(known) >= 2:
            (b0, c0), (b1, c1) = (known[0], self.t_round[known[0]]), (known[-1], self.t_round[known[-1]])
            slope = max((c1 - c0) / max(b1 - b0, 1), 0.0)
            return max(c0 + slope * (b - b0), 1e-6)
        base = self.t_ar if self.t_ar is not None else 1.0
        if len(known) == 1:
            k = known[0]
            return self.t_round[k] * (1 + self.slope * (b - 1)) / (1 + self.slope * (k - 1))
        return base * (1 + self.slope * (b - 1))

    def choose(self) -> tuple[str, int]:
        cands = range(self.bmin, self.bmax + 1)
        best = max(cands, key=lambda b: self.expected(b) / self.cost(b))
        if self.t_ar is None:                       # need one AR measurement before comparing
            return ("ar", self.ar_burst)
        if not self.t_round:                        # and one round
            return ("spec", best)
        spec_rate = self.expected(best) / self.cost(best)
        # hysteresis: AR only when it is estimated at least `margin` faster than the best block (round costs carry
        # read-back overhead that AR bursts hide; switching on a near-tie loses)
        if spec_rate * (1.0 + self.margin) >= 1.0 / self.t_ar or self.ar_since_probe >= self.probe_every:
            self.ar_since_probe = 0
            self.rounds_since_explore += 1
            if self.rounds_since_explore >= self.explore_every:
                self.rounds_since_explore = 0
                self._explore_flip = not self._explore_flip
                nb = best + 1 if self._explore_flip else best - 1
                if self.bmin <= nb <= self.bmax:
                    return ("spec", nb)
            return ("spec", best)
        return ("ar", self.ar_burst)

    def update_spec(self, b: int, accepted: int, dt: float, after_ar: bool = False) -> None:
        """after_ar: the round right after an AR burst feeds the drafter all of the burst's context rows and is
        slower than a steady-state round; its acceptance counts, its time does not."""
        for i in range(1, self.bmax + 1):
            self.n_try[i] *= self.decay; self.n_acc[i] *= self.decay
        for i in range(1, b):
            if i <= accepted + 1:
                self.n_try[i] += 1.0
            if i <= accepted:
                self.n_acc[i] += 1.0
        if not after_ar or b not in self.t_round:
            self.t_round[b] = dt if b not in self.t_round else 0.7 * self.t_round[b] + 0.3 * dt

    def update_ar(self, dt_per_token: float, tokens: int) -> None:
        self.t_ar = dt_per_token if self.t_ar is None else 0.7 * self.t_ar + 0.3 * dt_per_token
        self.ar_since_probe += tokens


def _dflash2_rounds(target: GLMDFlashTarget, draft, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                    top_p: float = 1.0, top_k: int = 0, block_size: int | None = None, prefill_step: int = 2048,
                    stats: SpecStats | None = None, controls=None, adaptive: bool = False,
                    prefix_store: "PrefixStore | None" = None):
    """Yields (new_tokens, drafted) per cycle: first the token sampled from the prompt (drafted 0), then each verify
    round's accepted drafts + bonus token (drafted = verify width - 1). Upstream dflash loop with GLM verify/rollback.
    controls: optional vmlx_engine.dflash2_sampling.DFlash2SamplingControls (min_p, logit_bias, penalties) applied to
    the TARGET distribution of every verified row, as the Qwen lane does (rejection sampling stays exact)."""
    import dflash.model_mlx as runtime
    from dflash.model_mlx import DFlash2DraftModel, _rejection_sample, _sampling_probs, _trim_recent_cache, make_sampler

    sampler = make_sampler(temperature, top_p, top_k)
    bs0 = int(block_size or draft.config.block_size)
    mask_id = int(draft.config.mask_token_id)
    prompt = mx.array(prompt_ids); st = stats if stats is not None else SpecStats()
    dcache = draft.make_cache(); draft.bind(target)
    hidden_limit = draft.config.sliding_window - 1 if all(t == "sliding_attention" for t in draft.config.layer_types) else None
    t0 = time.perf_counter()
    hit = prefix_store.lookup(prompt_ids) if prefix_store is not None else None
    reused, cache = 0, None
    if hit is not None:
        reused, snap = hit
        cache = _clone_caches(snap.cache)
    if cache is None:
        cache = target.make_cache()
    if reused == prompt.size:                       # whole prompt already prefilled
        logits, hidden = snap.logits, snap.hidden
    else:
        # Prefill the suffix; with a prefix store, stop at every absolute multiple of PREFIX_SNAPSHOT_EVERY and keep a
        # snapshot there too, so a later prompt that diverges mid-history (same long prefix, different last turn)
        # restores the last boundary before the divergence instead of starting cold. KDA recurrent state cannot be
        # truncated, so only stored boundaries are restorable.
        every = PREFIX_SNAPSHOT_EVERY if prefix_store is not None else 0
        pos, hidden = reused, (snap.hidden if reused else None)
        while pos < prompt.size:
            end = min(prompt.size, (pos // every + 1) * every) if every else prompt.size
            logits, h = _prefill(target, prompt[pos:end], cache, prefill_step, hidden_limit)
            hidden = h if hidden is None else mx.concatenate([hidden, h], axis=1)
            if hidden_limit is not None:
                hidden = hidden[:, -hidden_limit:]
            pos = end
            if every and pos < prompt.size:
                mx.eval(logits, hidden, [c.state for c in cache])
                prefix_store.put(prompt_ids[:pos], cache, hidden, logits[:, -1:])
    st.cached_tokens = reused
    st.cache_source = prefix_store.last_source if (prefix_store is not None and reused) else ""
    if prefix_store is not None:
        mx.eval(logits, hidden, [c.state for c in cache])
        prefix_store.put(prompt_ids, cache, hidden, logits[:, -1:])
    for c in dcache:
        c.offset = prompt.size - hidden.shape[1]
    history = list(prompt_ids)
    first = logits[:, -1:]
    if controls is not None:
        first = controls.process(first, history)
        tok = runtime._sample_probs(controls.probabilities(runtime, first, temperature, top_p, top_k)) if temperature > 0 \
            else mx.argmax(first, axis=-1)
    else:
        tok = sampler(first)
    mx.eval(tok, hidden)
    st.prompt_tokens, st.prefill_s = prompt.size, time.perf_counter() - t0
    t1 = time.perf_counter()
    tokens = [int(tok.item())]; n = 1
    try:
        yield [tokens[-1]], 0
        if tokens[-1] in eos:
            return
        policy = AdaptiveSpec(bs0) if adaptive else None

        def sample_row(lg, hist):
            last = lg[:, -1:]
            if controls is not None:
                last = controls.process(last, hist)
                return (runtime._sample_probs(controls.probabilities(runtime, last, temperature, top_p, top_k))
                        if temperature > 0 else mx.argmax(last, axis=-1))
            return sampler(last)

        def ar_step(tok_arr):
            from vmlx_engine.metal.affine_moe_pair_decode import affine_moe_ar_scope
            with affine_moe_ar_scope():
                lg = target.forward(tok_arr, cache, verify=False, taps=True)
            return lg, target.hidden()

        def ar_burst(k):
            """k plain target steps (pipelined when no history-dependent controls); keeps the drafter's tap rows.
            Invariant kept: the last emitted token is not forwarded (an extra step launched past EOS / max_tokens only
            touches this generation's private cache)."""
            emitted, rows = [], []
            pipelined = controls is None
            lg, h = ar_step(mx.array([[tokens[-1]]])); nx = sample_row(lg, history + tokens); mx.async_eval(nx, h)
            for j in range(k):
                launched = None
                if pipelined and j + 1 < k:
                    lg2, h2 = ar_step(nx.reshape(1, 1)); nx2 = sample_row(lg2, None); mx.async_eval(nx2, h2)
                    launched = (nx2, h2)
                t = int(nx.item()); emitted.append(t); rows.append(h)
                if t in eos or n + len(emitted) >= max_tokens or j + 1 >= k:
                    break
                if launched is None:
                    lg, h = ar_step(nx.reshape(1, 1)); nx = sample_row(lg, history + tokens + emitted)
                    mx.async_eval(nx, h)
                else:
                    nx, h = launched
            return emitted, rows

        after_ar = False
        while n < max_tokens:
            mode, b = policy.choose() if policy is not None else ("spec", bs0)
            if mode == "ar":
                after_ar = True
                ta = time.perf_counter()
                new, rows = ar_burst(min(b, max_tokens - n))
                policy.update_ar((time.perf_counter() - ta) / max(len(new), 1), len(new))
                hidden = mx.concatenate([hidden] + rows, axis=1)
                if hidden_limit is not None and hidden.shape[1] > hidden_limit:
                    drop = hidden.shape[1] - hidden_limit          # outside the drafter's sliding window anyway
                    hidden = hidden[:, drop:]
                    for c in dcache:
                        c.offset += drop
                st.ar_steps += len(new)
                stop = next((i for i, t in enumerate(new) if t in eos), None)
                if stop is not None:
                    new = new[: stop + 1]
                tokens.extend(new); n += len(new)
                yield new, 0
                if stop is not None:
                    break
                continue
            bs = min(b, max_tokens - n + 1)
            if bs <= 1:
                break
            tr = time.perf_counter()
            block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
            if isinstance(draft, DFlash2DraftModel):
                d_tok, d_idx, d_prob = draft.propose(block, hidden, dcache, temperature, logits_start=1)
            else:
                dl = draft(block, hidden, dcache, logits_start=1)
                d_idx = None
                if temperature > 0:
                    d_prob = _sampling_probs(dl, temperature, top_p, top_k); d_tok = mx.argmax(d_prob, axis=-1)
                else:
                    d_prob = None; d_tok = mx.argmax(dl, axis=-1)
            if (trim_n := dcache[0].offset - (prompt.size + n - 1)) > 0:
                _trim_recent_cache(dcache, trim_n)
            mx.async_eval(d_tok)
            verify_in = mx.concatenate([mx.array([[tokens[-1]]]), d_tok], axis=1)
            logits = target.forward(verify_in, cache, verify=True)
            hidden = target.hidden()
            if controls is not None:
                logits = controls.process(logits, history + tokens, d_tok)
            if temperature > 0:
                t_prob = controls.probabilities(runtime, logits, temperature, top_p, top_k) if controls is not None else \
                    _sampling_probs(logits, temperature, top_p, top_k)
                mx.async_eval(t_prob, hidden)
            else:
                t_tok = mx.argmax(logits, axis=-1); mx.async_eval(t_tok, hidden)
            dl_ = d_tok[0].tolist()
            if temperature > 0:
                accepted, bonus = _rejection_sample(d_tok, t_prob, d_prob, d_idx)
            else:
                tl_ = t_tok[0].tolist()
                accepted = next((i for i in range(len(dl_)) if dl_[i] != tl_[i]), len(dl_))
                bonus = tl_[accepted]
            new = (dl_[:accepted] + [bonus])[: max_tokens - n]
            st.rounds += 1; st.accepted.append(accepted); st.blocks.append(bs)
            if policy is not None:
                policy.update_spec(bs, accepted, time.perf_counter() - tr, after_ar=after_ar)
            after_ar = False
            stop = next((i for i, t in enumerate(new) if t in eos), None)
            if stop is not None:
                new = new[: stop + 1]
            tokens.extend(new); n += len(new)
            yield new, bs - 1
            if stop is not None:
                break
            target.rollback(cache, bs - accepted - 1)
            hidden = hidden[:, : accepted + 1, :]
    finally:
        st.tokens, st.decode_s = n, time.perf_counter() - t1


def generate_dflash2(target: GLMDFlashTarget, draft, prompt_ids, max_tokens: int, eos: set, temperature: float = 0.0,
                     top_p: float = 1.0, block_size: int | None = None, prefill_step: int = 2048,
                     stats: SpecStats | None = None, adaptive: bool = False,
                     prefix_store: "PrefixStore | None" = None) -> Iterator[int]:
    """DFlash2 speculative generation as a token iterator. Greedy output == generate_ar's (also with adaptive=True:
    AR bursts and verify rounds both emit the target's argmax)."""
    for new, _ in _dflash2_rounds(target, draft, prompt_ids, max_tokens, eos, temperature, top_p, 0, block_size,
                                  prefill_step, stats, adaptive=adaptive, prefix_store=prefix_store):
        yield from new


_TARGETS: dict = {}


def target_for(model, layer_ids) -> GLMDFlashTarget:
    """One tap adapter per loaded model (the taps wrap the model's layers in place)."""
    key = id(model)
    t = _TARGETS.get(key)
    if t is None or t.layer_ids != tuple(int(i) for i in layer_ids):
        t = GLMDFlashTarget(model, layer_ids); _TARGETS[key] = t
    return t


def is_glm5_next(model) -> bool:
    cfg = getattr(model, "config", None)
    mt = getattr(cfg, "model_type", None) if cfg is not None and not isinstance(cfg, dict) else (cfg or {}).get("model_type")
    return mt == "glm5_next" or getattr(model, "model_type", None) == "glm5_next"


def stream_glm_dflash2(model, tokenizer, draft, prompt: str, *, max_tokens: int, temperature: float, top_p: float = 1.0,
                       top_k: int = 0, stop=None, prompt_tokens=None, media=None, sampling_controls=None):
    """Server lane: yields upstream ``dflash.model_mlx.GenerationResponse`` chunks (text segment, tokens, accepted = tokens
    emitted this cycle, drafted = verify width - 1) for a glm5_next target. Text-only: media requests must take the
    ordinary VLM path (the caller checks ``media``)."""
    from dflash.model_mlx import GenerationResponse
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if media is not None:
        # mllm._dflash2_media_plan returns None for glm5_next (no M-RoPE prefill interface), so image/video requests
        # always take the native VLM path; reaching here with media would be a caller bug.
        raise ValueError("glm5_next DFlash2 lane received a media plan; media requests use the native VLM path")
    tw = tokenizer if isinstance(tokenizer, TokenizerWrapper) else TokenizerWrapper(tokenizer)
    if prompt_tokens is None:
        specials = tuple(t for t in ("[gMASK]", getattr(tw, "bos_token", None)) if t)
        prompt_tokens = tw.encode(prompt, add_special_tokens=not any(prompt.startswith(t) for t in specials))
    eos = set(getattr(tw, "eos_token_ids", None) or ()) | {154820, 154827, 154829}
    detok = tw.detokenizer; detok.reset()
    if stop:
        from vmlx_engine.dflash2_runtime import _StopDetokenizer
        detok = _StopDetokenizer(detok, [stop] if isinstance(stop, str) else list(stop))
    target = target_for(model, draft.config.target_layer_ids)
    st = SpecStats(); tic = time.perf_counter(); n = 0
    store = (prefix_store_for(target)
             if os.environ.get("VMLX_GLM_DFLASH2_PREFIX_CACHE", "1") != "0" else None)
    for new, drafted in _dflash2_rounds(target, draft, list(prompt_tokens), int(max_tokens), eos, float(temperature),
                                        float(top_p), int(top_k), stats=st, controls=sampling_controls,
                                        adaptive=os.environ.get("VMLX_GLM_DFLASH2_ADAPTIVE", "1") != "0",
                                        prefix_store=store):
        if n == 0:
            tic = time.perf_counter()
        for t in new:
            if t not in eos:                  # stop tokens end the turn; their text (<|user|>) is not content
                detok.add_token(t)
        n += len(new)
        finished = new[-1] in eos or n >= max_tokens or getattr(detok, "matched", False)
        reason = None
        if finished:
            detok.finalize(); reason = "stop" if (new[-1] in eos or getattr(detok, "matched", False)) else "length"
        r = GenerationResponse(detok.last_segment, list(new), len(new), st.prompt_tokens,
                               st.prompt_tokens / max(st.prefill_s, 1e-9), n, n / max(time.perf_counter() - tic, 1e-9),
                               mx.get_peak_memory() / 1e9, reason)
        r.drafted = int(drafted)
        r.cached_tokens = int(st.cached_tokens)        # prompt tokens restored from the prefix store (usage)
        # usage prompt_tokens_details.cache_detail, like the batched lanes' "native-glm+disk" / "block-disk"
        r.cache_detail = {"ram": "glm-dflash2-prefix", "disk": "glm-dflash2-prefix+disk"}.get(st.cache_source, "")
        if finished and store is not None:
            # Publish the terminal immediately; SimpleEngine waits for this
            # request's SSD receipt only at the next admission or teardown.
            r.persistence_future = store.last_persistence
        yield r
        if finished:
            return
