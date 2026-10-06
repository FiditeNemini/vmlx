# SPDX-License-Identifier: Apache-2.0
"""SSD (L2) tier for the DFlash2 session store.

WHY THIS EXISTS
---------------
A bundle that ships a DFlash2 drafter (Qwen3.8-27B JANG_4D / JANGH2) runs on
SimpleEngine (``--no-continuous-batching``), so the scheduler's prefix cache
and block-disk SSD tier are not in play.  Its only multiturn reuse was the
in-RAM ``_DFlash2SessionStore`` (4 entries): lost on restart, and a 5th
conversation evicted the 1st.  This module writes every stored session entry
to the managed SSD pool and reads it back on a RAM miss.

WHAT IS STORED (one safetensors file + JSON sidecar per entry)
--------------------------------------------------------------
Exactly what the RAM store holds: the target prompt cache (full-attention KV
and GatedDeltaNet recurrent state, via each cache object's ``state`` /
``meta_state``), the drafter cache when it was kept, the drafter's bridging
hidden slice (``draft_hidden_gap``), the token ids and ``cache_len``.  Full
precision, no re-quantisation: a restored entry is bit-identical to the RAM
entry it was written from.

HOW A PROMPT FINDS AN ENTRY
---------------------------
The RAM store matches "stored tokens are a prefix of the prompt".  The disk
store is keyed by the exact token prefix (sha256 of the uint32 ids), so the
lookup must guess where an entry could end.  Entries only ever end at two
places, both visible in the next prompt:
  * ``boundary`` entries end right BEFORE an ``<|im_start|>`` (the assistant
    generation tag of an earlier prompt), and
  * ``turn`` entries end right AFTER an EOS id (``<|im_end|>``) that closed a
    generated reply.
So the candidates are those positions in the new prompt, tried longest first.
The stored ids are compared with the prompt after load (hash collisions and
torn files are misses, never wrong state).

COST / TRADE-OFFS
-----------------
* Writes happen on a background thread after the arrays are evaluated on the
  generation thread, so the reply is not delayed.  A full queue drops the
  newest write (counted); the RAM tier still has that entry.
* Two writes per turn (boundary + turn entry).  Both are needed: reasoning
  templates strip <think> from history, so the next USER turn only matches
  the boundary entry, while a TOOL continuation matches the turn entry.
* Size is bounded by the same aggregate SSD budget as every other managed
  cache (``--block-disk-cache-max-gb`` / ``--block-disk-cache-max-percent``),
  with LRU eviction by the pool.  ``--disable-block-disk-cache`` turns it off.
* The namespace includes both bundles' file fingerprints (size + mtime of
  weights/configs) and the runtime package fingerprint, so a re-quantised
  bundle or upgraded runtime cannot replay old state.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from array import array
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

SCHEMA = "dflash2_session_v1"


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(array("I", tokens).tobytes()).hexdigest()


def signature_for(tokens: list[int]) -> str:
    return f"{len(tokens)}:{_token_digest(tokens)}"


def candidate_cuts(prompt: list[int], im_start_id: Optional[int], eos_ids: Iterable[int]) -> list[int]:
    """Prefix lengths at which a stored entry can end, longest first."""
    eos = {int(t) for t in eos_ids}
    cuts = set()
    for i, t in enumerate(prompt):
        if im_start_id is not None and t == im_start_id and i > 0:
            cuts.add(i)                    # boundary: tokens[:i], i = tag index
        if t in eos:
            cuts.add(i + 1)                # turn: tokens end with the EOS id
    return sorted((c for c in cuts if 0 < c < len(prompt)), reverse=True)


def _has_meta_setter(cache: Any) -> bool:
    prop = getattr(type(cache), "meta_state", None)
    return isinstance(prop, property) and prop.fset is not None


def _pack(prefix: str, caches: list, arrays: dict, layers: list) -> None:
    for i, c in enumerate(caches):
        state = list(c.state) if isinstance(c.state, (list, tuple)) else [c.state]
        nones = []
        for j, a in enumerate(state):
            if a is None:
                nones.append(j)
            else:
                arrays[f"{prefix}{i}.{j}"] = a
        meta = None
        if _has_meta_setter(c):
            m = c.meta_state
            meta = list(m) if isinstance(m, (list, tuple)) else m
        layers.append({"cls": type(c).__name__, "n": len(state), "none": nones,
                       "tuple": isinstance(c.state, (list, tuple)), "meta": meta})


def _unpack(prefix: str, factory: Callable[[], list], arrays: dict, layers: list) -> Optional[list]:
    fresh = factory()
    if len(fresh) != len(layers):
        return None
    for i, (dst, info) in enumerate(zip(fresh, layers)):
        if type(dst).__name__ != info["cls"]:
            return None
        state = [None if j in info["none"] else arrays[f"{prefix}{i}.{j}"] for j in range(info["n"])]
        dst.state = state if info["tuple"] else state[0]
        if info["meta"] is not None:
            dst.meta_state = tuple(info["meta"]) if isinstance(info["meta"], list) else info["meta"]
    return fresh


class DFlash2SessionSSD:
    """Write-behind SSD tier keyed by exact token prefix (see module doc)."""

    queue_depth = 2

    def __init__(self, *, root: str | Path, max_size_bytes: int, model_key: str):
        from .utils.omni_session_disk_store import OmniSessionDiskStore

        self.store = OmniSessionDiskStore(
            root=root, model_key=model_key, max_size_bytes=int(max_size_bytes), schema=SCHEMA
        )
        self.stats = {"writes": 0, "write_bytes": 0, "write_s": 0.0, "write_errors": 0,
                      "dropped": 0, "hits": 0, "hit_tokens": 0, "load_s": 0.0, "misses": 0}
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=self.queue_depth)
        self._worker = threading.Thread(target=self._run, name="dflash2-ssd-writer", daemon=True)
        self._worker.start()

    # ---- write ---------------------------------------------------------------
    def put(self, entry: dict) -> None:
        """Snapshot ``entry`` (a RAM-store dict) and queue it for SSD."""
        import mlx.core as mx

        arrays: dict = {}
        target_layers: list = []
        _pack("t", entry["target_cache"], arrays, target_layers)
        draft_layers = None
        if entry.get("draft_cache") is not None:
            draft_layers = []
            _pack("d", entry["draft_cache"], arrays, draft_layers)
            if entry.get("draft_hidden_gap") is not None:
                arrays["gap"] = entry["draft_hidden_gap"]
        tokens = [int(t) for t in entry["tokens"]]
        arrays["tokens"] = mx.array(tokens, dtype=mx.uint32)
        # Evaluate on the generation thread; the writer only copies bytes.
        mx.eval(list(arrays.values()))
        meta = {"kind": entry.get("kind", "turn"), "cache_len": int(entry["cache_len"]),
                "target": target_layers, "draft": draft_layers}
        try:
            self._q.put_nowait((signature_for(tokens), arrays, meta))
        except queue.Full:
            self.stats["dropped"] += 1
            logger.info("DFlash2 SSD write skipped (writer busy); entry stays in RAM only")

    def _run(self) -> None:
        import mlx.core as mx

        while True:
            sig, arrays, meta = self._q.get()
            t0 = time.perf_counter()
            try:
                path = self.store.save(
                    sig,
                    lambda tmp: mx.save_safetensors(str(tmp), arrays, {"dflash2": json.dumps(meta)}),
                )
                self.stats["writes"] += 1
                self.stats["write_bytes"] += path.stat().st_size
                self.stats["write_s"] += time.perf_counter() - t0
                logger.info("DFlash2 SSD stored %s entry: %s tokens, %.0f MB, %.2fs",
                            meta["kind"], sig.split(":")[0], path.stat().st_size / 1e6,
                            time.perf_counter() - t0)
            except Exception:
                self.stats["write_errors"] += 1
                logger.warning("DFlash2 SSD write failed", exc_info=True)
            finally:
                # A daemon waiting in get() otherwise retains the last full
                # snapshot indefinitely through these loop-local references.
                # Release completed payloads before signalling the write done.
                del arrays, meta
                self._q.task_done()

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait for queued writes (tests / shutdown)."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    # ---- read ----------------------------------------------------------------
    def take_matching(self, prompt: list[int], *, im_start_id: Optional[int], eos_ids: Iterable[int],
                      make_target: Callable[[], list], make_draft: Callable[[], list]) -> Optional[dict]:
        import mlx.core as mx

        t0 = time.perf_counter()
        for cut in candidate_cuts(prompt, im_start_id, eos_ids):
            prefix = prompt[:cut]
            loaded = self.store.load(signature_for(prefix), lambda p: mx.load(str(p), return_metadata=True))
            if loaded is None:
                continue
            arrays, metadata = loaded
            try:
                meta = json.loads(metadata["dflash2"])
                if arrays["tokens"].tolist() != prefix or int(meta["cache_len"]) >= len(prompt):
                    continue
                target = _unpack("t", make_target, arrays, meta["target"])
                if target is None:
                    continue
                draft = None
                if meta.get("draft") is not None:
                    draft = _unpack("d", make_draft, arrays, meta["draft"])
                mx.eval([a for a in arrays.values()])
            except Exception:
                logger.warning("DFlash2 SSD entry unreadable; treating as a miss", exc_info=True)
                continue
            dt = time.perf_counter() - t0
            self.stats["hits"] += 1
            self.stats["hit_tokens"] += int(meta["cache_len"])
            self.stats["load_s"] += dt
            return {"kind": meta["kind"], "tokens": prefix, "cache_len": int(meta["cache_len"]),
                    "target_cache": target, "draft_cache": draft,
                    "draft_hidden_gap": arrays.get("gap") if draft is not None else None,
                    "load_s": dt}
        self.stats["misses"] += 1
        return None

    def clear(self) -> int:
        self.flush(10.0)
        return self.store.clear()


def model_identity(target_path: str, draft_path: str) -> str:
    """Namespace key: both bundles' file fingerprints + runtime packages."""
    from .model_bundle_integrity import _bundle_fingerprint
    from .prefix_cache import runtime_cache_fingerprint

    parts = []
    for label, p in (("target", target_path), ("draft", draft_path)):
        root = Path(p).expanduser().resolve()
        parts.append(f"{label}={root}:{_bundle_fingerprint(root)[:16]}")
    parts.append(runtime_cache_fingerprint())
    return "|".join(parts)
