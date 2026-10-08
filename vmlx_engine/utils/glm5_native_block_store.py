"""Native GLM checkpoint manifests over the existing durable SSD transports.

Sequence payloads are content addressed; a checkpoint contains only its KDA
state and ordered block references. Missing dependencies fail as a cache miss.
This adapter is qualified independently before serving-path adoption.
"""
import hashlib
import json
import threading
from collections import OrderedDict

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache

from .glm5_native_blocks import (
    NativeBoundary, NativeSequenceBlock, restore_native_sequence,
    split_native_sequence,
)

SCHEMA = "glm5_native_blocks_v1"


def _block_hash(block, parent):
    h = hashlib.sha256(SCHEMA.encode())
    h.update(parent or b"")
    h.update(json.dumps([block.start, block.end]).encode())
    prepared = []
    for fragment in block.layers:
        if fragment is None:
            prepared.append(None)
            continue
        for array in fragment:
            raw = mx.contiguous(array).view(mx.uint8).reshape(-1)
            prepared.append((json.dumps([list(array.shape), str(array.dtype)]).encode(), raw))
    # Realize the bounded block as one graph, not one GPU submission per
    # layer/slot during the subsequent CPU digest transfers.
    mx.eval([item[1] for item in prepared if item is not None])
    for item in prepared:
        if item is None:
            h.update(b"recurrent")
        else:
            header, raw = item
            h.update(header)
            for start in range(0, raw.size, 1024 * 1024):
                h.update(np.array(raw[start:start + 1024 * 1024]))
    return h.digest()


def _fragment_layout(block):
    """Shapes and dtypes of a block's arrays (metadata only, no GPU sync)."""
    return tuple(
        None if fragment is None else tuple((tuple(array.shape), str(array.dtype)) for array in fragment)
        for fragment in block.layers
    )


def _digest_memo(states, offset):
    """Per-live-cache memo of full-block digests, or None when there is no live MLA cache to own it.

    Stored on the first Glm5MLACache object (one per request's live cache). A smaller boundary than any
    previously seen (trim / rollback) clears it: positions past the trim may be rewritten.
    """
    from ..models.glm5_next.glm5_next import Glm5MLACache

    owner = next((layer for layer in states if isinstance(layer, Glm5MLACache)), None)
    if owner is None:
        return None
    memo = getattr(owner, "_vmlx_native_block_digests", None)
    if memo is None or int(offset) < memo.get("__max_offset__", 0):
        memo = {}
        try:
            owner._vmlx_native_block_digests = memo
        except AttributeError:
            return None
    memo["__max_offset__"] = max(int(offset), memo.get("__max_offset__", 0))
    return memo


class Glm5NativeBlockStore:
    """Wrap an SSM checkpoint store and a same-namespace BlockDiskStore.

    Both transports must share the aggregate budget. Publication follows the
    block durability barrier. A later eviction may invalidate the checkpoint;
    fetch never returns a partially reconstructed state.
    """
    def __init__(self, checkpoints, blocks, *, block_size=256):
        self.checkpoints = checkpoints
        self.blocks = blocks
        self.block_size = block_size
        self.last_block_write = None
        self._lifecycle = threading.RLock()
        self._closed = False
        self._invalid_blocks = set()
        self._published_references = OrderedDict()

    def __getattr__(self, name):
        return getattr(self.checkpoints, name)

    def store(self, key, states, is_complete, token_ids, num_tokens):
        with self._lifecycle:
            if self._closed:
                return False
            return self._store(key, states, is_complete, token_ids, num_tokens)

    def _store(self, key, states, is_complete, token_ids, num_tokens):
        boundary, fragments = split_native_sequence(states, self.block_size)
        if not is_complete or boundary.offset != num_tokens or num_tokens > len(token_ids):
            return False
        references, hashes = [], []
        parent = None
        written = reused = 0
        memo = _digest_memo(states, boundary.offset)
        for fragment in fragments:
            # A FULL block of a live cache is immutable (split_native_sequence),
            # so its digest is computed once per cache object, not re-hashed
            # (GPU->CPU copy + sha256) at every later checkpoint. Re-hashing
            # the whole prefix made each per-chunk checkpoint O(history): a 32k
            # GLM prefill spent 22.9 s in 100 synchronous stores (0.05 s at 1k
            # -> 0.34 s at 32k). The parent digest and the fragment LAYOUT are
            # part of the memo key: a dense-only prefix keeps a zero-length DSA
            # pool that later materializes for the same block, and keying on
            # (start, end, parent) alone restored only 2,048 of 32k tokens.
            full = fragment.end - fragment.start == self.block_size
            memo_key = (fragment.start, fragment.end, parent, _fragment_layout(fragment))
            digest = memo.get(memo_key) if (full and memo is not None) else None
            if digest is None:
                digest = _block_hash(fragment, parent)
                if full and memo is not None:
                    memo[memo_key] = digest
            repair = digest in self._invalid_blocks
            if not repair and self.blocks.has_block(digest):
                reused += 1
            else:
                entries = [
                    ("skip",) if arrays is None else
                    ("cumulative", list(arrays), [fragment.end], SCHEMA)
                    for arrays in fragment.layers
                ]
                if not self.blocks.write_block_async(
                    digest, entries, fragment.end - fragment.start, parent_hash=parent,
                    replace_existing=repair,
                ):
                    return False
                written += 1
            references.append([fragment.start, fragment.end, digest.hex()])
            hashes.append(digest)
            parent = digest
        if self.blocks.wait_for_blocks(hashes, timeout=30.0) != set(hashes):
            return False
        self._invalid_blocks.difference_update(hashes)
        descriptor = {
            "schema": SCHEMA, "offset": boundary.offset,
            "metadata": boundary.metadata, "pooled": boundary.pooled,
            "blocks": references,
        }
        manifest = ArraysCache(1)
        manifest.cache = [mx.array(list(json.dumps(descriptor).encode()), dtype=mx.uint8)]
        packets = [manifest]
        for state in boundary.recurrent:
            if state is not None:
                packet = ArraysCache(len(state))
                packet.cache = list(state)
                packets.append(packet)
        accepted = self.checkpoints.store(key, packets, True, token_ids, num_tokens)
        if accepted:
            self._published_references[key] = tuple(hashes)
            self._published_references.move_to_end(key)
            while len(self._published_references) > 256:
                self._published_references.popitem(last=False)
        self.last_block_write = {"new_blocks": written, "reused_blocks": reused,
                                 "sequence_tokens": num_tokens, "admitted": accepted}
        return accepted

    def fetch(self, key):
        record = self.checkpoints.fetch(key)
        if not record or not record[1]:
            return None
        try:
            packets = record[0]
            raw = packets[0].cache[0]
            if raw.dtype != mx.uint8 or raw.ndim != 1 or raw.size > 4 * 1024 * 1024:
                return None
            descriptor = json.loads(np.array(raw).tobytes())
            if descriptor.get("schema") != SCHEMA:
                return None
            metadata = tuple(tuple(m) for m in descriptor["metadata"])
            pooled = tuple(descriptor["pooled"])
            if len(metadata) != len(pooled):
                return None
            recurrent, packet_index = [], 1
            for pool in pooled:
                if pool is None:
                    recurrent.append(tuple(packets[packet_index].cache))
                    packet_index += 1
                else:
                    recurrent.append(None)
            if packet_index != len(packets):
                return None
            boundary = NativeBoundary(descriptor["offset"], metadata, tuple(recurrent), pooled)
            fragments, parent = [], None
            for start, end, key_hex in descriptor["blocks"]:
                digest = bytes.fromhex(key_hex)
                if len(digest) != 32:
                    return None
                entries = self.blocks.read_block(digest)
                if entries is None or len(entries) != len(metadata):
                    self._invalid_blocks.add(digest)
                    return None
                arrays = []
                for entry, pool in zip(entries, pooled):
                    if pool is None:
                        if tuple(entry) != ("skip",):
                            return None
                        arrays.append(None)
                    else:
                        if len(entry) != 4 or entry[0] != "cumulative" or entry[3] != SCHEMA:
                            return None
                        arrays.append(tuple(entry[1]))
                fragment = NativeSequenceBlock(start, end, tuple(arrays))
                if _block_hash(fragment, parent) != digest:
                    self._invalid_blocks.add(digest)
                    return None
                fragments.append(fragment)
                parent = digest
            return restore_native_sequence(boundary, fragments), True
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            return None

    def has_complete(self, key):
        return self.fetch(key) is not None

    def wait_for_write(self, key, timeout=5.0):
        if not self.checkpoints.wait_for_write(key, timeout):
            return False
        references = self._published_references.get(key)
        if references is None:
            return self.has_complete(key)
        # These hashes came from state frozen by this writer, so a durability
        # fence needs file/index publication, not a second GPU reconstruction.
        return self.checkpoints.has_complete(key) and all(
            self.blocks.has_block(digest) for digest in references
        )

    def shutdown(self, timeout=None):
        with self._lifecycle:
            self._closed = True
            complete = self.checkpoints.shutdown(timeout)
            self.blocks.shutdown()
            return complete

    def clear(self):
        with self._lifecycle:
            self.checkpoints.clear()
            self.blocks.clear()
            self._invalid_blocks.clear()
            self._published_references.clear()

    def stats(self):
        result = self.checkpoints.stats()
        sequence = self.blocks.get_stats()
        result["checkpoint_bytes"] = result.get("bytes", 0)
        result["sequence_blocks"] = sequence
        result["bytes"] = result["checkpoint_bytes"] + sequence["disk_size_bytes"]
        result["native_storage"] = SCHEMA
        result["last_block_write"] = self.last_block_write
        return result
