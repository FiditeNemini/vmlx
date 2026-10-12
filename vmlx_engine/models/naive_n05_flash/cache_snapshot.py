"""Bounded, detached N-1 snapshots for Naive's mixed attention/indexer cache."""

import mlx.core as mx
from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache


def _compact_swa_length(entry, arrays):
    """Admit only oversized chronological, sink-free native SWA state."""
    if type(entry) is not RotatingKVCache or len(arrays) != 2:
        return None
    keys, values = arrays
    if keys.ndim != 4 or values.ndim != 4:
        return None
    length = int(keys.shape[2])
    keep, size, offset, idx = map(int, entry.meta_state)
    if (int(values.shape[2]) == length and keep == 0 and size > 0
            and length > size and idx == length and offset >= length):
        return size
    return None


def snapshot_size(cache, expected_tokens=None):
    """Validate the boundary and estimate detached copy bytes without allocation."""
    offsets = []
    total = 0

    def visit(entry):
        nonlocal total
        if type(entry) is CacheList:
            for child in entry.caches:
                visit(child)
        elif type(entry) in (KVCache, RotatingKVCache):
            offsets.append(int(entry.offset))
            arrays = entry.state
            compact_length = _compact_swa_length(entry, arrays)
            for array in arrays:
                total += (int(array.nbytes) if compact_length is None else
                          int(array.nbytes) // int(array.shape[2]) * compact_length)
        else:
            raise ValueError("unsupported Naive prompt-cache component")

    for entry in cache:
        visit(entry)
    if not offsets or min(offsets) <= 0 or len(set(offsets)) != 1:
        raise ValueError("Naive prompt-cache components have unequal boundaries")
    if expected_tokens is not None and offsets[0] != expected_tokens:
        raise ValueError("Naive prompt-cache boundary differs from its token key")
    return total


def cache_offset(cache):
    """The single token offset shared by every leaf of a Naive cache (ValueError when they disagree)."""
    offsets = set()

    def visit(entry):
        if type(entry) is CacheList:
            for child in entry.caches:
                visit(child)
        else:
            offsets.add(int(entry.offset))

    for entry in cache:
        visit(entry)
    if len(offsets) != 1:
        raise ValueError("Naive cache components have unequal offsets")
    return offsets.pop()


def _clone_leaf(entry):
    arrays = entry.state
    metadata = entry.meta_state
    if _compact_swa_length(entry, arrays) is not None:
        from ...prefix_cache import _rotating_terminal_window

        keys, values, size, keep, offset, idx = _rotating_terminal_window(
            *arrays, metadata, expected_offset=entry.offset,
            preserve_bounded_ring=True,
        )
        arrays = (keys, values)
        metadata = tuple(map(str, (keep, size, offset, idx)))
    # Detach even normalized suffix views from their oversized source backing.
    arrays = tuple(array * 1 for array in arrays)
    mx.eval(*arrays)
    return type(entry).from_state(arrays, metadata)


def clone_prompt_cache(cache):
    """Copy arrays plus ring metadata; FP32 indexer keys and empty V stay native."""
    snapshot_size(cache)

    def clone(entry):
        if type(entry) is CacheList:
            return CacheList(*(clone(child) for child in entry.caches))
        return _clone_leaf(entry)

    return [clone(entry) for entry in cache]


def clone_rotating_windows(cache):
    """Boundary capture for periodic checkpoints: copy only the sliding-window leaves.

    Naive's sliding window is small (128 rows), so this costs a few MB however long the prompt is. The append-only
    full-attention and indexer leaves are taken later as slices of the N-1 snapshot (`assemble_boundary_checkpoint`).
    Returns a structure mirroring the cache: ("list", children) for CacheList, the clone for RotatingKVCache, None
    for KVCache.
    """
    snapshot_size(cache)

    def capture(entry):
        if type(entry) is CacheList:
            return ("list", [capture(child) for child in entry.caches])
        if type(entry) is RotatingKVCache:
            return _clone_leaf(entry)
        return None

    return [capture(entry) for entry in cache]


def assemble_boundary_checkpoint(snapshot, windows, boundary):
    """Exact cache at an earlier prompt boundary: the N-1 snapshot's append-only leaves sliced to `boundary` plus
    the sliding windows captured when the prefill stood exactly there."""
    boundary = int(boundary)

    def build(entry, win):
        if type(entry) is CacheList:
            if not (isinstance(win, tuple) and win and win[0] == "list" and len(win[1]) == len(entry.caches)):
                raise ValueError("boundary windows do not match the cache structure")
            return CacheList(*(build(child, w) for child, w in zip(entry.caches, win[1])))
        if type(entry) is RotatingKVCache:
            if type(win) is not RotatingKVCache or int(win.offset) != boundary:
                raise ValueError("boundary window is missing or at another offset")
            return win
        if type(entry) is not KVCache or win is not None:
            raise ValueError("unsupported Naive boundary component")
        if int(entry.offset) < boundary:
            raise ValueError("snapshot is shorter than the boundary")
        keys, values = entry.state
        leaf = KVCache()
        leaf.keys, leaf.values = keys[..., :boundary, :], values[..., :boundary, :]
        leaf.offset = boundary
        return leaf

    if len(snapshot) != len(windows):
        raise ValueError("boundary windows do not match the cache structure")
    out = [build(entry, win) for entry, win in zip(snapshot, windows)]
    snapshot_size(out, expected_tokens=boundary)
    return out


def terminal_cache_key(cache, all_tokens, prompt_tokens):
    """Bind consumed native state to raw tokens, including generated history.

    The single-request generator normally consumes the reported token through
    lookahead, but skips that forward at a length limit. Use the actual common
    native offset rather than assuming either accounting convention. No rewind
    of the rotating window or rendered-text retokenization is permitted.
    """
    snapshot_size(cache)
    first = cache[0]
    while type(first) is CacheList:
        first = first.caches[0]
    offset = int(first.offset)
    tokens = list(all_tokens)
    prompt = list(prompt_tokens)
    if (
        not prompt
        or tokens[:len(prompt)] != prompt
        or offset < len(prompt)
        or len(tokens) - offset not in (0, 1)
    ):
        raise ValueError("Naive terminal state differs from its raw token sequence")
    return tokens[:offset]
