"""Bounded, detached N-1 snapshots for Naive's mixed attention/indexer cache."""

import mlx.core as mx
from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache


def snapshot_size(cache, expected_tokens=None):
    """Validate one common native boundary before allocating any copies."""
    offsets = []
    total = 0

    def visit(entry):
        nonlocal total
        if type(entry) is CacheList:
            for child in entry.caches:
                visit(child)
        elif type(entry) in (KVCache, RotatingKVCache):
            offsets.append(int(entry.offset))
            for array in entry.state:
                total += int(array.nbytes)
        else:
            raise ValueError("unsupported Naive prompt-cache component")

    for entry in cache:
        visit(entry)
    if not offsets or min(offsets) <= 0 or len(set(offsets)) != 1:
        raise ValueError("Naive prompt-cache components have unequal boundaries")
    if expected_tokens is not None and offsets[0] != expected_tokens:
        raise ValueError("Naive prompt-cache boundary differs from its token key")
    return total


def clone_prompt_cache(cache):
    """Copy arrays plus ring metadata; FP32 indexer keys and empty V stay native."""
    snapshot_size(cache)

    def clone(entry):
        if type(entry) is CacheList:
            return CacheList(*(clone(child) for child in entry.caches))
        arrays = tuple(array * 1 for array in entry.state)
        mx.eval(*arrays)
        return type(entry).from_state(arrays, entry.meta_state)

    return [clone(entry) for entry in cache]


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
