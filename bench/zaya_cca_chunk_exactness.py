#!/usr/bin/env python3
"""Is ZAYA CCA prefill chunk-boundary exact?

ISSUE-LEDGER L63: reusing a 24-token prefix on Zaya-8B changes the answer at
temperature 0 (cold 228 tokens / sha 335824b8, warm 119 / 74b2e5c7, each
reproducible). L63a narrowed it: the restore guards verify a chain CARRIES
terminal CCA conv_state/prev_hs, never that replaying it is EQUIVALENT to a
one-pass cold prefill.

This harness tests that hypothesis directly, with no cache and no engine
plumbing involved. It prefills the same tokens two ways --

    one-pass :  model(tokens,        cache=fresh)
    chunked  :  model(tokens[:k],    cache=fresh)  then  model(tokens[k:], cache=same)

-- and compares every array in the resulting caches, per layer.

If the states differ, prefix reuse can never be answer-exact for this family no
matter how the blocks are stored, and the fix has to be either
block-boundary-aligned reuse or a re-derive on restore. If they match, the
divergence is downstream of the state and the hunt continues there.

Usage:
    .venv/bin/python bench/zaya_cca_chunk_exactness.py \
        /Volumes/EricsLLMDrive/jangq-ai/Zaya-8B-JANG_4M [split_at]

`split_at` defaults to 24 -- the exact reuse length observed in L63.
"""

from __future__ import annotations

import sys


def _arrays_of(layer, prefix=""):
    """Yield (name, mx.array) for every array reachable on a cache layer.

    Cache layers are not uniform across families: ZAYA layers are
    CacheList(KVCache, ArraysCache) and ZayaNoStateCache, so walk generically
    rather than assuming attribute names.
    """
    import mlx.core as mx

    seen = []
    if layer is None:
        return seen
    if isinstance(layer, mx.array):
        return [(prefix, layer)]
    if isinstance(layer, (list, tuple)):
        for i, sub in enumerate(layer):
            seen.extend(_arrays_of(sub, f"{prefix}[{i}]"))
        return seen

    # `.state` is the canonical serialization surface every cache class exposes,
    # and it is what the block store itself persists -- so comparing it is the
    # same thing a restore would replay. Walk it recursively; ZAYA nests it as
    # tuples inside a CacheList.
    try:
        state = getattr(layer, "state", None)
    except Exception:
        state = None
    if state is not None and not isinstance(state, (str, bytes)):
        seen.extend(_arrays_of(state, f"{prefix}.state"))

    # CacheList exposes sub-caches as `.caches` (NOT `.cache`); missing this is
    # why an earlier version of this harness compared ZERO arrays and then
    # cheerfully printed "bit-identical".
    try:
        sub_caches = getattr(layer, "caches", None)
    except Exception:
        sub_caches = None
    if isinstance(sub_caches, (list, tuple)):
        for i, sub in enumerate(sub_caches):
            seen.extend(_arrays_of(sub, f"{prefix}.caches[{i}]"))

    for name in ("keys", "values", "conv_state", "prev_hs", "offset"):
        try:
            value = getattr(layer, name)
        except Exception:
            continue
        if isinstance(value, mx.array):
            seen.append((f"{prefix}.{name}", value))

    # Deduplicate by identity so `.state` and `.keys` pointing at the same
    # buffer are not counted twice.
    unique, ids = [], set()
    for name, arr in seen:
        if id(arr) in ids:
            continue
        ids.add(id(arr))
        unique.append((name, arr))
    return unique


def main() -> int:
    import mlx.core as mx

    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    model_path = sys.argv[1]
    split_at = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    from vmlx_engine.utils.tokenizer import load_model_with_fallback

    print(f"loading {model_path} ...")
    model, tokenizer = load_model_with_fallback(model_path)
    lm = getattr(model, "language_model", model)

    prompt = (
        "Write a detailed paragraph about the water cycle, then list five rivers. "
        + " ".join(f"Context line {i}: rivers carry sediment downstream." for i in range(40))
    )
    ids = tokenizer.encode(prompt)
    if len(ids) <= split_at + 8:
        print(f"prompt too short ({len(ids)} tokens) for split_at={split_at}")
        return 2
    tokens = mx.array([ids])
    print(f"prompt tokens={len(ids)}  split_at={split_at}")

    def prefill(chunks):
        cache = lm.make_cache() if hasattr(lm, "make_cache") else model.make_cache()
        for chunk in chunks:
            lm(chunk, cache=cache)
            mx.eval([c for c in cache if c is not None])
        return cache

    one_pass = prefill([tokens])
    chunked = prefill([tokens[:, :split_at], tokens[:, split_at:]])

    print(f"layers: one_pass={len(one_pass)} chunked={len(chunked)}")
    total = mismatched = skipped = 0
    worst = []

    for i, (a_layer, b_layer) in enumerate(zip(one_pass, chunked)):
        a_arrays = dict(_arrays_of(a_layer, f"L{i}"))
        b_arrays = dict(_arrays_of(b_layer, f"L{i}"))
        for name, a in a_arrays.items():
            b = b_arrays.get(name)
            if b is None or a.shape != b.shape:
                skipped += 1
                continue
            total += 1
            diff = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
            mx.eval(diff)
            d = float(diff)
            if d > 0:
                mismatched += 1
                worst.append((d, name, tuple(a.shape)))

    worst.sort(reverse=True)
    print()
    # POSITIVE CONTROL. An earlier version of this harness walked `.cache`
    # instead of `.caches`, found nothing, and printed "bit-identical" over a
    # comparison of ZERO arrays. A verdict from an empty comparison is worse
    # than no verdict, so refuse to emit one.
    if total == 0:
        print("arrays compared : 0")
        print(
            "\nABORT: the cache walker reached no arrays, so nothing was "
            "compared. This is NOT a pass -- fix the walker before reading any "
            "verdict from this script."
        )
        return 3
    print(f"arrays compared : {total}")
    print(f"arrays DIFFERING: {mismatched}")
    print(f"skipped (shape/absent): {skipped}")
    if worst:
        print("\nlargest absolute deltas:")
        for d, name, shape in worst[:12]:
            print(f"  {d:.6e}  {name}  shape={shape}")
        print(
            "\nVERDICT: chunked prefill does NOT reproduce one-pass state. "
            "Prefix reuse cannot be answer-exact for this family without "
            "boundary-aligned reuse or a re-derive on restore."
        )
    else:
        print(
            "\nVERDICT: chunked prefill is bit-identical to one-pass. "
            "The L63 divergence is NOT in the state itself -- look downstream "
            "(restore path, block assembly, or sampling entry state)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
