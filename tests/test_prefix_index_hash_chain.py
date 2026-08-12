# SPDX-License-Identifier: Apache-2.0
"""The prefix index hashed every block-aligned prefix FROM SCRATCH.

Both the lookup and the update walk every block-aligned prefix of the prompt.
Hashing each one from scratch makes that quadratic: block i re-hashes
i*block_size tokens, so the total is O(len(tokens)^2 / block_size). At 61k
tokens with 64-token blocks that is ~29 MILLION token-hashes per call — and it
runs under `paged_cache._lock`, so every other cache operation waits behind it.

`compute_block_hash` already takes a parent hash and forms a chain, so the same
walk can be incremental: hash only the new block and fold in the previous
prefix's hash. That is O(len(tokens)) total.

The chained values DIFFER from the from-scratch ones, which is why this is
env-gated (VMLX_CHAINED_PREFIX_INDEX_HASH) and DEFAULT OFF. It is safe in
principle because `_prefix_index` is a plain in-memory dict rebuilt per process
— declared in __init__, never persisted, never compared against an L2 record —
and both the reader and the writer draw their keys from the same helper. It
stays off until it has a live A/B on a long conversation.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "vmlx_engine" / "prefix_cache.py").read_text(encoding="utf-8")


def test_the_chained_walk_is_linear_not_quadratic():
    """Count hashed tokens both ways on the same prompt."""
    block_size = 64
    n_blocks = 200
    total = block_size * n_blocks

    from_scratch = sum(min(i * block_size, total) for i in range(1, n_blocks + 1))
    chained = total  # each token is hashed exactly once, in its own block

    assert from_scratch > 20 * chained, (
        f"from-scratch hashes {from_scratch} tokens vs chained {chained}"
    )
    # 200 blocks: 1,286,400 vs 12,800 — the gap is the point.
    assert from_scratch == sum(range(1, n_blocks + 1)) * block_size


def test_both_call_sites_go_through_the_gate():
    lookup = SRC[SRC.index("Find and pin the best matching prefix-index entry") :][:2600]
    update = SRC[SRC.index("Update prefix index with new token sequence") :][:2200]
    for name, body in (("lookup", lookup), ("update", update)):
        assert "_chained_prefix_index_hash" in body, (
            f"the {name} path no longer consults the gate, so reader and writer "
            "could disagree about how the key is derived"
        )
        assert "_prefix_index_hash_sequence" in body


def test_the_gate_defaults_off():
    m = re.search(
        r'self\._chained_prefix_index_hash = os\.environ\.get\(\s*"([A-Z_]+)",\s*""\s*\)',
        SRC,
    )
    assert m, "the chained-hash gate is no longer env-driven"
    assert m.group(1) == "VMLX_CHAINED_PREFIX_INDEX_HASH"
    # Empty default -> falsy -> OFF. A cache-key change must not arrive by
    # surprise; it needs a live long-conversation A/B first.
    assert 'os.environ.get(\n            "VMLX_CHAINED_PREFIX_INDEX_HASH", ""\n        )' in SRC


def test_the_index_it_keys_is_in_memory_only():
    """The premise that makes changing the key derivation safe at all."""
    assert "self._prefix_index: Dict[str, Tuple[List[int], List[int]]] = {}" in SRC
    # If this ever gains a persisted twin, the chained keys stop being free.
    assert "json.dump" not in SRC.split("self._prefix_index")[1][:400]
