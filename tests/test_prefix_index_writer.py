# SPDX-License-Identifier: Apache-2.0
"""Index construction preserves existing keys without rehashing all prefixes."""

import mlx.core as mx
import pytest

from vmlx_engine.cache_key import CACHE_EXTRA_SCOPES_KEY
from vmlx_engine.paged_cache import PagedCacheManager, compute_block_hash
from vmlx_engine.prefix_cache import BlockAwarePrefixCache


def _cache(block_size, chained=False):
    result = BlockAwarePrefixCache(
        model=None,
        paged_cache_manager=PagedCacheManager(block_size=block_size, max_blocks=256),
    )
    result._chained_prefix_index_hash = chained
    return result


def _extra(kind):
    return {
        "none": None,
        "empty": {},
        "global": {"lora": [1, "23"], "conditions": {"ab": "c"}},
        "media": {
            "image": "image-a", "video": "video-b",
            CACHE_EXTRA_SCOPES_KEY: {"image": 4, "video": 8},
        },
        "mixed": {
            "model": "same", "image": "image-a",
            CACHE_EXTRA_SCOPES_KEY: {"image": 1},
        },
        "bf16": {"embedding": mx.array([1.0, 2.0], dtype=mx.bfloat16)},
    }[kind]


@pytest.mark.parametrize("chained", [False, True], ids=["legacy", "chained"])
@pytest.mark.parametrize("block_size,count", [(1, 1), (1, 3), (4, 0), (4, 1), (4, 4), (4, 9), (64, 131)])
@pytest.mark.parametrize("extra_kind", ["none", "empty", "global", "media", "mixed", "bf16"])
def test_writer_keys_equal_existing_full_prefix_key_consumer(chained, block_size, count, extra_kind):
    cache = _cache(block_size, chained)
    tokens = [(-1 if i % 2 else 1) * (i * 100000000003) for i in range(count)]
    blocks = list(range((count + block_size - 1) // block_size))
    extra = _extra(extra_kind)
    # The unchanged full-prefix consumer is the compatibility oracle. Check
    # EVERY boundary, not just the last key or final chain tip.
    expected = {}
    for i in range(1, len(blocks) + 1):
        prefix = tokens[:i * block_size]
        expected[cache._prefix_index_key(prefix, extra)] = (
            prefix, blocks[:i], cache._prefix_index_extra_marker(extra, len(prefix)),
        )
    cache._update_prefix_index(tokens, blocks, extra)
    assert cache._prefix_index == expected
    # The request's mutable lists must not become the index's live storage.
    tokens[:] = [-999]
    blocks[:] = [-999]
    assert cache._prefix_index == expected


@pytest.mark.parametrize("extra_kind", ["none", "global", "media"])
def test_legacy_writer_serializes_each_token_at_most_twice(extra_kind):
    calls = []

    class CountedToken(int):
        def __str__(self):
            calls.append("str")
            return str(int(self))

        def __repr__(self):
            calls.append("repr")
            return repr(int(self))

    tokens = [CountedToken(i) for i in range(4097)]
    cache = _cache(64)
    blocks = list(range((len(tokens) + 63) // 64))
    cache._update_prefix_index(tokens, blocks, _extra(extra_kind))
    assert len(cache._prefix_index) == len(blocks)
    assert len(calls) <= 2 * len(tokens), (
        f"writer serialized {len(calls)} token values for {len(tokens)} tokens"
    )


@pytest.mark.parametrize("extra,root,parent", [
    (None, "b14a57538f62c0d262f111292d7d0cd3fe5640bdcc0c0992e3939de9d0a8deb1", "2aa148b8ecb07c2bead0f14e6cd1ad082a4be2aa80483ccb8d35e01480ef145e"),
    ({}, "71a888a3812eaf34d70194496646d1eb41b1012278f67e38f9e6d344b7476880", "0b5dda9525978c587721323dccc1e0d2aedf50dda45c544e9853126f1ba0a9ee"),
    ({"a": "bc"}, "6e01406cffdaea17f64ea3e68ffd99ba6d5d464c2b95346497b3a2d25e4b727f", "e31bf014ed456eb49527e11d056c927f4f649af9df2d1c42e734aaf23e085eca"),
    ({"ab": "c"}, "f5f53e77fcbf37ee7e08dfb6db8d0fc5d131abf306b51378ffd8c0d3bbb19807", "47c6e466c15adc60592ddd55f2e5515bf1451fb8f2b9b797d5f1d9fe46cd621d"),
    (mx.array([1, 2], dtype=mx.bfloat16), "0174c197273f1f9d049bfad4505ab3487619242516199523196c770c7d85d0d2", "c83f3dd2233c14cb4d1731fcbe4322f1882963af7364375407e3be150de54dac"),
    (mx.array([1, 2], dtype=mx.uint16), "60e6b30f9883b91aeee89c937b44fc5f084465480d74abf5e7b7c350062a846a", "6aa033446cfb46563c28c858aac645d66f74bde89036755d4e436fbfdf3872eb"),
])
def test_native_block_hash_serialization_matches_prechange_golden(extra, root, parent):
    # Captured at b2041c78 before extracting the canonical extra-key encoder.
    tokens = [1, -2, 987654321]
    assert compute_block_hash(None, tokens, extra).hex() == root
    assert compute_block_hash(bytes(range(32)), tokens, extra).hex() == parent
