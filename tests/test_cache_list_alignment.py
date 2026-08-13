"""CacheList prefix-cache alignment.

Families that build one ``CacheList`` per layer got NO prefix cache at all: the
paged/L2 store refused every nonempty CacheList wrapper, so every layer failed
alignment, every store was skipped, and each turn re-prefilled from scratch.
falcon_h1 was found this way -- two turns, zero cache hits, four
"Skipping paged cache store" warnings and no other signal.

``CacheList(KVCache(), KVCache())`` (deepseek_v32, longcat_flash,
longcat_flash_ngram) is position-indexed on both sides and aligns cleanly.
``CacheList(ArraysCache(...), KVCache())`` (falcon_h1, baichuan_m1) carries a
cumulative sub that cannot be sliced to a token boundary and must still refuse.
"""

import mlx.core as mx
import pytest

from vmlx_engine.scheduler import (
    _align_attention_state_dict,
    _align_cache_list_state_dict,
    _blocking_cache_list_sub_class,
)


def _kv_sub(length, cls_name="KVCache", heads=4, dim=8):
    return {
        "class_name": cls_name,
        "state": (
            mx.ones((1, heads, length, dim)),
            mx.ones((1, heads, length, dim)),
        ),
        "meta_state": (str(length),),
    }


def _quantized_sub(length, cls_name="QuantizedKVCache", heads=4, dim=8):
    part = lambda: tuple(mx.ones((1, heads, length, dim)) for _ in range(3))
    return {
        "class_name": cls_name,
        "state": (part(), part()),
        "meta_state": (str(length), "64", "8"),
    }


def _cache_list(*subs):
    return {
        "class_name": "CacheList",
        "state": None,
        "meta_state": None,
        "sub_caches": list(subs),
    }


def _seq_len(sub):
    keys = sub["state"][0]
    if isinstance(keys, tuple):
        return int(keys[0].shape[-2])
    return int(keys.shape[-2])


class TestAlignAttentionStateDict:
    def test_plain_4d_state_is_sliced_to_target(self):
        aligned = _align_attention_state_dict(_kv_sub(100), 60)
        assert aligned is not None
        assert _seq_len(aligned) == 60
        assert aligned["meta_state"] == ("60",)

    def test_plain_3d_state_is_sliced_on_axis_1(self):
        sub = {
            "class_name": "KVCache",
            "state": (mx.ones((4, 100, 8)), mx.ones((4, 100, 8))),
            "meta_state": ("100",),
        }
        aligned = _align_attention_state_dict(sub, 60)
        assert aligned is not None
        assert int(aligned["state"][0].shape[1]) == 60

    def test_quantized_components_are_each_sliced(self):
        aligned = _align_attention_state_dict(_quantized_sub(100), 60)
        assert aligned is not None
        keys, values = aligned["state"]
        assert all(int(t.shape[-2]) == 60 for t in keys)
        assert all(int(t.shape[-2]) == 60 for t in values)
        # group_size/bits must survive untouched.
        assert aligned["meta_state"][1:] == ("64", "8")

    def test_state_shorter_than_target_refuses(self):
        assert _align_attention_state_dict(_kv_sub(30), 60) is None

    def test_key_value_length_mismatch_refuses(self):
        sub = _kv_sub(100)
        sub["state"] = (mx.ones((1, 4, 100, 8)), mx.ones((1, 4, 90, 8)))
        assert _align_attention_state_dict(sub, 60) is None

    def test_nonpositive_target_refuses(self):
        assert _align_attention_state_dict(_kv_sub(100), 0) is None
        assert _align_attention_state_dict(_kv_sub(100), -5) is None

    def test_unknown_layout_refuses(self):
        assert _align_attention_state_dict(
            {"class_name": "Weird", "state": ("a", "b"), "meta_state": ()}, 10
        ) is None

    def test_wrapped_rotating_buffer_refuses(self):
        # A rotating cache whose meta cannot prove temporal order must never be
        # sliced -- this is the guard that stopped Gemma 4 word-looping.
        sub = _kv_sub(100, cls_name="RotatingKVCache")
        sub["meta_state"] = ()
        assert _align_attention_state_dict(sub, 60) is None

    def test_original_state_is_not_mutated(self):
        sub = _kv_sub(100)
        _align_attention_state_dict(sub, 60)
        assert _seq_len(sub) == 100
        assert sub["meta_state"] == ("100",)


class TestAlignCacheList:
    def test_two_plain_kv_subs_both_align(self):
        """deepseek_v32 / longcat_flash shape -- the case that was losing all reuse."""
        aligned = _align_cache_list_state_dict(_cache_list(_kv_sub(100), _kv_sub(100)), 60)
        assert aligned is not None
        assert [_seq_len(s) for s in aligned["sub_caches"]] == [60, 60]
        assert aligned["class_name"] == "CacheList"

    def test_mixed_plain_and_quantized_subs_align(self):
        aligned = _align_cache_list_state_dict(
            _cache_list(_kv_sub(100), _quantized_sub(100)), 60
        )
        assert aligned is not None
        assert [_seq_len(s) for s in aligned["sub_caches"]] == [60, 60]

    @pytest.mark.parametrize("cumulative", ["ArraysCache", "MambaCache", "BatchMambaCache"])
    def test_cumulative_sub_refuses(self, cumulative):
        """falcon_h1 / baichuan_m1 shape: recurrent state has no token boundary."""
        sd = _cache_list(_kv_sub(100, cls_name=cumulative), _kv_sub(100))
        assert _align_cache_list_state_dict(sd, 60) is None
        assert _blocking_cache_list_sub_class(sd) == cumulative

    def test_one_unalignable_sub_refuses_whole_layer(self):
        # A half-aligned CacheList would restore one sub at the key boundary and
        # the other past it -- worse than no entry.
        sd = _cache_list(_kv_sub(100), _kv_sub(30))
        assert _align_cache_list_state_dict(sd, 60) is None

    def test_no_state_placeholder_sub_passes_through(self):
        sub = {"class_name": "ZayaNoStateCache", "state": (), "meta_state": ()}
        aligned = _align_cache_list_state_dict(_cache_list(_kv_sub(100), sub), 60)
        assert aligned is not None
        assert aligned["sub_caches"][1] is sub

    def test_empty_sub_caches_refuses(self):
        assert _align_cache_list_state_dict(_cache_list(), 60) is None

    def test_nonpositive_target_refuses(self):
        assert _align_cache_list_state_dict(_cache_list(_kv_sub(100)), 0) is None

    def test_original_sub_caches_are_not_mutated(self):
        sd = _cache_list(_kv_sub(100), _kv_sub(100))
        _align_cache_list_state_dict(sd, 60)
        assert [_seq_len(s) for s in sd["sub_caches"]] == [100, 100]


class _FakeLayer:
    def __init__(self, name):
        self.__class__ = type(name, (_FakeLayer,), {})


def _layer(name):
    cls = type(name, (), {})
    return cls()


class _FakeCacheList:
    def __init__(self, *caches):
        self.caches = caches


_FakeCacheList.__name__ = "CacheList"


class _FakeModel:
    def __init__(self, cache):
        self._cache = cache

    def make_cache(self):
        return self._cache


class TestExpandCacheClassNames:
    def test_resolves_wrapper_to_contents(self):
        from vmlx_engine.utils.cache_types import expand_cache_class_names

        cache = [_FakeCacheList(_layer("ArraysCache"), _layer("KVCache"))]
        assert expand_cache_class_names(cache) == {"ArraysCache", "KVCache"}

    def test_all_kv_wrapper_resolves_to_kv_only(self):
        from vmlx_engine.utils.cache_types import expand_cache_class_names

        cache = [_FakeCacheList(_layer("KVCache"), _layer("KVCache"))]
        assert expand_cache_class_names(cache) == {"KVCache"}

    def test_flat_layers_pass_through(self):
        from vmlx_engine.utils.cache_types import expand_cache_class_names

        cache = [_layer("KVCache"), _layer("MambaCache")]
        assert expand_cache_class_names(cache) == {"KVCache", "MambaCache"}

    def test_empty_wrapper_keeps_its_own_name(self):
        """An unreadable wrapper must not silently vanish into 'no non-KV types'."""
        from vmlx_engine.utils.cache_types import expand_cache_class_names

        assert expand_cache_class_names([_FakeCacheList()]) == {"CacheList"}

    def test_nested_wrappers_resolve(self):
        from vmlx_engine.utils.cache_types import expand_cache_class_names

        cache = [_FakeCacheList(_FakeCacheList(_layer("ArraysCache")), _layer("KVCache"))]
        assert expand_cache_class_names(cache) == {"ArraysCache", "KVCache"}


class TestHybridDetection:
    """The classification that decided falcon_h1 had no SSM at all."""

    def _detect(self, cache):
        from vmlx_engine.scheduler import Scheduler

        return Scheduler._is_hybrid_model(_FakeModel(cache))

    def test_falcon_h1_shape_is_hybrid(self):
        cache = [_FakeCacheList(_layer("ArraysCache"), _layer("KVCache")) for _ in range(4)]
        assert self._detect(cache) is True

    def test_deepseek_v32_longcat_shape_is_not_hybrid(self):
        cache = [_FakeCacheList(_layer("KVCache"), _layer("KVCache")) for _ in range(4)]
        assert self._detect(cache) is False

    def test_plain_kv_is_not_hybrid(self):
        assert self._detect([_layer("KVCache") for _ in range(4)]) is False

    def test_flat_mamba_hybrid_still_detected(self):
        assert self._detect([_layer("KVCache"), _layer("MambaCache")]) is True

    def test_mllm_detector_agrees_with_llm_detector(self):
        """Both detectors carried the same wrong assumption (ec08b6d75/a0c6538);
        they must not drift apart again."""
        from vmlx_engine.mllm_scheduler import MLLMScheduler
        from vmlx_engine.scheduler import Scheduler

        for cache in (
            [_FakeCacheList(_layer("ArraysCache"), _layer("KVCache"))],
            [_FakeCacheList(_layer("KVCache"), _layer("KVCache"))],
            [_layer("KVCache")],
            [_layer("KVCache"), _layer("MambaCache")],
        ):
            model = _FakeModel(cache)
            assert Scheduler._is_hybrid_model(model) == MLLMScheduler._is_hybrid_model(
                model
            )


class TestStoreUsesSharedAlignment:
    def test_cache_list_branch_calls_the_shared_helper(self):
        """Anti-reinlining guard.

        The store previously carried its own refusal for CacheList. If someone
        re-inlines alignment there instead of calling the helper, these tests
        keep passing while the store silently diverges again.
        """
        import inspect

        from vmlx_engine import scheduler

        source = inspect.getsource(scheduler)
        # Must be >1: the `def` line alone satisfies a bare substring check, so
        # an "is it referenced" assertion passes even with every call deleted.
        # Verified by deleting the call site and watching this fail.
        assert source.count("_align_cache_list_state_dict") > 1
        assert "aligned_cl = _align_cache_list_state_dict(" in source
        assert source.count("cannot align nonempty CacheList") == 1
        # The old unconditional bail must not come back.
        assert "Until recursive alignment is implemented" not in source

    def test_neither_detector_discards_the_wrapper_name(self):
        """Discarding "CacheList" assumes the wrapper only ever holds KV layers.

        That is true for deepseek_v32/longcat and false for falcon_h1, and it
        is what classified falcon as plain KV. Both schedulers must resolve the
        wrapper instead.
        """
        import inspect

        from vmlx_engine import mllm_scheduler, scheduler

        for module in (scheduler, mllm_scheduler):
            source = inspect.getsource(module)
            assert 'discard("CacheList")' not in source, module.__name__
            assert "expand_cache_class_names(cache)" in source, module.__name__
