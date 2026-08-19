"""DFlash2 session-store prefix reuse.

The DFlash2 lane runs through SimpleEngine, which has no paged prefix-cache
stack, so before the session store every turn re-prefilled the entire
conversation (measured 15.5 s/turn at 7k tokens). These tests pin the pure
bookkeeping: prefix matching, LRU ownership transfer, and the exit-checkpoint
correction that shrinks the final cycle's committed cache positions down to
the emitted tokens (the runtime's EOS path returns before its rollback, so
rejected draft positions are still in the cache at exit).
"""

import pytest

from vmlx_engine.dflash2_runtime import (
    _DFlash2SessionStore,
    _checkpoint_correction,
    _prefix_reuse_enabled,
)


def _entry(model_key, tokens, tag=None, cache_len=None):
    return {
        "model_key": model_key,
        "tokens": list(tokens),
        "cache_len": len(tokens) - 1 if cache_len is None else cache_len,
        "target_cache": tag or object(),
        "draft_cache": None,
    }


class TestTakeMatching:
    def test_exact_prefix_hit(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3, 4]))
        got = store.take_matching("m", [1, 2, 3, 4, 5, 6])
        assert got is not None
        assert got["tokens"] == [1, 2, 3, 4]

    def test_hit_removes_the_entry(self):
        """Ownership transfers: the caches are mutated by the resumed
        generation, so a second consumer must never see the same entry."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3]))
        assert store.take_matching("m", [1, 2, 3, 4]) is not None
        assert store.take_matching("m", [1, 2, 3, 4]) is None

    def test_divergent_history_misses(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3, 4]))
        assert store.take_matching("m", [1, 2, 9, 4, 5]) is None

    def test_stored_longer_than_prompt_misses(self):
        """A shorter prompt cannot resume a longer conversation: hybrid GDN
        state cannot be trimmed back to an arbitrary earlier point."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3, 4, 5]))
        assert store.take_matching("m", [1, 2, 3]) is None

    def test_prompt_equal_to_stored_is_a_hit(self):
        """Regeneration from the same point: the delta is exactly the stored
        next-token, which the resumed prefill forwards."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3]))
        assert store.take_matching("m", [1, 2, 3]) is not None

    def test_model_key_mismatch_misses(self):
        """Token ids are only meaningful per (model, draft) pair; a different
        model's conversation must never be resumed."""
        store = _DFlash2SessionStore()
        store.put(_entry(("m1", "d1"), [1, 2, 3]))
        assert store.take_matching(("m2", "d1"), [1, 2, 3, 4]) is None

    def test_longest_matching_conversation_wins(self):
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2], tag="short"))
        store.put(_entry("m", [1, 2, 3, 4], tag="long"))
        got = store.take_matching("m", [1, 2, 3, 4, 5])
        assert got["target_cache"] == "long"


class TestPromptBoundarySnapshots:
    """Boundary entries have cache_len == len(tokens): the whole prompt is
    forwarded but no output token exists yet. They rescue reuse when the
    template strips the previous turn's <think> block from history."""

    def test_boundary_hit_needs_a_nonempty_delta(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2, 3], cache_len=3))
        # Prompt identical to the cached positions: nothing left to prefill,
        # so this must miss (the loop needs at least one token for logits).
        assert store.take_matching("m", [1, 2, 3]) is None
        store.put(_entry("m", [1, 2, 3], cache_len=3))
        assert store.take_matching("m", [1, 2, 3, 4]) is not None

    def test_end_of_turn_beats_boundary_when_it_covers_more(self):
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2, 3], cache_len=3, tag="boundary"))
        store.put(_entry("m", [1, 2, 3, 4, 5], cache_len=4, tag="turn"))
        got = store.take_matching("m", [1, 2, 3, 4, 5, 6])
        assert got["target_cache"] == "turn"

    def test_boundary_wins_when_think_strip_breaks_the_turn_entry(self):
        """The realistic reasoning-template shape: the end-of-turn entry
        contains think tokens (7, 8) the next prompt does not."""
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2, 3], cache_len=3, tag="boundary"))
        store.put(_entry("m", [1, 2, 3, 7, 8, 9], cache_len=5, tag="turn"))
        got = store.take_matching("m", [1, 2, 3, 9, 4, 5])
        assert got["target_cache"] == "boundary"

    def test_empty_store_misses(self):
        assert _DFlash2SessionStore().take_matching("m", [1, 2]) is None


class TestLru:
    def test_cap_evicts_oldest(self):
        store = _DFlash2SessionStore(max_entries=2)
        store.put(_entry("m", [1, 1]))
        store.put(_entry("m", [2, 2]))
        store.put(_entry("m", [3, 3]))
        assert store.take_matching("m", [1, 1, 9]) is None
        assert store.take_matching("m", [2, 2, 9]) is not None
        assert store.take_matching("m", [3, 3, 9]) is not None

    def test_zero_cached_positions_never_matches(self):
        """A single-token conversation caches nothing; resuming it would be
        a full prefill pretending to be a hit."""
        store = _DFlash2SessionStore()
        store.put(_entry("m", [7]))
        assert store.take_matching("m", [7, 9]) is None

    def test_clear(self):
        store = _DFlash2SessionStore()
        store.put(_entry("m", [1, 2]))
        store.clear()
        assert store.take_matching("m", [1, 2, 3]) is None


class TestCheckpointCorrection:
    def test_clean_cycle_needs_no_correction(self):
        """Normal loop end: the cycle's own rollback already trimmed to
        accepted+1 committed positions and all were emitted."""
        assert _checkpoint_correction(4, 4) is None

    def test_no_cycles_needs_no_correction(self):
        assert _checkpoint_correction(0, 0) is None

    def test_eos_exit_keeps_only_emitted_positions(self):
        """EOS at draft index 1 with block 5: upstream returns before its
        rollback, so all 5 verify positions are committed but only 2 tokens
        were emitted. rollback(accepted=1, trim=3) keeps exactly 2."""
        accepted, trim = _checkpoint_correction(5, 2)
        assert accepted == 1
        assert trim == 3
        assert accepted + 1 + trim == 5

    def test_length_truncation_after_cycle_rollback(self):
        """max_tokens truncation: the cycle rolled back to accepted+1=4
        committed positions but only 3 tokens were emitted."""
        accepted, trim = _checkpoint_correction(4, 3)
        assert accepted == 2
        assert trim == 1

    def test_single_emitted_token_from_full_block(self):
        accepted, trim = _checkpoint_correction(5, 1)
        assert accepted == 0
        assert trim == 4


class TestEnvGate:
    @pytest.mark.parametrize("value", ["0", "off", "false", "no", "OFF", "False"])
    def test_falsey_spellings_disable(self, monkeypatch, value):
        monkeypatch.setenv("VMLX_DFLASH2_PREFIX_REUSE", value)
        assert _prefix_reuse_enabled() is False

    @pytest.mark.parametrize("value", [None, "1", "on", "true", "maybe"])
    def test_default_and_truthy_enable(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("VMLX_DFLASH2_PREFIX_REUSE", raising=False)
        else:
            monkeypatch.setenv("VMLX_DFLASH2_PREFIX_REUSE", value)
        assert _prefix_reuse_enabled() is True


class TestAssistantTagCut:
    class _Tok:
        def convert_tokens_to_ids(self, s):
            assert s == "<|im_start|>"
            return 100

    def test_cuts_at_last_tag(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        # sys/user turns then the generation tag at index 6
        prompt = [100, 1, 2, 100, 3, 4, 100, 5]
        assert _assistant_tag_cut(self._Tok(), prompt, 0) == 6

    def test_tag_inside_cached_region_is_rejected(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        prompt = [100, 1, 2, 100, 3]
        assert _assistant_tag_cut(self._Tok(), prompt, 4) is None

    def test_no_tag_returns_none(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        assert _assistant_tag_cut(self._Tok(), [1, 2, 3], 0) is None

    def test_tokenizer_without_vocab_returns_none(self):
        from vmlx_engine.dflash2_runtime import _assistant_tag_cut

        assert _assistant_tag_cut(object(), [100, 1], 0) is None


class TestDescribeMisses:
    def test_reports_common_prefix_per_entry(self):
        store = _DFlash2SessionStore(max_entries=4)
        store.put(_entry("m", [1, 2, 3, 7, 8], cache_len=4))
        store.put(_entry("m", [1, 2], cache_len=2))
        out = store.describe_misses("m", [1, 2, 3, 9])
        assert out == ["turn len=5 cached=4 common=3", "turn len=2 cached=2 common=2"]

    def test_other_models_excluded(self):
        store = _DFlash2SessionStore()
        store.put(_entry("other", [1, 2, 3]))
        assert store.describe_misses("m", [1, 2, 3]) == []
