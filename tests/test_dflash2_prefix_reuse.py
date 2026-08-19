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


def _entry(model_key, tokens, tag=None):
    return {
        "model_key": model_key,
        "tokens": list(tokens),
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

    def test_empty_store_misses(self):
        assert _DFlash2SessionStore().take_matching("m", [1, 2]) is None


class TestLru:
    def test_cap_evicts_oldest(self):
        store = _DFlash2SessionStore(max_entries=2)
        store.put(_entry("m", [1]))
        store.put(_entry("m", [2]))
        store.put(_entry("m", [3]))
        assert store.take_matching("m", [1, 9]) is None
        assert store.take_matching("m", [2, 9]) is not None
        assert store.take_matching("m", [3, 9]) is not None

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
