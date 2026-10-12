"""Unstripped cache keys do not need the generation-prompt suffix hash (scheduler._generation_prompt_strip_active).

Mixed full/SWA models (Naive N0.5), openPangu and GLM-5.3 key their cache on the full prompt, suffix tokens included.
The extra `generation_prompt` hash re-keyed the whole shared history from block 0, so a thinking toggle re-prefilled
everything: Naive N0.5, 41-46k-token chat, cached 0, ~70 s TTFT (app and raw API, 2026-10-12).
"""
from types import SimpleNamespace

from vmlx_engine import scheduler as S


def test_salt_helper_drops_only_the_generation_prompt_key():
    keys = {"generation_prompt": "v1:5:abc", "media": "img:1", "__vmlx_native_cache_shape__": {"x": 1}}
    assert S._without_generation_prompt_salt(keys) == {"media": "img:1", "__vmlx_native_cache_shape__": {"x": 1}}
    assert S._without_generation_prompt_salt({"generation_prompt": "v1:5:abc"}) is None
    assert S._without_generation_prompt_salt(None) is None
    plain = {"media": "img:1"}
    assert S._without_generation_prompt_salt(plain) is plain


def _strip_active(**flags):
    return S.Scheduler._generation_prompt_strip_active(SimpleNamespace(**flags))


def test_strip_is_inactive_exactly_for_full_prompt_key_families(monkeypatch):
    monkeypatch.setattr(S, "_GEN_PROMPT_STRIP_DISABLED", False)
    assert _strip_active() is True
    assert _strip_active(_mixed_attention_cache_model=True) is False
    assert _strip_active(_uses_openpangu_cache=True) is False
    assert _strip_active(_uses_glm5_next_cache=True) is False
    monkeypatch.setattr(S, "_GEN_PROMPT_STRIP_DISABLED", True)
    assert _strip_active() is False


def test_fetch_and_add_request_use_the_same_rule():
    import inspect

    fetch_src = inspect.getsource(S.Scheduler)
    # One predicate decides both the fetch strip and the salt drop; no second copy of the family list.
    assert fetch_src.count("def _generation_prompt_strip_active") == 1
    assert "if not self._generation_prompt_strip_active():\n            # Mixed full/SWA models" in fetch_src
    add_src = inspect.getsource(S.Scheduler.add_request)
    assert "_without_generation_prompt_salt" in add_src
