"""Role-aware prompt-disk boundaries for first-turn cache snapshots."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

from vmlx_engine.engine.batched import BatchedEngine


def _engine_with_tokenizer() -> BatchedEngine:
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: text.split()
    )
    engine._processor = None

    def _render(self, messages, tools=None, **kwargs):
        del tools, kwargs
        return " ".join(f"{m['role']}:{m.get('content', '')}" for m in messages)

    engine._apply_chat_template = MethodType(_render, engine)
    return engine


def test_single_user_turn_produces_user_cache_boundary() -> None:
    engine = _engine_with_tokenizer()

    boundaries = engine._compute_segment_boundaries(
        [{"role": "user", "content": "retain this base"}],
        tools=None,
        num_images=0,
        enable_thinking=True,
        extra_template_kwargs=None,
    )

    assert boundaries == [(3, "user")]


def test_single_system_turn_produces_system_cache_boundary() -> None:
    engine = _engine_with_tokenizer()

    boundaries = engine._compute_segment_boundaries(
        [{"role": "system", "content": "shared stable prefix"}],
        tools=None,
        num_images=0,
        enable_thinking=False,
        extra_template_kwargs=None,
    )

    assert boundaries == [(3, "system")]


def test_empty_or_assistant_only_history_keeps_safe_fallback() -> None:
    engine = _engine_with_tokenizer()

    assert engine._compute_segment_boundaries([], None, 0, True, None) == []
    assert engine._compute_segment_boundaries(
        [{"role": "assistant", "content": "orphan"}], None, 0, True, None
    ) == []
