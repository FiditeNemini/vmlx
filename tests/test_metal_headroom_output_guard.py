# SPDX-License-Identifier: Apache-2.0

"""Contracts for projected Metal headroom output-token guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def test_explicit_max_tokens_over_safe_headroom_rejects(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)

    with pytest.raises(HTTPException) as exc:
        server._resolve_max_tokens(8192, "tight-headroom-model")

    assert exc.value.status_code == 413
    assert "8192" in str(exc.value.detail)
    assert "1024" in str(exc.value.detail)
    assert "Metal" in str(exc.value.detail)


def test_explicit_max_tokens_rejects_when_projected_safe_cap_is_zero(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 0)

    with pytest.raises(HTTPException) as exc:
        server._resolve_max_tokens(1, "no-headroom-model")

    assert exc.value.status_code == 413
    assert "requested=1" in str(exc.value.detail)
    assert "safe_cap=0" in str(exc.value.detail)


def test_implicit_default_over_safe_headroom_clamps(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)

    assert server._resolve_max_tokens(None, "tight-headroom-model") == 1024


def test_implicit_default_with_zero_safe_cap_clamps_to_one(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 0)

    assert server._resolve_max_tokens(None, "no-headroom-model") == 1


def test_session_default_over_safe_headroom_clamps_instead_of_rejecting(
    monkeypatch, caplog
):
    """A session-level --max-tokens default above the cap clamps with a
    warning. It previously raised the per-request 413, which made EVERY
    request that omitted max_tokens fail while the session looked healthy
    (live 2026-08-07: UI-saved Max Output 12288 vs DSV4 safe_cap 10063 —
    plain chat/completions requests were unservable). The request author
    sent no max_tokens and cannot fix the config, so the hard 413 is
    reserved for per-request values."""
    import logging

    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", True)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)
    monkeypatch.setattr(server, "_projected_guard_warned", set())

    with caplog.at_level(logging.DEBUG, logger=server.logger.name):
        assert server._resolve_max_tokens(None, "tight-headroom-model") == 1024

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "headroom guard clamped" in r.message
    ]
    assert len(warnings) == 1
    assert "session-default (--max-tokens)" in warnings[0].message


def test_reasoning_model_fallback_gets_larger_headroom(monkeypatch):
    """A reasoning-capable model with no bundle/CLI cap must fall back to the
    larger reasoning budget so the hidden <think> phase cannot starve the
    visible answer (observed live: MiniMax-M2.7 reasoning ate the whole flat
    4096 and the answer truncated at 183 chars, finish=length)."""
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    # Ample projected headroom so the guard does not clamp the fallback.
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": None)

    monkeypatch.setattr(server, "_reasoning_parser", object())
    assert (
        server._resolve_max_tokens(None, "reasoning-model")
        == server._FALLBACK_MAX_OUTPUT_TOKENS_REASONING
    )
    assert (
        server._FALLBACK_MAX_OUTPUT_TOKENS_REASONING > server._FALLBACK_MAX_OUTPUT_TOKENS
    )


def test_non_reasoning_model_keeps_modest_fallback(monkeypatch):
    """Non-reasoning models keep the modest 4096 loop guard."""
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": None)

    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_current_model_config", lambda: None)
    assert (
        server._resolve_max_tokens(None, "plain-model")
        == server._FALLBACK_MAX_OUTPUT_TOKENS
    )


def test_reasoning_family_with_parser_disabled_still_gets_larger_fallback(monkeypatch):
    """A reasoning family whose active parser is disabled/failed (None) but
    whose registry entry is reasoning-capable (reasoning_parser set OR
    think_in_template) still reasons via the chat template and must get the
    larger fallback — keying on the active parser alone would miss it."""
    from types import SimpleNamespace
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": None)
    monkeypatch.setattr(server, "_reasoning_parser", None)

    # Registry says reasoning_parser is set even though the active instance is None.
    monkeypatch.setattr(
        server, "_current_model_config",
        lambda: SimpleNamespace(reasoning_parser="qwen3", think_in_template=False),
    )
    assert (
        server._resolve_max_tokens(None, "reasoning-family-parser-off")
        == server._FALLBACK_MAX_OUTPUT_TOKENS_REASONING
    )

    # think_in_template alone (template injects <think>) also qualifies.
    monkeypatch.setattr(
        server, "_current_model_config",
        lambda: SimpleNamespace(reasoning_parser=None, think_in_template=True),
    )
    assert (
        server._resolve_max_tokens(None, "think-in-template-model")
        == server._FALLBACK_MAX_OUTPUT_TOKENS_REASONING
    )


def test_projection_uses_loaded_model_config(monkeypatch):
    from vmlx_engine import server

    class FakeMX:
        @staticmethod
        def get_active_memory():
            return int(105.41 * 1024**3)

        @staticmethod
        def device_info():
            return {"max_recommended_working_set_size": int(107.52 * 1024**3)}

    cfg = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=4,
        head_dim=8,
        torch_dtype="bfloat16",
    )
    engine = SimpleNamespace(model=SimpleNamespace(config=cfg))

    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(
        server,
        "_metal_projection_stats",
        lambda: (
            int(105.41 * 1024**3),
            int(107.52 * 1024**3),
        ),
    )
    monkeypatch.setenv("VMLX_METAL_PROJECTED_TOKEN_TRANSIENT_MULTIPLIER", "4")
    monkeypatch.setenv("VMLX_METAL_PROJECTED_TOKEN_BUDGET_FRACTION", "0.5")

    cap = server._metal_projected_output_token_cap("tight-headroom-model")

    assert cap is not None
    assert cap > 0


def test_implicit_clamp_warning_dedupes_on_repeat_resolutions(monkeypatch, caplog):
    import logging

    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)
    monkeypatch.setattr(server, "_projected_guard_warned", set())

    def clamp_warnings():
        return [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "headroom guard clamped" in r.message
        ]

    with caplog.at_level(logging.DEBUG, logger=server.logger.name):
        # Simulate the /health poll re-resolving the same implicit default.
        for _ in range(5):
            assert server._resolve_max_tokens(None, "tight-headroom-model") == 1024
        assert len(clamp_warnings()) == 1

        # A changed cap is new information and must warn again.
        monkeypatch.setattr(
            server, "_metal_projected_output_token_cap", lambda model_name="": 512
        )
        assert server._resolve_max_tokens(None, "tight-headroom-model") == 512
        assert server._resolve_max_tokens(None, "tight-headroom-model") == 512
        assert len(clamp_warnings()) == 2


def test_implicit_clamp_warning_dedupes_path_and_api_name(monkeypatch, caplog):
    """Boot resolves with the filesystem path, requests with the API model
    name; both describe the same loaded model and must warn once total."""
    import logging

    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_default_max_tokens", 4096)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model_name, key: None)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model_name="": 1024)
    monkeypatch.setattr(server, "_projected_guard_warned", set())

    with caplog.at_level(logging.DEBUG, logger=server.logger.name):
        assert (
            server._resolve_max_tokens(
                None, "/Users/example/.mlxstudio/models/DeepSeek-V4-Flash-0731-JANG"
            )
            == 1024
        )
        assert (
            server._resolve_max_tokens(None, "models/DeepSeek-V4-Flash-0731-JANG")
            == 1024
        )
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "headroom guard clamped" in r.message
    ]
    assert len(warnings) == 1


def test_buffer_ceiling_info_logs_once_per_binding_cap(monkeypatch, caplog):
    """The /health poll re-runs the cap projection every few seconds; the
    buffer-ceiling INFO must fire once per distinct binding cap, not per poll
    (byte_cap drifts with live memory and is excluded from the dedup key)."""
    import logging
    from types import SimpleNamespace

    from vmlx_engine import server
    from vmlx_engine.utils import memory_limits

    monkeypatch.setattr(
        server,
        "_loaded_model_config_for_memory_projection",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        server,
        "_metal_projection_stats",
        lambda: (int(100 * 1024**3), int(107 * 1024**3)),
    )
    monkeypatch.setattr(
        memory_limits, "estimate_kv_bytes_per_token_from_config", lambda cfg: 4096
    )
    monkeypatch.setattr(memory_limits, "retained_buffers_per_token", lambda cfg: 43.0)
    monkeypatch.setattr(memory_limits, "metal_resource_limit", lambda: 499000)
    byte_caps = iter([20000, 21000, 22000, 23000])
    monkeypatch.setattr(
        memory_limits,
        "projected_output_token_cap",
        lambda **kw: next(byte_caps),
    )
    buffer_cap_holder = {"cap": 10063}
    monkeypatch.setattr(
        memory_limits,
        "projected_output_token_cap_by_buffers",
        lambda **kw: buffer_cap_holder["cap"],
    )
    monkeypatch.setattr(server, "_buffer_ceiling_logged", set())

    def ceiling_infos():
        return [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "binding output limit" in r.message
        ]

    with caplog.at_level(logging.DEBUG, logger=server.logger.name):
        for _ in range(3):
            assert server._metal_projected_output_token_cap("m") == 10063
        assert len(ceiling_infos()) == 1

        # A changed binding cap is new information and must log again.
        buffer_cap_holder["cap"] = 9000
        assert server._metal_projected_output_token_cap("m") == 9000
        assert len(ceiling_infos()) == 2
