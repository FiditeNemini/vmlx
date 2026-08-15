"""Stamped reasoning-effort clamp — out-of-set effort must never be silent.

Field-failure class this pins (2026-08-15 directive): a client sends a
generic OpenAI tier ("high") to a bundle stamped low/medium/xhigh; the chat
template ignores the unknown value and silently renders its default tier,
so the user believes they got "high" while the model ran something else.
The policy coerces to the NEAREST stamped tier and logs the substitution.
DSV4 keeps its strict 400 encoder and Hy3 its own normalizer — both are
skipped by the policy.
"""

from __future__ import annotations

import logging

import pytest

import vmlx_engine.server as server


@pytest.fixture()
def stamped(monkeypatch):
    def fake_contract(bundle_path):
        return ("low", "medium", "xhigh"), "medium"

    monkeypatch.setattr(
        server, "_stamped_reasoning_effort_contract", fake_contract
    )
    monkeypatch.setattr(server, "_is_hy3_model", lambda _k: False)
    monkeypatch.setattr(
        server, "_model_family_for_defaults", lambda _k="": "qwen3_8"
    )
    return None


def test_in_set_effort_passes_through_unchanged(stamped):
    chat, ct = {}, {"reasoning_effort": "medium"}
    server._apply_stamped_effort_policy(chat, ct, model_key="/tmp/bundle")
    assert ct["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in chat


def test_out_of_set_high_coerces_to_nearest_stamped_tier(stamped, caplog):
    chat, ct = {}, {"reasoning_effort": "high"}
    with caplog.at_level(logging.WARNING):
        server._apply_stamped_effort_policy(chat, ct, model_key="/tmp/bundle")
    # "high" sits between medium and xhigh on the ladder; nearest stamped
    # tier at equal distance resolves to the lower one (min is stable), and
    # either way the substitution is LOGGED — never silent.
    assert ct["reasoning_effort"] in {"medium", "xhigh"}
    assert chat["reasoning_effort"] == ct["reasoning_effort"]
    assert any(
        "not in this bundle's stamped set" in rec.message for rec in caplog.records
    )


def test_unknown_token_falls_to_stamped_default_with_notice(stamped, caplog):
    chat, ct = {}, {"reasoning_effort": "turbo"}
    with caplog.at_level(logging.WARNING):
        server._apply_stamped_effort_policy(chat, ct, model_key="/tmp/bundle")
    assert ct["reasoning_effort"] == "medium"
    assert any(
        "not in this bundle's stamped set" in rec.message for rec in caplog.records
    )


def test_dsv4_family_is_skipped(monkeypatch):
    monkeypatch.setattr(
        server,
        "_stamped_reasoning_effort_contract",
        lambda _p: (("low", "high"), "low"),
    )
    monkeypatch.setattr(server, "_is_hy3_model", lambda _k: False)
    monkeypatch.setattr(
        server, "_model_family_for_defaults", lambda _k="": "deepseek_v4"
    )
    chat, ct = {}, {"reasoning_effort": "medium"}
    server._apply_stamped_effort_policy(chat, ct, model_key="/tmp/bundle")
    # untouched — DSV4's own encoder must keep rejecting loudly
    assert ct["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in chat


def test_hy3_family_is_skipped(monkeypatch):
    monkeypatch.setattr(
        server,
        "_stamped_reasoning_effort_contract",
        lambda _p: (("low", "high"), "low"),
    )
    monkeypatch.setattr(server, "_is_hy3_model", lambda _k: True)
    chat, ct = {}, {"reasoning_effort": "no_think"}
    server._apply_stamped_effort_policy(chat, ct, model_key="/tmp/bundle")
    assert ct["reasoning_effort"] == "no_think"


def test_unstamped_bundle_is_untouched(monkeypatch):
    monkeypatch.setattr(
        server, "_stamped_reasoning_effort_contract", lambda _p: ((), None)
    )
    monkeypatch.setattr(server, "_is_hy3_model", lambda _k: False)
    monkeypatch.setattr(
        server, "_model_family_for_defaults", lambda _k="": "qwen3_6"
    )
    chat, ct = {}, {"reasoning_effort": "high"}
    server._apply_stamped_effort_policy(chat, ct, model_key="/tmp/bundle")
    assert ct["reasoning_effort"] == "high"
    assert "reasoning_effort" not in chat
