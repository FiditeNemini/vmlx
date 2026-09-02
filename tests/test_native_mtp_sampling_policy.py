"""Native-MTP sampling-policy resolution contract.

With native MTP enabled the app launches the engine with
``deterministic-defaults`` (Auto) or ``greedy-only`` (Deterministic).
The contract, per product decision:

- ``deterministic-defaults``: the STARTUP DEFAULTS are pinned to greedy
  (temperature 0, top_p 1, top_k dropped) so chat settings and API requests
  that do not specify sampling run greedy — but an EXPLICIT request value
  (API kwargs) wins.
- ``greedy-only``: every request is hard-pinned to greedy regardless of
  what it sends.

The Auto behavior regressed on 2026-08-26 ("Preserve sampled defaults for
native MTP Auto mode" switched Auto to compatible-only, which keeps the
bundle's sampled temperature); these tests pin the restored contract at the
engine's resolution layer, which every API surface shares.
"""

import pytest

from vmlx_engine import server


@pytest.fixture
def _sampling_state(monkeypatch):
    monkeypatch.setattr(server, "_default_temperature", None)
    monkeypatch.setattr(server, "_default_top_p", None)
    monkeypatch.setattr(server, "_default_top_k", None)
    yield


def test_greedy_only_hard_pins_even_explicit_request_values(
    monkeypatch, _sampling_state
):
    monkeypatch.setattr(server, "_native_mtp_sampling_policy", "greedy-only")
    assert server._resolve_temperature(0.8) == 0.0
    assert server._resolve_top_p(0.5) == 1.0
    assert server._resolve_top_k(40) == 0


def test_deterministic_defaults_pins_defaults_but_request_kwargs_win(
    monkeypatch, _sampling_state
):
    monkeypatch.setattr(
        server, "_native_mtp_sampling_policy", "deterministic-defaults"
    )
    # The startup block sets these when the policy is deterministic-defaults;
    # mirror that state.
    monkeypatch.setattr(server, "_default_temperature", 0.0)
    monkeypatch.setattr(server, "_default_top_p", 1.0)
    monkeypatch.setattr(server, "_default_top_k", 0)

    # No request value -> pinned greedy startup defaults.
    assert server._resolve_temperature(None) == 0.0
    assert server._resolve_top_p(None) == 1.0
    assert server._resolve_top_k(None) == 0

    # Explicit API kwargs win.
    assert server._resolve_temperature(0.8) == 0.8
    assert server._resolve_top_p(0.5) == 0.5
    assert server._resolve_top_k(40) == 40


def test_startup_block_derives_greedy_defaults_for_enabled_policies():
    """The argv startup block pins defaults for BOTH enabled policies —
    asserted structurally so a future edit cannot drop one of them."""

    import inspect

    source = inspect.getsource(server.main)
    assert '"deterministic-defaults", "greedy-only"' in source.replace(
        "'", '"'
    )
