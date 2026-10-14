"""Projected memory estimates must not rewrite a model's generation contract."""
import pytest


@pytest.mark.parametrize("estimate", [0, 1, 78_446])
def test_native_bundle_output_limit_survives_projected_headroom(monkeypatch, estimate):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
    monkeypatch.setattr(server, "_bundle_sampling_default", lambda model, key: 131_072)
    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model="": estimate)
    assert server._resolve_max_tokens(None, "naive-n05-native") == 131_072


@pytest.mark.parametrize("explicit", [True, False])
def test_estimate_does_not_refuse_or_shorten_requested_output(monkeypatch, explicit):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model="": 0)
    assert server._apply_projected_output_guard(8192, explicit=explicit) == 8192


def test_status_and_request_agree_without_consuming_warning_dedup(monkeypatch):
    from vmlx_engine import server

    monkeypatch.setattr(server, "_metal_projected_output_token_cap", lambda model="": 1)
    monkeypatch.setattr(server, "_projected_guard_warned", set())
    assert server._apply_projected_output_guard(131_072, explicit=False, emit_warning=False) == 131_072
    assert server._projected_guard_warned == set()
    assert server._apply_projected_output_guard(131_072, explicit=False) == 131_072
    assert len(server._projected_guard_warned) == 1
