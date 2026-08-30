from types import SimpleNamespace


def _rows(contract):
    return {row["id"]: row for row in contract["features"]}


def test_acceleration_family_resolves_nested_model_types():
    from vmlx_engine.acceleration_contract import acceleration_family_from_config

    assert acceleration_family_from_config({"model_type": "qwen4_exp"}) == "qwen4_exp"
    assert (
        acceleration_family_from_config(
            {"model_type": "wrapper", "text_config": {"model_type": "qwen3_5"}}
        )
        == "qwen3_5"
    )
    assert (
        acceleration_family_from_config(
            {"llm_config": {"model_type": "qwen3_5_moe_text"}}
        )
        == "qwen3_5_moe"
    )
    assert acceleration_family_from_config({"model_type": "unknown"}) is None


def test_qwen4_contract_never_calls_requested_kernel_installed(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_QWEN4_FUSED_GDN_CONV", "1")
    contract = build_acceleration_contract("qwen4_exp")
    rows = _rows(contract)

    assert contract["native_state"] == ["QSA", "GDN", "PLE", "n-gram", "MoE"]
    assert rows["projection_groups"]["state"] == "configured_unattested"
    assert rows["gdn_conv_state"]["requested"] is True
    assert rows["gdn_conv_state"]["state"] == "configured_unattested"
    assert rows["affine_moe_pair"]["requested"] is True
    assert rows["affine_moe_pair"]["selection_source"] == "default"
    assert rows["affine_moe"]["requested"] is False
    assert rows["affine_moe"]["state"] == "disabled"
    assert contract["summary"]["installed"] == 0


def test_qwen4_affine_moe_pair_default_has_explicit_opt_out(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_QWEN4_FUSED_MOE_PAIR", raising=False)
    rows = _rows(build_acceleration_contract("qwen4_exp"))
    assert rows["affine_moe_pair"]["requested"] is True
    assert rows["affine_moe_pair"]["state"] == "configured_unattested"

    monkeypatch.setenv("VMLX_QWEN4_FUSED_MOE_PAIR", "0")
    rows = _rows(build_acceleration_contract("qwen4_exp"))
    assert rows["affine_moe_pair"]["requested"] is False
    assert rows["affine_moe_pair"]["state"] == "disabled"


def test_runtime_attestation_distinguishes_installed_from_observed(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.setenv("VMLX_GLM5_FUSED_KDA_CONV", "1")
    contract = build_acceleration_contract(
        "glm5_next",
        {
            "features": {
                "projection_groups": {"installed": 80},
                "startup_warmup": {"installed": True, "observed_calls": 1},
                "kda_conv_state": {"installed": 34, "observed_calls": 0},
            }
        },
    )
    rows = _rows(contract)

    assert rows["projection_groups"]["state"] == "installed_unobserved"
    assert rows["startup_warmup"]["state"] == "active_observed"
    assert rows["kda_conv_state"]["state"] == "installed_unobserved"
    assert contract["summary"] == {
        "requested": 4,
        "installed": 3,
        "observed": 1,
        "source_only_or_unattested": 1,
    }


def test_dsv4_contract_preserves_production_defaults(monkeypatch):
    from vmlx_engine.acceleration_contract import build_acceleration_contract

    monkeypatch.delenv("VMLX_DSV4_FUSED_MOE_PAIR", raising=False)
    monkeypatch.delenv("VMLX_DSV4_LM_HEAD_MODE", raising=False)
    monkeypatch.delenv("VMLX_DSV4_ROPE_CACHE", raising=False)
    contract = build_acceleration_contract("deepseek_v4")
    rows = _rows(contract)

    assert rows["runtime_patch"]["requested"] is True
    assert rows["fused_moe_pair"]["requested"] is True
    assert rows["lm_head"]["selection"] == "qmm"
    assert rows["lm_head"]["requested"] is True
    assert rows["rope_cache"]["requested"] is True
    assert rows["affine_moe"]["requested"] is False


def test_model_attestation_merges_and_returns_a_copy():
    from vmlx_engine.acceleration_contract import (
        find_acceleration_attestation,
        record_acceleration_attestation,
    )

    model = SimpleNamespace()
    record_acceleration_attestation(
        model,
        "glm5_next",
        {"projection_groups": {"installed": 79}},
    )
    record_acceleration_attestation(
        model,
        "glm5_next",
        {
            "projection_groups": {"observed_calls": 2},
            "startup_warmup": {"installed": True, "observed_calls": 1},
        },
    )

    found = find_acceleration_attestation([object(), model])
    assert found["family"] == "glm5_next"
    assert found["features"]["projection_groups"] == {
        "installed": 79,
        "observed_calls": 2,
    }
    found["features"]["projection_groups"]["installed"] = 0
    assert model._vmlx_acceleration_attestation["features"]["projection_groups"][
        "installed"
    ] == 79


def test_server_acceleration_surface_embeds_family_contract(monkeypatch, tmp_path):
    import vmlx_engine.server as server

    (tmp_path / "config.json").write_text(
        '{"model_type":"glm5_next","quantization":{"bits":4,"group_size":128}}'
    )
    monkeypatch.setattr(
        server,
        "_mlx_metal_na_status",
        lambda: {"available": True, "nax_symbols": 1, "naxtile_symbols": 1},
    )
    monkeypatch.setattr(
        server,
        "_host_supports_metal_na",
        lambda: {"supported": True, "brand": "Apple M5 Max"},
    )
    monkeypatch.setattr(
        server,
        "_loaded_acceleration_attestation",
        lambda: {
            "family": "glm5_next",
            "features": {"startup_warmup": {"installed": True, "observed_calls": 1}},
        },
    )

    status = server._model_acceleration_status(str(tmp_path))

    assert status["metal_na_active_on_host"] is True
    family = status["family_runtime"]
    assert family["schema"] == "vmlx-runtime-acceleration-v1"
    assert family["family"] == "glm5_next"
    assert _rows(family)["startup_warmup"]["state"] == "active_observed"


def test_server_reports_observed_qwen_kernel_only_after_qualifying_call(
    monkeypatch, tmp_path
):
    import vmlx_engine.server as server
    from vmlx_engine.metal import gdn_conv_decode

    (tmp_path / "config.json").write_text('{"model_type":"qwen4_exp"}')
    monkeypatch.setenv("VMLX_QWEN4_FUSED_GDN_CONV", "1")
    monkeypatch.setattr(server, "_loaded_acceleration_attestation", lambda: None)
    monkeypatch.setattr(gdn_conv_decode, "_OBSERVED", False)

    before = server._family_acceleration_contract(str(tmp_path))
    assert _rows(before)["gdn_conv_state"]["state"] == "configured_unattested"

    monkeypatch.setattr(gdn_conv_decode, "_OBSERVED", True)
    after = server._family_acceleration_contract(str(tmp_path))
    assert _rows(after)["gdn_conv_state"]["state"] == "active_observed"
    assert _rows(after)["gdn_conv_state"]["runtime"]["observed_calls"] == 1
