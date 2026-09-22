import json
from types import SimpleNamespace

import pytest

from vmlx_engine.models.mimo_v26_contract import read_mimo_v26_contract


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "mimo_v2", "vision_config": {"depth": 28},
    }))
    (tmp_path / "jang_config.json").write_text(json.dumps({
        "weight_format": "mixed_affine_mxfp4",
        "has_vision": False, "has_audio": False, "has_video": False,
        "preserved_modalities": ["vision", "audio", "video"],
        "capabilities": {"modalities": {"text": True, "vision": False}},
    }))
    return tmp_path


def test_routing_uses_bundle_contract_even_when_vlm_is_forced(bundle, monkeypatch):
    import vmlx_engine.server as server
    from vmlx_engine.api.utils import is_mllm_model

    monkeypatch.setattr(server, "_smelt_enabled", False)
    monkeypatch.setattr(server, "_force_text_only", False, raising=False)
    assert is_mllm_model(str(bundle)) is False
    assert is_mllm_model(str(bundle), force_mllm=True) is False


def test_discovery_never_registers_legacy_runtime(bundle, monkeypatch):
    import vmlx_engine.server as server

    def legacy():
        pytest.fail("capability lookup imported the V2.5 runtime")

    monkeypatch.setattr(server, "_mimo_v2_runtime_module", legacy)
    assert server._mimo_v2_runtime_modalities(str(bundle)) == ["text"]


def test_direct_vlm_load_rejects_before_legacy_registration(bundle, monkeypatch):
    from vmlx_engine.models import mllm

    def legacy():
        pytest.fail("V2.6 loader registered the V2.5 runtime")

    monkeypatch.setattr(mllm, "_register_mimo_v2_mlx_vlm_runtime", legacy)
    with pytest.raises(ValueError, match="legacy V2.5"):
        mllm._register_local_mlx_vlm_runtime_if_needed(bundle)


def test_media_qualification_uses_fresh_route(bundle, monkeypatch):
    import vmlx_engine.server as server
    from vmlx_engine.api.utils import is_mllm_model

    monkeypatch.setattr(server, "_smelt_enabled", False)
    monkeypatch.setattr(server, "_force_text_only", False, raising=False)
    monkeypatch.setenv("VMLX_MIMO26_MEDIA_TEST", "1")
    assert is_mllm_model(str(bundle)) is True
    from vmlx_engine.models import mllm
    def legacy():
        pytest.fail("qualification imported the V2.5 runtime")
    monkeypatch.setattr(mllm, "_register_mimo_v2_mlx_vlm_runtime", legacy)
    mllm._register_local_mlx_vlm_runtime_if_needed(bundle)


def test_thinking_off_preserves_native_mimo_history(bundle, monkeypatch):
    import vmlx_engine.server as server
    monkeypatch.setattr(server, "_model_path", str(bundle))
    messages = [{"role": "assistant", "reasoning_content": "prior reasoning", "content": "answer"}]
    assert server._strip_prior_reasoning_for_thinking_off(messages) == messages


def test_reasoning_only_turn_survives_request_preparation(bundle, monkeypatch):
    import vmlx_engine.server as server
    monkeypatch.setattr(server, "_model_path", str(bundle))
    messages = [{"role": "assistant", "reasoning_content": "unfinished reasoning", "content": ""}]
    assert server._drop_contentless_assistant_turns(messages) == messages


def test_legacy_format_and_other_families_are_not_reclassified(bundle):
    path = bundle / "jang_config.json"
    path.write_text(json.dumps({"weight_format": "jang"}))
    assert read_mimo_v26_contract(bundle) is None
    path.write_text(json.dumps({"weight_format": "mixed_affine_mxfp4"}))
    (bundle / "config.json").write_text('{"model_type": "another_model"}')
    assert read_mimo_v26_contract(bundle) is None


def test_fresh_batched_decode_has_no_legacy_token_constraints():
    from types import SimpleNamespace
    from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.model = SimpleNamespace(_mimo_v26_runtime=True)
    request = SimpleNamespace(enable_thinking=False, extra_kwargs={"tool_choice": "required"})
    assert generator._mimo_v2_thinking_off_logits_processors(request) == []
    assert generator._mimo_v2_required_tool_prefix_processors(request) == []


def test_media_cache_admission_is_runtime_specific_and_has_kill_switch(monkeypatch):
    from vmlx_engine.mllm_batch_generator import _mllm_media_prefix_cache_family_enabled as allowed
    monkeypatch.delenv("VMLINUX_MLLM_MEDIA_PREFIX_CACHE", raising=False)
    monkeypatch.delenv("VMLINUX_MLLM_MEDIA_PREFIX_CACHE_UNSAFE_ACK", raising=False)
    assert allowed("mimo_v2", mimo_v26_runtime=True)
    assert not allowed("mimo_v2")
    assert not allowed("another_family", mimo_v26_runtime=True)
    monkeypatch.setenv("VMLINUX_MLLM_MEDIA_PREFIX_CACHE", "0")
    assert not allowed("mimo_v2", mimo_v26_runtime=True)


def test_fresh_cache_detection_uses_instantiated_slots_without_legacy_metadata():
    from vmlx_engine.mllm_scheduler import MLLMScheduler
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.model = SimpleNamespace(_mimo_v26_runtime=True)
    KV = type("KVCache", (), {})
    SWA = type("RotatingKVCache", (), {})
    language = SimpleNamespace(make_cache=lambda: [KV(), SWA()], args=SimpleNamespace(hybrid_layer_pattern=[0,1]))
    assert scheduler._model_has_mixed_attention(language)
    language.make_cache = lambda: [KV()]
    assert not scheduler._model_has_mixed_attention(language)


def test_fresh_storage_telemetry_preserves_native_rotating_slots():
    from vmlx_engine.server import _native_cache_status
    scheduler = SimpleNamespace(model=SimpleNamespace(_mimo_v26_runtime=True),
        _kv_cache_bits=8, _kv_cache_group_size=64, _tq_active=False,
        block_aware_cache=object(), paged_cache_manager=SimpleNamespace(_disk_store=object()))
    status = _native_cache_status(scheduler, family="mimo_v2",
        cfg=SimpleNamespace(cache_subtype="mimo_v2_asymmetric_swa"))
    policy = status["storage_quantization"]
    assert policy["bits"] == 8
    assert policy["applies_to"] == "full_attention_kv_only"
    assert policy["sliding_window_policy"] == "native_rotating_kv_state"
