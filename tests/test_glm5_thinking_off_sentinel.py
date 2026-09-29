"""GLM5-next has native effort levels, not a thinking-off generation rail."""

import pytest

from vmlx_engine.utils.chat_template_kwargs import ensure_thinking_off_sentinel


@pytest.mark.parametrize("tools_present", [False, True])
def test_glm5_next_does_not_close_native_generation_rail(tools_present):
    prompt = "<|user|>Read the inventory.<|assistant|><think>"
    assert ensure_thinking_off_sentinel(
        prompt, family_name="glm5_next", tools_present=tools_present
    ) == prompt


def test_glm5_next_registry_rejects_instruct_mode(tmp_path, monkeypatch):
    import json
    from fastapi import HTTPException
    from vmlx_engine import server
    from vmlx_engine.model_config_registry import get_model_config_registry

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    config = get_model_config_registry().lookup(str(tmp_path))
    assert config.supports_instruct_mode is False
    assert config.supported_reasoning_efforts == ["low", "high", "max"]
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    for value, kwargs in ((False, {}), (None, {"enable_thinking": False})):
        with pytest.raises(HTTPException, match="does not expose a native thinking-off"):
            server._resolve_enable_thinking(
                request_value=value, ct_kwargs=kwargs, tools_present=False,
                model_key=str(tmp_path),
            )
    for effort in ("low", "high", "max"):
        assert server._resolve_enable_thinking(
            request_value=None, ct_kwargs={"reasoning_effort": effort},
            tools_present=False, model_key=str(tmp_path), reasoning_effort=effort,
        ) is True


@pytest.mark.parametrize("surface", ["chat", "responses"])
@pytest.mark.parametrize("controls", [
    {"enable_thinking": False},
    {"reasoning_effort": "none"},
    {"reasoning": {"effort": "none"}},
    {"thinking_mode": "off"},
    {"chat_template_kwargs": {"enable_thinking": False}},
])
def test_glm5_native_contract_after_api_alias_normalization(
    tmp_path, monkeypatch, surface, controls
):
    import json
    from fastapi import HTTPException
    from vmlx_engine import server
    from vmlx_engine.api.models import ChatCompletionRequest, ResponsesRequest

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "glm5_next", "capabilities": {
            "supports_thinking": True, "think_in_template": True,
            "reasoning_parser": "glm_think_block",
        },
    }))
    monkeypatch.setattr(server, "_model_path", str(tmp_path))
    monkeypatch.setattr(server, "_default_enable_thinking", None)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {})
    if surface == "chat":
        request = ChatCompletionRequest(
            model=str(tmp_path), messages=[{"role": "user", "content": "Hello"}],
            **controls,
        )
    else:
        request = ResponsesRequest(model=str(tmp_path), input="Hello", **controls)
    kwargs = server._merge_ct_kwargs(
        request.chat_template_kwargs, request.reasoning_effort,
        enable_thinking=request.enable_thinking,
    )
    with pytest.raises(HTTPException, match="does not expose a native thinking-off"):
        server._resolve_enable_thinking(
            request_value=request.enable_thinking, ct_kwargs=kwargs,
            tools_present=False, model_key=str(tmp_path),
            reasoning_effort=request.reasoning_effort,
        )


def test_glm5_next_keeps_closed_history_and_does_not_invent_rail():
    for prompt in (
        "<|assistant|><think></think>",
        "<|assistant|><think>earlier reasoning</think>answer",
        "<|assistant|>",
    ):
        assert ensure_thinking_off_sentinel(
            prompt, family_name="glm5_next"
        ) == prompt


@pytest.mark.parametrize("family", ["glm4_moe", "glm_moe_dsa", "unknown"])
def test_glm_name_does_not_broaden_native_contract(family):
    prompt = "<|assistant|><think>"
    assert ensure_thinking_off_sentinel(
        prompt, family_name=family, model_name="GLM-5.3-Flash"
    ) == prompt + "\n</think>\n\n"
