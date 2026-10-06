"""Template data must not overwrite generation controls or call arguments."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.engine.simple import SimpleEngine


def test_template_kwargs_do_not_collide_with_generation_arguments():
    async def run():
        captured = {}
        def generate(*, messages, max_tokens, temperature, top_p, **kwargs):
            captured.update(max_tokens=max_tokens, temperature=temperature, **kwargs)
            yield SimpleNamespace(text="ok", finish_reason="stop", prompt_tokens=1,
                                  completion_tokens=1)
        with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("fake")
        engine._loaded = True
        engine._model = SimpleNamespace(stream_chat=generate)
        async def call(fn):
            return fn()
        engine._run_model_call = call
        extra = {"max_tokens": "template variable", "temperature": "template value",
                 "enable_thinking": True, "reasoning_effort": "high", "custom": 42}
        outputs = [o async for o in engine.stream_chat([], max_tokens=17, temperature=0.2,
                   enable_thinking=False, reasoning_effort="low", chat_template_kwargs=extra)]
        assert outputs[-1].text == "ok"
        assert captured["max_tokens"] == 17 and captured["temperature"] == 0.2
        assert captured["enable_thinking"] is False and captured["reasoning_effort"] == "low"
        assert captured["chat_template_kwargs"] == extra
    asyncio.run(run())


def test_mllm_template_receives_custom_values_but_request_controls_win():
    from vmlx_engine.models.mllm import MLXMultimodalLM

    model = object.__new__(MLXMultimodalLM)
    model.processor = SimpleNamespace()
    model.config = {"model_type": "qwen3_5"}
    model.model_name = "fake"
    with patch("mlx_vlm.prompt_utils.get_chat_template", return_value="PROMPT") as render:
        prompt = model._apply_chat_template(
            [{"role": "user", "content": "hello"}], enable_thinking=False,
            reasoning_effort="low", chat_template_kwargs={
                "custom": 42, "max_tokens": "template variable", "tokenize": True,
                "add_generation_prompt": False, "enable_thinking": True,
                "thinking": True, "reasoning_effort": "high",
            },
        )
    assert prompt == "PROMPT"
    assert render.call_args.kwargs == {
        "add_generation_prompt": True, "custom": 42, "max_tokens": "template variable",
        "enable_thinking": False, "thinking": False, "reasoning_effort": "low",
    }
