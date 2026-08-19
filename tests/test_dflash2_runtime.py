from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch


def test_simple_engine_counts_multi_token_mllm_chunks_cumulatively():
    from vmlx_engine.engine.simple import _advance_mllm_completion_tokens

    assert _advance_mllm_completion_tokens(
        1, SimpleNamespace(text="four tokens", completion_tokens=5)
    ) == 5
    assert _advance_mllm_completion_tokens(
        5, SimpleNamespace(text="", completion_tokens=5)
    ) == 5
    assert _advance_mllm_completion_tokens(
        5, SimpleNamespace(text="legacy", completion_tokens=0)
    ) == 6


def test_qwen_mtp_model_wrapper_accepts_upstream_capture_contract():
    from vmlx_engine.patches.mlx_vlm_mtp.qwen35_vl import _patch_qwen_model

    class FakeQwenModel:
        def __call__(
            self,
            inputs,
            inputs_embeds=None,
            mask=None,
            cache=None,
            position_ids=None,
        ):
            return "original"

    qlang = SimpleNamespace(Qwen3_5Model=FakeQwenModel)
    _patch_qwen_model(qlang)

    signature = inspect.signature(FakeQwenModel.__call__)
    assert "capture_layer_ids" in signature.parameters
    assert "hidden_sink" in signature.parameters
    assert FakeQwenModel()(None, capture_layer_ids=None, hidden_sink=None) == "original"


def test_stream_chat_routes_text_only_dflash2_before_mlx_vlm_generator():
    from vmlx_engine.models.mllm import MLXMultimodalLM

    model = object.__new__(MLXMultimodalLM)
    model._loaded = True
    model.model_name = "target"
    model.model = object()
    model.processor = SimpleNamespace(tokenizer=object())
    model.config = {}
    model._extract_multimodal_messages = lambda _messages: (
        [{"role": "user", "content": "hello"}],
        [],
        [],
        [],
    )
    model._normalize_text_only_messages_for_processor = (
        lambda messages, has_media: messages
    )
    model._apply_chat_template = lambda messages, enable_thinking, tools=None: "PROMPT"

    chunks = [
        SimpleNamespace(
            text="one",
            tokens=[1],
            accepted=None,
            prompt_tokens=7,
            generation_tps=0.0,
            finish_reason=None,
        ),
        SimpleNamespace(
            text=" two",
            tokens=[2, 3],
            accepted=2,
            prompt_tokens=7,
            generation_tps=55.0,
            finish_reason="stop",
        ),
    ]

    with (
        patch("vmlx_engine.speculative.is_dflash2_enabled", return_value=True),
        patch("vmlx_engine.speculative.get_draft_model", return_value=object()),
        patch(
            "vmlx_engine.dflash2_runtime.stream_dflash2_generate",
            return_value=iter(chunks),
        ) as bridge,
        patch("mlx_vlm.stream_generate") as ordinary,
    ):
        outputs = list(
            model.stream_chat(
                [{"role": "user", "content": "hello"}],
                max_tokens=10,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
        )

    bridge.assert_called_once()
    ordinary.assert_not_called()
    assert [output.text for output in outputs] == ["one", " two"]
    assert outputs[-1].completion_tokens == 3
    assert outputs[-1].finish_reason == "stop"
