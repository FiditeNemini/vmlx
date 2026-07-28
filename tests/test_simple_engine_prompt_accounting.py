# SPDX-License-Identifier: Apache-2.0
"""Focused tests for SimpleEngine LLM prompt-token accounting."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vmlx_engine.engine.simple import SimpleEngine


class _FakeBosTokenizer:
    bos_token = "<s>"

    def __init__(self):
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        token_ids: list[int] = []
        if text.startswith(self.bos_token):
            token_ids.append(1)
            text = text[len(self.bos_token) :]
        if add_special_tokens:
            token_ids.insert(0, 1)
        token_ids.extend(range(2, 2 + len(text.split())))
        return token_ids

    def apply_chat_template(self, _messages, **_kwargs) -> str:
        return "<s>rendered assistant"


class _FakeLLM:
    def __init__(self, *, stream_ends_explicitly: bool = True):
        self.tokenizer = _FakeBosTokenizer()
        self.stream_ends_explicitly = stream_ends_explicitly
        self.generate_prompts: list[str] = []
        self.stream_prompts: list[str] = []

    def generate(self, *, prompt: str, **_kwargs):
        self.generate_prompts.append(prompt)
        return SimpleNamespace(
            text="ok",
            tokens=[9],
            prompt_tokens=0,
            completion_tokens=1,
            finish_reason="stop",
        )

    def stream_generate(self, *, prompt: str, **_kwargs):
        self.stream_prompts.append(prompt)
        yield SimpleNamespace(
            text="ok",
            prompt_tokens=0,
            finished=self.stream_ends_explicitly,
            finish_reason="stop" if self.stream_ends_explicitly else None,
        )


class _FakeProcessorModel:
    def __init__(self):
        self.processor = SimpleNamespace(tokenizer=_FakeBosTokenizer())
        self.generate_prompts: list[str] = []

    def generate(self, *, prompt: str, **_kwargs):
        self.generate_prompts.append(prompt)
        return SimpleNamespace(
            text="ok",
            tokens=[9],
            prompt_tokens=0,
            completion_tokens=1,
            finish_reason="stop",
        )


def _loaded_engine(model, *, is_mllm: bool = False) -> SimpleEngine:
    with patch(
        "vmlx_engine.engine.simple.is_mllm_model",
        return_value=is_mllm,
    ):
        engine = SimpleEngine("fake-llm")
    engine._model = model
    engine._loaded = True

    async def _run_inline(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    engine._run_model_call = _run_inline
    return engine


@pytest.mark.parametrize(
    ("prompt", "expected_add_special_tokens"),
    [
        ("<s>alpha beta", False),
        ("alpha beta", True),
    ],
)
def test_prompt_limit_accounting_matches_mlx_lm_bos_policy(
    prompt: str,
    expected_add_special_tokens: bool,
):
    model = _FakeLLM()
    engine = _loaded_engine(model)

    assert engine._prompt_token_count(prompt) == 3
    assert model.tokenizer.encode_calls[-1] == (
        prompt,
        expected_add_special_tokens,
    )


def test_mimo_text_only_accounting_counts_rendered_bos_once():
    tokenizer = _FakeBosTokenizer()
    with patch("vmlx_engine.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine("fake-mimo")
    engine._model = SimpleNamespace(
        model=SimpleNamespace(language_model=object()),
    )
    captured_prompts: list[str] = []

    def _generate(_model, _tokenizer, *, prompt: str, **_kwargs):
        captured_prompts.append(prompt)
        return "ok"

    with (
        patch("mlx_lm.generate", side_effect=_generate),
        patch("vmlx_engine.sampling.make_sampler", return_value=object()),
    ):
        output = engine._mimo_text_only_generate(
            prompt="<s>alpha beta",
            tokenizer=tokenizer,
            max_tokens=4,
            temperature=0.6,
            top_p=0.95,
            stop=None,
            enable_thinking=True,
            kwargs={},
        )

    assert output.prompt_tokens == 3
    assert captured_prompts == ["<s>alpha beta"]
    assert tokenizer.encode_calls[-1] == ("<s>alpha beta", False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_add_special_tokens"),
    [
        ("<s>alpha beta", False),
        ("alpha beta", True),
    ],
)
async def test_nonstream_generate_fallback_counts_without_rewriting_prompt(
    prompt: str,
    expected_add_special_tokens: bool,
):
    model = _FakeLLM()
    engine = _loaded_engine(model)

    output = await engine.generate(prompt, max_tokens=1)

    assert output.prompt_tokens == 3
    assert model.generate_prompts == [prompt]
    assert model.generate_prompts[0].encode() == prompt.encode()
    assert model.tokenizer.encode_calls[-1] == (
        prompt,
        expected_add_special_tokens,
    )


@pytest.mark.asyncio
async def test_nonstream_processor_model_does_not_guess_mllm_prompt_tokens():
    model = _FakeProcessorModel()
    engine = _loaded_engine(model, is_mllm=True)
    prompt = "<s>alpha beta"

    output = await engine.generate(prompt, max_tokens=1)

    assert output.prompt_tokens == 0
    assert model.generate_prompts == [prompt]
    assert model.generate_prompts[0].encode() == prompt.encode()
    assert model.processor.tokenizer.encode_calls == []


@pytest.mark.asyncio
async def test_nonstream_chat_fallback_counts_rendered_bos_once():
    model = _FakeLLM()
    engine = _loaded_engine(model)

    output = await engine.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=1,
    )

    assert output.prompt_tokens == 3
    assert model.generate_prompts == ["<s>rendered assistant"]
    assert model.generate_prompts[0].encode() == b"<s>rendered assistant"
    assert model.tokenizer.encode_calls[-1] == ("<s>rendered assistant", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_ends_explicitly", [True, False])
async def test_stream_fallback_counts_terminal_and_exhausted_prompts_once(
    stream_ends_explicitly: bool,
):
    model = _FakeLLM(stream_ends_explicitly=stream_ends_explicitly)
    engine = _loaded_engine(model)
    prompt = "<s>alpha beta"

    chunks = [
        chunk
        async for chunk in engine.stream_generate(prompt, max_tokens=4)
    ]

    assert chunks[-1].finished is True
    assert chunks[-1].prompt_tokens == 3
    assert model.stream_prompts == [prompt]
    assert model.stream_prompts[0].encode() == prompt.encode()
    assert model.tokenizer.encode_calls[-1] == (prompt, False)
