"""Pins for the two silent API failures fixed here."""
import pytest

from vmlx_engine.api.models import ChatCompletionRequest
from vmlx_engine.api.ollama_adapter import *  # noqa - for the mapper below

MSGS = [{"role": "user", "content": "hi"}]


class TestMaxCompletionTokens:
    """OpenAI deprecated max_tokens for chat; current SDKs send only this.

    The request model ignores unknown fields, so before it was declared a
    client setting only max_completion_tokens silently got the default output
    cap — a truncated or runaway generation with no error anywhere.
    """

    def test_alone_it_drives_max_tokens(self):
        assert ChatCompletionRequest(model="m", messages=MSGS,
                                     max_completion_tokens=16).max_tokens == 16

    def test_legacy_spelling_still_works(self):
        assert ChatCompletionRequest(model="m", messages=MSGS,
                                     max_tokens=32).max_tokens == 32

    def test_both_agreeing_is_accepted(self):
        assert ChatCompletionRequest(model="m", messages=MSGS, max_tokens=8,
                                     max_completion_tokens=8).max_tokens == 8

    def test_both_disagreeing_is_rejected_not_guessed(self):
        with pytest.raises(ValueError, match="disagree"):
            ChatCompletionRequest(model="m", messages=MSGS, max_tokens=8,
                                  max_completion_tokens=9)

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError):
            ChatCompletionRequest(model="m", messages=MSGS, max_completion_tokens=0)


class TestOllamaStreamingHonoursFormat:
    """Ollama's `format` must survive into the STREAMING path.

    Streaming is Ollama's default. The streaming branch builds its own
    chat_kwargs instead of delegating, so the mapped response_format used to be
    dropped — JSON mode worked at stream:false and silently did nothing at
    stream:true, which reads as flaky rather than broken.
    """

    def test_streaming_branch_injects_and_forwards_response_format(self):
        import inspect

        from vmlx_engine import server

        src = inspect.getsource(server.ollama_chat)
        assert "_vmlx_response_format" in src, (
            "ollama streaming branch never forwards response_format to the engine"
        )
        assert "_inject_json_instruction" in src, (
            "ollama streaming branch never injects the JSON system prompt"
        )
