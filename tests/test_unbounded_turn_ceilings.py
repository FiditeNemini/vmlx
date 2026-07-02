# SPDX-License-Identifier: Apache-2.0
"""In-stream hard ceilings for unbounded (model-owned max_tokens) turns.

Live finding (2026-07-02 openPangu UI soak, docs/OPENPANGU-V2-CAMPAIGN.md):
with model-owned max_tokens a reasoning turn decoded 50-90+ minutes unbounded;
the post-stream bounded answer backstop can never fire if the stream never
ends. Explicit max_tokens WAS honored. Two in-stream protections were added:

1. Scheduler thinking-budget backstop (generalized from the proven M3
   mechanism, commit b5af37d23): logits processor force-injects the family's
   close-think token once a model-owned turn passes VMLX_THINK_BUDGET
   thinking tokens without closing the reasoning rail. Gated to
   THINK_BUDGET_FAMILIES (openpangu_v2 only for now).
2. Server stream-loop absolute ceiling: model-owned turns are force-ended
   with finish_reason="length" at VMLX_MODEL_OWNED_MAX_TOKENS tokens,
   independent of scheduler/generator length accounting.

These tests pin: both ceilings fire, explicit max_tokens is unaffected, and
non-listed families are unaffected.
"""

import inspect

import mlx.core as mx
import pytest

from vmlx_engine.request import Request, SamplingParams
from vmlx_engine.scheduler import Scheduler, THINK_BUDGET_FAMILIES

VOCAB = 256
OPEN_ID = 100
CLOSE_ID = 101


class _FakeTokenizer:
    """Encodes the openpangu think tags to single ids."""

    def __init__(self, multi_token: bool = False):
        self._multi = multi_token

    def encode(self, text, add_special_tokens=False):
        if self._multi:
            return [1, 2]
        if text == "<think>":
            return [OPEN_ID]
        if text == "</think>":
            return [CLOSE_ID]
        return [7]


def _make_scheduler(tokenizer=None) -> Scheduler:
    """Bare Scheduler with only the attrs _request_logits_processors needs."""
    sched = Scheduler.__new__(Scheduler)
    sched.tokenizer = tokenizer or _FakeTokenizer()
    sched._actual_tokenizer = sched.tokenizer
    sched._long_repetition_context = False
    return sched


def _make_request(*, model_owned: bool, max_tokens: int = 4096, rep: float = 1.0):
    return Request(
        request_id="req-test",
        prompt="q",
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            repetition_penalty=rep,
            model_owned_max_tokens=model_owned,
        ),
    )


def _patch_family(monkeypatch, family: str):
    import vmlx_engine.server as server

    monkeypatch.setattr(
        server, "_model_family_for_defaults", lambda name="": family
    )


def _drive(processor, n_tokens: int, close_at: int | None = None):
    """Feed a growing token context to the wrapped processor; return the
    logits output for the final step (context length n_tokens)."""
    logits_in = mx.zeros((1, VOCAB))
    tokens = [OPEN_ID]  # prompt-open tail token survives the prefix trim
    out = logits_in
    while len(tokens) < n_tokens:
        tokens.append(
            CLOSE_ID if (close_at is not None and len(tokens) == close_at) else 5
        )
        out = processor(list(tokens), logits_in)
    return out


class TestThinkingBudgetBackstop:
    """Ceiling 1: close-think token injection in the scheduler."""

    def test_registry_lists_openpangu_only(self):
        assert set(THINK_BUDGET_FAMILIES) == {"openpangu_v2"}
        spec = THINK_BUDGET_FAMILIES["openpangu_v2"]
        assert spec["open_tag"] == "<think>"
        assert spec["close_tag"] == "</think>"
        assert spec["default_budget"] >= 4096

    def test_close_token_injected_at_budget(self, monkeypatch):
        _patch_family(monkeypatch, "openpangu_v2")
        monkeypatch.setenv("VMLX_THINK_BUDGET", "64")
        sched = _make_scheduler()
        req = _make_request(model_owned=True)
        procs = sched._request_logits_processors(req, [1, 2, 3, OPEN_ID])
        assert procs is not None and len(procs) == 1
        proc = procs[0]

        # Before the budget: logits pass through unchanged.
        early = proc([OPEN_ID, 5, 5, 5], mx.zeros((1, VOCAB)))
        assert mx.array_equal(early, mx.zeros((1, VOCAB)))

        # Past the budget without closing the rail: close token forced.
        forced = _drive(proc, 80)
        assert int(mx.argmax(forced, axis=-1).item()) == CLOSE_ID
        # One-hot mask: everything else is strongly suppressed.
        assert float(forced[0, CLOSE_ID].item()) == 0.0
        assert float(forced[0, 5].item()) <= -1e8

    def test_rail_closed_before_budget_disarms(self, monkeypatch):
        _patch_family(monkeypatch, "openpangu_v2")
        monkeypatch.setenv("VMLX_THINK_BUDGET", "64")
        sched = _make_scheduler()
        req = _make_request(model_owned=True)
        proc = sched._request_logits_processors(req, [1, 2, 3, OPEN_ID])[0]
        # Model closes </think> at token 10 — past-budget steps pass through.
        out = _drive(proc, 80, close_at=10)
        assert mx.array_equal(out, mx.zeros((1, VOCAB)))

    def test_explicit_max_tokens_not_armed(self, monkeypatch):
        """Client-set max_tokens (model_owned=False) never gets the backstop."""
        _patch_family(monkeypatch, "openpangu_v2")
        sched = _make_scheduler()
        req = _make_request(model_owned=False)
        assert sched._request_logits_processors(req, [1, 2, 3, OPEN_ID]) is None

    def test_non_openpangu_family_not_armed(self, monkeypatch):
        _patch_family(monkeypatch, "qwen3_5")
        sched = _make_scheduler()
        req = _make_request(model_owned=True)
        assert sched._request_logits_processors(req, [1, 2, 3, OPEN_ID]) is None

    def test_thinking_off_prompt_not_armed(self, monkeypatch):
        """Rail pre-closed in the prompt tail (thinking off) → no backstop."""
        _patch_family(monkeypatch, "openpangu_v2")
        sched = _make_scheduler()
        req = _make_request(model_owned=True)
        assert (
            sched._request_logits_processors(req, [1, 2, 3, CLOSE_ID]) is None
        )

    def test_multi_token_tags_disable_backstop(self, monkeypatch):
        """Tags that don't encode to single ids cannot be forced — disabled."""
        _patch_family(monkeypatch, "openpangu_v2")
        sched = _make_scheduler(tokenizer=_FakeTokenizer(multi_token=True))
        req = _make_request(model_owned=True)
        assert sched._request_logits_processors(req, [1, 2, 3]) is None

    def test_env_zero_disables(self, monkeypatch):
        _patch_family(monkeypatch, "openpangu_v2")
        monkeypatch.setenv("VMLX_THINK_BUDGET", "0")
        sched = _make_scheduler()
        req = _make_request(model_owned=True)
        assert sched._request_logits_processors(req, [1, 2, 3, OPEN_ID]) is None

    def test_budget_scales_below_output_budget(self, monkeypatch):
        """Backstop fires before max_tokens so the answer always has room."""
        _patch_family(monkeypatch, "openpangu_v2")
        monkeypatch.delenv("VMLX_THINK_BUDGET", raising=False)
        sched = _make_scheduler()
        req = _make_request(model_owned=True, max_tokens=512)
        proc = sched._request_logits_processors(req, [OPEN_ID])[0]
        # effective budget = min(8192, max(96, 512 // 2)) = 256
        below = _drive(proc, 200)
        assert mx.array_equal(below, mx.zeros((1, VOCAB)))
        fresh = sched._request_logits_processors(req, [OPEN_ID])[0]
        forced = _drive(fresh, 300)
        assert int(mx.argmax(forced, axis=-1).item()) == CLOSE_ID

    def test_coexists_with_repetition_penalty(self, monkeypatch):
        _patch_family(monkeypatch, "openpangu_v2")
        sched = _make_scheduler()
        req = _make_request(model_owned=True, rep=1.1)
        procs = sched._request_logits_processors(req, [1, OPEN_ID])
        assert procs is not None and len(procs) == 2


class TestModelOwnedResolution:
    """Server-side model-owned detection + ceiling resolution."""

    def test_absent_and_zero_are_model_owned(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
        assert server._max_tokens_is_model_owned(None) is True
        assert server._max_tokens_is_model_owned(0) is True

    def test_explicit_request_cap_is_not_model_owned(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_default_max_tokens_explicit", False)
        assert server._max_tokens_is_model_owned(64) is False

    def test_cli_override_is_not_model_owned(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_default_max_tokens_explicit", True)
        assert server._max_tokens_is_model_owned(None) is False

    def test_ceiling_default_and_env(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.delenv("VMLX_MODEL_OWNED_MAX_TOKENS", raising=False)
        assert server._model_owned_output_ceiling() == 32768
        monkeypatch.setenv("VMLX_MODEL_OWNED_MAX_TOKENS", "1000")
        assert server._model_owned_output_ceiling() == 1000
        monkeypatch.setenv("VMLX_MODEL_OWNED_MAX_TOKENS", "0")
        assert server._model_owned_output_ceiling() == 0


class TestStreamAbsoluteCeiling:
    """Ceiling 2: server stream loop force-ends unbounded model-owned turns."""

    class _Tokenizer:
        has_thinking = False

        def encode(self, text, add_special_tokens=False):
            return [1]

    def _engine(self, *, endless: bool, chunks: int = 40):
        from vmlx_engine.engine.base import GenerationOutput

        tokenizer = self._Tokenizer()

        class _Engine:
            def __init__(self):
                self.tokenizer = tokenizer
                self.aborted: list[str] = []

            async def stream_chat(self, *, messages, **kwargs):
                i = 0
                while True:
                    i += 1
                    finished = (not endless) and i >= chunks
                    yield GenerationOutput(
                        text="x" * i,
                        new_text="x",
                        prompt_tokens=3,
                        completion_tokens=i,
                        finished=finished,
                        finish_reason="stop" if finished else None,
                    )
                    if finished:
                        return

            async def abort_request(self, request_id):
                self.aborted.append(request_id)

        return _Engine()

    def _patch_server(self, monkeypatch):
        import vmlx_engine.server as server

        monkeypatch.setattr(server, "_default_timeout", 10.0)
        monkeypatch.setattr(server, "_model_name", "loaded-model")
        monkeypatch.setattr(server, "_model_path", None)
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        return server

    @pytest.mark.asyncio
    async def test_model_owned_turn_force_ends_with_length(self, monkeypatch):
        import json

        server = self._patch_server(monkeypatch)
        monkeypatch.setenv("VMLX_MODEL_OWNED_MAX_TOKENS", "25")
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        engine = self._engine(endless=True)
        request = ChatCompletionRequest(
            model="loaded-model",
            messages=[Message(role="user", content="hard question")],
            stream=True,
        )
        chunks = []
        async for line in server.stream_chat_completion(
            engine,
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
            max_tokens=4096,
            _model_owned_max_tokens=True,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        # The endless stream terminated (we got here at all), the engine
        # request was aborted, and the client saw finish_reason="length".
        assert engine.aborted, "engine.abort_request must be called"
        finish_reasons = [
            c["choices"][0].get("finish_reason")
            for c in chunks
            if c.get("choices")
        ]
        assert "length" in finish_reasons
        content = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c.get("choices")
        )
        assert 0 < len(content) <= 30

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_stream_unaffected(self, monkeypatch):
        import json

        server = self._patch_server(monkeypatch)
        monkeypatch.setenv("VMLX_MODEL_OWNED_MAX_TOKENS", "25")
        from vmlx_engine.api.models import ChatCompletionRequest, Message

        engine = self._engine(endless=False, chunks=40)
        request = ChatCompletionRequest(
            model="loaded-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
            max_tokens=40,
        )
        chunks = []
        async for line in server.stream_chat_completion(
            engine,
            [m.model_dump(exclude_none=True) for m in request.messages],
            request,
            fastapi_request=None,
            max_tokens=40,
        ):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))

        # No _model_owned_max_tokens flag → wall disarmed even past 25 tokens.
        assert not engine.aborted
        content = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c.get("choices")
        )
        assert len(content) == 40
        finish_reasons = [
            c["choices"][0].get("finish_reason")
            for c in chunks
            if c.get("choices")
        ]
        assert "stop" in finish_reasons
        assert "length" not in finish_reasons


class TestModelOwnedFlagPlumbing:
    """Source contracts: the flag must thread server → engine → SamplingParams."""

    def test_sampling_params_field_default_off(self):
        assert SamplingParams().model_owned_max_tokens is False
        assert SamplingParams(model_owned_max_tokens=True).model_owned_max_tokens

    def test_batched_engine_threads_flag(self):
        from vmlx_engine.engine import batched

        src = inspect.getsource(batched.BatchedEngine.stream_generate)
        assert '_model_owned_max_tokens' in src
        assert "model_owned_max_tokens=model_owned_max_tokens" in src
        src_gen = inspect.getsource(batched.BatchedEngine.generate)
        assert '_model_owned_max_tokens' in src_gen
        assert "model_owned_max_tokens=model_owned_max_tokens" in src_gen

    def test_chat_handlers_set_flag(self):
        import vmlx_engine.server as server

        src = inspect.getsource(server)
        assert src.count('chat_kwargs["_model_owned_max_tokens"] = True') >= 2

    def test_answer_passes_strip_flag(self):
        """Bounded second passes must not re-arm the unbounded-turn guards."""
        import vmlx_engine.server as server

        src = inspect.getsource(server)
        assert src.count('.pop("_model_owned_max_tokens", None)') >= 4
