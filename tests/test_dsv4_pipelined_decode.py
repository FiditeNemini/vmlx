"""Pipelined DSV4 decode fast path: parity, stop handling, eligibility."""

import mlx.core as mx

from vmlx_engine.utils.dsv4_batch_generator import (
    DSV4BatchGenerator,
    _Request,
)

VOCAB = 16


class _FakeModel:
    """Deterministic next-token model: argmax(logits) == (input + 1) % VOCAB."""

    def __init__(self):
        self.calls = 0

    def __call__(self, tokens, cache):
        self.calls += 1
        nxt = (tokens[:, -1].astype(mx.int32) + 1) % VOCAB
        logits = mx.zeros((1, 1, VOCAB))
        one_hot = (mx.arange(VOCAB)[None, :] == nxt[:, None]).astype(mx.float32)
        return logits + one_hot[:, None, :]


def _bare_generator(pipelined: bool) -> DSV4BatchGenerator:
    gen = DSV4BatchGenerator.__new__(DSV4BatchGenerator)
    gen.model = _FakeModel()
    gen.fallback_sampler = lambda x: mx.argmax(x, axis=-1)
    gen.stop_tokens = set()
    gen._warmed_up = True
    gen._requests = []
    gen._device = mx.default_device()
    gen._stream = mx.default_stream(gen._device)
    gen._pipelined_decode = pipelined
    gen.capture_prompt_snapshot = False
    return gen


def _decode_request(max_tokens: int, **kwargs) -> _Request:
    return _Request(
        uid=0,
        prompt_tokens=[1],
        context_tokens=[1],
        cache=[],
        out_tokens=[1],
        max_tokens=max_tokens,
        prompt_processed=True,
        **kwargs,
    )


def _run(gen: DSV4BatchGenerator, req: _Request, steps: int):
    gen._requests = [req]
    tokens = []
    for _ in range(steps):
        if req.finish_reason is not None:
            break
        _, gen_resps = gen.next()
        tokens.extend(resp.token for resp in gen_resps)
    return tokens


def test_pipelined_tokens_match_sync_path():
    sync_gen = _bare_generator(pipelined=False)
    sync_req = _decode_request(max_tokens=8)
    sync_tokens = _run(sync_gen, sync_req, 8)

    pipe_gen = _bare_generator(pipelined=True)
    pipe_req = _decode_request(max_tokens=8)
    pipe_tokens = _run(pipe_gen, pipe_req, 8)

    assert sync_tokens == pipe_tokens
    # max_tokens counts out_tokens including the prefill-boundary token, so
    # 8 - 1 = 7 decode tokens.
    assert pipe_tokens == [2, 3, 4, 5, 6, 7, 8]
    assert sync_req.finish_reason == "length"
    assert pipe_req.finish_reason == "length"
    # Length finish must not leave a speculative forward pending.
    assert pipe_req.pending_sampled is None


def test_pipelined_stop_token_drops_pending_speculative():
    gen = _bare_generator(pipelined=True)
    gen.stop_tokens = {4}
    req = _decode_request(max_tokens=64)
    tokens = _run(gen, req, 8)

    assert tokens == [2, 3, 4]
    assert req.finish_reason == "stop"
    assert req.pending_sampled is None


def test_processors_route_to_sync_path():
    gen = _bare_generator(pipelined=True)
    req = _decode_request(
        max_tokens=4,
        logits_processors=[lambda ctx, logits: logits],
    )
    tokens = _run(gen, req, 4)

    assert tokens == [2, 3, 4]
    # The pipeline must never prime for processor-bearing requests: the sync
    # path assumes the live cache holds exactly the committed tokens.
    assert req.pending_sampled is None


def test_pipeline_keeps_exactly_one_forward_in_flight():
    gen = _bare_generator(pipelined=True)
    req = _decode_request(max_tokens=8)
    gen._requests = [req]
    _, first = gen.next()
    assert [r.token for r in first] == [2]
    # Prime + speculative for the next step: two model calls on step one,
    # then one per subsequent step.
    assert gen.model.calls == 2
    _, second = gen.next()
    assert [r.token for r in second] == [3]
    assert gen.model.calls == 3
