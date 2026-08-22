# SPDX-License-Identifier: Apache-2.0
"""Media-expanded prefill must chunk, so a long VL chat stops hitting a wall.

The one-shot forward pushed the WHOLE media-expanded prompt through the
language model in a single command buffer. Measured on an M5 Max: a
28,483-token prompt returned kIOGPUCommandBufferCallbackErrorOutOfMemory with
89GB free -- the problem is one enormous allocation, not total memory. Since a
chat client re-sends the image every turn, the prompt only grows, so a VL
conversation dies and never recovers.

The vision tower needs the whole image; the language model does not need the
whole sequence. Every wrapper here already exposes that seam.
"""

import os

import pytest

from vmlx_engine.mllm_batch_generator import (
    _MEDIA_PREFILL_CHUNK_FLOOR,
    _MEDIA_PREFILL_CHUNK_MIN_SEQ,
    _media_chunk_boundaries,
    _media_embed_kwarg_name,
    _media_placeholder_runs,
    _named_params,
)


class TestCapabilityDetection:
    """`**kwargs` is not support. It only looks like support."""

    def test_kwargs_absorption_does_not_count_as_a_named_parameter(self):
        def swallows(a, **kwargs):
            pass

        assert _named_params(swallows) == {"a"}
        assert "inputs_embeds" not in _named_params(swallows)

    def test_detects_both_spellings(self):
        class _A:
            def __call__(self, inputs, inputs_embeds=None, cache=None):
                pass

        class _B:
            def __call__(self, inputs, input_embeddings=None, cache=None):
                pass

        assert _media_embed_kwarg_name(_A()) == "inputs_embeds"
        assert _media_embed_kwarg_name(_B()) == "input_embeddings"

    def test_a_model_that_only_swallows_kwargs_is_not_chunkable(self):
        """Several wrappers here take position_ids into **kwargs and drop it.

        Treating that as support would build a chunked prefill on a model
        that ignores the per-chunk positions and returns confident garbage.
        """

        class _Swallower:
            def __call__(self, inputs, cache=None, **kwargs):
                pass

        assert _media_embed_kwarg_name(_Swallower()) is None

    def test_none_model_is_not_chunkable(self):
        assert _media_embed_kwarg_name(None) is None


class TestChunkBoundaries:
    def test_plain_split_when_there_is_no_media_run(self):
        assert _media_chunk_boundaries(10, 4, []) == [4, 8, 10]

    def test_a_boundary_inside_a_media_run_snaps_to_the_run_start(self):
        """Harmless today, not necessarily tomorrow.

        Post-merge every family here builds masks from the cache offset, so a
        mid-run split matches the one-shot result. It stops matching the
        moment a family builds a mask from whole-sequence image geometry --
        gemma4's config already asks for bidirectional vision attention even
        though the MLX language model does not implement it, and qwen3_vl's
        deepstack injection is keyed to visual rows in the current window.
        Snapping costs nothing, so it is not worth being clever about.
        """
        # run occupies [5, 12); a naive split at 8 would land inside it
        bounds = _media_chunk_boundaries(20, 8, [(5, 12)])
        assert 8 not in bounds
        assert bounds[0] == 5

    def test_snapping_always_makes_forward_progress(self):
        """A run starting at the current position must not stall the loop."""
        bounds = _media_chunk_boundaries(30, 4, [(0, 25)])
        assert bounds == sorted(set(bounds))
        assert bounds[-1] == 30
        assert all(b > 0 for b in bounds)
        # strictly increasing => the loop terminates
        assert all(b2 > b1 for b1, b2 in zip(bounds, bounds[1:]))

    def test_a_run_covering_the_whole_sequence_still_terminates(self):
        assert _media_chunk_boundaries(16, 4, [(0, 16)])[-1] == 16

    def test_degenerate_inputs(self):
        assert _media_chunk_boundaries(0, 8, []) == [0]
        assert _media_chunk_boundaries(10, 0, []) == [10]


class TestPlaceholderRuns:
    def test_finds_half_open_spans(self):
        ids = [1, 2, 99, 99, 99, 3, 4, 99, 5]
        assert _media_placeholder_runs(ids, {99}) == [(2, 5), (7, 8)]

    def test_run_touching_the_end(self):
        assert _media_placeholder_runs([1, 99, 99], {99}) == [(1, 3)]

    def test_no_media_ids_means_no_runs(self):
        assert _media_placeholder_runs([1, 2, 3], set()) == []
        assert _media_placeholder_runs(None, {99}) == []


class TestChunkSizing:
    def test_the_floor_is_large_on_purpose(self):
        """A SMALLER CHUNK DOES NOT REDUCE WEIGHT STREAMING -- IT MULTIPLIES IT.

        The chunk bounds only the terms that scale with it; the weights are
        re-read in full on every chunk. dots3 restreams ~85GB of expert
        weights per chunk, so a 64-token chunk paid that 32x more often than a
        2048-token one. Anyone tempted to shrink this to "be safe" is making
        long prompts slower, not safer.
        """
        assert _MEDIA_PREFILL_CHUNK_FLOOR >= 4096

    def test_short_media_prompts_stay_one_shot(self):
        """One-shot reads the weights once and was never the failing shape."""
        assert _MEDIA_PREFILL_CHUNK_MIN_SEQ >= _MEDIA_PREFILL_CHUNK_FLOOR


class _OneShotModel:
    """Callable stand-in for a VLM wrapper. `__call__` must live on the TYPE.

    A SimpleNamespace with a `__call__` attribute is NOT callable -- Python
    looks dunders up on the type, not the instance -- which is exactly how the
    first version of these tests managed to "fail" against working code.
    """

    def __init__(self, calls, **attrs):
        self._calls = calls
        for key, value in attrs.items():
            setattr(self, key, value)

    def __call__(self, ids, **kwargs):
        self._calls.append("one-shot")
        return "out"


class _EmbedsLM:
    def __call__(self, inputs, inputs_embeds=None, cache=None):
        return None


class _NoEmbedsLM:
    def __call__(self, inputs, cache=None, **kwargs):
        return None


class TestMediaForwardFallbacks:
    def _gen(self, model, lm):
        from vmlx_engine.mllm_batch_generator import MLLMBatchGenerator

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.prefill_step_size = 2048
        gen.model = model
        gen.language_model = lm
        return gen

    def _run(self, gen):
        from types import SimpleNamespace

        return gen._media_forward(
            SimpleNamespace(request_id="r"),
            _FakeIds(30000),
            30000,
            [object()],
            {},
        )

    def test_falls_back_to_one_shot_without_an_embeddings_api(self):
        calls = []
        gen = self._gen(_OneShotModel(calls), _NoEmbedsLM())
        assert self._run(gen) == "out"
        assert calls == ["one-shot"]

    def test_env_kill_switch_forces_one_shot(self, monkeypatch):
        monkeypatch.setenv("VMLX_DISABLE_MEDIA_CHUNKED_PREFILL", "1")
        calls = []
        gen = self._gen(_OneShotModel(calls), _EmbedsLM())
        self._run(gen)
        assert calls == ["one-shot"]

    def test_wrapper_can_declare_itself_unchunkable(self):
        calls = []
        gen = self._gen(
            _OneShotModel(calls, no_chunked_prefill=True), _EmbedsLM()
        )
        self._run(gen)
        assert calls == ["one-shot"]

    def test_short_prompts_stay_one_shot_even_when_chunkable(self):
        """One-shot reads the weights once; it was never the failing shape."""
        from types import SimpleNamespace

        calls = []
        gen = self._gen(_OneShotModel(calls), _EmbedsLM())
        gen.model.get_input_embeddings = lambda ids, **kw: SimpleNamespace(
            inputs_embeds=_FakeIds(1000)
        )
        gen._media_forward(
            SimpleNamespace(request_id="r"), _FakeIds(1000), 1000,
            [object()], {},
        )
        assert calls == ["one-shot"]

    def test_a_failing_embedding_merge_falls_back_instead_of_erroring(self):
        from types import SimpleNamespace

        def _boom(ids, **kw):
            raise RuntimeError("merge exploded")

        calls = []
        gen = self._gen(_OneShotModel(calls), _EmbedsLM())
        gen.model.get_input_embeddings = _boom
        assert self._run(gen) == "out"
        assert calls == ["one-shot"]


class _FakeIds:
    """Minimal stand-in for an mx.array of token ids."""

    def __init__(self, n):
        self._n = n
        self.ndim = 2

    def __getitem__(self, item):
        return self

    def tolist(self):
        return list(range(self._n))
