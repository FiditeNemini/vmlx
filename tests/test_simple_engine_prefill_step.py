"""--prefill-step-size reaches every SimpleEngine lane (user report 2026-10-11).

The batched engines take it from SchedulerConfig; SimpleEngine (DFlash2 lanes, --no-continuous-batching) used each
library's own 2048 default, so a user's smaller step (the documented remedy for Metal OOM on big-MoE prefills) was
silently ignored: a 158-token prompt still prefilled in one chunk with --prefill-step-size 8.
"""
from types import SimpleNamespace

import pytest

from vmlx_engine.utils import prefill_step


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    prefill_step.set_prefill_step_size(prefill_step.DEFAULT_PREFILL_STEP_SIZE)


def test_default_matches_every_library_default():
    """An unset flag must not change any lane: mlx-lm, mlx-vlm and both DFlash2 loops default to 2048 too."""
    import inspect
    import importlib

    assert prefill_step.prefill_step_size() == 2048
    for module in ("mlx_lm.generate", "mlx_vlm.generate"):
        step = inspect.signature(importlib.import_module(module).generate_step).parameters["prefill_step_size"]
        assert step.default == 2048, module
    from vmlx_engine import dflash2_runtime, glm_dflash2

    assert inspect.signature(dflash2_runtime._stream_generate_resumable).parameters["prefill_step_size"].default == 2048
    assert inspect.signature(glm_dflash2._dflash2_rounds).parameters["prefill_step"].default == 2048


def test_llm_lane_forwards_the_step(monkeypatch):
    import mlx_lm

    from vmlx_engine.models.llm import MLXLanguageModel

    seen = {}

    def fake_stream_generate(model, tokenizer, **kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    llm = MLXLanguageModel.__new__(MLXLanguageModel)
    llm._loaded, llm.model, llm.tokenizer, llm.model_name = True, object(), object(), "fixture"
    llm._create_sampler = lambda *a, **k: None
    prefill_step.set_prefill_step_size(8)
    list(llm.stream_generate("hi", max_tokens=1))
    assert seen["prefill_step_size"] == 8


def test_glm_dflash2_lane_forwards_the_step(monkeypatch):
    pytest.importorskip("dflash.model_mlx")
    import mlx_lm.tokenizer_utils as tu

    import vmlx_engine.glm_dflash2 as glm

    class Detok:
        last_segment = ""
        def reset(self): pass
        def add_token(self, t): pass
        def finalize(self): pass

    class FakeWrapper:
        eos_token_ids = {99}
        detokenizer = Detok()

    seen = {}

    def rounds(*args, prefill_step=None, **kwargs):
        seen["prefill_step"] = prefill_step
        yield [99], 0

    monkeypatch.setattr(tu, "TokenizerWrapper", FakeWrapper)
    monkeypatch.setattr(glm, "target_for", lambda model, ids: object())
    monkeypatch.setattr(glm, "_dflash2_rounds", rounds)
    monkeypatch.setenv("VMLX_GLM_DFLASH2_PREFIX_CACHE", "0")
    prefill_step.set_prefill_step_size(8)
    draft = SimpleNamespace(config=SimpleNamespace(target_layer_ids=[1]))
    list(glm.stream_glm_dflash2(object(), FakeWrapper(), draft, "", max_tokens=4, temperature=0.0,
                                prompt_tokens=[1, 2, 3]))
    assert seen["prefill_step"] == 8


def test_qwen_dflash2_lane_forwards_the_step(monkeypatch):
    pytest.importorskip("dflash.model_mlx")
    import contextlib

    import dflash.model_mlx as runtime

    from vmlx_engine import dflash2_runtime

    seen = {}

    def resumable(*args, prefill_step_size=None, **kwargs):
        seen["prefill_step_size"] = prefill_step_size
        return iter(())

    monkeypatch.setattr(dflash2_runtime, "_stream_generate_resumable", resumable)
    monkeypatch.setattr(dflash2_runtime, "_adapter_for", lambda model: object())
    monkeypatch.setattr(runtime, "wired_limit", lambda *a, **k: contextlib.nullcontext())
    import vmlx_engine.glm_dflash2 as glm

    monkeypatch.setattr(glm, "is_glm5_next", lambda model: False)
    prefill_step.set_prefill_step_size(8)
    draft = SimpleNamespace(config=SimpleNamespace(block_size=8))
    list(dflash2_runtime.stream_dflash2_generate(object(), object(), draft, "hi", max_tokens=4, temperature=0.0))
    assert seen["prefill_step_size"] == 8


def test_vlm_lanes_forward_the_step():
    """mlx-vlm stream_generate is reached from two mllm entry points; both must pass the configured step."""
    from pathlib import Path

    source = Path("vmlx_engine/models/mllm.py").read_text(encoding="utf-8")
    assert source.count('kwargs.setdefault("prefill_step_size", prefill_step_size())') == 2
