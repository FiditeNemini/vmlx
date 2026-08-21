"""LFM2-VL image requests died on an upstream arity bug (mlx-vlm 0.5.0).

`Model.__call__` calls `self.get_input_embeddings(input_ids, pixel_values,
spatial_shapes, pixel_attention_mask)` — four positional arguments — while
the method is declared `(self, input_ids=None, pixel_values=None, **kwargs)`
and reads the extras from kwargs. Every image prompt raised
"get_input_embeddings() takes from 1 to 3 positional arguments but 5 were
given" before any Metal work. Text-only prompts never hit it.
"""

import pytest


def test_lfm2_vl_accepts_the_positional_call_its_own_caller_makes():
    pytest.importorskip("mlx_vlm")
    lfm2_vl = pytest.importorskip("mlx_vlm.models.lfm2_vl.lfm2_vl")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()

    seen = {}

    def fake_original(self, input_ids=None, pixel_values=None, **kwargs):
        seen.update(
            input_ids=input_ids,
            pixel_values=pixel_values,
            spatial_shapes=kwargs.get("spatial_shapes"),
            pixel_attention_mask=kwargs.get("pixel_attention_mask"),
        )
        return "embeddings"

    patched = lfm2_vl.Model.get_input_embeddings
    assert getattr(patched, "_vmlx_lfm2_vl_positional", False), "compat patch not applied"

    # Exactly the call upstream's own __call__ makes.
    result = patched(
        object(), "IDS", "PIXELS", "SHAPES", "MASK", _original=fake_original
    )
    assert result == "embeddings"
    assert seen == {
        "input_ids": "IDS",
        "pixel_values": "PIXELS",
        "spatial_shapes": "SHAPES",
        "pixel_attention_mask": "MASK",
    }


def test_keyword_callers_are_unaffected():
    pytest.importorskip("mlx_vlm")
    lfm2_vl = pytest.importorskip("mlx_vlm.models.lfm2_vl.lfm2_vl")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()
    seen = {}

    def fake_original(self, input_ids=None, pixel_values=None, **kwargs):
        seen.update(ids=input_ids, px=pixel_values, extra=dict(kwargs))
        return "ok"

    out = lfm2_vl.Model.get_input_embeddings(
        object(),
        input_ids="IDS",
        pixel_values=None,
        spatial_shapes="S",
        _original=fake_original,
    )
    assert out == "ok"
    assert seen["ids"] == "IDS" and seen["px"] is None
    assert seen["extra"]["spatial_shapes"] == "S"


def test_patch_is_idempotent():
    pytest.importorskip("mlx_vlm")
    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()
    from mlx_vlm.models.lfm2_vl import lfm2_vl

    first = lfm2_vl.Model.get_input_embeddings
    mlx_vlm_compat._patch_lfm2_vl_input_embeddings()
    assert lfm2_vl.Model.get_input_embeddings is first


def test_language_model_unwraps_input_embeddings_features():
    """Second upstream defect: Model.__call__ passes the whole
    InputEmbeddingsFeatures dataclass as `inputs_embeds`, but
    LanguageModel.__call__ is typed Optional[mx.array] and indexes it —
    "'InputEmbeddingsFeatures' object has no attribute 'shape'"."""
    pytest.importorskip("mlx_vlm")
    language = pytest.importorskip("mlx_vlm.models.lfm2_vl.language")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()
    assert getattr(language.LanguageModel.__call__, "_vmlx_lfm2_vl_unwrap", False)

    class _Features:
        inputs_embeds = "ARRAY"

    seen = {}

    def fake_original(self, inputs, mask=None, cache=None, inputs_embeds=None, **kw):
        seen["inputs_embeds"] = inputs_embeds
        return "logits"

    out = language.LanguageModel.__call__(
        object(), "IDS", None, None, _Features(), _original=fake_original
    )
    assert out == "logits"
    assert seen["inputs_embeds"] == "ARRAY", "features object was not unwrapped"


def test_language_model_passes_plain_arrays_through_untouched():
    pytest.importorskip("mlx_vlm")
    language = pytest.importorskip("mlx_vlm.models.lfm2_vl.language")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()
    seen = {}

    class _Arr:
        shape = (1, 2, 3)

    arr = _Arr()

    def fake_original(self, inputs, mask=None, cache=None, inputs_embeds=None, **kw):
        seen["inputs_embeds"] = inputs_embeds
        return "logits"

    language.LanguageModel.__call__(
        object(), "IDS", None, None, arr, _original=fake_original
    )
    assert seen["inputs_embeds"] is arr


def test_llama_style_mlp_names_are_aliased_back_to_lfm2_names():
    """jjang-ai/mlxstudio#132: a conversion renamed LFM2's feed_forward
    w1/w2/w3 to the llama gate_proj/up_proj/down_proj spelling, and the bundle
    failed to load with 270 unhandled parameters (reproduced on the Python
    engine by renaming the official bundle's keys, weights untouched)."""
    lfm2_vl = pytest.importorskip("mlx_vlm.models.lfm2_vl.lfm2_vl")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()

    weights = {
        "language_model.model.layers.0.feed_forward.gate_proj.weight": 1,
        "language_model.model.layers.0.feed_forward.up_proj.weight": 2,
        "language_model.model.layers.0.feed_forward.down_proj.weight": 3,
        "language_model.model.layers.0.self_attn.q_proj.weight": 4,
    }
    out = lfm2_vl.Model.sanitize(object(), dict(weights))

    assert out["language_model.model.layers.0.feed_forward.w1.weight"] == 1
    assert out["language_model.model.layers.0.feed_forward.w3.weight"] == 2
    assert out["language_model.model.layers.0.feed_forward.w2.weight"] == 3
    assert not any("gate_proj" in k or "up_proj" in k or "down_proj" in k for k in out)
    # non-MLP tensors are untouched
    assert out["language_model.model.layers.0.self_attn.q_proj.weight"] == 4


def test_correctly_named_lfm2_bundle_is_passed_through_untouched():
    """A bundle that already uses w1/w2/w3 must not be rewritten."""
    lfm2_vl = pytest.importorskip("mlx_vlm.models.lfm2_vl.lfm2_vl")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()

    weights = {
        "language_model.model.layers.0.feed_forward.w1.weight": 1,
        "language_model.model.layers.0.feed_forward.w2.weight": 2,
        "language_model.model.layers.0.feed_forward.w3.weight": 3,
    }
    assert lfm2_vl.Model.sanitize(object(), dict(weights)) == weights


def test_alias_never_clobbers_an_existing_canonical_tensor():
    """If both spellings are present, the canonical one wins and the llama
    duplicate is left alone rather than overwriting real weights."""
    lfm2_vl = pytest.importorskip("mlx_vlm.models.lfm2_vl.lfm2_vl")

    from vmlx_engine.utils import mlx_vlm_compat

    mlx_vlm_compat.apply()

    weights = {
        "language_model.model.layers.0.feed_forward.w1.weight": "canonical",
        "language_model.model.layers.0.feed_forward.gate_proj.weight": "duplicate",
    }
    out = lfm2_vl.Model.sanitize(object(), dict(weights))
    assert out["language_model.model.layers.0.feed_forward.w1.weight"] == "canonical"
