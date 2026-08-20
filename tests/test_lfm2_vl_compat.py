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
