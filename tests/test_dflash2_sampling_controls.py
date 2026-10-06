"""Small tensor checks for requested DFlash2 target distributions; no models."""
import mlx.core as mx
import dflash.model_mlx as runtime
import pytest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from vmlx_engine.dflash2_sampling import DFlash2SamplingControls


def test_bias_applies_to_every_verify_row_including_bonus():
    controls = DFlash2SamplingControls(logit_bias={"0": 5})
    logits = mx.array([[[0., 2., 1.], [0., 3., 2.], [0., 4., 3.]]])
    result = controls.process(logits, [])
    assert mx.argmax(result, axis=-1).tolist() == [[0, 0, 0]]
    assert mx.array_equal(logits, mx.array([[[0., 2., 1.], [0., 3., 2.], [0., 4., 3.]]]))


def test_penalties_use_causal_proposal_prefix_and_reset_after_rejection():
    controls = DFlash2SamplingControls(logit_bias={"2": 1}, frequency_penalty=.5,
                                      presence_penalty=.25)
    logits = mx.array([[[1., 2., 3.]] * 3])
    result = controls.process(logits, [0, 1], mx.array([[1, 2]]))
    assert mx.allclose(result, mx.array([[[.25, 1.25, 4.], [.25, .75, 4.], [.25, .75, 3.25]]]))
    # Both proposals rejected; their counts must not survive in the target.
    next_result = controls.process(logits[:, :1], [0, 1, 0])
    assert mx.allclose(next_result, mx.array([[[-.25, 1.25, 4.]]]))


def test_repetition_penalty_is_sign_aware_and_causal():
    controls = DFlash2SamplingControls(repetition_penalty=2)
    result = controls.process(mx.array([[[2., -4., 1.]] * 2]), [1, 1, 0], mx.array([[2]]))
    assert mx.allclose(result, mx.array([[[1., -8., 1.], [1., -8., .5]]]))


def test_min_p_is_normalized_and_zero_preserves_existing_distribution():
    logits = mx.log(mx.array([[[.7, .2, .1]]]))
    controls = DFlash2SamplingControls(min_p=.2)
    result = controls.probabilities(runtime, logits, 1., 1., 0)
    assert mx.allclose(result, mx.array([[[7/9, 2/9, 0.]]]))
    assert mx.array_equal(DFlash2SamplingControls().probabilities(runtime, logits, .8, .9, 2),
                          runtime._sampling_probs(logits, .8, .9, 2))


@pytest.mark.parametrize("penalty", [0., -1., float("inf"), float("nan")])
def test_invalid_repetition_penalty_is_not_silently_replaced(penalty):
    with pytest.raises(ValueError, match="finite and positive"):
        DFlash2SamplingControls(repetition_penalty=penalty)


def test_default_runtime_uses_existing_sampling_path():
    from vmlx_engine.dflash2_runtime import stream_dflash2_generate
    draft = SimpleNamespace(config=SimpleNamespace(block_size=8))
    with patch("vmlx_engine.dflash2_runtime._adapter_for", return_value=object()), \
         patch.object(runtime, "wired_limit", return_value=nullcontext()), \
         patch("vmlx_engine.dflash2_runtime._stream_generate_resumable", return_value=iter([])) as loop:
        assert list(stream_dflash2_generate(object(), object(), draft, "prompt",
                    max_tokens=10, temperature=0.0)) == []
    assert loop.call_args.kwargs["sampling_controls"] is None
