# SPDX-License-Identifier: Apache-2.0
"""The VL outer models must drop MTP weights when no MTP head was built.

`--disable-native-mtp` took the whole server down on any bundle that ships MTP
weights. The outer sanitize re-homed `mtp.*` to `language_model.mtp.*`
unconditionally, so with the head switched off those 31 tensors reached
`load_weights` -- which is strict -- and startup aborted with
"Received 31 parameters not in model". Both outer models had the same bug; the
inner language-model sanitize already dropped them correctly.
"""

import types

import pytest

from vmlx_engine.patches import mlx_lm_mtp
from vmlx_engine.patches.mlx_vlm_mtp import qwen35_vl


@pytest.fixture(autouse=True)
def _mtp_inactive():
    """Default to MTP off; the drop only fires when it is positively off."""
    previous = mlx_lm_mtp.is_mtp_active()
    mlx_lm_mtp.set_mtp_active(False)
    yield
    mlx_lm_mtp.set_mtp_active(previous)


class _Weight:
    """Stands in for an mx array: sanitize only inspects ndim/shape here."""

    ndim = 2
    shape = (2, 2)


def _patched_model(patcher):
    class _Model:
        pass

    patcher(types.SimpleNamespace(Model=_Model))
    model = _Model()
    model.config = types.SimpleNamespace(
        text_config=types.SimpleNamespace(tie_word_embeddings=False)
    )
    model._vmlx_norms_are_mlx_ready = True
    model.language_model = types.SimpleNamespace()
    return _Model, model


def _weights():
    return {
        "mtp.fc.weight": _Weight(),
        "language_model.mtp.norm.weight": _Weight(),
        "model.language_model.layers.0.self_attn.q_proj.weight": _Weight(),
    }


@pytest.mark.parametrize(
    "patcher",
    [qwen35_vl._patch_outer_model, qwen35_vl._patch_moe_outer_model],
    ids=["dense", "moe"],
)
def test_outer_sanitize_drops_mtp_weights_when_head_absent(patcher):
    cls, model = _patched_model(patcher)

    kept = cls.sanitize(model, _weights())

    assert not [key for key in kept if "mtp" in key], (
        "MTP tensors survived with no head to load them into — "
        "strict load_weights aborts startup on this"
    )
    # the ordinary weights must still come through
    assert any("q_proj" in key for key in kept)


@pytest.mark.parametrize(
    "patcher",
    [qwen35_vl._patch_outer_model, qwen35_vl._patch_moe_outer_model],
    ids=["dense", "moe"],
)
def test_outer_sanitize_keeps_mtp_weights_when_head_present(patcher):
    cls, model = _patched_model(patcher)
    model.language_model.mtp = object()

    kept = cls.sanitize(model, _weights())

    assert [key for key in kept if "mtp" in key], (
        "dropping MTP tensors when the head exists would silently disable "
        "speculative decode"
    )


@pytest.mark.parametrize(
    "patcher",
    [qwen35_vl._patch_outer_model, qwen35_vl._patch_moe_outer_model],
    ids=["dense", "moe"],
)
def test_outer_sanitize_keeps_mtp_weights_while_runtime_is_active(patcher):
    """A head that simply has not been constructed yet must not lose weights.

    Keying the drop on the attribute alone would discard a live head's tensors
    whenever sanitize happens to run against a not-yet-populated model.
    """
    cls, model = _patched_model(patcher)
    mlx_lm_mtp.set_mtp_active(True)

    kept = cls.sanitize(model, _weights())

    assert [key for key in kept if "mtp" in key]
