"""One-time proposal-head stamp: derive, confirm bundle, persist, honor.

The in-process verdict always applies; the WRITE into a user's bundle is
gated on confirmation — the bundle's own config.json must name a measured
JANG Qwen3.8 model_type and declare the same lm_head layout the runtime
loaded. Unconfirmable bundles keep the verdict but are never stamped.
"""

import json

from vmlx_engine.native_mtp_proposal_stamp import (
    STAMP_FILENAME,
    read_proposal_stamp,
    resolve_proposal_head_plan,
)

Q8_G64 = {"bits": 8, "group_size": 64, "mode": "affine", "tied": False}
Q6_G64 = {"bits": 6, "group_size": 64, "mode": "affine", "tied": False}
Q4_G128 = {"bits": 4, "group_size": 128, "mode": "affine", "tied": False}
Q6_G128 = {"bits": 6, "group_size": 128, "mode": "affine", "tied": False}
MXFP8_G32 = {"bits": 8, "group_size": 32, "mode": "mxfp8", "tied": False}


def _write_config(
    bundle, layout, model_type="qwen4_exp", head_key="lm_head", tied=False
):
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "tie_word_embeddings": tied,
                "jang_config": {"calibrated": True},
                "quantization": {
                    head_key: {
                        "bits": layout["bits"],
                        "group_size": layout["group_size"],
                        "mode": layout["mode"],
                    }
                },
            }
        )
    )


def test_first_launch_confirms_and_stamps_eligible_q8_g64(tmp_path):
    _write_config(tmp_path, Q8_G64)
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["proposal_bits"] == 4
    assert plan["stamped"] is True
    assert plan["stamp_source"] == "new"
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["eligible"] is True
    assert stamp["proposal_bits"] == 4
    assert stamp["source"]["bits"] == 8


def test_low_bit_heads_stamp_ineligible(tmp_path):
    # 27B D-tiers (q4/q6 g128) and Flash-Next 2L (q6/g64).
    for layout, model_type in (
        (Q4_G128, "qwen3_5"),
        (Q6_G128, "qwen3_5"),
        (Q6_G64, "qwen4_exp"),
    ):
        for f in tmp_path.glob("*.json"):
            f.unlink()
        family = "qwen3_5" if model_type == "qwen3_5" else "qwen4_exp"
        _write_config(
            tmp_path, layout, model_type=model_type,
            head_key="language_model.lm_head",
        )
        plan = resolve_proposal_head_plan(tmp_path, layout, family=family)
        assert plan["eligible"] is False
        assert plan["reason"] == "native_head_already_low_bit"
        assert plan["stamped"] is True


def test_mxfp8_head_reason_matches_converter_stamper(tmp_path):
    # 27B-MXFP8-CRACK ships an mxfp8 q8/g32 head; the converter-side fork
    # stamped it "unmeasured_layout_q8_g32" — a re-derive must reproduce it.
    _write_config(tmp_path, MXFP8_G32, model_type="qwen3_5")
    plan = resolve_proposal_head_plan(tmp_path, MXFP8_G32, family="qwen3_5")
    assert plan["eligible"] is False
    assert plan["reason"] == "unmeasured_layout_q8_g32"
    assert plan["stamped"] is True


def test_existing_matching_stamp_is_honored_not_rewritten(tmp_path):
    _write_config(tmp_path, Q8_G64)
    resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    stamp_file = tmp_path / STAMP_FILENAME
    data = json.loads(stamp_file.read_text())
    data["eligible"] = False
    data["reason"] = "user_opt_out"
    del data["proposal_bits"]
    stamp_file.write_text(json.dumps(data))

    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is False
    assert plan["reason"] == "user_opt_out"
    assert plan["stamp_source"] == "existing"
    assert json.loads(stamp_file.read_text())["reason"] == "user_opt_out"


def test_existing_stamp_honored_even_without_config(tmp_path):
    # Converter-side stamps shipped on HF must be honored even if the engine
    # could not itself confirm the bundle (confirmation gates only WRITES).
    _write_config(tmp_path, Q8_G64)
    resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    (tmp_path / "config.json").unlink()
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamp_source"] == "existing"


def test_source_layout_change_rederives(tmp_path):
    _write_config(tmp_path, Q8_G64)
    resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    _write_config(tmp_path, Q4_G128)
    plan = resolve_proposal_head_plan(tmp_path, Q4_G128, family="qwen4_exp")
    assert plan["eligible"] is False
    assert plan["stamp_source"] == "new"
    assert json.loads((tmp_path / STAMP_FILENAME).read_text())["source"]["bits"] == 4


def test_unconfirmed_bundle_type_gets_verdict_but_no_stamp(tmp_path):
    # (a) no config.json at all
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamped"] is False
    assert not (tmp_path / STAMP_FILENAME).exists()
    # (b) config names a different model type
    _write_config(tmp_path, Q8_G64, model_type="llama")
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamped"] is False
    # (c) family/model_type cross-mismatch (27B config, qwen4_exp family)
    _write_config(tmp_path, Q8_G64, model_type="qwen3_5")
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["stamped"] is False
    assert not (tmp_path / STAMP_FILENAME).exists()


def test_config_layout_disagreement_stamps_from_loaded_head(tmp_path):
    # Config defaults lie; the loaded weights don't. A declared layout that
    # disagrees with the loaded head must NOT veto the write — the stamp is
    # a cache of a pure function of the loaded head.
    _write_config(tmp_path, Q4_G128, model_type="qwen4_exp")
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamped"] is True
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["source"] == Q8_G64  # loaded truth, not the config claim


def test_2l_misstamp_self_heals_on_first_load(tmp_path):
    # The shipped 2L misstamp: a converter agent stamped from the bundle's
    # 6-bit tier DEFAULT while the per-module lm_head override is q8/g64.
    # A runtime following the contract must treat the mismatched stamp as
    # absent, re-derive from the loaded head, and overwrite with truth.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "tie_word_embeddings": False,
                "jang_config": {"calibrated": True},
                # Top-level tier default says 6-bit; no per-module entry.
                "quantization": {"bits": 6, "group_size": 64, "mode": "affine"},
            }
        )
    )
    bad = {
        "version": 1,
        "family": "qwen4_exp",
        "source": {"bits": 6, "group_size": 64, "mode": "affine", "tied": False},
        "eligible": False,
        "reason": "native_head_already_low_bit",
        "basis": "misstamped from config default",
    }
    (tmp_path / STAMP_FILENAME).write_text(json.dumps(bad))

    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["proposal_bits"] == 4
    assert plan["stamp_source"] == "new"
    healed = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert healed["eligible"] is True
    assert healed["source"]["bits"] == 8


def test_tied_embeddings_ineligible(tmp_path):
    tied = dict(Q8_G64, tied=True)
    _write_config(tmp_path, Q8_G64, model_type="qwen3_5", tied=True)
    plan = resolve_proposal_head_plan(tmp_path, tied, family="qwen3_5")
    assert plan["eligible"] is False
    assert plan["reason"] == "tied_embeddings"
    assert plan["stamped"] is True


def test_write_failure_is_fail_open(tmp_path):
    bundle = tmp_path / "ro"
    bundle.mkdir()
    _write_config(bundle, Q8_G64)
    bundle.chmod(0o500)
    try:
        plan = resolve_proposal_head_plan(bundle, Q8_G64, family="qwen4_exp")
        assert plan["eligible"] is True
        assert plan["stamped"] is False
    finally:
        bundle.chmod(0o700)


def test_no_bundle_path_still_returns_plan():
    plan = resolve_proposal_head_plan(None, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamped"] is False


def test_corrupt_stamp_ignored_and_rewritten(tmp_path):
    _write_config(tmp_path, Q8_G64)
    (tmp_path / STAMP_FILENAME).write_text("{not json")
    assert read_proposal_stamp(tmp_path) is None
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamp_source"] == "new"


def test_bundle_wide_default_quant_layout_confirms(tmp_path):
    # Bundles without an explicit lm_head entry declare a top-level default.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "tie_word_embeddings": False,
                "jang_config": {"calibrated": True},
                "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
            }
        )
    )
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamped"] is True


def test_uncalibrated_speed_pack_gets_no_stamp_and_no_eligible_verdict(tmp_path):
    # Same architecture, plain benchmark quant, NO jang marker: the premise
    # ("valid because JANG bundles are calibrated") is absent, so neither the
    # stamp nor the eligible verdict is earned.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "tie_word_embeddings": False,
                "quantization": {
                    "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"}
                },
            }
        )
    )
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is False
    assert plan["reason"] == "uncalibrated_bundle"
    assert plan["stamped"] is False
    assert not (tmp_path / STAMP_FILENAME).exists()


def test_standalone_jang_config_sidecar_counts_as_calibrated(tmp_path):
    # 27B-style marker: jang_config.json file, nothing embedded in config.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "tie_word_embeddings": False,
                "quantization": {
                    "language_model.lm_head": {
                        "bits": 4, "group_size": 128, "mode": "affine"
                    }
                },
            }
        )
    )
    (tmp_path / "jang_config.json").write_text(json.dumps({"calibrated": True}))
    plan = resolve_proposal_head_plan(tmp_path, Q4_G128, family="qwen3_5")
    assert plan["reason"] == "native_head_already_low_bit"
    assert plan["stamped"] is True
