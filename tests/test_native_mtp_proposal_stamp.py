"""One-time proposal-head stamp: derive, persist, honor, fail-open."""

import json

from vmlx_engine.native_mtp_proposal_stamp import (
    STAMP_FILENAME,
    read_proposal_stamp,
    resolve_proposal_head_plan,
)

Q8_G64 = {"bits": 8, "group_size": 64, "mode": "affine", "tied": False}
Q4_G128 = {"bits": 4, "group_size": 128, "mode": "affine", "tied": False}
Q6_G128 = {"bits": 6, "group_size": 128, "mode": "affine", "tied": False}


def test_first_launch_stamps_eligible_q8_g64(tmp_path):
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["proposal_bits"] == 4
    assert plan["stamped"] is True
    assert plan["stamp_source"] == "new"
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["eligible"] is True
    assert stamp["proposal_bits"] == 4
    assert stamp["source"]["bits"] == 8


def test_27b_low_bit_heads_stamp_ineligible(tmp_path):
    for layout in (Q4_G128, Q6_G128):
        for f in tmp_path.glob(STAMP_FILENAME):
            f.unlink()
        plan = resolve_proposal_head_plan(tmp_path, layout, family="qwen3_5")
        assert plan["eligible"] is False
        assert plan["reason"] == "native_head_already_low_bit"
        stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
        assert stamp["eligible"] is False


def test_existing_matching_stamp_is_honored_not_rewritten(tmp_path):
    resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    stamp_file = tmp_path / STAMP_FILENAME
    first_mtime = stamp_file.stat().st_mtime_ns
    # Hand-edit the verdict; a matching source layout must be honored as-is.
    data = json.loads(stamp_file.read_text())
    data["eligible"] = False
    data["reason"] = "user_opt_out"
    del data["proposal_bits"]
    stamp_file.write_text(json.dumps(data))

    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is False
    assert plan["reason"] == "user_opt_out"
    assert plan["stamp_source"] == "existing"
    # File untouched by the second resolve.
    assert json.loads(stamp_file.read_text())["reason"] == "user_opt_out"
    assert stamp_file.stat().st_mtime_ns >= first_mtime


def test_source_layout_change_rederives(tmp_path):
    resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    plan = resolve_proposal_head_plan(tmp_path, Q4_G128, family="qwen4_exp")
    assert plan["eligible"] is False
    assert plan["stamp_source"] == "new"
    stamp = json.loads((tmp_path / STAMP_FILENAME).read_text())
    assert stamp["source"]["bits"] == 4


def test_tied_embeddings_ineligible(tmp_path):
    tied = dict(Q8_G64, tied=True)
    plan = resolve_proposal_head_plan(tmp_path, tied, family="qwen3_5")
    assert plan["eligible"] is False
    assert plan["reason"] == "tied_embeddings"


def test_write_failure_is_fail_open(tmp_path):
    bundle = tmp_path / "ro"
    bundle.mkdir()
    bundle.chmod(0o500)
    try:
        plan = resolve_proposal_head_plan(bundle, Q8_G64, family="qwen4_exp")
        # Verdict still applies in-process; only persistence failed.
        assert plan["eligible"] is True
        assert plan["stamped"] is False
        assert plan["stamp_source"] == "none"
    finally:
        bundle.chmod(0o700)


def test_no_bundle_path_still_returns_plan():
    plan = resolve_proposal_head_plan(None, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamped"] is False


def test_corrupt_stamp_ignored_and_rewritten(tmp_path):
    (tmp_path / STAMP_FILENAME).write_text("{not json")
    assert read_proposal_stamp(tmp_path) is None
    plan = resolve_proposal_head_plan(tmp_path, Q8_G64, family="qwen4_exp")
    assert plan["eligible"] is True
    assert plan["stamp_source"] == "new"
