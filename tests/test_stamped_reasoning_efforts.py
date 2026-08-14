# SPDX-License-Identifier: Apache-2.0
"""Effort levels must come from the artifact, never a family guess.

Qwen3.8 accepts exactly low/medium/xhigh and its chat template RAISES on
anything else, so publishing vMLX's generic low/medium/high/xhigh/max would
hand it two values it rejects outright -- template exceptions, not graceful
degradation. Qwen3.6 ships no effort tiers at all, and absence has to read as
"no effort control" rather than "unconstrained".
"""

import json

import pytest

from vmlx_engine.server import _stamped_reasoning_effort_contract


def _bundle(tmp_path, reasoning):
    (tmp_path / "jang_config.json").write_text(json.dumps({"reasoning": reasoning}))
    return str(tmp_path)


def test_reads_the_qwen38_stamp(tmp_path):
    levels, default = _stamped_reasoning_effort_contract(
        _bundle(
            tmp_path,
            {
                "supported_reasoning_efforts": ["low", "medium", "xhigh"],
                "default_reasoning_effort": "xhigh",
            },
        )
    )
    assert levels == ("low", "medium", "xhigh")
    assert default == "xhigh"
    assert "high" not in levels and "max" not in levels


def test_reads_the_older_dsv4_muse_spelling(tmp_path):
    """Both spellings name one fact; a second reader is how a stamp goes stale."""
    levels, default = _stamped_reasoning_effort_contract(
        _bundle(
            tmp_path,
            {
                "reasoning_effort_levels": ["low", "medium", "high", "xhigh"],
                "default_effort": "high",
            },
        )
    )
    assert levels == ("low", "medium", "high", "xhigh")
    assert default == "high"


def test_unstamped_bundle_means_no_effort_control(tmp_path):
    """Qwen3.6 is deliberately unstamped -- absence is not 'anything goes'."""
    levels, default = _stamped_reasoning_effort_contract(
        _bundle(tmp_path, {"supported": True, "parser": "qwen3", "default": "on"})
    )
    assert levels == ()
    assert default is None


@pytest.mark.parametrize("missing", ["", "/nonexistent/bundle"])
def test_absent_bundle_is_quiet(missing):
    assert _stamped_reasoning_effort_contract(missing) == ((), None)


def test_normalizes_case_and_drops_a_default_outside_the_levels(tmp_path):
    levels, default = _stamped_reasoning_effort_contract(
        _bundle(
            tmp_path,
            {
                "supported_reasoning_efforts": ["LOW", " Medium ", "xhigh"],
                "default_reasoning_effort": "high",
            },
        )
    )
    assert levels == ("low", "medium", "xhigh")
    # "high" is not offered by this bundle, so it cannot be its default.
    assert default is None
