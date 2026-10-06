"""Row-indexed copies of the single-row MoE kernels change addressing only (qwen4_rows_exact_moe)."""
import re

import pytest

from vmlx_engine.metal import qwen4_rows_exact_moe as rows


def test_pair_rows_source_offsets_every_row_dependent_address():
    src = """
        uint tid = thread_position_in_grid.x;
        uint expert = (uint)expert_ids[route];
        float value = (float)x[input_base + item];
            output[(size_t)route * 640u + out_d] = (T)activated;
"""
    out = rows._rowify_pair(src, hidden=2560, top_k=10, inter=640)
    assert "uint trow = thread_position_in_grid.y;" in out
    assert "expert_ids[trow * 10u + route]" in out
    assert "x[trow * 2560u + input_base + item]" in out
    assert "output[(size_t)trow * 6400u + (size_t)route * 640u + out_d]" in out
    # nothing else in the arithmetic changed
    assert re.sub(r"trow[^;\]+]*", "", out).count("simd_sum") == src.count("simd_sum")


def test_pair_rows_refuses_an_unrecognised_layout():
    with pytest.raises(ValueError):
        rows._rowify_pair("uint tid = thread_position_in_grid.x;", hidden=2560, top_k=10, inter=640)


def test_exact_down_rows_source_builds_from_the_single_row_kernel():
    from vmlx_engine.metal.qwen4_exact_down import _SOURCE
    assert all(tok in _SOURCE for tok in ("expert_ids[route]", "activated+route*640u+start",
                                          "route_scores[route]", "output[first+lane]"))
