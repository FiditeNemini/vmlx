# SPDX-License-Identifier: Apache-2.0
"""BOTH scheduler paths must bound the legacy prefix cache by BYTES.

An entry count does not bound memory: entries here are whole-KV snapshots whose
size grows with context, so 100 entries at 90k context is hundreds of GB. That
is the measured 93.48GB failure mode.

The MLLM path was fixed first, and the fix carried a comment asserting that "the
LLM path in scheduler.py has always passed max_bytes". That was FALSE and it hid
the same hole for longer: scheduler.py passes the PARAMETER, but
SchedulerConfig defaults prefix_cache_max_bytes to None, so the value was
unbounded. A confidently-worded wrong comment is worse than no comment.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_llm_path_derives_a_byte_budget():
    src = (ROOT / "vmlx_engine" / "scheduler.py").read_text(encoding="utf-8")
    idx = src.index("PrefixCacheManager(")
    window = src[idx : idx + 900]
    assert "_resolve_prefix_cache_byte_budget" in window, (
        "the LLM path passes prefix_cache_max_bytes straight through again; "
        "SchedulerConfig defaults it to None, so the cache is bounded by entry "
        "count only"
    )


def test_mllm_path_still_derives_a_byte_budget():
    src = (ROOT / "vmlx_engine" / "mllm_scheduler.py").read_text(encoding="utf-8")
    assert "_resolve_prefix_cache_byte_budget" in src


def test_no_claim_that_the_llm_path_always_bounded_bytes():
    """Guard the specific false statement that concealed the LLM hole."""
    src = (ROOT / "vmlx_engine" / "mllm_scheduler.py").read_text(encoding="utf-8")
    assert not re.search(
        r"LLM path in\s*#?\s*scheduler\.py has always passed max_bytes", src
    ), "the false 'LLM path always passed max_bytes' claim is back"


def test_budget_resolution_is_bounded_when_nothing_is_configured_explicitly():
    from vmlx_engine.mllm_scheduler import _resolve_prefix_cache_byte_budget

    class Cfg:
        prefix_cache_max_bytes = None
        cache_memory_mb = None
        cache_memory_percent = 0.15

    budget = _resolve_prefix_cache_byte_budget(Cfg())
    assert budget is not None and budget > 0, (
        "with a cache_memory_percent configured the budget must be finite"
    )
