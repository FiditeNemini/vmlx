"""VLM sessions report the TTFT EWMA the Perf/Cache panels read (ewma_ttft_seconds).

MLLMScheduler.get_stats never had the key, so `/health.scheduler.ewma_ttft_seconds` was None and the TTFT card was
hidden on every VLM session (T-PERF wiring map 2026-10-11). Clock and smoothing match the text Scheduler.
"""
import ast
import inspect
import textwrap
import time
from types import SimpleNamespace

from vmlx_engine.mllm_scheduler import MLLMScheduler


def test_first_sample_seeds_then_smooths_like_the_text_scheduler():
    s = MLLMScheduler.__new__(MLLMScheduler)          # no __init__: defaults must hold
    s._note_first_token(SimpleNamespace(_schedule_time=time.perf_counter() - 2.0))
    assert s._ttft_sample_count == 1 and 1.9 < s._ewma_ttft < 3.0
    seeded = s._ewma_ttft
    s._record_ttft_sample(1.0)
    assert abs(s._ewma_ttft - (0.1 * 1.0 + 0.9 * seeded)) < 1e-9
    assert s._ttft_sample_count == 2


def test_unscheduled_request_records_nothing():
    s = MLLMScheduler.__new__(MLLMScheduler)
    s._note_first_token(SimpleNamespace())
    assert getattr(s, "_ttft_sample_count", 0) == 0


def test_response_loop_notes_the_first_token_before_appending_it():
    src = textwrap.dedent(inspect.getsource(MLLMScheduler._process_batch_responses))
    note = src.index("self._note_first_token(request)")
    guard = src.rindex("if not is_error and not request.output_tokens:", 0, note)
    append = src.index("request.output_tokens.append(response.token)", note)
    assert guard < note < append


def test_get_stats_publishes_the_key():
    tree = ast.parse(textwrap.dedent(inspect.getsource(MLLMScheduler.get_stats)))
    keys = {k.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
            for k in node.keys if isinstance(k, ast.Constant)}
    assert "ewma_ttft_seconds" in keys
