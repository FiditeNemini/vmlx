"""Ollama clients compute throughput from duration fields we did not send.

`eval_count / eval_duration * 1e9` is how Open WebUI, Continue.dev and
`ollama run --verbose` display tok/s. vMLX sent `total_duration: 0` on the
non-streaming path and NO duration fields at all on the streaming one, so
every such client showed nothing — while a comment in the chat handler
claimed the streaming path carried tok/s. Measured live before the fix:

    top-level keys: created_at, done, done_reason, eval_count, message,
                    model, prompt_eval_count, total_duration
    total_duration = 0

The split is measured, never estimated: prompt_eval_duration is the real time
to the first content token. Where no first-token timestamp exists (the
non-streaming path cannot see one) the split is OMITTED — reporting total time
as decode time would understate tok/s with a number that looks precise.
"""

import pytest

from vmlx_engine.api.ollama_adapter import (
    ollama_duration_fields,
    openai_chat_response_to_ollama,
    openai_chat_response_to_ollama_generate,
)

NS = 1_000_000_000


def test_streaming_split_comes_from_the_first_token_timestamp():
    fields = ollama_duration_fields(0, 2 * NS, 10 * NS)
    assert fields["total_duration"] == 10 * NS
    assert fields["prompt_eval_duration"] == 2 * NS
    assert fields["eval_duration"] == 8 * NS
    assert fields["load_duration"] == 0


def test_the_split_is_omitted_rather_than_guessed():
    """Without a first-token timestamp we know the total and nothing else."""
    fields = ollama_duration_fields(0, None, 10 * NS)
    assert fields["total_duration"] == 10 * NS
    assert "prompt_eval_duration" not in fields
    assert "eval_duration" not in fields, (
        "eval_duration must not be filled in from the total — a client would "
        "divide by it and report a confidently wrong tok/s"
    )


def test_a_client_can_compute_tokens_per_second():
    fields = ollama_duration_fields(0, 1 * NS, 5 * NS)
    tps = 40 / (fields["eval_duration"] / NS)  # 40 tokens in 4s
    assert tps == pytest.approx(10.0)


def test_no_timestamps_means_no_fields_at_all():
    assert ollama_duration_fields(None, None, None) == {}
    assert ollama_duration_fields(0, None, None) == {}


def test_durations_are_never_negative_or_out_of_order():
    """A first-token stamp after the end stamp must not produce a negative
    eval_duration; clocks and retries can surprise us."""
    fields = ollama_duration_fields(0, 20 * NS, 10 * NS)
    assert fields["total_duration"] == 10 * NS
    assert fields["prompt_eval_duration"] == 10 * NS
    assert fields["eval_duration"] == 0
    fields_back = ollama_duration_fields(10 * NS, None, 0)
    assert fields_back["total_duration"] == 0


def _resp():
    return {
        "choices": [{"message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"completion_tokens": 4, "prompt_tokens": 11},
    }


def test_non_streaming_chat_reports_a_real_total_not_zero():
    out = openai_chat_response_to_ollama(_resp(), "m", started_ns=0)
    assert out["total_duration"] > 0, "still reporting the hardcoded zero"
    assert out["eval_count"] == 4
    assert out["prompt_eval_count"] == 11


def test_non_streaming_generate_reports_a_real_total_not_zero():
    out = openai_chat_response_to_ollama_generate(_resp(), "m", started_ns=0)
    assert out["total_duration"] > 0


def test_non_streaming_without_a_start_stamp_is_unchanged():
    """Callers that do not measure keep the previous shape rather than
    getting a fabricated duration."""
    out = openai_chat_response_to_ollama(_resp(), "m")
    assert out["total_duration"] == 0
    assert "eval_duration" not in out


# A deep-sleep JIT wake reloads the model inside the request. Measured live
# (DSV4 102 GB, frozen 1.6.44 head): wall time 51 s, but the terminal row
# said total_duration=2.766 s and load_duration=0 — the entire wake was
# dropped. Real Ollama reports model load in load_duration and includes it
# in total_duration. The wake middleware stamps request.state.vmlx_wake_ns;
# the adapter folds it in WITHOUT touching the prefill/decode split.


def test_wake_time_lands_in_load_and_total_but_never_in_the_split():
    fields = ollama_duration_fields(0, 1 * NS, 3 * NS, load_ns=48 * NS)
    assert fields["load_duration"] == 48 * NS
    assert fields["total_duration"] == 51 * NS
    # tok/s must be wake-independent: split unchanged by load_ns
    assert fields["prompt_eval_duration"] == 1 * NS
    assert fields["eval_duration"] == 2 * NS


def test_no_wake_still_reports_a_measured_zero_load():
    fields = ollama_duration_fields(0, 1 * NS, 3 * NS)
    assert fields["load_duration"] == 0
    assert fields["total_duration"] == 3 * NS


def test_a_negative_or_none_wake_stamp_clamps_to_zero():
    assert ollama_duration_fields(0, None, NS, load_ns=-5)["load_duration"] == 0
    assert ollama_duration_fields(0, None, NS, load_ns=None)["load_duration"] == 0


def test_non_streaming_chat_carries_the_wake_stamp():
    resp = {"choices": [{"message": {"content": "hi"}}], "usage": {}}
    row = openai_chat_response_to_ollama(resp, "m", started_ns=0, load_ns=7 * NS)
    assert row["load_duration"] == 7 * NS
    assert row["total_duration"] >= 7 * NS


def test_non_streaming_generate_carries_the_wake_stamp():
    resp = {"choices": [{"message": {"content": "hi"}}], "usage": {}}
    row = openai_chat_response_to_ollama_generate(
        resp, "m", started_ns=0, load_ns=7 * NS
    )
    assert row["load_duration"] == 7 * NS
    assert row["total_duration"] >= 7 * NS
