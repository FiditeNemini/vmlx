"""Output-token budget must survive a cache-error reschedule.

Regression coverage for the honest max_tokens-enforcement fix (task #51):
a recoverable cache error reschedules a running request, clearing its
per-attempt output_token_ids and restarting it from prefill. Before the fix
the re-insert used the FULL resolved max_tokens again, so a request that kept
hitting a recoverable error could stream far past its cap (the openPangu
composite-cache runaway). The lifetime ``total_output_tokens`` counter plus
``remaining_output_budget`` bound the request to its ORIGINAL max_tokens across
any number of restarts.
"""

from vmlx_engine.request import Request, RequestStatus
from vmlx_engine.request import SamplingParams


def _make_request(max_tokens: int) -> Request:
    return Request(
        request_id="r-budget",
        prompt="hi",
        sampling_params=SamplingParams(max_tokens=max_tokens),
        prompt_token_ids=[1, 2, 3],
        num_prompt_tokens=3,
    )


def test_fresh_request_budget_equals_max_tokens():
    req = _make_request(4096)
    assert req.total_output_tokens == 0
    assert req.remaining_output_budget == 4096


def test_budget_shrinks_by_emitted_tokens():
    req = _make_request(4096)
    for i in range(3000):
        req.append_output_token(i % 100)
    assert req.total_output_tokens == 3000
    assert req.num_output_tokens == 3000
    assert req.remaining_output_budget == 4096 - 3000


def test_lifetime_counter_survives_reschedule_reset():
    """Simulate what _reschedule_running_requests does: clear per-attempt
    output state but leave total_output_tokens intact."""
    req = _make_request(4096)
    for i in range(3000):
        req.append_output_token(i % 100)

    # Emulate the reschedule reset (scheduler clears these; NOT total_*).
    req.status = RequestStatus.WAITING
    req.output_token_ids = []
    req.num_computed_tokens = 0

    assert req.num_output_tokens == 0  # per-attempt view reset
    assert req.total_output_tokens == 3000  # lifetime view preserved
    # Re-insert must use the remaining budget, not a fresh 4096.
    assert req.remaining_output_budget == 1096

    # Generate the remaining budget; total reaches exactly the cap.
    for i in range(1096):
        req.append_output_token(i % 100)
    assert req.total_output_tokens == 4096
    assert req.remaining_output_budget == 1  # floored, never 0


def test_budget_never_negative_or_zero_even_when_over():
    req = _make_request(64)
    for i in range(200):  # over-run (defensive; shouldn't happen in practice)
        req.append_output_token(i % 10)
    assert req.total_output_tokens == 200
    # Floor keeps a valid (>=1) insert budget.
    assert req.remaining_output_budget == 1
