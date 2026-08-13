"""`Scheduler.request_progress` must be an honest, monotonic token count.

Found 2026-08-13 while checking whether MiniMax-M2.7-Small had blown its
`max_tokens` cap: the engine logged "still progressing (17496 tokens)" against
a resolved cap of 16384. It had not — the counter was returning twice the
generated token count, so the real figure was ~8.7k. The suspected defect was
not real; the counter reporting it was.

Two things were wrong with `num_computed_tokens + total_output_tokens`:

1. `num_computed_tokens` is incremented in exactly one place,
   `Request.append_output_token`, so it counts OUTPUT tokens. Nothing advances
   it during prefill. The sum was 2x the generated count while both the
   docstring and two operator-facing log lines called it "tokens".
2. `_reschedule_running_requests` zeroes `num_computed_tokens` on a recovery
   restart but deliberately preserves `total_output_tokens`, so the
   "monotonic" counter halved mid-request. `server.py` only credits
   `progress > last_progress` as liveness, so after a restart a healthy
   request had to regenerate everything it had already emitted before counting
   as alive — and could be killed as wedged in the meantime.
"""

from types import SimpleNamespace

from vmlx_engine.request import Request
from vmlx_engine.scheduler import Scheduler


class _FakeScheduler:
    """Bind the real method to a bare object holding only `requests`.

    Constructing a real Scheduler needs a loaded model. The method under test
    touches nothing but `self.requests`, so binding it directly keeps this a
    unit test instead of an integration one.
    """

    def __init__(self, requests):
        self.requests = requests

    request_progress = Scheduler.request_progress


def _request(prompt_len=64):
    return Request(
        request_id="r1",
        prompt="hello",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SimpleNamespace(max_tokens=128),
    )


def test_unknown_request_is_none_not_zero():
    # None and 0 mean different things to the streaming grace logic: None is
    # "cannot read", 0 is "registered and prefilling".
    assert _FakeScheduler({}).request_progress("nope") is None


def test_counts_generated_tokens_exactly_once():
    req = _request()
    for tok in range(10):
        req.append_output_token(tok)
    sched = _FakeScheduler({"r1": req})

    # The bug returned 20 here: num_computed_tokens (10) + total_output (10).
    assert sched.request_progress("r1") == 10


def test_prefill_reports_zero_not_the_prompt_length():
    # server.py's grace branch is written against this exact contract: a
    # registered request that is prefilling healthily reports 0, not None.
    req = _request(prompt_len=4096)
    assert _FakeScheduler({"r1": req}).request_progress("r1") == 0


def test_survives_a_recovery_restart_without_going_backwards():
    # The regression this exists to prevent. Reproduces what
    # _reschedule_running_requests does to a request: clear the generation
    # state, zero num_computed_tokens, keep total_output_tokens.
    req = _request()
    for tok in range(100):
        req.append_output_token(tok)
    sched = _FakeScheduler({"r1": req})
    before = sched.request_progress("r1")

    req.output_token_ids = []
    req.output_text = ""
    req.num_computed_tokens = 0  # exactly what the reschedule path does

    after = sched.request_progress("r1")
    assert after >= before, (
        "progress went backwards across a recovery restart; the timeout logic "
        "credits only progress > last_progress, so a healthy request would be "
        "killed as wedged"
    )
    assert after == 100

    # And it keeps climbing from the true total, not from zero.
    req.append_output_token(999)
    assert sched.request_progress("r1") == 101


def test_matches_the_number_an_operator_would_check():
    # The log line says "%d tokens" and an operator compares it against
    # max_tokens. Those have to be the same unit or the comparison is noise.
    req = _request()
    for tok in range(16384):
        req.append_output_token(tok)

    assert _FakeScheduler({"r1": req}).request_progress("r1") == 16384
