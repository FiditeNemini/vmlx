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


class TestMLLMRequestProgress:
    """The MLLM scheduler had the SAME go-backwards defect, and kept it.

    Found in review of `6b5f11ca5`, which fixed the text scheduler and then
    asserted in its own docstring that "both are monotonic, which is the only
    property the callers require". That was false.

    `MLLMScheduler.request_progress` summed `num_output_tokens`, which is
    assigned from `len(request.output_tokens)` — and the recovery-retry path
    calls `output_tokens.clear()` and sets `num_output_tokens = 0`. The value
    therefore DROPPED mid-request, and because it stayed positive it matched
    neither the "progressing" branch (`progress > last_progress`) nor the
    zero/unknown grace branch in `server.py`, so a healthy retrying request
    could be killed as wedged.
    """

    class _FakeMLLM:
        from vmlx_engine.mllm_scheduler import MLLMScheduler as _S

        request_progress = _S.request_progress

        def __init__(self, requests):
            self.requests = requests
            import threading

            self._queue_lock = threading.RLock()

    @staticmethod
    def _req(prompt=100, generated=50, base=0):
        return SimpleNamespace(
            num_prompt_tokens=prompt,
            num_output_tokens=generated,
            total_output_tokens=base + generated,
        )

    def test_unknown_request_is_none(self):
        assert self._FakeMLLM({}).request_progress("nope") is None

    def test_counts_prompt_plus_generated(self):
        sched = self._FakeMLLM({"r": self._req(prompt=100, generated=50)})
        assert sched.request_progress("r") == 150

    def test_does_not_go_backwards_across_a_retry(self):
        # Reproduces mllm_scheduler's retry path: output_tokens cleared,
        # num_output_tokens zeroed, lifetime total carried forward.
        req = self._req(prompt=100, generated=50)
        sched = self._FakeMLLM({"r": req})
        before = sched.request_progress("r")

        req._retry_output_base = req.total_output_tokens
        req.num_output_tokens = 0  # what the retry path does
        after = sched.request_progress("r")

        assert after >= before, (
            "MLLM progress went backwards across a recovery retry; the shared "
            "timeout credits only progress > last_progress, so a healthy "
            "request would be killed as wedged"
        )
        assert after == 150

        # And it climbs past the pre-retry peak as regeneration proceeds.
        req.num_output_tokens = 5
        req.total_output_tokens = req._retry_output_base + 5
        assert sched.request_progress("r") == 155

    def test_reads_lifetime_counter_not_the_resettable_one(self):
        # Guard the specific regression: if this ever reads num_output_tokens
        # again, a retry silently reintroduces the defect.
        req = SimpleNamespace(
            num_prompt_tokens=10, num_output_tokens=0, total_output_tokens=77
        )
        assert self._FakeMLLM({"r": req}).request_progress("r") == 87


class TestMLLMPrefillProgress:
    """Prefill must register as ADVANCING progress, not zero.

    Measured live: a 196k-token span exhausted all bounded grace windows
    (900s) and was killed as wedged while the GPU was legitimately chunking.
    The generator now advances `_prefill_tokens_done` per chunk; the probe
    takes max(num_prompt_tokens, _prefill_tokens_done) — NOT their sum, which
    would double-count the prompt once decode starts (the exact 2x defect
    fixed for the text scheduler earlier the same day).
    """

    class _FakeMLLM:
        from vmlx_engine.mllm_scheduler import MLLMScheduler as _S

        request_progress = _S.request_progress

        def __init__(self, requests):
            self.requests = requests
            import threading

            self._queue_lock = threading.RLock()

    def test_prefill_chunks_advance_progress(self):
        req = SimpleNamespace(
            num_prompt_tokens=0, total_output_tokens=0, _prefill_tokens_done=0
        )
        sched = self._FakeMLLM({"r": req})
        readings = []
        for chunk_end in (2048, 4096, 8192):
            req._prefill_tokens_done = chunk_end
            readings.append(sched.request_progress("r"))
        assert readings == [2048, 4096, 8192], (
            "each prefill chunk must be visible as increased progress or the "
            "timeout kills a healthy long prefill"
        )

    def test_no_double_count_when_decode_starts(self):
        # After prefill: _prefill_tokens_done == prompt len; first output sets
        # num_prompt_tokens to the same value. Sum would jump to 2x.
        req = SimpleNamespace(
            num_prompt_tokens=8192, total_output_tokens=3, _prefill_tokens_done=8192
        )
        assert self._FakeMLLM({"r": req}).request_progress("r") == 8195

    def test_monotonic_across_the_prefill_to_decode_boundary(self):
        req = SimpleNamespace(
            num_prompt_tokens=0, total_output_tokens=0, _prefill_tokens_done=8192
        )
        sched = self._FakeMLLM({"r": req})
        before = sched.request_progress("r")
        req.num_prompt_tokens = 8192  # first output token lands
        req.total_output_tokens = 1
        assert sched.request_progress("r") == before + 1
