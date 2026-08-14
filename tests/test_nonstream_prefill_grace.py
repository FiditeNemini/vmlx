"""The non-stream chat wait loop must grant zero-progress grace during prefill.

Measured live 2026-08-13: a 150k-token prompt on a healthy Qwen3.6-2D engine
was killed with a 504 at exactly 300.0s while the GPU was mid-prefill. Both
schedulers report 0 generated tokens for the whole prefill (the MLLM lane sets
num_prompt_tokens only when the FIRST output token arrives), and the
non-stream `_await_chat_with_disconnect_abort` extended only on
`progress > 0` — so prefill, the one phase long enough to hit the timeout,
was precisely the phase with no protection. The STREAMING keepalive already
implements bounded zero/None grace; this pins the non-stream twin to the same
contract. One contract, two enforcement points, again.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1].joinpath("vmlx_engine/server.py").read_text()


def _function_body(name: str) -> str:
    i = SRC.index(f"async def {name}")
    j = SRC.index("\ndef ", i)
    return SRC[i:j]


def test_nonstream_wait_has_zero_progress_grace():
    body = _function_body("_await_chat_with_disconnect_abort")
    assert "unknown_progress_windows" in body, (
        "non-stream chat wait must grant bounded grace when progress is 0/None "
        "(prefill), like the streaming keepalive does"
    )
    assert "_UNKNOWN_PROGRESS_GRACE_WINDOWS" in body


def test_grace_is_bounded_not_infinite():
    body = _function_body("_await_chat_with_disconnect_abort")
    # The grace must compare against the shared bound, not loop forever.
    assert re.search(
        r"unknown_progress_windows\s*<\s*_UNKNOWN_PROGRESS_GRACE_WINDOWS", body
    ), "a wedged request must still die after the bounded number of windows"


def test_real_progress_resets_the_grace_budget():
    body = _function_body("_await_chat_with_disconnect_abort")
    assert "unknown_progress_windows = 0" in body, (
        "positive evidence must clear the ambiguity budget so a later "
        "unreadable stretch gets full grace again — same as streaming"
    )


def test_hard_timeout_stays_wall_clock():
    body = _function_body("_await_chat_with_disconnect_abort")
    # The grace lives inside `if not hard_timeout:` — an explicit user
    # timeout must never be extended.
    i = body.index("if not hard_timeout:")
    j = body.index("_stop_drain()", i)
    graced = body[i:j]
    assert "unknown_progress_windows" in graced
