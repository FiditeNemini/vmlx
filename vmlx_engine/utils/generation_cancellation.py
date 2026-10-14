# SPDX-License-Identifier: Apache-2.0
"""Request-owned cooperative cancellation on the dedicated model worker."""
from contextlib import contextmanager
import threading


class GenerationCancelled(RuntimeError):
    """The caller explicitly stopped this generation at a safe boundary."""


_worker = threading.local()


@contextmanager
def generation_cancellation_scope(check):
    previous = getattr(_worker, "check", None)
    _worker.check = check
    try:
        yield
    finally:
        _worker.check = previous


def raise_if_generation_cancelled():
    check = getattr(_worker, "check", None)
    if check is not None and check():
        raise GenerationCancelled("Generation stopped at a prefill boundary")
