"""Scoped guards for the exact packed-row Deferred PLE AR transaction."""
from contextvars import ContextVar

_active = ContextVar("qwen4_pending_packed_ple", default=None)


class Declined(RuntimeError):
    """Pre-build refusal: the unchanged forward is safe."""


class UnsafeDeferredEvaluation(BaseException):
    """Bypass kernel fallback catches; the owning boundary makes RuntimeError."""


def guard_consumer_eval():
    if _active.get() is not None:
        raise UnsafeDeferredEvaluation("Deferred PLE first-use consumer was not ready")


def pending():
    return _active.get() is not None


def defer_embedding(layer, input_ids, cache):
    """Return the pending packed embedding, or None for the unchanged path."""
    scope = _active.get()
    if scope is None:
        return None
    if tuple(input_ids.shape) != (1, 1):
        raise RuntimeError("Deferred PLE reached a non-AR embedding")
    return scope.bind_packed(layer, input_ids, cache)
