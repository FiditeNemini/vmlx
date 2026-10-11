"""``--prefill-step-size`` for the lanes that do not run through the batched scheduler.

The batched engines read it from SchedulerConfig. SimpleEngine (DFlash2 lanes, non-batched serving) called mlx-lm,
mlx-vlm and the DFlash2 prefill loops with their own 2048 default, so a user's smaller step (the documented remedy
for Metal single-buffer OOM on big MoE prefills) was silently ignored there (user report 2026-10-11: a 158-token
prompt still prefilled in one chunk with --prefill-step-size 8). Every one of those defaults is 2048, the CLI default,
so an unset flag changes nothing.
"""

DEFAULT_PREFILL_STEP_SIZE = 2048
_prefill_step_size = DEFAULT_PREFILL_STEP_SIZE


def set_prefill_step_size(value) -> None:
    global _prefill_step_size
    try:
        _prefill_step_size = max(1, int(value))
    except (TypeError, ValueError):
        _prefill_step_size = DEFAULT_PREFILL_STEP_SIZE


def prefill_step_size() -> int:
    return _prefill_step_size
