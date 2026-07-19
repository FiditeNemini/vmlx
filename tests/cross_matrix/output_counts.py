"""Shared subprocess count parsing for cross-matrix proof runners."""

from __future__ import annotations

import re


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean(output: str) -> str:
    return _ANSI_ESCAPE.sub("", output)


def _last_int(pattern: str, output: str) -> int | None:
    matches = re.findall(pattern, output, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_counts(output: str) -> dict[str, int | None]:
    """Parse pytest, Vitest, and nested-runner result counts.

    Vitest wraps both labels and numbers in ANSI SGR sequences. Parse its
    explicit ``Tests`` summary before generic ``N passed`` text so the earlier
    ``Test Files`` count can never be mistaken for the number of tests.
    """

    clean = _clean(output)
    passed = _last_int(r"\bTests\s+(\d+)\s+passed\b", clean)
    if passed is None:
        passed = _last_int(r"\bpassed=(\d+)\b", clean)
    if passed is None:
        passed = _last_int(r"\b(\d+)\s+passed\b", clean)

    # Do not read ``passed=42 skipped=None`` as ``42 skipped``.
    skipped = _last_int(r"(?<![=\w])(\d+)\s+skipped\b", clean)
    deselected = _last_int(r"\bdeselected=(\d+)\b", clean)
    if deselected is None:
        deselected = _last_int(r"\b(\d+)\s+deselected\b", clean)
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


def parse_vitest_counts(output: str) -> dict[str, int | None]:
    """Return the specialized field names used by the panel-settings runner."""

    clean = _clean(output)
    counts = parse_counts(clean)
    return {
        "test_files_passed": _last_int(
            r"\bTest Files\s+(\d+)\s+passed\b", clean
        ),
        "tests_passed": counts["passed"],
        "tests_skipped": counts["skipped"],
    }
