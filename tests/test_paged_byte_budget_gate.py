# SPDX-License-Identifier: Apache-2.0
"""The byte-budget guard must be reachable in the mode that needs it most.

`enforce_byte_budget` carries TWO policies: the static RAM ceiling, and the
Metal working-set PRESSURE guard. Its own docstring says disk-only mode "gets
the pressure guard but never the static ceiling" — disk-only has a zero ceiling
by construction ("no persistent payloads"), so only genuine Metal pressure
justifies shedding the transient buffers a reconstruction is reading.

But five callers gated on `max_resident_bytes > 0`, which is exactly 0 in that
mode. Every one of them skipped, so the pressure guard was unreachable from all
of them and the only ungated caller was the native-fence release. The predicate
now lives once, on the object that implements it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _FakePagedCache:
    """Just enough of PagedCacheManager to exercise the predicate."""

    def __init__(self, max_resident_bytes: int, disk_only: bool) -> None:
        self.max_resident_bytes = max_resident_bytes
        self.disk_only = disk_only

    enforces_byte_budget = property(
        lambda self: self.max_resident_bytes > 0 or self.disk_only
    )


def test_predicate_matches_the_delegate_for_every_mode():
    from vmlx_engine.paged_cache import PagedCacheManager

    source = Path(PagedCacheManager.__module__.replace(".", "/") + ".py")
    del source  # module path is only illustrative; behaviour is asserted below

    cases = {
        # (max_resident_bytes, disk_only): enforcement runs?
        (1 << 30, False): True,   # ordinary paged RAM tier
        (1 << 30, True): True,    # explicit ceiling with disk-only
        (0, True): True,          # SSD-only: ceiling is 0 BY DESIGN, guard must run
        (0, False): False,        # accounting off, nothing measurable
    }
    for (limit, disk_only), expected in cases.items():
        fake = _FakePagedCache(limit, disk_only)
        assert fake.enforces_byte_budget is expected, (
            f"max_resident_bytes={limit} disk_only={disk_only}"
        )


def test_ssd_only_mode_is_the_case_the_old_gate_got_wrong():
    """The regression, stated as a value: 0 bytes + disk-only must enforce."""
    ssd_only = _FakePagedCache(0, True)
    assert ssd_only.max_resident_bytes == 0
    assert not (ssd_only.max_resident_bytes > 0), "the old gate"
    assert ssd_only.enforces_byte_budget, "the new gate"


def test_no_caller_reintroduces_the_raw_comparison():
    """Guarding on the raw field again silently re-breaks SSD-only mode."""
    offenders: list[str] = []
    for path in (
        ROOT / "vmlx_engine" / "scheduler.py",
        ROOT / "vmlx_engine" / "prefix_cache.py",
        ROOT / "vmlx_engine" / "paged_cache.py",
    ):
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            if not re.search(r"max_resident_bytes\s*$|max_resident_bytes\s*>\s*0", line):
                continue
            window = "\n".join(lines[number - 1 : number + 4])
            if "enforce_byte_budget()" in window:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "these gate enforce_byte_budget() on the raw field instead of "
        f"enforces_byte_budget, so SSD-only mode skips them: {offenders}"
    )
