"""One-time per-bundle proposal-head eligibility stamp.

JANG Qwen3.8 bundles (Flash-Next and 27B) ship AWQ-folded, imatrix/GPTQ
calibrated weights, so the q4 proposal-head decision is a pure function of
the bundle's lm_head layout — it does not need re-deriving on every launch.
The first launch checks the layout once and stamps the verdict into the
bundle as ``vmlx_mtp_proposal_head.json`` (next to the existing
``vmlx_mtp_tuning.json`` sidecar); later launches honor the stamp as-is.

Contract:
- An existing stamp whose recorded source layout still matches the loaded
  head is authoritative — it is never rewritten.
- A stamp whose recorded source no longer matches (bundle was requantized)
  is treated as absent and re-derived.
- Writes are FAIL-OPEN: a read-only volume or write error only logs; the
  in-process verdict still applies. No launch is ever blocked by this file.

Eligibility (matches the measured 2026-09-03 Flash-Next A/B):
- untied, affine, q8/g64 lm_head  -> eligible, proposal_bits=4
- head already <= 6 bits          -> ineligible: native head is already cheap
  (all current 27B D-tiers: q4/g128 or q6/g128)
- tied embeddings / non-affine    -> ineligible (nothing to requantize)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STAMP_FILENAME = "vmlx_mtp_proposal_head.json"
STAMP_VERSION = 1
STAMP_BASIS = (
    "settled 2026-09-03 A/B on Qwen3.8-Flash-Next-JANG_4S fixed-D3: "
    "q4 proposal head +2.3-3.0% count / +1.8-2.8% code over three "
    "same-thermal pairs; target verify keeps the checkpoint head"
)


def _stamp_path(bundle_path: str | Path) -> Path:
    return Path(bundle_path) / STAMP_FILENAME


def _normalized_source(source_layout: dict[str, Any]) -> dict[str, Any]:
    return {
        "bits": source_layout.get("bits"),
        "group_size": source_layout.get("group_size"),
        "mode": str(source_layout.get("mode") or "affine"),
        "tied": bool(source_layout.get("tied", False)),
    }


def read_proposal_stamp(bundle_path: str | Path | None) -> Optional[dict[str, Any]]:
    """Return the bundle's stamp if present and structurally valid."""

    if not bundle_path:
        return None
    try:
        path = _stamp_path(bundle_path)
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - a broken stamp must not block load
        logger.warning("Ignoring unreadable %s: %s", STAMP_FILENAME, exc)
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != STAMP_VERSION:
        return None
    if not isinstance(data.get("source"), dict):
        return None
    if not isinstance(data.get("eligible"), bool):
        return None
    return data


def write_proposal_stamp(
    bundle_path: str | Path, record: dict[str, Any]
) -> bool:
    """Atomically write the stamp. Fail-open: errors log and return False."""

    try:
        target = _stamp_path(bundle_path)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".vmlx_mtp_proposal_head.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        logger.info(
            "Stamped %s for %s: eligible=%s%s",
            STAMP_FILENAME,
            Path(bundle_path).name,
            record.get("eligible"),
            (
                f" proposal_bits={record.get('proposal_bits')}"
                if record.get("eligible")
                else f" reason={record.get('reason')}"
            ),
        )
        return True
    except Exception as exc:  # noqa: BLE001 - advisory only, never block launch
        logger.warning(
            "Could not stamp %s in %s (continuing without): %s",
            STAMP_FILENAME,
            bundle_path,
            exc,
        )
        return False


def _derive_plan(source: dict[str, Any], family: str) -> dict[str, Any]:
    bits = source.get("bits")
    group_size = source.get("group_size")
    mode = source.get("mode")
    if source.get("tied"):
        return {"eligible": False, "reason": "tied_embeddings"}
    if mode != "affine" or not isinstance(bits, int):
        return {"eligible": False, "reason": "non_affine_or_unquantized_head"}
    if bits <= 6:
        # 27B D-tiers land here (q4/g128, q6/g128): drafting already pays a
        # cheap head; a lower-bit copy has nothing left to save.
        return {"eligible": False, "reason": "native_head_already_low_bit"}
    if bits == 8 and group_size == 64:
        return {"eligible": True, "proposal_bits": 4}
    return {"eligible": False, "reason": f"unmeasured_layout_q{bits}_g{group_size}"}


def resolve_proposal_head_plan(
    bundle_path: str | Path | None,
    source_layout: dict[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    """One-time check-and-stamp; an existing matching stamp is authoritative.

    Returns the plan dict: ``{"eligible": bool, "proposal_bits": int?,
    "reason": str?, "stamped": bool, "stamp_source": "existing"|"new"|"none"}``.
    """

    source = _normalized_source(source_layout)

    stamp = read_proposal_stamp(bundle_path)
    if stamp is not None:
        if _normalized_source(stamp.get("source") or {}) == source:
            plan = {
                "eligible": bool(stamp.get("eligible")),
                "stamped": True,
                "stamp_source": "existing",
            }
            if plan["eligible"]:
                plan["proposal_bits"] = int(stamp.get("proposal_bits") or 4)
            else:
                plan["reason"] = str(stamp.get("reason") or "stamped_ineligible")
            return plan
        logger.info(
            "%s source layout changed (%s -> %s); re-deriving",
            STAMP_FILENAME,
            stamp.get("source"),
            source,
        )

    plan = _derive_plan(source, family)
    record: dict[str, Any] = {
        "version": STAMP_VERSION,
        "family": family,
        "source": source,
        "eligible": plan["eligible"],
        "basis": STAMP_BASIS,
    }
    if plan["eligible"]:
        record["proposal_bits"] = plan["proposal_bits"]
    else:
        record["reason"] = plan["reason"]

    stamped = bool(bundle_path) and write_proposal_stamp(bundle_path, record)
    plan["stamped"] = stamped
    plan["stamp_source"] = "new" if stamped else "none"
    return plan


__all__ = [
    "STAMP_FILENAME",
    "read_proposal_stamp",
    "resolve_proposal_head_plan",
    "write_proposal_stamp",
]
