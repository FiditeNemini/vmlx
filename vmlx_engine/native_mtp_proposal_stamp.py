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
    if not isinstance(bits, int):
        return {"eligible": False, "reason": "non_affine_or_unquantized_head"}
    if mode != "affine":
        # Known-quantized but unmeasured mode (e.g. the 27B-MXFP8 head):
        # reason matches the converter-side stamper so a re-derive after a
        # deleted stamp reproduces the shipped verdict byte-for-byte.
        return {
            "eligible": False,
            "reason": f"unmeasured_layout_q{bits}_g{group_size}",
        }
    if bits <= 6:
        # 27B D-tiers (q2/q4/q6 g128) and Flash-Next 2L (q6/g64) land here:
        # drafting already pays a cheap head; a lower-bit copy has nothing
        # left to save.
        return {"eligible": False, "reason": "native_head_already_low_bit"}
    if bits == 8 and group_size == 64:
        return {"eligible": True, "proposal_bits": 4}
    return {"eligible": False, "reason": f"unmeasured_layout_q{bits}_g{group_size}"}


# The measured families this stamp may be WRITTEN for. Other model types can
# still use the in-process verdict, but a persistent stamp lands in a user's
# bundle only when the bundle's own config confirms it is one of these.
_STAMPABLE_MODEL_TYPES: dict[str, frozenset[str]] = {
    "qwen4_exp": frozenset({"qwen4_exp", "qwen4_exp_text"}),
    "qwen3_5": frozenset(
        {"qwen3_5", "qwen3_5_text", "qwen3_5_vl", "qwen3_5_moe", "qwen3_5_moe_text"}
    ),
}


def _bundle_confirms_type(
    bundle_path: str | Path, source: dict[str, Any], family: str
) -> tuple[bool, str]:
    """Confirm the bundle TYPE (never the layout) before writing a stamp.

    The stamp is a cache of a pure function of the LOADED lm_head — the
    loaded module is the only layout authority, because config defaults lie
    (the 2L misstamp: a converter agent read the bundle-wide 6-bit tier
    default while the per-module lm_head override was q8/g64). So this
    check deliberately does NOT compare the config's declared quantization
    against the loaded head; it only confirms this is a measured JANG
    Qwen3.8 bundle of the resolving family, so the sidecar never lands in
    an unrelated model's directory. A config-vs-loaded head disagreement
    is logged as evidence of the default-vs-override case, not a veto —
    vetoing on it would leave a bad shipped stamp permanently uncorrected
    and break the self-healing property.
    """

    try:
        config = json.loads((Path(bundle_path) / "config.json").read_text())
    except Exception as exc:  # noqa: BLE001 - unconfirmable, not fatal
        return False, f"config_unreadable:{type(exc).__name__}"
    if not isinstance(config, dict):
        return False, "config_not_a_mapping"

    model_type = str(config.get("model_type") or "").strip().lower()
    allowed = _STAMPABLE_MODEL_TYPES.get(family, frozenset())
    if model_type not in allowed:
        return False, f"model_type_not_stampable:{model_type or 'missing'}"

    # The eligibility rule's premise is "valid because JANG bundles are
    # AWQ+imatrix/GPTQ calibrated". A plain benchmark quant of the same
    # architecture (speed-audit packs) has the same model_type but has not
    # earned the verdict: require the JANG calibration marker — either the
    # standalone jang_config.json sidecar (27B-style) or the jang_config /
    # jang block embedded in config.json (Flash-Next-style).
    has_jang_marker = (
        (Path(bundle_path) / "jang_config.json").is_file()
        or isinstance(config.get("jang_config"), dict)
        or isinstance(config.get("jang"), dict)
    )
    if not has_jang_marker:
        return False, "not_a_jang_bundle"

    quant = config.get("quantization")
    if isinstance(quant, dict):
        head_entry: Any = None
        for key in ("language_model.lm_head", "lm_head", "model.lm_head"):
            candidate = quant.get(key)
            if isinstance(candidate, dict):
                head_entry = candidate
                break
        if head_entry is None:
            head_entry = {
                k: v for k, v in quant.items() if not isinstance(v, dict)
            }
        declared = {
            "bits": head_entry.get("bits"),
            "group_size": head_entry.get("group_size"),
            "mode": str(head_entry.get("mode") or "affine"),
        }
        loaded = {
            "bits": source.get("bits"),
            "group_size": source.get("group_size"),
            "mode": source.get("mode"),
        }
        if declared != loaded:
            logger.info(
                "config.json head declaration %s differs from loaded head %s "
                "in %s — stamping from the loaded head (config defaults lie)",
                declared,
                loaded,
                Path(bundle_path).name,
            )
    return True, "confirmed"


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

    # Confirm the bundle TYPE from its own config before persisting anything
    # into the user's bundle (layout truth stays with the loaded head). The
    # in-process verdict above applies regardless; only the WRITE is gated.
    stamped = False
    if bundle_path:
        confirmed, why = _bundle_confirms_type(bundle_path, source, family)
        if confirmed:
            stamped = write_proposal_stamp(bundle_path, record)
        else:
            if why == "not_a_jang_bundle" and plan.get("eligible"):
                # An uncalibrated q8 head has not earned an eligible
                # verdict — the measured win was on calibrated JANG heads.
                # Downgrade the in-process verdict too, not just the write.
                plan = {"eligible": False, "reason": "uncalibrated_bundle"}
            logger.info(
                "Not stamping %s in %s (unconfirmed bundle: %s); "
                "verdict=%s",
                STAMP_FILENAME,
                Path(bundle_path).name,
                why,
                "uncalibrated_bundle" if why == "not_a_jang_bundle" else "kept",
            )
    plan["stamped"] = stamped
    plan["stamp_source"] = "new" if stamped else "none"
    return plan


__all__ = [
    "STAMP_FILENAME",
    "read_proposal_stamp",
    "resolve_proposal_head_plan",
    "write_proposal_stamp",
]
