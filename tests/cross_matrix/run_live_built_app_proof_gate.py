#!/usr/bin/env python3
"""Fail-closed guard for live built-app runtime proof.

This is intentionally strict. A source test, API-only run, notarized artifact,
or old checklist row is not enough to claim a model/runtime lane is done. The
proof JSON must come from a dev or installed build driven like a real user and
must explicitly assert every required surface in ``live_built_app_gate``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_TRUE_PATHS = (
    "proof.current_run_not_stale",
    "proof.no_unverified_assumptions",
    "source.traced_call_path",
    "source.verified_config_values",
    "source.verified_model_artifact_files",
    "source.no_unverified_function_or_variable_claims",
    "app.launched_built_app",
    "app.loaded_model_from_ui",
    "app.visible_chat_completed",
    "runtime.model_load_log_captured",
    "runtime.live_values_captured",
    "runtime.ui_transcript_captured",
    "implementation.no_fake_guards_or_behavior_enforcement",
    "model.coherent_single_turn",
    "model.coherent_multiturn",
    "model.no_incoherent_looping",
    "model.no_parser_or_template_leaks",
    "tools.live_ui_tool_call",
    "tools.tool_result_continuation",
    "reasoning.off_works",
    "reasoning.auto_works",
    "reasoning.on_works",
    "reasoning.streaming_separated",
    "generation_config.used_model_bundle_defaults",
    "modality.correct_family_route",
    "modality.no_unsupported_media_claim",
    "cache.expected_architecture_selected",
    "cache.prefix_hit_observed",
    "cache.ssd_l2_hit_or_persist_observed",
    "cache.no_incompatible_turboquant_kv",
    "responses.previous_response_id_reuse",
    "responses.content_delta_streaming",
    "responses.reasoning_delta_streaming",
    "responses.tool_argument_delta_streaming",
)


def _get(obj: dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", required=True, help="Path to live proof JSON")
    parser.add_argument(
        "--model-family",
        help="Optional expected model family, e.g. gemma4, minimax_m3, deepseek_v4",
    )
    parser.add_argument(
        "--min-tok-s",
        type=float,
        default=None,
        help="Optional minimum live generated token/s for this proof",
    )
    args = parser.parse_args(argv)

    proof_path = Path(args.proof)
    failures: list[str] = []
    try:
        proof = json.loads(proof_path.read_text())
    except Exception as exc:
        print(f"status=fail proof={proof_path} error=invalid_json:{exc}")
        return 2

    if proof.get("status") != "pass":
        failures.append(f"top-level status is {proof.get('status')!r}, expected 'pass'")

    gate = proof.get("live_built_app_gate")
    if not isinstance(gate, dict):
        failures.append("missing live_built_app_gate object")
        gate = {}

    for dotted in REQUIRED_TRUE_PATHS:
        if _get(gate, dotted) is not True:
            failures.append(f"live_built_app_gate.{dotted} is not true")

    if args.model_family:
        observed = _get(gate, "model.family")
        if observed != args.model_family:
            failures.append(
                f"live_built_app_gate.model.family is {observed!r}, expected {args.model_family!r}"
            )

    if args.min_tok_s is not None:
        observed_tps = _get(gate, "speed.generated_tok_s")
        if not isinstance(observed_tps, (int, float)):
            failures.append("live_built_app_gate.speed.generated_tok_s is missing/non-numeric")
        elif float(observed_tps) < args.min_tok_s:
            failures.append(
                f"live_built_app_gate.speed.generated_tok_s={observed_tps} < {args.min_tok_s}"
            )

    if failures:
        print(f"status=fail proof={proof_path}")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"status=pass proof={proof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
