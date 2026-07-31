#!/bin/bash
set -euo pipefail
R20_FIXED_PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$R20_FIXED_PATH"
umask 077

# Submit immutable private snapshots of the exact r20 build-manifest DMGs to
# Apple, corroborate each Accepted submission with independent info and log
# records, then staple and verify the original public artifacts. This generic,
# token-free script never uploads a release, tags Git, publishes packages, or
# mutates update feeds.

SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd -P)"
# shellcheck source=panel/scripts/verify-release-dmgs.sh
source "$SCRIPT_DIR/verify-release-dmgs.sh"

NOTARY_PROFILE="${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-${VMLX_NOTARY_KEYCHAIN_PROFILE:-}}"
NOTARY_KEYCHAIN="${VMLINUX_NOTARY_KEYCHAIN:-${VMLX_NOTARY_KEYCHAIN:-}}"
PRE_NOTARY_MANIFEST="${VMLX_R20_PRE_NOTARY_MANIFEST:-${VMLINUX_R20_PRE_NOTARY_MANIFEST:-}}"
FINAL_NOTARY_MANIFEST="${VMLX_R20_FINAL_NOTARY_MANIFEST:-${VMLINUX_R20_FINAL_NOTARY_MANIFEST:-}}"

if [[ -n "${VMLX_NOTARY_KEYCHAIN_PROFILE:-}" ]] \
  && [[ -n "${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-}" ]] \
  && [[ "$VMLX_NOTARY_KEYCHAIN_PROFILE" != "$VMLINUX_NOTARY_KEYCHAIN_PROFILE" ]]; then
  echo "ERROR: VMLX_NOTARY_KEYCHAIN_PROFILE and VMLINUX_NOTARY_KEYCHAIN_PROFILE disagree" >&2
  exit 1
fi
if [[ -z "$NOTARY_PROFILE" ]]; then
  echo "ERROR: set VMLX_NOTARY_KEYCHAIN_PROFILE from the private release environment" >&2
  exit 1
fi
if [[ -n "${VMLX_NOTARY_KEYCHAIN:-}" ]] \
  && [[ -n "${VMLINUX_NOTARY_KEYCHAIN:-}" ]] \
  && [[ "$VMLX_NOTARY_KEYCHAIN" != "$VMLINUX_NOTARY_KEYCHAIN" ]]; then
  echo "ERROR: VMLX_NOTARY_KEYCHAIN and VMLINUX_NOTARY_KEYCHAIN disagree" >&2
  exit 1
fi
if [[ -n "${VMLX_R20_PRE_NOTARY_MANIFEST:-}" ]] \
  && [[ -n "${VMLINUX_R20_PRE_NOTARY_MANIFEST:-}" ]] \
  && [[ "$VMLX_R20_PRE_NOTARY_MANIFEST" != "$VMLINUX_R20_PRE_NOTARY_MANIFEST" ]]; then
  echo "ERROR: VMLX_R20_PRE_NOTARY_MANIFEST and VMLINUX_R20_PRE_NOTARY_MANIFEST disagree" >&2
  exit 1
fi
if [[ -n "${VMLX_R20_FINAL_NOTARY_MANIFEST:-}" ]] \
  && [[ -n "${VMLINUX_R20_FINAL_NOTARY_MANIFEST:-}" ]] \
  && [[ "$VMLX_R20_FINAL_NOTARY_MANIFEST" != "$VMLINUX_R20_FINAL_NOTARY_MANIFEST" ]]; then
  echo "ERROR: VMLX_R20_FINAL_NOTARY_MANIFEST and VMLINUX_R20_FINAL_NOTARY_MANIFEST disagree" >&2
  exit 1
fi

notarytool_args=(--keychain-profile "$NOTARY_PROFILE")
if [[ -n "$NOTARY_KEYCHAIN" ]]; then
  notarytool_args=(--keychain "$NOTARY_KEYCHAIN" "${notarytool_args[@]}")
fi

json_path() {
  local payload="$1"
  shift
  "$PYTHON_BIN" -I -c \
    'import json,sys; value=json.loads(sys.argv[1]); [value := value[key] for key in sys.argv[2:]]; print(value)' \
    "$payload" "$@"
}

json_file_path() {
  local path="$1"
  shift
  "$PYTHON_BIN" -I - "$path" "$@" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in sys.argv[2:]:
    value = value[key]
print(value)
PY
}

regenerate_blockmap() {
  local dmg="$1"
  run_manifest_app_builder \
    "$PRE_NOTARY_MANIFEST" \
    "$EXPECTED_PRE_MANIFEST_SHA256" \
    blockmap \
    --input "$dmg" \
    --output "$dmg.blockmap"
}

capture_private_command() {
  local label="$1"
  local result_dir="$2"
  local output_name="$3"
  local stderr_name="$4"
  shift 4

  artifact_chain capture-private-command \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --result-dir "$result_dir" \
    --output-name "$output_name" \
    --stderr-name "$stderr_name" \
    --label "$label" \
    -- "$@" >/dev/null
}

capture_apple_records() {
  local flavor="$1"
  local snapshot_dmg="$2"
  local snapshot_sha256="$3"
  local result_dir="$4"
  local submit_path="$result_dir/${flavor}.submit.json"
  local info_path="$result_dir/${flavor}.info.json"
  local log_path="$result_dir/${flavor}.log.json"
  local snapshot_record
  local submission_id

  snapshot_record="$(file_record "$snapshot_dmg" "immutable submitted ${flavor} DMG")"
  if [[ "$(json_field "$snapshot_record" sha256)" != "$snapshot_sha256" ]]; then
    echo "ERROR: immutable ${flavor} submission snapshot does not match build handoff" >&2
    exit 1
  fi
  "$APPLE_HDIUTIL" verify "$snapshot_dmg"
  "$APPLE_CODESIGN" --verify --strict --verbose=2 "$snapshot_dmg"
  require_developer_id_signature "$snapshot_dmg" 0
  assert_file_record "$snapshot_dmg" "$snapshot_record" "${flavor} snapshot before submit"
  capture_private_command \
    "${flavor}-submit" \
    "$result_dir" \
    "${flavor}.submit.json" \
    "${flavor}.submit.stderr.log" \
    "$APPLE_XCRUN" notarytool submit "$snapshot_dmg" \
      "${notarytool_args[@]}" \
      --wait \
      --output-format json
  assert_file_record "$snapshot_dmg" "$snapshot_record" "${flavor} snapshot after submit"
  submission_id="$(json_file_path "$submit_path" id)"
  capture_private_command \
    "${flavor}-info" \
    "$result_dir" \
    "${flavor}.info.json" \
    "${flavor}.info.stderr.log" \
    "$APPLE_XCRUN" notarytool info "$submission_id" \
      "${notarytool_args[@]}" \
      --output-format json
  assert_file_record "$snapshot_dmg" "$snapshot_record" "${flavor} snapshot after info"
  capture_private_command \
    "${flavor}-log" \
    "$result_dir" \
    "${flavor}.log.json" \
    "${flavor}.log.stderr.log" \
    "$APPLE_XCRUN" notarytool log "$submission_id" \
      "${notarytool_args[@]}" \
      --output-format json
  assert_file_record "$snapshot_dmg" "$snapshot_record" "${flavor} snapshot after log"
  artifact_chain query-apple-online \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --capture-dir "$result_dir/${flavor}.fresh-query" \
    --submission-id "$submission_id" \
    --expected-dmg-sha256 "$snapshot_sha256" \
    --expected-archive-name "$(basename "$snapshot_dmg")" \
    --expected-team-id "$EXPECTED_APPLE_TEAM_ID" \
    "${notarytool_args[@]}" >/dev/null
  printf '%s\n' "$submission_id"
}

staple_and_verify_original() {
  local flavor="$1"
  local dmg="$DIST_DIR_ABS/vMLX-${VERSION}-${flavor}-arm64.dmg"
  local blockmap="$dmg.blockmap"
  local operation_dir="$PRIVATE_EVIDENCE_ROOT/notary-working/vMLX-${VERSION}-${EXPECTED_SOURCE_COMMIT:0:12}/$flavor"
  local operation_dmg="$operation_dir/$(basename "$dmg")"
  local operation_blockmap="$operation_dmg.blockmap"
  local pre_dmg_sha256
  local pre_blockmap_sha256
  local before_dmg
  local before_blockmap
  local post_dmg
  local post_blockmap

  pre_dmg_sha256="$(json_file_path "$PRE_NOTARY_MANIFEST" artifacts "$flavor" dmg_sha256)"
  pre_blockmap_sha256="$(json_file_path "$PRE_NOTARY_MANIFEST" artifacts "$flavor" blockmap_sha256)"
  before_dmg="$(file_record "$dmg" "pre-staple ${flavor} DMG")"
  before_blockmap="$(file_record "$blockmap" "pre-staple ${flavor} blockmap")"
  if [[ "$(json_field "$before_dmg" sha256)" != "$pre_dmg_sha256" ]] \
    || [[ "$(json_field "$before_blockmap" sha256)" != "$pre_blockmap_sha256" ]]; then
    echo "ERROR: ${flavor} public artifacts changed before stapling" >&2
    exit 1
  fi

  if [[ -e "$operation_dir" || -L "$operation_dir" ]]; then
    echo "ERROR: refusing reused ${flavor} private notary operation directory" >&2
    exit 1
  fi
  mkdir -p "$operation_dir"
  chmod 0700 "$operation_dir"
  artifact_chain copy-operation-file \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --source "$dmg" \
    --out "$operation_dmg" \
    --expected-sha256 "$pre_dmg_sha256" \
    --writable \
    --label "pre-staple ${flavor} DMG" >/dev/null
  "$APPLE_XCRUN" stapler staple "$operation_dmg"
  post_dmg="$(file_record "$operation_dmg" "post-staple ${flavor} DMG")"
  if [[ "$(json_field "$post_dmg" sha256)" == "$pre_dmg_sha256" ]]; then
    echo "ERROR: ${flavor} DMG digest did not change after staple" >&2
    exit 1
  fi
  assert_file_record "$operation_dmg" "$post_dmg" "${flavor} DMG before staple validation"
  "$APPLE_XCRUN" stapler validate "$operation_dmg"
  "$APPLE_HDIUTIL" verify "$operation_dmg"
  "$APPLE_CODESIGN" --verify --strict --verbose=2 "$operation_dmg"
  require_developer_id_signature "$operation_dmg" 0
  "$APPLE_SPCTL" --assess --type open --context context:primary-signature --verbose=4 "$operation_dmg"
  assert_file_record "$operation_dmg" "$post_dmg" "${flavor} private DMG after Apple validation"

  regenerate_blockmap "$operation_dmg"
  assert_file_record "$operation_dmg" "$post_dmg" "${flavor} DMG after blockmap regeneration"
  post_blockmap="$(file_record "$operation_blockmap" "post-staple ${flavor} blockmap")"
  if [[ "$(json_field "$post_blockmap" sha256)" == "$pre_blockmap_sha256" ]]; then
    echo "ERROR: ${flavor} blockmap digest did not change after regeneration" >&2
    exit 1
  fi
  artifact_chain install-operation-file \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --source "$operation_dmg" \
    --destination "$dmg" \
    --expected-source-sha256 "$(json_field "$post_dmg" sha256)" \
    --expected-destination-sha256 "$pre_dmg_sha256" \
    --label "final ${flavor} DMG" >/dev/null
  artifact_chain install-operation-file \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --source "$operation_blockmap" \
    --destination "$blockmap" \
    --expected-source-sha256 "$(json_field "$post_blockmap" sha256)" \
    --expected-destination-sha256 "$pre_blockmap_sha256" \
    --label "final ${flavor} blockmap" >/dev/null
  assert_file_record "$dmg" "$(file_record "$dmg" "installed ${flavor} DMG")" "${flavor} installed DMG"
  assert_file_record "$blockmap" "$(file_record "$blockmap" "installed ${flavor} blockmap")" "${flavor} installed blockmap"
}

notarize_release_chain() {
  require_r20_release_context
  require_pre_notary_handoff_environment
  if [[ -z "$PRE_NOTARY_MANIFEST" ]]; then
    echo "ERROR: VMLX_R20_PRE_NOTARY_MANIFEST is required from the build handoff" >&2
    exit 1
  fi
  if [[ -z "$FINAL_NOTARY_MANIFEST" ]]; then
    FINAL_NOTARY_MANIFEST="$PRIVATE_EVIDENCE_ROOT/artifact-handoffs/vMLX-${VERSION}-${EXPECTED_SOURCE_COMMIT:0:12}-post-notary.json"
  fi

  artifact_chain check-pre \
    --root "$ROOT_DIR" \
    --dist "$DIST_DIR_ABS" \
    --version "$VERSION" \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --manifest "$PRE_NOTARY_MANIFEST" \
    --expected-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
    --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-source-tree "$EXPECTED_SOURCE_TREE" \
    --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" >/dev/null

  local snapshot_dir="$PRIVATE_EVIDENCE_ROOT/notary-snapshots/vMLX-${VERSION}-${EXPECTED_SOURCE_COMMIT:0:12}"
  local result_dir="$PRIVATE_EVIDENCE_ROOT/notary-records/vMLX-${VERSION}-${EXPECTED_SOURCE_COMMIT:0:12}"
  local snapshot_result
  local sequoia_snapshot
  local tahoe_snapshot
  local sequoia_snapshot_sha
  local tahoe_snapshot_sha
  local sequoia_id
  local tahoe_id
  artifact_chain create-private-directory \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --path "$result_dir" \
    --label "notary result directory" >/dev/null

  snapshot_result="$(
    artifact_chain create-snapshots \
      --root "$ROOT_DIR" \
      --dist "$DIST_DIR_ABS" \
      --version "$VERSION" \
      --private-root "$PRIVATE_EVIDENCE_ROOT" \
      --manifest "$PRE_NOTARY_MANIFEST" \
      --expected-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
      --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
      --expected-source-tree "$EXPECTED_SOURCE_TREE" \
      --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" \
      --snapshot-dir "$snapshot_dir"
  )"
  sequoia_snapshot="$(json_path "$snapshot_result" snapshots sequoia dmg_path)"
  tahoe_snapshot="$(json_path "$snapshot_result" snapshots tahoe dmg_path)"
  sequoia_snapshot_sha="$(json_path "$snapshot_result" snapshots sequoia dmg_sha256)"
  tahoe_snapshot_sha="$(json_path "$snapshot_result" snapshots tahoe dmg_sha256)"

  # Both Apple submissions are completed and their IDs proven distinct before
  # either public DMG is mutated by stapling.
  sequoia_id="$(
    capture_apple_records \
      sequoia "$sequoia_snapshot" "$sequoia_snapshot_sha" "$result_dir"
  )"
  tahoe_id="$(
    capture_apple_records \
      tahoe "$tahoe_snapshot" "$tahoe_snapshot_sha" "$result_dir"
  )"
  if [[ "$sequoia_id" == "$tahoe_id" ]]; then
    echo "ERROR: Sequoia and Tahoe reused one Apple submission ID" >&2
    exit 1
  fi

  staple_and_verify_original sequoia
  staple_and_verify_original tahoe

  local final_result
  local final_sha256
  final_result="$(
    artifact_chain write-final \
      --root "$ROOT_DIR" \
      --dist "$DIST_DIR_ABS" \
      --version "$VERSION" \
      --pre-manifest "$PRE_NOTARY_MANIFEST" \
      --expected-pre-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
      --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
      --expected-source-tree "$EXPECTED_SOURCE_TREE" \
      --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" \
      --private-root "$PRIVATE_EVIDENCE_ROOT" \
      --out "$FINAL_NOTARY_MANIFEST" \
      --sequoia-submission-id "$sequoia_id" \
      --sequoia-snapshot-dmg "$sequoia_snapshot" \
      --tahoe-submission-id "$tahoe_id" \
      --tahoe-snapshot-dmg "$tahoe_snapshot"
  )"
  final_sha256="$(json_path "$final_result" sha256)"
  artifact_chain check-final \
    --root "$ROOT_DIR" \
    --dist "$DIST_DIR_ABS" \
    --version "$VERSION" \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --manifest "$FINAL_NOTARY_MANIFEST" \
    --expected-manifest-sha256 "$final_sha256" \
    --expected-pre-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
    --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-source-tree "$EXPECTED_SOURCE_TREE" \
    --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" >/dev/null
  EXPECTED_FINAL_MANIFEST_SHA256="$final_sha256"
  export VMLX_R20_FINAL_NOTARY_MANIFEST="$FINAL_NOTARY_MANIFEST"
  export VMLX_R20_FINAL_NOTARY_MANIFEST_SHA256="$final_sha256"
  echo "==> Running final owner verification with fresh Apple queries and private mounts"
  verify_final_release_chain

  echo "==> Private no-clobber post-notary artifact handoff"
  printf 'VMLX_R20_FINAL_NOTARY_MANIFEST=%s\n' "$FINAL_NOTARY_MANIFEST"
  printf 'VMLX_R20_FINAL_NOTARY_MANIFEST_SHA256=%s\n' "$final_sha256"
  printf 'VMLX_R20_PRE_NOTARY_MANIFEST_SHA256=%s\n' "$EXPECTED_PRE_MANIFEST_SHA256"
  printf 'VMLX_R20_EXPECTED_SOURCE_COMMIT=%s\n' "$EXPECTED_SOURCE_COMMIT"
  printf 'VMLX_R20_EXPECTED_SOURCE_TREE=%s\n' "$EXPECTED_SOURCE_TREE"
  printf 'VMLX_R20_EXPECTED_PREFLIGHT_SHA256=%s\n' "$EXPECTED_PREFLIGHT_SHA256"
}

notarize_release_chain "$@"
