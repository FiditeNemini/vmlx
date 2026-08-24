#!/bin/bash
set -euo pipefail
R20_FIXED_PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PATH="$R20_FIXED_PATH"
umask 077

# Verify the final public r20 DMG chain after signing, notarization, stapling,
# and post-staple blockmap regeneration. This token-free script does not build,
# sign, notarize, upload, tag, publish, or mutate release feeds.

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PANEL_DIR="$(cd -P "$SCRIPT_DIR/.." && pwd -P)"
ROOT_DIR="$(cd -P "$PANEL_DIR/.." && pwd -P)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

VERSION="$(
  "$PYTHON_BIN" -I - "$PANEL_DIR/package.json" <<'PY'
import json
import pathlib
import sys

print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["version"])
PY
)"
if [[ -n "${VMLX_R20_EXPECTED_VERSION:-}" ]] \
  && [[ "$VMLX_R20_EXPECTED_VERSION" != "$VERSION" ]]; then
  echo "ERROR: VMLX_R20_EXPECTED_VERSION does not match package version $VERSION" >&2
  exit 1
fi
export VMLX_R20_EXPECTED_VERSION="$VERSION"
DIST_DIR_ABS="$PANEL_DIR/release"
ARTIFACT_CHAIN_HELPER="$ROOT_DIR/tests/cross_matrix/run_packaged_integrity_contract.py"
EXPECTED_APPLE_TEAM_ID="55KGF2S5AY"
EXPECTED_CODESIGN_IDENTITY="Developer ID Application: ShieldStack LLC (55KGF2S5AY)"
RELEASE_SCOPE="${VMLX_RELEASE_SCOPE:-${VMLINUX_RELEASE_SCOPE:-}}"
PRIVATE_EVIDENCE_ROOT="${VMLX_R20_PRIVATE_EVIDENCE_ROOT:-${VMLINUX_R20_PRIVATE_EVIDENCE_ROOT:-}}"
EXPECTED_PRE_MANIFEST_SHA256="${VMLX_R20_PRE_NOTARY_MANIFEST_SHA256:-${VMLINUX_R20_PRE_NOTARY_MANIFEST_SHA256:-}}"
EXPECTED_FINAL_MANIFEST_SHA256="${VMLX_R20_FINAL_NOTARY_MANIFEST_SHA256:-${VMLINUX_R20_FINAL_NOTARY_MANIFEST_SHA256:-}}"
EXPECTED_SOURCE_COMMIT="${VMLX_R20_EXPECTED_SOURCE_COMMIT:-${VMLINUX_R20_EXPECTED_SOURCE_COMMIT:-}}"
EXPECTED_SOURCE_TREE="${VMLX_R20_EXPECTED_SOURCE_TREE:-${VMLINUX_R20_EXPECTED_SOURCE_TREE:-}}"
EXPECTED_PREFLIGHT_SHA256="${VMLX_R20_EXPECTED_PREFLIGHT_SHA256:-${VMLINUX_R20_EXPECTED_PREFLIGHT_SHA256:-}}"
NOTARY_PROFILE="${VMLINUX_NOTARY_KEYCHAIN_PROFILE:-${VMLX_NOTARY_KEYCHAIN_PROFILE:-}}"
NOTARY_KEYCHAIN="${VMLINUX_NOTARY_KEYCHAIN:-${VMLX_NOTARY_KEYCHAIN:-}}"
APPLE_XCRUN="/usr/bin/xcrun"
APPLE_CODESIGN="/usr/bin/codesign"
APPLE_HDIUTIL="/usr/bin/hdiutil"
APPLE_SPCTL="/usr/sbin/spctl"

if [[ -z "$NOTARY_PROFILE" ]]; then
  echo "ERROR: set VMLX_NOTARY_KEYCHAIN_PROFILE from the private release environment" >&2
  exit 1
fi

assert_unshadowed_tool() {
  local name="$1"
  local expected="$2"
  local resolved
  resolved="$(type -P "$name" || true)"
  if [[ "$(type -t "$name" || true)" != "file" || "$resolved" != "$expected" ]]; then
    echo "ERROR: Apple release tool $name is shadowed; expected $expected, found ${resolved:-missing}" >&2
    exit 1
  fi
  if [[ ! -x "$expected" ]]; then
    echo "ERROR: required Apple release tool is unavailable: $expected" >&2
    exit 1
  fi
}

assert_unshadowed_apple_tools() {
  assert_unshadowed_tool xcrun "$APPLE_XCRUN"
  assert_unshadowed_tool codesign "$APPLE_CODESIGN"
  assert_unshadowed_tool hdiutil "$APPLE_HDIUTIL"
  assert_unshadowed_tool spctl "$APPLE_SPCTL"
}

notarytool_args=(--keychain-profile "$NOTARY_PROFILE")
if [[ -n "$NOTARY_KEYCHAIN" ]]; then
  notarytool_args=(--keychain "$NOTARY_KEYCHAIN" "${notarytool_args[@]}")
fi

run_manifest_tool_action() {
  local manifest="$1"
  local expected_manifest_sha256="$2"
  local action="$3"
  shift 3
  if [[ "$action" == "app-builder" ]] \
    && [[ -n "${VMLINUX_APP_BUILDER_BIN:-}" || -n "${VMLX_APP_BUILDER_BIN:-}" ]]; then
    echo "ERROR: production verification forbids app-builder binary overrides" >&2
    exit 1
  fi
  artifact_chain run-bound-tool-action \
    --binding-kind manifest \
    --document "$manifest" \
    --expected-document-sha256 "$expected_manifest_sha256" \
    --action "$action" \
    --cwd "$PANEL_DIR" \
    -- "$@"
}

run_manifest_asar_action() {
  local manifest="$1"
  local expected_manifest_sha256="$2"
  shift 2
  run_manifest_tool_action \
    "$manifest" "$expected_manifest_sha256" asar "$@"
}

run_manifest_app_builder() {
  local manifest="$1"
  local expected_manifest_sha256="$2"
  shift 2
  run_manifest_tool_action \
    "$manifest" "$expected_manifest_sha256" app-builder "$@"
}

if [[ -n "${VMLX_RELEASE_SCOPE:-}" && -n "${VMLINUX_RELEASE_SCOPE:-}" ]] \
  && [[ "$VMLX_RELEASE_SCOPE" != "$VMLINUX_RELEASE_SCOPE" ]]; then
  echo "ERROR: VMLX_RELEASE_SCOPE and VMLINUX_RELEASE_SCOPE disagree" >&2
  exit 1
fi
if [[ -n "${VMLX_R20_PRIVATE_EVIDENCE_ROOT:-}" ]] \
  && [[ -n "${VMLINUX_R20_PRIVATE_EVIDENCE_ROOT:-}" ]] \
  && [[ "$VMLX_R20_PRIVATE_EVIDENCE_ROOT" != "$VMLINUX_R20_PRIVATE_EVIDENCE_ROOT" ]]; then
  echo "ERROR: VMLX_R20_PRIVATE_EVIDENCE_ROOT and VMLINUX_R20_PRIVATE_EVIDENCE_ROOT disagree" >&2
  exit 1
fi

artifact_chain() {
  "$PYTHON_BIN" "$ARTIFACT_CHAIN_HELPER" artifact-chain "$@"
}

json_field() {
  local payload="$1"
  local field="$2"
  "$PYTHON_BIN" -I -c \
    'import json,sys; value=json.loads(sys.argv[1]); print(value[sys.argv[2]])' \
    "$payload" "$field"
}

json_file_field() {
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

file_record() {
  local path="$1"
  local label="$2"
  artifact_chain file-record --path "$path" --label "$label"
}

assert_file_record() {
  local path="$1"
  local record="$2"
  local label="$3"
  artifact_chain check-file \
    --path "$path" \
    --label "$label" \
    --expected-sha256 "$(json_field "$record" sha256)" \
    --expected-device "$(json_field "$record" device)" \
    --expected-inode "$(json_field "$record" inode)" \
    --expected-size "$(json_field "$record" size)" >/dev/null
}

require_pre_notary_handoff_environment() {
  local name
  for name in \
    EXPECTED_PRE_MANIFEST_SHA256 \
    EXPECTED_SOURCE_COMMIT \
    EXPECTED_SOURCE_TREE \
    EXPECTED_PREFLIGHT_SHA256; do
    if [[ -z "${!name}" ]]; then
      echo "ERROR: missing independent r20 build handoff value: $name" >&2
      exit 1
    fi
  done
}

require_final_notary_handoff_environment() {
  require_pre_notary_handoff_environment
  if [[ -z "$EXPECTED_FINAL_MANIFEST_SHA256" ]]; then
    echo "ERROR: VMLX_R20_FINAL_NOTARY_MANIFEST_SHA256 is required from the notarization handoff" >&2
    exit 1
  fi
}

require_r20_release_context() {
  assert_unshadowed_apple_tools
  if [[ "$PATH" != "$R20_FIXED_PATH" ]]; then
    echo "ERROR: r20 verification PATH is not sanitized" >&2
    exit 1
  fi
  if [[ -n "${VMLX_RELEASE_OUTPUT_DIR:-}" || -n "${VMLINUX_RELEASE_OUTPUT_DIR:-}" ]]; then
    echo "ERROR: r20 verification output is fixed at $PANEL_DIR/release; overrides are forbidden" >&2
    exit 1
  fi
  if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: the r20 artifact-chain verifier requires a release version like 1.2.3, found $VERSION" >&2
    exit 1
  fi
  if [[ "$RELEASE_SCOPE" != "r20_production" ]]; then
    echo "ERROR: set VMLX_RELEASE_SCOPE=r20_production for final r20 verification" >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: missing authoritative release Python: $PYTHON_BIN" >&2
    exit 1
  fi
  if [[ ! -f "$ARTIFACT_CHAIN_HELPER" ]]; then
    echo "ERROR: missing artifact-chain validator: $ARTIFACT_CHAIN_HELPER" >&2
    exit 1
  fi
  if [[ -z "$PRIVATE_EVIDENCE_ROOT" ]]; then
    echo "ERROR: VMLX_R20_PRIVATE_EVIDENCE_ROOT is required and must be outside every Git worktree" >&2
    exit 1
  fi
  artifact_chain check-private-root --private-root "$PRIVATE_EVIDENCE_ROOT"
}

require_developer_id_signature() {
  local path="$1"
  local require_runtime="${2:-0}"
  local signature

  signature="$("$APPLE_CODESIGN" -dv --verbose=4 "$path" 2>&1)"
  printf '%s\n' "$signature"

  if ! grep -Fqx "Authority=$EXPECTED_CODESIGN_IDENTITY" <<<"$signature"; then
    echo "ERROR: $path is not signed by the exact ShieldStack Developer ID Application certificate" >&2
    exit 1
  fi
  if ! grep -Fqx "TeamIdentifier=$EXPECTED_APPLE_TEAM_ID" <<<"$signature"; then
    echo "ERROR: $path is not signed with the exact ShieldStack Developer ID team identifier" >&2
    exit 1
  fi
  if grep -Fq "Signature=adhoc" <<<"$signature"; then
    echo "ERROR: $path is ad-hoc signed and is not a public release artifact" >&2
    exit 1
  fi
  if ! grep -q "^Timestamp=" <<<"$signature"; then
    echo "ERROR: $path is missing a trusted signing timestamp" >&2
    exit 1
  fi
  if [[ "$require_runtime" == "1" ]] \
    && ! grep -Eq "^CodeDirectory .*flags=.*runtime" <<<"$signature"; then
    echo "ERROR: $path is missing the hardened-runtime code-signing flag" >&2
    exit 1
  fi
}

verify_mounted_release_dmg() {
  local flavor="$1"
  local dmg="$2"
  local expected_sha256="$3"
  local final_manifest="$4"
  local bound_record

  echo "==> Verifying final ${flavor} DMG and mounted application: $dmg"
  bound_record="$(file_record "$dmg" "final ${flavor} DMG")"
  if [[ "$(json_field "$bound_record" sha256)" != "$expected_sha256" ]]; then
    echo "ERROR: final ${flavor} DMG does not match the independently validated manifest" >&2
    exit 1
  fi
  assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG before hdiutil verify"
  "$APPLE_HDIUTIL" verify "$dmg"
  assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after hdiutil verify"
  "$APPLE_CODESIGN" --verify --strict --verbose=2 "$dmg"
  assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after codesign verify"
  require_developer_id_signature "$dmg" 0
  assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after identity inspection"
  "$APPLE_XCRUN" stapler validate "$dmg"
  assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after stapler validate"
  "$APPLE_SPCTL" --assess --type open --context context:primary-signature --verbose=4 "$dmg"
  assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after Gatekeeper assessment"

  (
    set -euo pipefail
    local mount_dir
    local extracted_asar
    local attached=0
    mount_dir="$(mktemp -d "$PRIVATE_EVIDENCE_ROOT/.mount-${flavor}.XXXXXX")"
    extracted_asar="$(mktemp -d "$PRIVATE_EVIDENCE_ROOT/.mounted-${flavor}-asar.XXXXXX")"
    cleanup_mount() {
      if [[ "$attached" == "1" ]]; then
        "$APPLE_HDIUTIL" detach "$mount_dir" >/dev/null 2>&1 \
          || "$APPLE_HDIUTIL" detach -force "$mount_dir" >/dev/null 2>&1 \
          || true
      fi
      rm -rf "$mount_dir"
      rm -rf "$extracted_asar"
    }
    trap cleanup_mount EXIT

    "$APPLE_HDIUTIL" attach \
      -readonly \
      -nobrowse \
      -noautoopen \
      -mountpoint "$mount_dir" \
      "$dmg" >/dev/null
    attached=1
    assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after readonly mount"

    "$PYTHON_BIN" -I - "$mount_dir" <<'PY'
import os
import pathlib
import re
import sys
import uuid

root = pathlib.Path(sys.argv[1])
entries = {entry.name: entry for entry in root.iterdir()}
visible = {name for name in entries if not name.startswith(".")}
expected_visible = {"vMLX.app", "Applications"}
if visible != expected_visible:
    raise SystemExit(
        f"mounted DMG visible payload mismatch: "
        f"missing={sorted(expected_visible - visible)}, "
        f"extra={sorted(visible - expected_visible)}"
    )
allowed_presentation_metadata = {
    ".background",
    ".DS_Store",
    ".VolumeIcon.icns",
    ".fseventsd",
}
unexpected_hidden = {
    name for name in entries
    if name.startswith(".") and name not in allowed_presentation_metadata
}
if unexpected_hidden:
    raise SystemExit(
        f"mounted DMG has unexpected hidden payloads: {sorted(unexpected_hidden)}"
    )
background = entries.get(".background")
if background is not None:
    if background.is_symlink() or not background.is_dir():
        raise SystemExit("mounted DMG .background metadata is not a real directory")
    background_entries = list(background.iterdir())
    if (
        len(background_entries) != 1
        or background_entries[0].is_symlink()
        or not background_entries[0].is_file()
        or background_entries[0].name != "1.tiff"
    ):
        raise SystemExit("mounted DMG .background metadata has unexpected contents")
for metadata_name in (".DS_Store", ".VolumeIcon.icns"):
    metadata = entries.get(metadata_name)
    if metadata is not None and (metadata.is_symlink() or not metadata.is_file()):
        raise SystemExit(f"mounted DMG {metadata_name} metadata is not a regular file")
fseventsd = entries.get(".fseventsd")
if fseventsd is not None:
    if fseventsd.is_symlink() or not fseventsd.is_dir():
        raise SystemExit("mounted DMG .fseventsd metadata is not a real directory")
    fsevent_entries = list(fseventsd.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in fsevent_entries):
        raise SystemExit("mounted DMG .fseventsd metadata contains a non-file entry")
    fsevent_names = {entry.name for entry in fsevent_entries}
    if "fseventsd-uuid" not in fsevent_names or any(
        name != "fseventsd-uuid" and re.fullmatch(r"[0-9a-f]{16}", name) is None
        for name in fsevent_names
    ):
        raise SystemExit("mounted DMG .fseventsd metadata has unexpected contents")
    fsevent_uuid = (fseventsd / "fseventsd-uuid").read_text(encoding="ascii").strip()
    try:
        canonical_fsevent_uuid = str(uuid.UUID(fsevent_uuid))
    except ValueError as exc:
        raise SystemExit("mounted DMG .fseventsd UUID is invalid") from exc
    if canonical_fsevent_uuid != fsevent_uuid.lower():
        raise SystemExit("mounted DMG .fseventsd UUID is not canonical")
app = entries["vMLX.app"]
applications = entries["Applications"]
if app.is_symlink() or not app.is_dir():
    raise SystemExit("mounted vMLX.app is not a real application directory")
if not applications.is_symlink() or os.readlink(applications) != "/Applications":
    raise SystemExit("mounted Applications entry is not the exact /Applications link")
PY

    local app="$mount_dir/vMLX.app"
    "$PYTHON_BIN" -I - "$app/Contents/Info.plist" "$VERSION" <<'PY'
import pathlib
import plistlib
import sys

info_path = pathlib.Path(sys.argv[1])
expected_version = sys.argv[2]
with info_path.open("rb") as handle:
    info = plistlib.load(handle)
if info.get("CFBundleIdentifier") != "net.vmlx.app":
    raise SystemExit("mounted app bundle identifier is not net.vmlx.app")
if info.get("CFBundleShortVersionString") != expected_version:
    raise SystemExit("mounted app short version does not match the release version")
if info.get("CFBundleVersion") != expected_version:
    raise SystemExit("mounted app bundle version does not match the release version")
PY
    "$APPLE_CODESIGN" --verify --deep --strict --verbose=2 "$app"
    require_developer_id_signature "$app" 1
    "$APPLE_SPCTL" --assess --type execute --verbose=4 "$app"
    run_manifest_asar_action "$final_manifest" "$EXPECTED_FINAL_MANIFEST_SHA256" extract \
      "$app/Contents/Resources/app.asar" \
      "$extracted_asar"
    artifact_chain check-mounted-app \
      --root "$ROOT_DIR" \
      --dist "$DIST_DIR_ABS" \
      --version "$VERSION" \
      --private-root "$PRIVATE_EVIDENCE_ROOT" \
      --final-manifest "$final_manifest" \
      --expected-final-manifest-sha256 "$EXPECTED_FINAL_MANIFEST_SHA256" \
      --expected-pre-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
      --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
      --expected-source-tree "$EXPECTED_SOURCE_TREE" \
      --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256" \
      --flavor "$flavor" \
      --mounted-app "$app" \
      --extracted-asar "$extracted_asar" >/dev/null
    assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after mounted-app checks"

    "$APPLE_HDIUTIL" detach "$mount_dir" >/dev/null
    attached=0
    assert_file_record "$dmg" "$bound_record" "final ${flavor} DMG after detach"
    trap - EXIT
    rm -rf "$mount_dir"
    rm -rf "$extracted_asar"
  )
}

verify_final_release_chain() {
  require_r20_release_context
  require_final_notary_handoff_environment
  local final_manifest="${VMLX_R20_FINAL_NOTARY_MANIFEST:-${VMLINUX_R20_FINAL_NOTARY_MANIFEST:-$PRIVATE_EVIDENCE_ROOT/vmlx-${VERSION}-r20-post-notary-manifest.json}}"
  if [[ -n "${VMLX_R20_FINAL_NOTARY_MANIFEST:-}" ]] \
    && [[ -n "${VMLINUX_R20_FINAL_NOTARY_MANIFEST:-}" ]] \
    && [[ "$VMLX_R20_FINAL_NOTARY_MANIFEST" != "$VMLINUX_R20_FINAL_NOTARY_MANIFEST" ]]; then
    echo "ERROR: VMLX_R20_FINAL_NOTARY_MANIFEST and VMLINUX_R20_FINAL_NOTARY_MANIFEST disagree" >&2
    exit 1
  fi

  artifact_chain check-final \
    --root "$ROOT_DIR" \
    --dist "$DIST_DIR_ABS" \
    --version "$VERSION" \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --manifest "$final_manifest" \
    --expected-manifest-sha256 "$EXPECTED_FINAL_MANIFEST_SHA256" \
    --expected-pre-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
    --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-source-tree "$EXPECTED_SOURCE_TREE" \
    --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256"

  local verification_root
  verification_root="$(
    mktemp -d "$PRIVATE_EVIDENCE_ROOT/.final-verification.XXXXXX"
  )"
  chmod 0700 "$verification_root"
  (
    set -euo pipefail
    trap 'rm -rf "$verification_root"' EXIT
    local flavor
    local expected_sha256
    local submitted_sha256
    local expected_blockmap_sha256
    local submitted_path
    local submitted_record
    local submission_id
    local public_dmg
    local snapshot_dmg
    local apple_capture_dir
    local recomputed_blockmap
    for flavor in sequoia tahoe; do
      expected_sha256="$(
        json_file_field "$final_manifest" artifacts "$flavor" dmg_post_notary_sha256
      )"
      submitted_sha256="$(
        json_file_field "$final_manifest" artifacts "$flavor" submitted_dmg_sha256
      )"
      expected_blockmap_sha256="$(
        json_file_field "$final_manifest" artifacts "$flavor" blockmap_post_notary_sha256
      )"
      submitted_path="$(
        json_file_field "$final_manifest" artifacts "$flavor" submitted_dmg_path
      )"
      submission_id="$(
        json_file_field "$final_manifest" artifacts "$flavor" notary_submission_id
      )"
      submitted_record="$(
        file_record "$submitted_path" "immutable submitted ${flavor} DMG"
      )"
      if [[ "$(json_field "$submitted_record" sha256)" != "$submitted_sha256" ]]; then
        echo "ERROR: submitted ${flavor} snapshot differs from the final manifest" >&2
        exit 1
      fi
      "$APPLE_HDIUTIL" verify "$submitted_path"
      "$APPLE_CODESIGN" --verify --strict --verbose=2 "$submitted_path"
      require_developer_id_signature "$submitted_path" 0
      assert_file_record \
        "$submitted_path" \
        "$submitted_record" \
        "submitted ${flavor} snapshot after exact-team verification"
      apple_capture_dir="$verification_root/${flavor}-fresh-apple"
      artifact_chain query-apple-online \
        --private-root "$PRIVATE_EVIDENCE_ROOT" \
        --capture-dir "$apple_capture_dir" \
        --submission-id "$submission_id" \
        --expected-dmg-sha256 "$submitted_sha256" \
        --expected-archive-name "$(basename "$submitted_path")" \
        --expected-team-id "$EXPECTED_APPLE_TEAM_ID" \
        "${notarytool_args[@]}" >/dev/null

      public_dmg="$DIST_DIR_ABS/vMLX-${VERSION}-${flavor}-arm64.dmg"
      mkdir -p "$verification_root/$flavor"
      snapshot_dmg="$verification_root/$flavor/$(basename "$public_dmg")"
      artifact_chain copy-operation-file \
        --private-root "$PRIVATE_EVIDENCE_ROOT" \
        --source "$public_dmg" \
        --out "$snapshot_dmg" \
        --expected-sha256 "$expected_sha256" \
        --label "final ${flavor} verification DMG" >/dev/null
      recomputed_blockmap="$verification_root/$flavor/recomputed.dmg.blockmap"
      run_manifest_app_builder \
        "$final_manifest" "$EXPECTED_FINAL_MANIFEST_SHA256" blockmap \
        --input "$snapshot_dmg" \
        --output "$recomputed_blockmap"
      artifact_chain check-recomputed-blockmap \
        --expected-blockmap "$public_dmg.blockmap" \
        --recomputed-blockmap "$recomputed_blockmap" \
        --expected-sha256 "$expected_blockmap_sha256" >/dev/null
      verify_mounted_release_dmg \
        "$flavor" \
        "$snapshot_dmg" \
        "$expected_sha256" \
        "$final_manifest"
    done
  )

  # Recompute the final manifest digest and every artifact/notary-result hash
  # after mounting so a concurrent mutation cannot pass on an earlier check.
  artifact_chain check-final \
    --root "$ROOT_DIR" \
    --dist "$DIST_DIR_ABS" \
    --version "$VERSION" \
    --private-root "$PRIVATE_EVIDENCE_ROOT" \
    --manifest "$final_manifest" \
    --expected-manifest-sha256 "$EXPECTED_FINAL_MANIFEST_SHA256" \
    --expected-pre-manifest-sha256 "$EXPECTED_PRE_MANIFEST_SHA256" \
    --expected-source-commit "$EXPECTED_SOURCE_COMMIT" \
    --expected-source-tree "$EXPECTED_SOURCE_TREE" \
    --expected-preflight-sha256 "$EXPECTED_PREFLIGHT_SHA256"
  echo "==> Final r20 DMG artifact chain verified: $final_manifest"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  verify_final_release_chain "$@"
fi
