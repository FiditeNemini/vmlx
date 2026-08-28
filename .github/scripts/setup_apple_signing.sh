#!/bin/bash
set -euo pipefail
umask 077

usage() {
  echo "Usage: $0 install|cleanup" >&2
  exit 2
}

STATE_DIR="${RUNNER_TEMP:?RUNNER_TEMP is required}/vmlx-apple-signing-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}"
KEYCHAIN_PATH="$STATE_DIR/release.keychain-db"
PASSWORD_FILE="$STATE_DIR/keychain-password"
SEARCH_LIST_FILE="$STATE_DIR/original-search-list"
P12_FILE="$STATE_DIR/developer-id.p12"
P8_FILE="$STATE_DIR/notary-api-key.p8"
PROFILE="vmlx-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}"

decode_secret() {
  local env_name="$1"
  local output="$2"
  /usr/bin/python3 - "$env_name" "$output" <<'PY'
import base64
import os
import pathlib
import sys

name, output = sys.argv[1:]
value = os.environ.get(name, "")
if not value:
    raise SystemExit(f"ERROR: required secret {name} is empty")
try:
    decoded = base64.b64decode(value, validate=True)
except Exception as exc:
    raise SystemExit(f"ERROR: {name} is not valid base64: {exc}")
pathlib.Path(output).write_bytes(decoded)
PY
}

restore_search_list() {
  if [[ ! -s "$SEARCH_LIST_FILE" ]]; then
    return 0
  fi
  local keychains=()
  local line
  while IFS= read -r line; do
    line="${line#*\"}"
    line="${line%\"*}"
    [[ -n "$line" ]] && keychains+=("$line")
  done < "$SEARCH_LIST_FILE"
  if [[ ${#keychains[@]} -gt 0 ]]; then
    /usr/bin/security list-keychains -d user -s "${keychains[@]}"
  fi
}

install_signing() {
  : "${APPLE_DEVELOPER_ID_P12_PASSWORD:?APPLE_DEVELOPER_ID_P12_PASSWORD is required}"
  : "${APPLE_NOTARY_KEY_ID:?APPLE_NOTARY_KEY_ID is required}"
  : "${APPLE_NOTARY_ISSUER_ID:?APPLE_NOTARY_ISSUER_ID is required}"
  mkdir -p "$STATE_DIR"
  /usr/bin/security list-keychains -d user > "$SEARCH_LIST_FILE"
  /usr/bin/openssl rand -hex 32 > "$PASSWORD_FILE"
  local password
  password="$(<"$PASSWORD_FILE")"
  decode_secret APPLE_DEVELOPER_ID_P12_BASE64 "$P12_FILE"
  decode_secret APPLE_NOTARY_KEY_P8_BASE64 "$P8_FILE"

  /usr/bin/security create-keychain -p "$password" "$KEYCHAIN_PATH"
  /usr/bin/security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH"
  /usr/bin/security unlock-keychain -p "$password" "$KEYCHAIN_PATH"

  local existing=()
  local line
  while IFS= read -r line; do
    line="${line#*\"}"
    line="${line%\"*}"
    [[ -n "$line" ]] && existing+=("$line")
  done < "$SEARCH_LIST_FILE"
  /usr/bin/security list-keychains -d user -s "$KEYCHAIN_PATH" "${existing[@]}"
  /usr/bin/security import "$P12_FILE" \
    -k "$KEYCHAIN_PATH" \
    -P "$APPLE_DEVELOPER_ID_P12_PASSWORD" \
    -T /usr/bin/codesign \
    -T /usr/bin/security
  /usr/bin/security set-key-partition-list \
    -S apple-tool:,apple:,codesign: \
    -s -k "$password" "$KEYCHAIN_PATH" >/dev/null

  /usr/bin/xcrun notarytool store-credentials "$PROFILE" \
    --key "$P8_FILE" \
    --key-id "$APPLE_NOTARY_KEY_ID" \
    --issuer "$APPLE_NOTARY_ISSUER_ID" \
    --keychain "$KEYCHAIN_PATH" >/dev/null

  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "keychain=$KEYCHAIN_PATH"
      echo "profile=$PROFILE"
      echo "state_dir=$STATE_DIR"
    } >> "$GITHUB_OUTPUT"
  fi
  /usr/bin/security find-identity -v -p codesigning "$KEYCHAIN_PATH"
}

cleanup_signing() {
  restore_search_list || true
  /usr/bin/security delete-keychain "$KEYCHAIN_PATH" >/dev/null 2>&1 || true
  /bin/rm -rf -- "$STATE_DIR"
}

case "${1:-}" in
  install) install_signing ;;
  cleanup) cleanup_signing ;;
  *) usage ;;
esac
