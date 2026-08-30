#!/usr/bin/env python3
"""Atomically update the fixed vMLX website release surfaces on the origin."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def atomic_write(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, required=True)
    parser.add_argument("--updater", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()

    updater = json.loads(args.updater.read_text(encoding="utf-8"))
    version = str(updater.get("version", ""))
    if not VERSION_RE.fullmatch(version):
        raise SystemExit("ERROR: updater version is invalid")
    downloads = updater.get("downloads", {})
    tahoe = downloads.get("tahoe", {})
    sequoia = downloads.get("sequoia", {})
    for flavor, entry in (("tahoe", tahoe), ("sequoia", sequoia)):
        expected = f"vMLX-{version}-{flavor}-arm64.dmg"
        if not str(entry.get("url", "")).endswith(f"/v{version}/{expected}"):
            raise SystemExit(f"ERROR: invalid {flavor} updater URL")

    site_root = args.site_root.resolve()
    updater_target = site_root / "update" / "latest.json"
    download_page = site_root / "download" / "index.html"
    if not download_page.is_file():
        raise SystemExit(f"ERROR: missing download page: {download_page}")

    html = download_page.read_text(encoding="utf-8")
    matches = re.findall(r'"softwareVersion"\s*:\s*"([0-9]+\.[0-9]+\.[0-9]+)"', html)
    if len(matches) != 1:
        raise SystemExit("ERROR: download page must contain exactly one softwareVersion")
    old_version = matches[0]
    if VERSION_RE.fullmatch(old_version) is None:
        raise SystemExit("ERROR: current download page version is invalid")

    if old_version == version:
        if not updater_target.is_file():
            raise SystemExit("ERROR: current-version website is missing its updater")
        current_updater = json.loads(updater_target.read_text(encoding="utf-8"))
        if current_updater != updater:
            raise SystemExit("ERROR: current-version website updater differs from candidate")
        for entry in (tahoe, sequoia):
            if entry["url"] not in html:
                raise SystemExit(
                    f"ERROR: current-version page is missing {entry['url']}"
                )
        print(f"Website release surfaces already match {version}")
        return 0

    backup = args.backup_root.resolve() / f"vmlx-{old_version}-before-{version}"
    if backup.exists():
        backup_page = backup / "download-index.html"
        backup_updater = backup / "latest.json"
        if not backup_page.is_file() or backup_page.read_text(encoding="utf-8") != html:
            raise SystemExit(f"ERROR: existing release backup is not reusable: {backup}")
        if updater_target.is_file():
            if not backup_updater.is_file() or backup_updater.read_bytes() != updater_target.read_bytes():
                raise SystemExit(f"ERROR: existing updater backup is not reusable: {backup}")
    else:
        backup.mkdir(parents=True)
        if updater_target.is_file():
            shutil.copy2(updater_target, backup / "latest.json")
        shutil.copy2(download_page, backup / "download-index.html")

    updated = html.replace(f"vMLX-{old_version}-", f"vMLX-{version}-")
    updated = updated.replace(f"/v{old_version}/", f"/v{version}/")
    updated = updated.replace(f'"softwareVersion": "{old_version}"', f'"softwareVersion": "{version}"')
    updated = re.sub(
        r'"fileSize"\s*:\s*"[0-9]+ bytes"',
        f'"fileSize": "{tahoe.get("bytes", 0)} bytes"',
        updated,
        count=1,
    )
    updated = re.sub(
        rf"(?<![0-9.]){re.escape(old_version)}(?![0-9.])",
        version,
        updated,
    )
    for entry in (tahoe, sequoia):
        if entry["url"] not in updated:
            raise SystemExit(f"ERROR: patched page is missing {entry['url']}")
    if old_version != version and old_version in updated:
        raise SystemExit(f"ERROR: patched page still contains old version {old_version}")

    updater_bytes = (json.dumps(updater, indent=2) + "\n").encode("utf-8")
    atomic_write(updater_target, updater_bytes, 0o644)
    atomic_write(download_page, updated.encode("utf-8"), 0o644)
    print(f"Updated website release surfaces from {old_version} to {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
