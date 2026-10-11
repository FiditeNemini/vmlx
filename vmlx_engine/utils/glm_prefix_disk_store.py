"""Metadata catalog for GLM snapshots in the shared SSD pool.

The GLM owner serializes its native tensors unchanged. This layer handles
publication, discovery and eviction ownership using the existing pool protocol.
"""
from __future__ import annotations

import json

from .omni_session_disk_store import OmniSessionDiskStore


class GlmPrefixDiskStore:
    SCHEMA = "glm_native_prefix_v2"

    def __init__(self, root, model_key: str, max_size_bytes: int):
        self.store = OmniSessionDiskStore(
            root=root, model_key=model_key, max_size_bytes=max_size_bytes,
            schema=self.SCHEMA,
        )
        self.directory = self.store.directory
        self.budget = self.store.budget
        self._revision = None
        self._records: dict[str, dict] = {}

    @staticmethod
    def signature(name: str, tokens: int) -> str:
        return f"{int(tokens)}:{name}"

    def records(self) -> dict[str, dict]:
        """Refresh after pool mutations, never delete another writer's temp file."""
        with self.budget.mutation_guard() as locked:
            if not locked:
                raise OSError("GLM SSD catalog lock unavailable")
            revision = self.budget.cache_contents_revision_locked()
            if revision is not None and revision == self._revision:
                return dict(self._records)
            records = {}
            for side in self.directory.glob("*.json"):
                if ".tmp." in side.name or side.is_symlink():
                    continue
                try:
                    meta = json.loads(side.read_text())
                    if meta.get("schema") != self.SCHEMA:
                        continue
                    signature = meta["signature"]
                    count, name = signature.split(":", 1)
                    count = int(count)
                    if count < 1 or len(name) != 40 or any(c not in "0123456789abcdef" for c in name):
                        continue
                    data, expected_side = self.store.paths(signature)
                    if side != expected_side or data.is_symlink() or not data.is_file():
                        continue
                    ds, ss = data.stat(), side.stat()
                    records[name] = {
                        "n": count, "bytes": ds.st_size + ss.st_size,
                        "used": max(ds.st_mtime, ss.st_mtime),
                    }
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    # Discovery is not a cleanup pass. Pool ownership governs
                    # deletion; an unreadable record is only a cache miss here.
                    continue
            self._records = records
            self._revision = revision
            return dict(records)

    def contains(self, name: str, tokens: int) -> bool:
        # Do not trust an in-memory catalog after another owner evicted a file.
        return self.store.load(self.signature(name, tokens), lambda _path: True) is True

    def save(self, name: str, tokens: int, writer):
        return self.store.save(self.signature(name, tokens), writer)

    def load(self, name: str, tokens: int, reader):
        # The reader must materialize lazy tensors before returning so the
        # shared eviction lock protects the complete disk read.
        return self.store.load(self.signature(name, tokens), reader)

    @staticmethod
    def file_identity(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def remove(self, name: str, tokens: int, expected_identity) -> bool:
        with self.budget.exclusive_mutation_guard() as locked:
            if not locked:
                raise OSError("GLM SSD removal lock unavailable")
            paths = self.store.paths(self.signature(name, tokens))
            if any(p.is_symlink() for p in paths):
                raise OSError("GLM SSD record must not be a symlink")
            if not paths[0].is_file() or self.file_identity(paths[0]) != expected_identity:
                return False  # Another writer replaced the failed record.
            removed = 0
            for path in paths:
                if path.is_file():
                    removed += path.stat().st_size
                    path.unlink()
            if removed:
                self.store._fsync_directory(self.directory)
                self.budget.account_finalized_write_locked(-removed)
            self._revision = None
            return True

    def clear(self) -> int:
        count = len(self.records())
        self.store.clear()
        self._records = {}
        self._revision = None
        return count

    def close(self) -> None:
        self.store.close()
