# SPDX-License-Identifier: Apache-2.0
"""Process-safe aggregate byte budget for block-cache SSD payloads.

``BlockDiskStore`` indexes one model/config namespace at a time.  Applying the
user's ``--block-disk-cache-max-gb`` value independently to every namespace
allows the physical cache root to grow by ``N * max_gb`` and hybrid models can
add a second, equally large SSM-companion budget.  This module owns the one
aggregate LRU trim for the whole configured block-cache root.

Only finalized cache payloads are managed:

* SQLite-indexed block ``.safetensors`` files, using index ``last_accessed``;
* paired SSM companion ``.safetensors`` + ``.json`` records, using mtime; and
* old finalized orphan payloads inside known cache layout directories.

Temporary/in-flight files are counted as physical usage but never deleted while
their writer lease is live.  Cooperating writers hold a shared ``flock`` while
publishing final files; eviction holds the exclusive lock.  Older/uncoordinated
writers are protected by an orphan grace period, and any ambiguous
database/path/lock condition aborts the trim without affecting inference.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import re
import sqlite3
import stat
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_NAME = ".vmlx-global-cache-budget.lock"
_LEASE_DIR_NAME = ".vmlx-global-cache-budget-leases"
_ACCOUNTING_NAME = ".vmlx-global-cache-budget-accounting.json"
_ROOT_MARKER_NAME = ".vmlx-block-cache-root-v1"
_NAMESPACE_MARKER_NAME = ".vmlx-block-cache-namespace-v1"
_STATE_VERSION = 1
_DEFAULT_ORPHAN_GRACE_SECONDS = 5 * 60
_DEFAULT_RECONCILE_INTERVAL_SECONDS = 30.0
_DEFAULT_BIRTH_VALIDATION_INTERVAL_SECONDS = 30.0

_ROOT_THREAD_LOCKS_GUARD = threading.Lock()
_ROOT_THREAD_LOCKS: dict[Path, threading.RLock] = {}


def _root_thread_lock(root: Path) -> threading.RLock:
    with _ROOT_THREAD_LOCKS_GUARD:
        return _ROOT_THREAD_LOCKS.setdefault(root, threading.RLock())


def _process_birth_identity(pid: int) -> str | None:
    """Return a stable process-start identity, not merely a reusable PID."""

    if pid <= 0:
        return None
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        raw = proc_stat.read_text()
        # Field 2 (comm) may contain spaces/parentheses.  Fields after its final
        # ')' start at field 3; starttime is field 22, hence index 19 here.
        remainder = raw[raw.rfind(")") + 2 :].split()
        if len(remainder) > 19:
            return f"proc-start-ticks:{remainder[19]}"
    except (OSError, ValueError):
        pass
    try:
        started = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(int(pid))],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return f"ps-lstart:{started}" if started else None


def ensure_managed_block_cache_namespace(
    namespace: str | os.PathLike[str],
) -> Path:
    """Claim only an empty or unambiguously legacy block-cache namespace."""

    path = Path(namespace).expanduser()
    if path.is_symlink():
        raise OSError(f"refusing symlinked block-cache namespace: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"invalid block-cache namespace: {path}")
    resolved_namespace = path.resolve(strict=True)

    recognized_types = {
        "blocks": "directory",
        "ssm_companion": "directory",
        "block_index.db": "file",
        "block_index.db-wal": "file",
        "block_index.db-shm": "file",
        "block_index.db-journal": "file",
    }
    for name, expected_type in recognized_types.items():
        entry = path / name
        # ``Path.exists`` is false for a broken symlink, so test the link first.
        if entry.is_symlink():
            raise OSError(f"refusing symlinked block-cache path: {entry}")
        if not entry.exists():
            continue
        if expected_type == "directory" and not entry.is_dir():
            raise OSError(f"expected block-cache directory: {entry}")
        if expected_type == "file" and not entry.is_file():
            raise OSError(f"expected block-cache file: {entry}")
        resolved_entry = entry.resolve(strict=True)
        if not resolved_entry.is_relative_to(resolved_namespace):
            raise OSError(f"block-cache path escaped its namespace: {entry}")

    marker = path / _NAMESPACE_MARKER_NAME
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise OSError(f"invalid block-cache namespace marker: {marker}")
        return resolved_namespace

    existing = {entry.name for entry in path.iterdir()}
    legacy_names = {
        "blocks",
        "block_index.db",
        "block_index.db-wal",
        "block_index.db-shm",
        "block_index.db-journal",
        "ssm_companion",
    }
    if existing and not existing.issubset(legacy_names):
        raise OSError(
            "refusing to claim non-empty unrecognized block-cache namespace "
            f"{path}"
        )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(marker, flags, 0o600)
    except FileExistsError as exc:
        if marker.is_symlink() or not marker.is_file():
            raise OSError(
                f"invalid block-cache namespace marker: {marker}"
            ) from exc
    else:
        os.close(fd)
    return resolved_namespace


@dataclass(frozen=True)
class _BudgetCandidate:
    kind: str
    size_bytes: int
    last_accessed_ns: int
    paths: tuple[Path, ...]
    database: Path | None = None
    block_hash: str | None = None
    indexed_file_name: str | None = None


@dataclass(frozen=True)
class GlobalBudgetResult:
    """Best-effort result from one aggregate trim."""

    max_size_bytes: int
    bytes_before: int
    bytes_after: int
    evicted_entries: int
    evicted_bytes: int
    protected_recent_orphans: int
    compliant: bool
    scan_performed: bool
    reconciled_at_ns: int
    accounted: bool
    accounting_generation: int
    reconciliation_generation: int
    error: str | None = None


class GlobalDiskCacheBudget:
    """Aggregate LRU budget shared by all namespaces below one cache root."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        max_size_bytes: int,
        *,
        orphan_grace_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS,
        reconcile_interval_seconds: float = _DEFAULT_RECONCILE_INTERVAL_SECONDS,
        allow_legacy_hashed_namespaces: bool = False,
        allow_legacy_direct_namespace: bool = False,
    ) -> None:
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self.root = root_path.resolve()
        self._requested_max_size_bytes = max(0, int(max_size_bytes))
        self._orphan_grace_ns = max(
            0,
            int(float(orphan_grace_seconds) * 1_000_000_000),
        )
        self._reconcile_interval_ns = max(
            0,
            int(float(reconcile_interval_seconds) * 1_000_000_000),
        )
        self._allow_legacy_hashed_namespaces = bool(
            allow_legacy_hashed_namespaces
        )
        self._allow_legacy_direct_namespace = bool(
            allow_legacy_direct_namespace
        )
        self._thread_lock = _root_thread_lock(self.root)
        self._last_result: GlobalBudgetResult | None = None
        self._last_reconcile_monotonic_ns = 0
        self._lease_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._lease_owner_pid = os.getpid()
        self._lease_owner_birth_identity = _process_birth_identity(
            self._lease_owner_pid
        )
        self._birth_validation_interval_ns = int(
            _DEFAULT_BIRTH_VALIDATION_INTERVAL_SECONDS * 1_000_000_000
        )
        self._birth_validation_cache: dict[tuple[int, str], tuple[int, bool]] = {}
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._ensure_root_marker()
        self._publish_budget(self._requested_max_size_bytes)
        self._atexit_callback = self.close
        atexit.register(self._atexit_callback)

    @property
    def requested_max_size_bytes(self) -> int:
        return self._requested_max_size_bytes

    @property
    def last_result(self) -> GlobalBudgetResult | None:
        return self._last_result

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def _open_lock(self) -> int:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.root / _LOCK_NAME, flags, 0o600)
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            os.close(fd)
            raise OSError("global cache budget lock is not a regular file")
        return fd

    def _ensure_root_marker(self) -> None:
        marker = self.root / _ROOT_MARKER_NAME
        try:
            flags = os.O_CREAT | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(marker, flags, 0o600)
            mode = os.fstat(fd).st_mode
            os.close(fd)
            if not stat.S_ISREG(mode):
                raise OSError("managed root marker is not a regular file")
        except OSError as exc:
            raise OSError(f"cannot establish managed block-cache root: {exc}") from exc

    @contextmanager
    def mutation_guard(self) -> Iterator[bool]:
        """Hold the process-shared publication lock for a finalized write.

        The boolean is false only when the lock cannot be acquired.  Callers
        must fail the cache operation closed in that case; publishing without
        the lock would make an eviction race possible.
        """

        with self._thread_lock:
            if self._closed:
                yield False
                return
            fd: int | None = None
            try:
                fd = self._open_lock()
                fcntl.flock(fd, fcntl.LOCK_SH)
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                logger.warning(
                    "Global block-cache publication lock unavailable; "
                    "cache operation must fail closed: %s",
                    exc,
                )
                yield False
                return
            try:
                yield True
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    @contextmanager
    def _exclusive_guard(self) -> Iterator[None]:
        with self._thread_lock:
            fd = self._open_lock()
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    @contextmanager
    def exclusive_mutation_guard(self) -> Iterator[bool]:
        """Hold the root-exclusive lock across publish/delete and accounting."""

        with self._thread_lock:
            if self._closed:
                yield False
                return
            fd: int | None = None
            try:
                fd = self._open_lock()
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as exc:
                if fd is not None:
                    os.close(fd)
                logger.warning(
                    "Global block-cache exclusive mutation lock unavailable; "
                    "cache operation must fail closed: %s",
                    exc,
                )
                yield False
                return
            try:
                yield True
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _lease_dir(self) -> Path:
        return self.root / _LEASE_DIR_NAME

    def _lease_path(self) -> Path:
        return self._lease_dir() / f"{self._lease_id}.json"

    def _publish_budget(self, max_size_bytes: int) -> None:
        """Publish this live process's requested cap as a root lease."""

        try:
            with self._exclusive_guard():
                lease_dir = self._lease_dir()
                with suppress(FileExistsError):
                    lease_dir.mkdir(parents=True, exist_ok=False)
                if lease_dir.is_symlink() or not lease_dir.is_dir():
                    raise OSError(
                        f"invalid global cache budget lease directory: {lease_dir}"
                    )
                state = {
                    "version": _STATE_VERSION,
                    "max_size_bytes": max(0, int(max_size_bytes)),
                    "updated_at_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "process_birth_identity": self._lease_owner_birth_identity,
                }
                temp = self._lease_dir() / (
                    f"{self._lease_id}.{threading.get_ident()}."
                    f"{uuid.uuid4().hex}.tmp"
                )
                try:
                    fd = os.open(
                        temp,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    try:
                        payload = json.dumps(state, sort_keys=True).encode("utf-8")
                        os.write(fd, payload)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    os.replace(temp, self._lease_path())
                finally:
                    with suppress(OSError):
                        temp.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not publish global block-cache budget: %s", exc)
            # The caller can disable Block Disk L2 and continue inference, but
            # a cache store must not run without its cap/lease coordination.
            raise

    def _remove_lease(self) -> bool:
        """Compatibility alias for tests and older direct callers."""

        return self.close()

    def close(self) -> bool:
        """Release this owner's live cap after every dependent writer stops.

        ``BlockDiskStore`` owns the coordinator; an SSM companion merely shares
        it and must not close it independently.  A failed removal leaves the
        atexit callback registered so process exit can retry instead of making
        a still-live lower ceiling disappear silently.
        """

        with self._lifecycle_lock:
            if self._closed:
                return True
            if os.getpid() != self._lease_owner_pid:
                # A forked child never owns the parent's lease.
                self._closed = True
                with suppress(Exception):
                    atexit.unregister(self._atexit_callback)
                return True
            try:
                # Keep the local root lock through the closed-state flip so a
                # new shared mutation cannot slip between lease removal and
                # observing that this coordinator is no longer an owner.
                with self._thread_lock, self._exclusive_guard():
                    self._lease_path().unlink(missing_ok=True)
                    self._closed = True
            except OSError as exc:
                logger.warning(
                    "Could not remove global block-cache budget lease %s: %s",
                    self._lease_id,
                    exc,
                )
                return False
            with suppress(Exception):
                atexit.unregister(self._atexit_callback)
            return True

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _lease_birth_matches(self, pid: int, stored_birth: str) -> bool:
        """Validate PID birth at a bounded cadence while holding root lock.

        A missing birth marker is a legacy/unavailable-probe lease.  Keep it
        live while its PID exists (fail closed: a possibly stale low cap may
        remain, but the user ceiling is never silently relaxed).  New leases on
        macOS/Linux normally have a birth marker.
        """

        if not stored_birth:
            return True
        if (
            pid == self._lease_owner_pid
            and stored_birth == (self._lease_owner_birth_identity or "")
        ):
            return True
        key = (pid, stored_birth)
        now_ns = time.monotonic_ns()
        cached = self._birth_validation_cache.get(key)
        if (
            cached is not None
            and now_ns - cached[0] < self._birth_validation_interval_ns
        ):
            return cached[1]
        current_birth = _process_birth_identity(pid)
        # Failure to probe must not relax a finite cap or expose an in-flight
        # temp owned by a process that may still be alive.
        matches = current_birth is None or current_birth == stored_birth
        self._birth_validation_cache[key] = (now_ns, matches)
        return matches

    def _effective_max_size_bytes_locked(self) -> int:
        lease_dir = self._lease_dir()
        if not lease_dir.exists():
            return self._requested_max_size_bytes
        active_caps: list[int] = []
        try:
            paths = list(lease_dir.iterdir())
        except OSError as exc:
            raise OSError(f"cannot inspect global cache budget leases: {exc}") from exc
        for path in paths:
            if path.name.endswith(".tmp"):
                continue
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise OSError(f"invalid global cache budget lease path: {path}")
            try:
                state = json.loads(path.read_text())
                if int(state.get("version") or 0) != _STATE_VERSION:
                    raise ValueError("unsupported lease version")
                pid = int(state["pid"])
                cap = max(0, int(state["max_size_bytes"]))
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning(
                    "Removing invalid atomically-published cache budget lease %s: %s",
                    path,
                    exc,
                )
                with suppress(OSError):
                    path.unlink()
                continue
            if not self._pid_is_alive(pid):
                with suppress(OSError):
                    path.unlink()
                continue
            stored_birth = str(state.get("process_birth_identity") or "")
            if not self._lease_birth_matches(pid, stored_birth):
                logger.info(
                    "Removing stale cache-budget lease whose PID was reused: %s",
                    path,
                )
                with suppress(OSError):
                    path.unlink()
                continue
            active_caps.append(cap)
        if not active_caps:
            return self._requested_max_size_bytes
        finite = [cap for cap in active_caps if cap > 0]
        # Unlimited may never relax another live process's finite ceiling.
        return min(finite) if finite else 0

    def _accounting_path(self) -> Path:
        return self.root / _ACCOUNTING_NAME

    def _read_accounting_locked(self) -> dict[str, int] | None:
        accounting_path = self._accounting_path()
        if accounting_path.is_symlink():
            raise OSError("global cache accounting path is a symlink")
        try:
            state = json.loads(accounting_path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise OSError(f"invalid global cache accounting state: {exc}") from exc
        try:
            if int(state.get("version") or 0) != _STATE_VERSION:
                raise ValueError("unsupported accounting state version")
            return {
                "bytes_estimate": max(0, int(state["bytes_estimate"])),
                "accounting_generation": max(
                    0,
                    int(state.get("accounting_generation") or 0),
                ),
                "reconciliation_generation": max(
                    0,
                    int(state.get("reconciliation_generation") or 0),
                ),
                "reconciled_at_ns": max(
                    0,
                    int(state.get("reconciled_at_ns") or 0),
                ),
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise OSError(f"invalid global cache accounting values: {exc}") from exc

    def _read_or_reset_accounting_locked(self) -> dict[str, int] | None:
        """Drop corrupt derived accounting so physical scan can rebuild it."""

        try:
            return self._read_accounting_locked()
        except OSError as exc:
            logger.warning(
                "Resetting invalid derived global cache accounting state: %s",
                exc,
            )
            with suppress(OSError):
                self._accounting_path().unlink()
            return None

    def _write_accounting_locked(self, state: dict[str, int]) -> None:
        temp = self.root / (
            f"{_ACCOUNTING_NAME}.{os.getpid()}.{threading.get_ident()}."
            f"{uuid.uuid4().hex}.tmp"
        )
        payload = {
            "version": _STATE_VERSION,
            **{key: max(0, int(value)) for key, value in state.items()},
        }
        try:
            fd = os.open(
                temp,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(fd, json.dumps(payload, sort_keys=True).encode("utf-8"))
            finally:
                os.close(fd)
            os.replace(temp, self._accounting_path())
        finally:
            with suppress(OSError):
                temp.unlink(missing_ok=True)

    def account_finalized_write(
        self,
        net_bytes_delta: int,
        *,
        require_reconciled: bool = False,
    ) -> GlobalBudgetResult:
        """Safely reconcile a mutation not held in our exclusive transaction.

        A caller that already released its physical-mutation lock cannot apply
        a delta without a scan: another process may have reconciled those bytes
        in between. Fast O(1) writers use ``exclusive_mutation_guard`` plus
        ``account_finalized_write_locked`` instead.
        """

        try:
            with self._exclusive_guard():
                return self._enforce_locked()
        except (OSError, sqlite3.Error) as exc:
            return GlobalBudgetResult(
                max_size_bytes=self._requested_max_size_bytes,
                bytes_before=0,
                bytes_after=0,
                evicted_entries=0,
                evicted_bytes=0,
                protected_recent_orphans=0,
                compliant=False,
                scan_performed=False,
                reconciled_at_ns=0,
                accounted=False,
                accounting_generation=0,
                reconciliation_generation=0,
                error=str(exc),
            )

    def account_finalized_write_locked(
        self,
        net_bytes_delta: int,
        *,
        require_reconciled: bool = False,
    ) -> GlobalBudgetResult:
        """Account a mutation while ``exclusive_mutation_guard`` is held.

        Keeping final publication/deletion and ledger mutation in one root
        critical section prevents both double-add and double-subtract races
        with another process's physical reconciliation.
        """

        delta = int(net_bytes_delta)
        if delta < 0:
            return self._enforce_locked()
        maximum = self._effective_max_size_bytes_locked()
        state = self._read_or_reset_accounting_locked()
        force_reconcile = bool(require_reconciled or state is None)
        if state is None:
            state = {
                "bytes_estimate": 0,
                "accounting_generation": 0,
                "reconciliation_generation": 0,
                "reconciled_at_ns": 0,
            }
        estimate = max(0, state["bytes_estimate"] + delta)
        generation = state["accounting_generation"] + 1
        self._write_accounting_locked(
            {
                **state,
                "bytes_estimate": estimate,
                "accounting_generation": generation,
            }
        )
        interval_due = (
            state["reconciled_at_ns"] <= 0
            or time.time_ns() - state["reconciled_at_ns"]
            >= self._reconcile_interval_ns
        )
        force_reconcile = bool(
            force_reconcile
            or interval_due
            or (maximum > 0 and estimate > maximum)
        )
        if force_reconcile:
            return self._enforce_locked()
        result = GlobalBudgetResult(
            max_size_bytes=maximum,
            bytes_before=estimate,
            bytes_after=estimate,
            evicted_entries=0,
            evicted_bytes=0,
            protected_recent_orphans=0,
            compliant=maximum <= 0 or estimate <= maximum,
            scan_performed=False,
            reconciled_at_ns=state["reconciled_at_ns"],
            accounted=True,
            accounting_generation=generation,
            reconciliation_generation=state["reconciliation_generation"],
        )
        self._last_result = result
        return result

    @staticmethod
    def _is_temp_path(path: Path) -> bool:
        name = path.name
        return (
            ".tmp." in name
            or name.endswith(".tmp")
            or name.endswith(".tmp.safetensors")
            or name.endswith(".tmp.json")
        )

    def _active_lease_ids_locked(self) -> set[str]:
        active: set[str] = set()
        lease_dir = self._lease_dir()
        if not lease_dir.exists():
            return active
        for path in lease_dir.iterdir():
            if path.suffix != ".json" or path.is_symlink() or not path.is_file():
                continue
            try:
                state = json.loads(path.read_text())
                pid = int(state["pid"])
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if not self._pid_is_alive(pid):
                continue
            stored_birth = str(state.get("process_birth_identity") or "")
            if not self._lease_birth_matches(pid, stored_birth):
                continue
            active.add(path.stem)
        return active

    @staticmethod
    def _temp_owner_lease(path: Path) -> str | None:
        match = re.search(r"\.(\d+-[0-9a-f]{32})\.", path.name)
        return match.group(1) if match else None

    def _safe_resolved_file(self, path: Path) -> Path | None:
        if self._is_temp_path(path) or path.is_symlink():
            return None
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.root):
                return None
            if not resolved.is_file() or resolved.is_symlink():
                return None
            return resolved
        except OSError:
            return None

    def _managed_namespace_dirs_locked(self) -> list[Path]:
        namespaces: list[Path] = []
        root_marker = self.root / _NAMESPACE_MARKER_NAME
        if root_marker.is_symlink():
            raise OSError(f"refusing symlinked namespace marker: {root_marker}")
        if root_marker.is_file():
            namespaces.append(self.root)
        elif (
            self._allow_legacy_direct_namespace
            and not (self.root / "block_index.db").is_symlink()
            and (self.root / "block_index.db").is_file()
            and not (self.root / "blocks").is_symlink()
            and (self.root / "blocks").is_dir()
        ):
            # A pre-namespace custom root is owned cache data but unscoped, so
            # it is never reused for a new model. It still participates in the
            # physical cap and can be evicted without touching unrelated files
            # beside the recognized layout.
            namespaces.append(self.root)
        try:
            children = list(self.root.iterdir())
        except OSError as exc:
            raise OSError(f"cannot inspect managed cache root: {exc}") from exc
        for child in children:
            if child.is_symlink() or not child.is_dir():
                continue
            if child.name == _LEASE_DIR_NAME:
                continue
            child_marker = child / _NAMESPACE_MARKER_NAME
            if child_marker.is_symlink():
                raise OSError(
                    f"refusing symlinked namespace marker: {child_marker}"
                )
            if child_marker.is_file():
                namespaces.append(child.resolve())
                continue
            if (
                self._allow_legacy_hashed_namespaces
                and re.fullmatch(r"(?:[0-9a-f]{12}|default)", child.name)
                and (child / "block_index.db").is_file()
                and (child / "blocks").is_dir()
            ):
                # The built-in default root predates namespace markers. Its
                # hash/default layout is unambiguous and may be budgeted, but
                # arbitrary custom-root children never receive this exception.
                namespaces.append(child.resolve())
        return sorted(set(namespaces))

    def _database_paths_locked(self) -> list[Path]:
        databases: list[Path] = []
        for namespace in self._managed_namespace_dirs_locked():
            database = namespace / "block_index.db"
            if not database.exists():
                continue
            if database.is_symlink():
                raise OSError(f"refusing symlinked block index: {database}")
            databases.append(database)
        return databases

    def _scan_locked(
        self,
        *,
        now_ns: int,
    ) -> tuple[list[_BudgetCandidate], int, int]:
        candidates: list[_BudgetCandidate] = []
        referenced: set[Path] = set()
        protected_recent_orphans = 0
        active_lease_ids = self._active_lease_ids_locked()

        all_files: set[Path] = set()
        for namespace in self._managed_namespace_dirs_locked():
            legacy_direct = bool(
                namespace == self.root
                and not (self.root / _NAMESPACE_MARKER_NAME).is_file()
            )
            walk_roots = (
                [
                    path
                    for path in (
                        self.root / "blocks",
                        self.root / "ssm_companion",
                    )
                    if path.is_dir() and not path.is_symlink()
                ]
                if legacy_direct
                else [namespace]
            )
            if legacy_direct:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    path = Path(f"{self.root / 'block_index.db'}{suffix}")
                    if path.is_file() and not path.is_symlink():
                        all_files.add(path.resolve())
            for walk_root in walk_roots:
                for current, directories, files in os.walk(
                    walk_root,
                    followlinks=False,
                ):
                    current_path = Path(current)
                    directories[:] = [
                        name
                        for name in directories
                        if not (current_path / name).is_symlink()
                        and not (
                            namespace == self.root
                            and current_path == self.root
                            and name == _LEASE_DIR_NAME
                        )
                    ]
                    for name in files:
                        path = current_path / name
                        if name == _NAMESPACE_MARKER_NAME or (
                            namespace == self.root
                            and current_path == self.root
                            and name
                            in {
                                _LOCK_NAME,
                                _ACCOUNTING_NAME,
                                _ROOT_MARKER_NAME,
                            }
                        ):
                            continue
                        if path.is_symlink():
                            raise OSError(f"refusing symlinked cache file: {path}")
                        try:
                            resolved = path.resolve(strict=True)
                        except OSError as exc:
                            raise OSError(
                                f"cannot inspect cache file {path}: {exc}"
                            ) from exc
                        if (
                            not resolved.is_relative_to(namespace)
                            or not resolved.is_file()
                        ):
                            raise OSError(
                                f"cache file escaped managed namespace: {path}"
                            )
                        all_files.add(resolved)

        for database in self._database_paths_locked():
            namespace = database.parent.resolve()
            conn = sqlite3.connect(str(database), timeout=1.0)
            try:
                rows = conn.execute(
                    "SELECT block_hash, file_name, last_accessed FROM blocks"
                ).fetchall()
            except sqlite3.Error as exc:
                raise OSError(f"cannot inspect block index {database}: {exc}") from exc
            finally:
                conn.close()
            for block_hash, file_name, last_accessed in rows:
                relative = Path(str(file_name))
                if relative.is_absolute() or ".." in relative.parts:
                    raise OSError(f"unsafe block index path in {database}")
                resolved = self._safe_resolved_file(namespace / relative)
                if resolved is None:
                    continue
                referenced.add(resolved)
                try:
                    block_stat = resolved.stat()
                    size = block_stat.st_size
                except OSError as exc:
                    raise OSError(f"cannot stat indexed block {resolved}: {exc}") from exc
                indexed_access_ns = max(
                    0,
                    int(float(last_accessed or 0.0) * 1_000_000_000),
                )
                candidates.append(
                    _BudgetCandidate(
                        kind="block",
                        size_bytes=max(0, int(size)),
                        # A successful read touches the finalized payload under
                        # a shared root lock before returning.  This durable FS
                        # signal keeps global LRU correct even if the optional
                        # async SQLite access update queue is saturated.
                        last_accessed_ns=max(
                            indexed_access_ns,
                            block_stat.st_mtime_ns,
                        ),
                        paths=(resolved,),
                        database=database,
                        block_hash=str(block_hash),
                        indexed_file_name=str(relative),
                    )
                )

        finalized_files: list[Path] = []
        for path in all_files:
            if self._is_temp_path(path):
                continue
            if path.suffix not in {".safetensors", ".json"}:
                continue
            relative_parts = path.relative_to(self.root).parts
            if "blocks" not in relative_parts and "ssm_companion" not in relative_parts:
                continue
            finalized_files.append(path)

        finalized_set = set(finalized_files)
        for data_path in sorted(finalized_set):
            if data_path in referenced or data_path.suffix != ".safetensors":
                continue
            relative_parts = data_path.relative_to(self.root).parts
            if "ssm_companion" not in relative_parts:
                continue
            side_path = data_path.with_suffix(".json")
            if side_path not in finalized_set:
                continue
            try:
                data_stat = data_path.stat()
                side_stat = side_path.stat()
            except OSError as exc:
                raise OSError(f"cannot stat SSM companion pair: {exc}") from exc
            referenced.add(data_path)
            referenced.add(side_path)
            candidates.append(
                _BudgetCandidate(
                    kind="ssm",
                    size_bytes=max(0, data_stat.st_size) + max(0, side_stat.st_size),
                    last_accessed_ns=max(data_stat.st_mtime_ns, side_stat.st_mtime_ns),
                    paths=(data_path, side_path),
                )
            )

        for path in finalized_files:
            if path in referenced:
                continue
            try:
                file_stat = path.stat()
            except OSError as exc:
                raise OSError(f"cannot stat orphan cache payload {path}: {exc}") from exc
            age_ns = max(0, now_ns - file_stat.st_mtime_ns)
            evictable = age_ns >= self._orphan_grace_ns
            if not evictable:
                protected_recent_orphans += 1
            referenced.add(path)
            candidates.append(
                _BudgetCandidate(
                    kind="orphan" if evictable else "recent_orphan",
                    size_bytes=max(0, file_stat.st_size),
                    last_accessed_ns=file_stat.st_mtime_ns,
                    paths=(path,),
                )
            )

        # Count every other physical cache-root file—including SQLite index,
        # WAL/SHM files and active temp files—without ever selecting it for
        # deletion.  This keeps the configured max honest while preserving
        # in-flight writers and typed metadata.  A temporary overage is
        # reported non-compliant and settles on the next post-write trim.
        for path in all_files - referenced:
            try:
                file_stat = path.stat()
            except OSError as exc:
                raise OSError(f"cannot stat cache metadata {path}: {exc}") from exc
            is_temp = self._is_temp_path(path)
            temp_owner = self._temp_owner_lease(path) if is_temp else None
            age_ns = max(0, now_ns - file_stat.st_mtime_ns)
            temp_is_stale = bool(
                is_temp
                and (
                    (
                        temp_owner is not None
                        and temp_owner not in active_lease_ids
                    )
                    or (
                        temp_owner is None
                        and age_ns >= self._orphan_grace_ns
                    )
                )
            )
            candidates.append(
                _BudgetCandidate(
                    kind=(
                        "stale_temp"
                        if temp_is_stale
                        else "protected_temp"
                        if is_temp
                        else "protected_metadata"
                    ),
                    size_bytes=max(0, file_stat.st_size),
                    last_accessed_ns=file_stat.st_mtime_ns,
                    paths=(path,),
                )
            )

        total = sum(candidate.size_bytes for candidate in candidates)
        return candidates, total, protected_recent_orphans

    def _evict_block_locked(self, candidate: _BudgetCandidate) -> int:
        if candidate.database is None or candidate.block_hash is None:
            return 0
        conn = sqlite3.connect(str(candidate.database), timeout=1.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT file_name, last_accessed FROM blocks WHERE block_hash = ?",
                (candidate.block_hash,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return 0
            file_name, last_accessed = row
            current_access_ns = max(
                0,
                int(float(last_accessed or 0.0) * 1_000_000_000),
            )
            if str(file_name) != candidate.indexed_file_name:
                conn.rollback()
                return 0
            try:
                current_mtime_ns = candidate.paths[0].stat().st_mtime_ns
            except OSError:
                current_mtime_ns = 0
            if max(current_access_ns, current_mtime_ns) > candidate.last_accessed_ns:
                conn.rollback()
                return 0
            path = candidate.paths[0]
            conn.execute(
                "DELETE FROM blocks WHERE block_hash = ?",
                (candidate.block_hash,),
            )
            conn.commit()
            try:
                size = path.stat().st_size
                path.unlink()
            except FileNotFoundError:
                size = candidate.size_bytes
            except OSError:
                # The committed index deletion makes the remaining payload a
                # non-retrievable orphan. It remains counted and is eligible
                # for a later safe-orphan trim; no stale retrievable row is left.
                return 0
            return max(0, int(size))
        except sqlite3.Error:
            with suppress(sqlite3.Error):
                conn.rollback()
            return 0
        finally:
            conn.close()

    @staticmethod
    def _evict_paths_locked(candidate: _BudgetCandidate) -> int:
        freed = 0
        for path in candidate.paths:
            try:
                size = path.stat().st_size
                path.unlink()
                freed += max(0, int(size))
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return freed

    def refresh_health(self) -> GlobalBudgetResult:
        """Refresh O(1) aggregate telemetry from the shared root ledger.

        Each process owns a coordinator instance, but all writers publish into
        one accounting file.  Reading only ``last_result`` can therefore stay
        stale forever in an idle process while another process advances or
        repairs the root.  This refresh reads the shared cap leases and ledger
        under the short root lock; it never claims a physical scan occurred.
        """

        previous = self._last_result
        try:
            with self._exclusive_guard():
                maximum = self._effective_max_size_bytes_locked()
                state = self._read_accounting_locked()
                if state is None:
                    raise OSError("global cache accounting has not been reconciled")
                estimate = max(0, int(state["bytes_estimate"]))
                reconciliation_generation = max(
                    0, int(state["reconciliation_generation"])
                )
                same_local_reconciliation = bool(
                    previous is not None
                    and previous.reconciliation_generation
                    == reconciliation_generation
                )
                result = GlobalBudgetResult(
                    max_size_bytes=maximum,
                    bytes_before=estimate,
                    bytes_after=estimate,
                    evicted_entries=(
                        previous.evicted_entries
                        if same_local_reconciliation and previous is not None
                        else 0
                    ),
                    evicted_bytes=(
                        previous.evicted_bytes
                        if same_local_reconciliation and previous is not None
                        else 0
                    ),
                    protected_recent_orphans=(
                        previous.protected_recent_orphans
                        if previous is not None
                        else 0
                    ),
                    compliant=maximum <= 0 or estimate <= maximum,
                    scan_performed=bool(
                        same_local_reconciliation
                        and previous is not None
                        and previous.scan_performed
                    ),
                    reconciled_at_ns=max(0, int(state["reconciled_at_ns"])),
                    accounted=True,
                    accounting_generation=max(
                        0, int(state["accounting_generation"])
                    ),
                    reconciliation_generation=reconciliation_generation,
                )
                self._last_result = result
                return result
        except (OSError, sqlite3.Error, KeyError, TypeError, ValueError) as exc:
            result = GlobalBudgetResult(
                max_size_bytes=self._requested_max_size_bytes,
                bytes_before=0,
                bytes_after=0,
                evicted_entries=0,
                evicted_bytes=0,
                protected_recent_orphans=0,
                compliant=False,
                scan_performed=False,
                reconciled_at_ns=0,
                accounted=False,
                accounting_generation=0,
                reconciliation_generation=0,
                error=str(exc),
            )
            self._last_result = result
            return result

    def enforce(self, *, force: bool = False) -> GlobalBudgetResult:
        """Trim aggregate finalized payload bytes to the configured ceiling.

        ``0`` means unlimited.  Any uncertainty fails closed: no candidate is
        removed unless the root lock, every block index scan, and path safety
        checks succeed.
        """

        monotonic_now = time.monotonic_ns()
        with self._thread_lock:
            if (
                not force
                and self._last_result is not None
                and monotonic_now - self._last_reconcile_monotonic_ns
                < self._reconcile_interval_ns
            ):
                previous = self._last_result
                return GlobalBudgetResult(
                    max_size_bytes=previous.max_size_bytes,
                    bytes_before=previous.bytes_after,
                    bytes_after=previous.bytes_after,
                    evicted_entries=0,
                    evicted_bytes=0,
                    protected_recent_orphans=previous.protected_recent_orphans,
                    compliant=previous.compliant,
                    scan_performed=False,
                    reconciled_at_ns=previous.reconciled_at_ns,
                    accounted=previous.accounted,
                    accounting_generation=previous.accounting_generation,
                    reconciliation_generation=previous.reconciliation_generation,
                    error=previous.error,
                )

        try:
            with self._exclusive_guard():
                return self._enforce_locked()
        except (OSError, sqlite3.Error) as exc:
            result = GlobalBudgetResult(
                max_size_bytes=self._requested_max_size_bytes,
                bytes_before=0,
                bytes_after=0,
                evicted_entries=0,
                evicted_bytes=0,
                protected_recent_orphans=0,
                compliant=False,
                scan_performed=True,
                reconciled_at_ns=0,
                accounted=False,
                accounting_generation=0,
                reconciliation_generation=0,
                error=str(exc),
            )
            self._last_result = result
            logger.warning(
                "Global block-cache budget enforcement skipped without deletion: %s",
                exc,
            )
            return result

    def _enforce_locked(self) -> GlobalBudgetResult:
        """Reconcile and trim while the root-exclusive lock is held."""

        max_size_bytes = self._effective_max_size_bytes_locked()
        stored_accounting = self._read_or_reset_accounting_locked()
        accounting = stored_accounting or {
            "bytes_estimate": 0,
            "accounting_generation": 0,
            "reconciliation_generation": 0,
            "reconciled_at_ns": 0,
        }
        now_ns = time.time_ns()
        candidates, total, _protected = self._scan_locked(now_ns=now_ns)
        before = total
        evicted_entries = 0
        evicted_bytes = 0
        if max_size_bytes > 0:
            trim_target = (
                max(0, int(max_size_bytes * 0.9))
                if total > max_size_bytes
                else max_size_bytes
            )
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    item.last_accessed_ns,
                    item.kind,
                    str(item.paths[0]),
                ),
            ):
                if total <= trim_target:
                    break
                if candidate.kind in {
                    "recent_orphan",
                    "protected_temp",
                    "protected_metadata",
                }:
                    continue
                if candidate.kind == "block":
                    freed = self._evict_block_locked(candidate)
                else:
                    freed = self._evict_paths_locked(candidate)
                if freed <= 0:
                    continue
                total = max(0, total - freed)
                evicted_entries += 1
                evicted_bytes += freed

        _, after, protected_after = self._scan_locked(now_ns=time.time_ns())
        reconciled_at_ns = time.time_ns()
        reconciliation_generation = accounting["reconciliation_generation"] + 1
        self._write_accounting_locked(
            {
                "bytes_estimate": after,
                "accounting_generation": accounting["accounting_generation"],
                "reconciliation_generation": reconciliation_generation,
                "reconciled_at_ns": reconciled_at_ns,
            }
        )
        result = GlobalBudgetResult(
            max_size_bytes=max_size_bytes,
            bytes_before=before,
            bytes_after=after,
            evicted_entries=evicted_entries,
            evicted_bytes=evicted_bytes,
            protected_recent_orphans=protected_after,
            compliant=max_size_bytes <= 0 or after <= max_size_bytes,
            scan_performed=True,
            reconciled_at_ns=reconciled_at_ns,
            accounted=True,
            accounting_generation=accounting["accounting_generation"],
            reconciliation_generation=reconciliation_generation,
        )
        self._last_result = result
        self._last_reconcile_monotonic_ns = time.monotonic_ns()
        if evicted_entries:
            logger.info(
                "Global block-cache eviction: removed %d entries (%.3fGB); "
                "aggregate physical usage now %.3fGB / %.3fGB",
                evicted_entries,
                evicted_bytes / 1024**3,
                after / 1024**3,
                max_size_bytes / 1024**3,
            )
        if not result.compliant:
            logger.warning(
                "Global block-cache remains above its ceiling because only "
                "recent/in-flight or non-removable payloads remain: "
                "%.3fGB / %.3fGB",
                after / 1024**3,
                max_size_bytes / 1024**3,
            )
        return result


def get_global_disk_cache_budget(
    root: str | os.PathLike[str],
    max_size_bytes: int,
    *,
    orphan_grace_seconds: float = _DEFAULT_ORPHAN_GRACE_SECONDS,
    reconcile_interval_seconds: float = _DEFAULT_RECONCILE_INTERVAL_SECONDS,
    allow_legacy_hashed_namespaces: bool = False,
    allow_legacy_direct_namespace: bool = False,
) -> GlobalDiskCacheBudget:
    """Create one independently leased coordinator for a cache-store owner.

    Multiple stores can coexist in one process and can request different caps.
    Reusing a singleton would overwrite a lease and let the newest store relax
    an older store's finite cap, so every owner deliberately publishes its own
    lease while sharing only the in-process root lock.
    """

    return GlobalDiskCacheBudget(
        Path(root).expanduser().resolve(),
        max_size_bytes,
        orphan_grace_seconds=orphan_grace_seconds,
        reconcile_interval_seconds=reconcile_interval_seconds,
        allow_legacy_hashed_namespaces=allow_legacy_hashed_namespaces,
        allow_legacy_direct_namespace=allow_legacy_direct_namespace,
    )
