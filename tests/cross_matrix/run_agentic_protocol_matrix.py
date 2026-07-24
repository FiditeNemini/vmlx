#!/usr/bin/env python3
"""Run a real two-tool coding-harness matrix against existing vMLX endpoints.

The runner is intentionally import-safe and does not launch, stop, or mutate a
server.  Callers provide the model and both direct/gateway base URLs.  It runs
the same bounded conversation through Chat Completions, Responses, Anthropic,
and Ollama, in streaming and non-streaming modes:

    reasoning -> file_info -> real result -> reasoning -> pwd -> real result
    -> final visible synthesis

Only two allowlisted read-only tools exist.  Private reasoning and exact
decompressed parser-input response bodies are not written to the output
artifact; the artifact retains hashes, lengths, timestamps, visible text, tool
metadata, and terminal classifications.  Callers may opt into private streaming
parser-input capture with ``--raw-artifact-dir`` when stream mode is requested.
That directory must resolve outside every Git worktree, and capture metadata
retains values only for an explicit safe-header allowlist.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import stat
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

PROTOCOLS = ("chat", "responses", "anthropic", "ollama")
MODES = ("stream", "nonstream")
FILE_INFO_PATH = "panel/package.json"
PWD_COMMAND = "pwd"
CONTROL_MARKERS = (
    "<think",
    "</think",
    "[think]",
    "[/think]",
    "<mm:think",
    "<tool_call",
    "</tool_call",
    "<tool_calls",
    "</tool_calls",
    "<tool_sep>",
    "<arg_key>",
    "<arg_value>",
    "<zyphra_tool_call",
    "<function=",
    "<parameter=",
    "<|tool_call",
    "<|tool_calls",
    "[tool_calls]",
    "[tool]",
    "[calling tool:",
    "<minimax:tool_call>",
    "]<]minimax[>[",
    "<|recipient|>",
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "<｜tool▁calls▁begin｜>",
    "<｜tool▁call▁begin｜>",
    "<｜dsml｜tool",
    "<｜dsml｜invoke",
    "<|python_tag|>",
    "```tool_code",
)
SAFE_CAPTURE_HEADER_NAMES = {
    "content-encoding",
    "content-length",
    "content-type",
    "transfer-encoding",
}
CAPTURE_LAYER = "requests.decompressed_iter_lines_input"
CAPTURE_SEMANTICS = (
    "Exact decompressed streaming response-body bytes delivered to "
    "requests.iter_lines before line splitting, Unicode decoding, or protocol "
    "parsing; excludes nonstream responses, HTTP transfer framing, and "
    "compressed transport octets."
)

TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "file_info": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "run_command": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
}

TOOL_DESCRIPTIONS = {
    "file_info": "Return current filesystem metadata for the one allowed path.",
    "run_command": "Run the one allowed read-only command in the repository.",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _canonical_request_payload(value: Any) -> Any:
    """Normalize only volatile tool-call IDs for cross-base body comparison."""
    if isinstance(value, list):
        return [_canonical_request_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    object_type = str(value.get("type") or "")
    for key, item in value.items():
        if key in {"call_id", "tool_call_id", "tool_use_id"} or (
            key == "id" and object_type in {"function", "function_call", "tool_use"}
        ):
            result[key] = "<tool-call-id>"
        else:
            result[key] = _canonical_request_payload(item)
    return result


def _request_public(stage: int, payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    canonical = json.dumps(
        _canonical_request_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return {
        "stage": int(stage),
        "body_chars": len(raw),
        "body_sha256": _sha256(raw),
        "canonical_body_sha256": _sha256(canonical),
        "tool_choice": copy.deepcopy(payload.get("tool_choice")),
        "stream": bool(payload.get("stream")),
        "enable_thinking": payload.get("enable_thinking", payload.get("think")),
        "max_output_tokens": payload.get(
            "max_output_tokens",
            payload.get(
                "max_tokens", (payload.get("options") or {}).get("num_predict")
            ),
        ),
    }


def _milliseconds(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 3)


def _human_size(size: int) -> str:
    value = float(size)
    units = ("B", "KB", "MB", "GB", "TB")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.1f} {unit}"


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate if candidate.is_dir() else candidate.parent


def _is_within_directory(path: Path, parent: Path) -> bool:
    """Use filesystem identity so APFS case aliases and symlinks cannot escape."""
    try:
        guarded_parent = parent.resolve(strict=True)
    except OSError:
        guarded_parent = parent.resolve()
    existing = _nearest_existing_directory(path)
    for candidate in (existing, *existing.parents):
        try:
            if candidate.samefile(guarded_parent):
                return True
        except OSError:
            continue
    return False


def _is_in_git_context(path: Path) -> bool:
    """Return whether an existing ancestor is in any worktree or Git metadata."""
    directory = _nearest_existing_directory(path)
    environment = dict(os.environ)
    for name in ("GIT_CEILING_DIRECTORIES", "GIT_DIR", "GIT_WORK_TREE"):
        environment.pop(name, None)
    environment["LC_ALL"] = "C"
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "rev-parse",
                "--is-inside-work-tree",
                "--is-inside-git-dir",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as exc:
        raise ValueError(
            "cannot safely inspect --raw-artifact-dir for Git containment"
        ) from exc
    if proc.returncode != 0:
        if "not a git repository" in proc.stderr.lower():
            return False
        raise ValueError("cannot safely inspect --raw-artifact-dir for Git containment")
    return any(line.strip().lower() == "true" for line in proc.stdout.splitlines())


def _reject_capture_destination(path: Path, guarded_worktree: Path) -> None:
    if _is_within_directory(path, guarded_worktree) or _is_in_git_context(path):
        raise ValueError(
            "--raw-artifact-dir must resolve outside every Git worktree "
            "and Git metadata directory"
        )


def _git_worktree_root(repo_root: Path) -> Path:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ValueError(
            "cannot enable raw capture without identifying the Git worktree"
        )
    return Path(proc.stdout.strip()).resolve()


def _sanitized_headers(headers: Any) -> list[dict[str, str]]:
    if headers is None:
        return []
    iterator = getattr(headers, "iteritems", None)
    pairs = iterator() if callable(iterator) else headers.items()
    return [
        {
            "name": str(name),
            "value": (
                str(value)
                if str(name).strip().lower().replace("_", "-")
                in SAFE_CAPTURE_HEADER_NAMES
                else "<redacted>"
            ),
        }
        for name, value in pairs
    ]


def _safe_request_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def _prepared_body_bytes(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    return str(body).encode("utf-8", errors="replace")


def _safe_capture_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    )
    return label.strip("-")[:80] or "request"


def _capture_route(
    base_label: str,
    protocol: str,
    capture_label: str,
) -> dict[str, str]:
    return {
        "base_label": str(base_label),
        "protocol": str(protocol),
        "capture_label": str(capture_label),
    }


def _capture_route_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("base_label") or ""),
        str(row.get("protocol") or ""),
        str(row.get("capture_label") or ""),
    )


class DecompressedParserInputCaptureSession:
    """Write decompressed bytes before line splitting, decoding, or parsing."""

    def __init__(
        self,
        *,
        body_path: Path,
        metadata_path: Path,
        base_label: str,
        protocol: str,
        capture_label: str,
        payload: dict[str, Any],
        response: requests.Response,
        started: float,
        started_at: str,
        on_finished: Callable[[str | None], None],
        on_finish_error: Callable[[str], None],
    ) -> None:
        descriptor = os.open(
            body_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        try:
            body_file = os.fdopen(descriptor, "wb")
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(OSError):
                body_path.unlink(missing_ok=True)
            raise
        try:
            self._body_file = body_file
            self._body_path = body_path
            self._metadata_path = metadata_path
            self._started = started
            self._digest = hashlib.sha256()
            self._bytes = 0
            self._first_byte_ms: float | None = None
            self._last_byte_ms: float | None = None
            self._finished = False
            self._on_finished = on_finished
            self._on_finish_error = on_finish_error

            prepared = response.request
            request_body = _prepared_body_bytes(
                prepared.body if prepared is not None else None
            )
            request_public = _request_public(0, payload)
            request_public.pop("stage", None)
            response_headers = getattr(response.raw, "headers", None)
            if response_headers is None:
                response_headers = response.headers
            self._metadata: dict[str, Any] = {
                "schema_version": 1,
                "capture_layer": CAPTURE_LAYER,
                "capture_semantics": CAPTURE_SEMANTICS,
                "base_label": base_label,
                "protocol": protocol,
                "capture_label": capture_label,
                "started_at": started_at,
                "request": {
                    "method": str(prepared.method if prepared is not None else "POST"),
                    "url": _safe_request_url(
                        str(prepared.url if prepared is not None else "")
                    ),
                    "headers": _sanitized_headers(
                        prepared.headers if prepared is not None else {}
                    ),
                    "body_bytes": len(request_body),
                    "body_sha256": hashlib.sha256(request_body).hexdigest(),
                    "payload": {
                        **request_public,
                        "model": str(payload.get("model") or ""),
                        "top_level_fields": sorted(str(key) for key in payload),
                    },
                },
                "response": {
                    "status_code": int(response.status_code),
                    "headers": _sanitized_headers(response_headers),
                    "headers_received_ms": _milliseconds(started),
                    "body_file": body_path.name,
                },
            }
        except BaseException:
            with suppress(Exception):
                body_file.close()
            with suppress(OSError):
                body_path.unlink(missing_ok=True)
            raise

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        now_ms = _milliseconds(self._started)
        if self._first_byte_ms is None:
            self._first_byte_ms = now_ms
        self._last_byte_ms = now_ms
        self._body_file.write(chunk)
        self._digest.update(chunk)
        self._bytes += len(chunk)

    def finish(
        self,
        *,
        completed_ms: float,
        error_type: str | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            try:
                self._body_file.flush()
            finally:
                self._body_file.close()
            response = self._metadata["response"]
            response.update(
                {
                    "first_byte_ms": self._first_byte_ms,
                    "last_byte_ms": self._last_byte_ms,
                    "completed_ms": completed_ms,
                    "body_bytes": self._bytes,
                    "body_sha256": self._digest.hexdigest(),
                }
            )
            if error_type:
                response["capture_error_type"] = error_type
            metadata_bytes = (
                json.dumps(self._metadata, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            descriptor = os.open(
                self._metadata_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(descriptor, "wb") as metadata_file:
                    metadata_file.write(metadata_bytes)
            except BaseException:
                self._metadata_path.unlink(missing_ok=True)
                raise
        except BaseException as exc:
            cleanup_error: OSError | None = None
            for artifact_path in (self._metadata_path, self._body_path):
                try:
                    artifact_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    if cleanup_error is None:
                        cleanup_error = cleanup_exc
            self._on_finish_error(type(exc).__name__)
            if cleanup_error is not None:
                raise cleanup_error from exc
            raise
        self._on_finished(error_type)


class DecompressedParserInputCaptureRecorder:
    """Allocate private parser-input captures outside every Git worktree."""

    def __init__(
        self,
        artifact_root: Path,
        git_worktree: Path,
        *,
        run_id: str | None = None,
    ) -> None:
        worktree = git_worktree.resolve(strict=True)
        root = artifact_root.expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise ValueError("--raw-artifact-dir must name a directory")
        _reject_capture_destination(root, worktree)
        root_existed = root.exists()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root = root.resolve(strict=True)
        try:
            _reject_capture_destination(root, worktree)
        except BaseException:
            if not root_existed:
                with suppress(OSError):
                    root.rmdir()
            raise
        identifier = run_id or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-pid{os.getpid()}"
        )
        self.run_id = _safe_capture_label(identifier)
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(mode=0o700)
        self.manifest_path = self.run_dir / "manifest.json"
        self._counter = 0
        self._expected_configured = False
        self._expected: list[dict[str, str]] = []
        self._started: list[dict[str, Any]] = []
        self._finished: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._final_summary: dict[str, Any] | None = None

    def configure_expected(
        self,
        routes: Iterable[tuple[str, str, str]],
    ) -> None:
        if self._expected_configured or self._counter:
            raise RuntimeError(
                "capture expectations must be configured once before use"
            )
        self._expected = [
            _capture_route(base_label, protocol, capture_label)
            for base_label, protocol, capture_label in routes
        ]
        keys = [_capture_route_key(row) for row in self._expected]
        if len(keys) != len(set(keys)):
            raise ValueError("capture expectations contain duplicate route labels")
        self._expected_configured = True

    def _record_error(
        self,
        route: dict[str, str],
        *,
        phase: str,
        error_type: str,
        sequence: int | None = None,
    ) -> None:
        row: dict[str, Any] = {
            **route,
            "phase": phase,
            "error_type": error_type,
        }
        if sequence is not None:
            row["sequence"] = sequence
        self._errors.append(row)

    def _record_finished(
        self,
        started_row: dict[str, Any],
        error_type: str | None,
    ) -> None:
        self._finished.append(dict(started_row))
        if error_type:
            self._record_error(
                _capture_route(
                    started_row["base_label"],
                    started_row["protocol"],
                    started_row["capture_label"],
                ),
                phase="stream_or_parse",
                error_type=error_type,
                sequence=int(started_row["sequence"]),
            )

    def _record_finish_error(
        self,
        started_row: dict[str, Any],
        error_type: str,
    ) -> None:
        self._record_error(
            _capture_route(
                started_row["base_label"],
                started_row["protocol"],
                started_row["capture_label"],
            ),
            phase="finish",
            error_type=error_type,
            sequence=int(started_row["sequence"]),
        )

    def begin(
        self,
        *,
        base_label: str,
        protocol: str,
        capture_label: str,
        payload: dict[str, Any],
        response: requests.Response,
        started: float,
        started_at: str,
    ) -> DecompressedParserInputCaptureSession:
        if self._final_summary is not None:
            raise RuntimeError("capture recorder is already finalized")
        self._counter += 1
        route = _capture_route(base_label, protocol, capture_label)
        stem = "-".join(
            (
                f"{self._counter:04d}",
                _safe_capture_label(base_label),
                _safe_capture_label(protocol),
                _safe_capture_label(capture_label),
            )
        )
        started_row: dict[str, Any] = {
            **route,
            "sequence": self._counter,
            "body_file": f"{stem}.decompressed-parser-input.bin",
            "metadata_file": f"{stem}.metadata.json",
        }
        self._started.append(started_row)
        if self._expected_configured:
            expected = Counter(_capture_route_key(row) for row in self._expected)
            actual = Counter(_capture_route_key(row) for row in self._started)
            route_key = _capture_route_key(route)
            if route_key not in expected or actual[route_key] > expected[route_key]:
                self._record_error(
                    route,
                    phase="setup",
                    error_type="UnexpectedCaptureRoute",
                    sequence=self._counter,
                )
                raise ValueError(
                    "capture route was not expected or was started more than once"
                )
        try:
            return DecompressedParserInputCaptureSession(
                body_path=self.run_dir / started_row["body_file"],
                metadata_path=self.run_dir / started_row["metadata_file"],
                base_label=base_label,
                protocol=protocol,
                capture_label=capture_label,
                payload=payload,
                response=response,
                started=started,
                started_at=started_at,
                on_finished=lambda error_type: self._record_finished(
                    started_row,
                    error_type,
                ),
                on_finish_error=lambda error_type: self._record_finish_error(
                    started_row,
                    error_type,
                ),
            )
        except BaseException as exc:
            self._record_error(
                route,
                phase="setup",
                error_type=type(exc).__name__,
                sequence=self._counter,
            )
            raise

    def _summary(self) -> dict[str, Any]:
        expected_counts = Counter(_capture_route_key(row) for row in self._expected)
        started_counts = Counter(_capture_route_key(row) for row in self._started)
        finished_counts = Counter(_capture_route_key(row) for row in self._finished)
        ordered_keys = list(expected_counts)
        ordered_keys.extend(key for key in started_counts if key not in expected_counts)
        for error in self._errors:
            key = _capture_route_key(error)
            if key not in ordered_keys:
                ordered_keys.append(key)
        routes = []
        for key in ordered_keys:
            base_label, protocol, capture_label = key
            route_errors = [
                {
                    "phase": row["phase"],
                    "error_type": row["error_type"],
                    **({"sequence": row["sequence"]} if "sequence" in row else {}),
                }
                for row in self._errors
                if _capture_route_key(row) == key
            ]
            artifacts = [
                {
                    "sequence": row["sequence"],
                    "body_file": row["body_file"],
                    "metadata_file": row["metadata_file"],
                }
                for row in self._started
                if _capture_route_key(row) == key
            ]
            routes.append(
                {
                    "base_label": base_label,
                    "protocol": protocol,
                    "capture_label": capture_label,
                    "expected": expected_counts[key],
                    "started": started_counts[key],
                    "finished": finished_counts[key],
                    "errors": route_errors,
                    "artifacts": artifacts,
                }
            )
        complete = (
            self._expected_configured
            and started_counts == expected_counts
            and finished_counts == expected_counts
            and not self._errors
        )
        return {
            "schema_version": 1,
            "enabled": True,
            "capture_layer": CAPTURE_LAYER,
            "capture_semantics": CAPTURE_SEMANTICS,
            "run_id": self.run_id,
            "expected": len(self._expected),
            "started": len(self._started),
            "finished": len(self._finished),
            "errors": len(self._errors),
            "complete": complete,
            "routes": routes,
        }

    def current_summary(self) -> dict[str, Any]:
        return self._summary()

    def finalize(self) -> dict[str, Any]:
        if self._final_summary is not None:
            return dict(self._final_summary)
        manifest = self._summary()
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        try:
            descriptor = os.open(
                self.manifest_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                with os.fdopen(descriptor, "wb") as manifest_file:
                    manifest_file.write(manifest_bytes)
            except BaseException:
                self.manifest_path.unlink(missing_ok=True)
                raise
        except BaseException as exc:
            self.manifest_path.unlink(missing_ok=True)
            self._record_error(
                _capture_route("", "", ""),
                phase="manifest",
                error_type=type(exc).__name__,
            )
            raise
        self._final_summary = {
            **manifest,
            "manifest_file": self.manifest_path.name,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        return dict(self._final_summary)


class _CapturingDecompressedStream:
    """Tee decompressed bytes before requests splits them into logical lines."""

    def __init__(
        self,
        raw: Any,
        capture: DecompressedParserInputCaptureSession,
    ) -> None:
        self._raw = raw
        self._capture = capture

    def stream(
        self,
        amount: int | None = None,
        decode_content: bool | None = None,
    ) -> Iterable[bytes]:
        for chunk in self._raw.stream(
            amount,
            decode_content=decode_content,
        ):
            capture_bytes = (
                chunk.encode("utf-8", errors="replace")
                if isinstance(chunk, str)
                else bytes(chunk)
            )
            self._capture.write(capture_bytes)
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _merge_name_fragment(existing: str, fragment: str) -> str:
    """Merge split names without duplicating full-name retransmissions."""
    if not fragment:
        return existing
    if not existing:
        return fragment
    if fragment.startswith(existing):
        return fragment
    if existing.endswith(fragment):
        return existing
    return existing + fragment


class FragmentedToolAssembler:
    """Accumulate OpenAI/Anthropic streamed function-call fragments by index."""

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, Any]] = {}

    def add(
        self,
        index: int,
        *,
        call_id: str = "",
        name: str = "",
        arguments: Any = None,
        complete: bool = False,
    ) -> None:
        target = self._parts.setdefault(
            int(index),
            {"id": "", "name": "", "arguments_text": "", "arguments_object": None},
        )
        if call_id:
            target["id"] = call_id
        target["name"] = _merge_name_fragment(str(target["name"]), str(name or ""))
        if isinstance(arguments, dict):
            target["arguments_object"] = dict(arguments)
            if complete:
                target["arguments_text"] = ""
        elif arguments is not None:
            fragment = str(arguments)
            if complete:
                target["arguments_text"] = fragment
                target["arguments_object"] = None
            else:
                target["arguments_text"] += fragment

    def calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index in sorted(self._parts):
            part = self._parts[index]
            raw: Any = part["arguments_object"]
            if raw is None:
                raw = part["arguments_text"] or "{}"
            try:
                arguments = _parse_arguments(raw)
                parse_error = None
            except Exception as exc:
                arguments = {}
                parse_error = str(exc)
            call = {
                "index": index,
                "id": str(part["id"]),
                "name": str(part["name"]),
                "arguments": arguments,
            }
            if parse_error:
                call["arguments_parse_error"] = parse_error
                call["arguments_sha256"] = _sha256(str(raw))
            calls.append(call)
        return calls


@dataclass
class EventCollector:
    protocol: str
    started: float
    events: list[dict[str, Any]] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    content_parts: list[str] = field(default_factory=list)
    terminals: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    response_id: str = ""
    tools: FragmentedToolAssembler = field(default_factory=FragmentedToolAssembler)

    def text(
        self, channel: str, text: str, kind: str, at_ms: float | None = None
    ) -> None:
        if not text:
            return
        if channel == "reasoning":
            self.reasoning_parts.append(text)
        elif channel == "content":
            self.content_parts.append(text)
        else:
            raise ValueError(f"unsupported text channel: {channel}")
        self.events.append(
            {
                "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
                "channel": channel,
                "kind": kind,
                "chars": len(text),
                "sha256": _sha256(text),
            }
        )

    def tool_fragment(
        self,
        index: int,
        *,
        call_id: str = "",
        name: str = "",
        arguments: Any = None,
        kind: str,
        complete: bool = False,
        at_ms: float | None = None,
    ) -> None:
        self.tools.add(
            index,
            call_id=call_id,
            name=name,
            arguments=arguments,
            complete=complete,
        )
        argument_text = (
            ""
            if arguments is None
            else json.dumps(arguments, sort_keys=True)
            if isinstance(arguments, dict)
            else str(arguments)
        )
        self.events.append(
            {
                "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
                "channel": "tool",
                "kind": kind,
                "index": int(index),
                "call_id": call_id,
                "name_fragment": name,
                "argument_chars": len(argument_text),
                "argument_sha256": _sha256(argument_text),
            }
        )

    def terminal(self, value: str, at_ms: float | None = None) -> None:
        if not value:
            return
        self.terminals.append(value)
        self.events.append(
            {
                "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
                "channel": "terminal",
                "kind": value,
            }
        )

    def error(self, kind: str, detail: str = "", at_ms: float | None = None) -> None:
        row = {
            "at_ms": _milliseconds(self.started) if at_ms is None else at_ms,
            "channel": "error",
            "kind": kind,
        }
        if detail:
            row["detail_chars"] = len(detail)
            row["detail_sha256"] = _sha256(detail)
        self.errors.append(dict(row))
        self.events.append(row)

    def result(self, status_code: int, elapsed_ms: float) -> dict[str, Any]:
        return {
            "status_code": int(status_code),
            "elapsed_ms": elapsed_ms,
            "response_id": self.response_id,
            "reasoning": "".join(self.reasoning_parts),
            "content": "".join(self.content_parts),
            "tool_calls": self.tools.calls(),
            "terminals": list(self.terminals),
            "errors": list(self.errors),
            "events": list(self.events),
        }


def tool_schemas(protocol: str, names: Iterable[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in names:
        parameters = TOOL_PARAMETERS[name]
        if protocol == "responses":
            result.append(
                {
                    "type": "function",
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "parameters": parameters,
                }
            )
        elif protocol == "anthropic":
            result.append(
                {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS[name],
                    "input_schema": parameters,
                }
            )
        else:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": TOOL_DESCRIPTIONS[name],
                        "parameters": parameters,
                    },
                }
            )
    return result


def tool_choice(
    protocol: str, mode: str, stage: int, second_tool_choice: str = "auto"
) -> Any:
    """Return native choice for tool stage 1/2 or final stage 3."""
    if protocol == "ollama":
        return None
    if stage == 3:
        return {"type": "none"} if protocol == "anthropic" else "none"
    if stage == 1 and mode == "stream":
        if protocol == "chat":
            return {"type": "function", "function": {"name": "file_info"}}
        if protocol == "responses":
            return {"type": "function", "name": "file_info"}
        return {"type": "tool", "name": "file_info"}
    if stage == 2:
        if second_tool_choice == "explicit":
            if protocol == "chat":
                return {"type": "function", "function": {"name": "run_command"}}
            if protocol == "responses":
                return {"type": "function", "name": "run_command"}
            return {"type": "tool", "name": "run_command"}
        if second_tool_choice == "required":
            return {"type": "any"} if protocol == "anthropic" else "required"
        return {"type": "auto"} if protocol == "anthropic" else "auto"
    return {"type": "any"} if protocol == "anthropic" else "required"


def validate_allowlisted_call(
    call: dict[str, Any], expected_name: str
) -> tuple[bool, str]:
    if call.get("arguments_parse_error"):
        return False, "arguments were not valid JSON"
    if call.get("name") != expected_name:
        return False, f"expected {expected_name}, got {call.get('name')!r}"
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return False, "arguments are not an object"
    expected = (
        {"path": FILE_INFO_PATH}
        if expected_name == "file_info"
        else {"command": PWD_COMMAND}
    )
    if arguments != expected:
        return False, f"arguments must equal {expected!r}, got {arguments!r}"
    if not call.get("id"):
        return False, "tool call has no id"
    return True, ""


def execute_allowlisted_tool(repo_root: Path, call: dict[str, Any]) -> dict[str, Any]:
    ok, error = validate_allowlisted_call(call, str(call.get("name") or ""))
    if not ok:
        raise ValueError(error)
    name = str(call["name"])
    if name == "file_info":
        target = (repo_root / FILE_INFO_PATH).resolve()
        expected = (repo_root.resolve() / FILE_INFO_PATH).resolve()
        if target != expected or not target.is_file():
            raise ValueError(f"allowed file is unavailable: {expected}")
        info = target.stat()
        result = {
            "path": FILE_INFO_PATH,
            "type": "file",
            "size_bytes": info.st_size,
            "size_human": _human_size(info.st_size),
            "modified_utc": datetime.fromtimestamp(info.st_mtime, tz=UTC).isoformat(),
            "permissions": f"{stat.S_IMODE(info.st_mode):04o}",
        }
    elif name == "run_command":
        completed = subprocess.run(
            [PWD_COMMAND],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"pwd failed with exit code {completed.returncode}")
        result = {
            "command": PWD_COMMAND,
            "stdout": completed.stdout.strip(),
            "exit_code": completed.returncode,
        }
    else:
        raise ValueError(f"tool is not allowlisted: {name}")
    output = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return {
        "name": name,
        "call_id": call["id"],
        "arguments": call["arguments"],
        "result": result,
        "output": output,
    }


def assistant_message(protocol: str, round_result: dict[str, Any]) -> dict[str, Any]:
    calls = round_result.get("tool_calls") or []
    if protocol == "chat":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": round_result.get("content") or "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call["arguments"], separators=(",", ":")
                        ),
                    },
                }
                for call in calls
            ],
        }
        if round_result.get("reasoning"):
            message["reasoning_content"] = round_result["reasoning"]
        return message
    if protocol == "anthropic":
        blocks: list[dict[str, Any]] = []
        if round_result.get("reasoning"):
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": round_result["reasoning"],
                    "signature": "dm1seA==",
                }
            )
        if round_result.get("content"):
            blocks.append({"type": "text", "text": round_result["content"]})
        blocks.extend(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": call["arguments"],
            }
            for call in calls
        )
        return {"role": "assistant", "content": blocks}
    if protocol == "ollama":
        message = {
            "role": "assistant",
            "content": round_result.get("content") or "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in calls
            ],
        }
        if round_result.get("reasoning"):
            message["thinking"] = round_result["reasoning"]
        return message
    raise ValueError(
        f"Responses uses function_call_output, not assistant_message: {protocol}"
    )


def history_after_tool(
    protocol: str,
    history: list[dict[str, Any]],
    round_result: dict[str, Any],
    execution: dict[str, Any],
    next_instruction: str,
) -> list[dict[str, Any]]:
    """Return protocol-native history after one real tool result."""
    if protocol == "responses":
        return [
            {
                "type": "function_call_output",
                "call_id": execution["call_id"],
                "output": execution["output"],
            },
            {"role": "user", "content": next_instruction},
        ]
    result = [*history, assistant_message(protocol, round_result)]
    if protocol == "chat":
        result.extend(
            [
                {
                    "role": "tool",
                    "tool_call_id": execution["call_id"],
                    "name": execution["name"],
                    "content": execution["output"],
                },
                {"role": "user", "content": next_instruction},
            ]
        )
    elif protocol == "anthropic":
        result.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": execution["call_id"],
                        "content": execution["output"],
                    },
                    {"type": "text", "text": next_instruction},
                ],
            }
        )
    elif protocol == "ollama":
        result.extend(
            [
                {
                    "role": "tool",
                    "tool_name": execution["name"],
                    "content": execution["output"],
                },
                {"role": "user", "content": next_instruction},
            ]
        )
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return result


def classify_terminal(
    protocol: str,
    terminals: list[str],
    *,
    stream: bool,
    expect_tool: bool,
) -> dict[str, Any]:
    """Classify one protocol-native terminal without conflating SSE framing."""
    values = [str(value) for value in terminals if value]
    if protocol == "chat":
        done = values.count("DONE")
        semantic = [value for value in values if value != "DONE"]
        expected = "tool_calls" if expect_tool else "stop"
        passed = semantic == [expected] and (done == 1 if stream else done == 0)
    elif protocol == "responses":
        semantic = values
        passed = semantic == ["response.completed"]
    elif protocol == "anthropic":
        message_stop = values.count("message_stop")
        semantic = [value for value in values if value != "message_stop"]
        expected = "tool_use" if expect_tool else "end_turn"
        passed = semantic == [expected] and (
            message_stop == 1 if stream else message_stop == 0
        )
    elif protocol == "ollama":
        semantic = values
        expected = "tool_calls" if expect_tool else "stop"
        passed = semantic == [expected]
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return {"pass": passed, "values": values, "semantic": semantic}


def _parse_stream_object(
    protocol: str,
    data: dict[str, Any],
    event_name: str | None,
    collector: EventCollector,
    at_ms: float,
) -> None:
    if not collector.response_id:
        response = data.get("response")
        message = data.get("message")
        collector.response_id = str(
            data.get("id")
            or data.get("response_id")
            or (response.get("id") if isinstance(response, dict) else "")
            or (message.get("id") if isinstance(message, dict) else "")
            or ""
        )
    if protocol == "chat":
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            collector.text(
                "reasoning",
                str(delta.get("reasoning_content") or delta.get("reasoning") or ""),
                "chat.reasoning.delta",
                at_ms,
            )
            collector.text(
                "content", str(delta.get("content") or ""), "chat.content.delta", at_ms
            )
            for fragment in delta.get("tool_calls") or []:
                function = fragment.get("function") or {}
                collector.tool_fragment(
                    int(fragment.get("index") or 0),
                    call_id=str(fragment.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=function.get("arguments"),
                    kind="chat.tool.delta",
                    at_ms=at_ms,
                )
            if choice.get("finish_reason") is not None:
                collector.terminal(str(choice["finish_reason"]), at_ms)
        return

    kind = str(data.get("type") or event_name or "")
    if protocol == "responses":
        if kind in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning.delta",
            "response.output_text.reasoning_delta",
        }:
            collector.text("reasoning", str(data.get("delta") or ""), kind, at_ms)
        elif kind == "response.output_text.delta":
            collector.text("content", str(data.get("delta") or ""), kind, at_ms)
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                collector.tool_fragment(
                    int(data.get("output_index") or 0),
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=item.get("arguments"),
                    kind=kind,
                    complete=kind == "response.output_item.done",
                    at_ms=at_ms,
                )
        elif kind == "response.function_call_arguments.delta":
            collector.tool_fragment(
                int(data.get("output_index") or 0),
                call_id=str(data.get("call_id") or ""),
                arguments=data.get("delta"),
                kind=kind,
                at_ms=at_ms,
            )
        elif kind == "response.function_call_arguments.done":
            collector.tool_fragment(
                int(data.get("output_index") or 0),
                call_id=str(data.get("call_id") or ""),
                arguments=data.get("arguments"),
                kind=kind,
                complete=True,
                at_ms=at_ms,
            )
        elif kind == "error":
            collector.error(kind, json.dumps(data, sort_keys=True), at_ms)
        elif kind in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }:
            collector.terminal(kind, at_ms)
        return

    if protocol == "anthropic":
        delta = data.get("delta") or {}
        if kind == "message_start":
            message = data.get("message") or {}
            collector.response_id = collector.response_id or str(
                message.get("id") or ""
            )
        elif kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                collector.tool_fragment(
                    int(data.get("index") or 0),
                    call_id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=block.get("input") if block.get("input") else None,
                    kind=kind,
                    complete=bool(block.get("input")),
                    at_ms=at_ms,
                )
        elif kind == "content_block_delta" and delta.get("type") in {
            "thinking_delta",
            "reasoning_delta",
        }:
            collector.text(
                "reasoning",
                str(delta.get("thinking") or delta.get("reasoning") or ""),
                str(delta.get("type")),
                at_ms,
            )
        elif kind == "content_block_delta" and delta.get("type") == "text_delta":
            collector.text("content", str(delta.get("text") or ""), "text_delta", at_ms)
        elif kind == "content_block_delta" and delta.get("type") == "input_json_delta":
            collector.tool_fragment(
                int(data.get("index") or 0),
                arguments=delta.get("partial_json"),
                kind="input_json_delta",
                at_ms=at_ms,
            )
        elif kind == "message_delta" and delta.get("stop_reason"):
            collector.terminal(str(delta["stop_reason"]), at_ms)
        elif kind == "error":
            collector.error(kind, json.dumps(data, sort_keys=True), at_ms)
            collector.terminal(kind, at_ms)
        elif kind == "message_stop":
            collector.terminal(kind, at_ms)
        return

    if protocol == "ollama":
        if data.get("error"):
            collector.error("ollama.error", str(data.get("error")), at_ms)
        message = data.get("message") or {}
        collector.text(
            "reasoning",
            str(message.get("thinking") or message.get("reasoning") or ""),
            "ollama.thinking",
            at_ms,
        )
        collector.text(
            "content", str(message.get("content") or ""), "ollama.content", at_ms
        )
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            collector.tool_fragment(
                index,
                call_id=str(call.get("id") or f"ollama_call_{index}"),
                name=str(function.get("name") or ""),
                arguments=function.get("arguments") or {},
                kind="ollama.tool",
                complete=True,
                at_ms=at_ms,
            )
        if data.get("done"):
            collector.terminal(str(data.get("done_reason") or "stop"), at_ms)
        return

    raise ValueError(f"unknown protocol: {protocol}")


def parse_nonstream(
    protocol: str, body: dict[str, Any], status_code: int, elapsed_ms: float
) -> dict[str, Any]:
    started = time.monotonic()
    collector = EventCollector(protocol=protocol, started=started)
    collector.response_id = str(body.get("id") or "")
    if protocol == "chat":
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        collector.text(
            "reasoning",
            str(message.get("reasoning_content") or message.get("reasoning") or ""),
            "chat.reasoning.complete",
            elapsed_ms,
        )
        collector.text(
            "content",
            str(message.get("content") or ""),
            "chat.content.complete",
            elapsed_ms,
        )
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            collector.tool_fragment(
                index,
                call_id=str(call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=function.get("arguments"),
                kind="chat.tool.complete",
                complete=True,
                at_ms=elapsed_ms,
            )
        collector.terminal(str(choice.get("finish_reason") or ""), elapsed_ms)
    elif protocol == "responses":
        for item_index, item in enumerate(body.get("output") or []):
            if item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    collector.text(
                        "reasoning",
                        str(summary.get("text") or ""),
                        "responses.reasoning.complete",
                        elapsed_ms,
                    )
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") in {"output_text", "text"}:
                        collector.text(
                            "content",
                            str(part.get("text") or ""),
                            "responses.content.complete",
                            elapsed_ms,
                        )
            elif item.get("type") == "function_call":
                collector.tool_fragment(
                    item_index,
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=item.get("arguments"),
                    kind="responses.tool.complete",
                    complete=True,
                    at_ms=elapsed_ms,
                )
        collector.terminal(f"response.{body.get('status') or 'completed'}", elapsed_ms)
    elif protocol == "anthropic":
        for index, block in enumerate(body.get("content") or []):
            if block.get("type") in {"thinking", "reasoning"}:
                collector.text(
                    "reasoning",
                    str(
                        block.get("thinking")
                        or block.get("reasoning")
                        or block.get("text")
                        or ""
                    ),
                    "anthropic.reasoning.complete",
                    elapsed_ms,
                )
            elif block.get("type") == "text":
                collector.text(
                    "content",
                    str(block.get("text") or ""),
                    "anthropic.content.complete",
                    elapsed_ms,
                )
            elif block.get("type") == "tool_use":
                collector.tool_fragment(
                    index,
                    call_id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=block.get("input") or {},
                    kind="anthropic.tool.complete",
                    complete=True,
                    at_ms=elapsed_ms,
                )
        collector.terminal(str(body.get("stop_reason") or ""), elapsed_ms)
    elif protocol == "ollama":
        _parse_stream_object(protocol, body, None, collector, elapsed_ms)
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return collector.result(status_code, elapsed_ms)


class ProtocolClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: int,
        *,
        base_label: str = "",
        raw_recorder: DecompressedParserInputCaptureRecorder | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.base_label = base_label
        self.raw_recorder = raw_recorder
        self.headers = {"content-type": "application/json"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"

    @staticmethod
    def route(protocol: str) -> str:
        return {
            "chat": "/v1/chat/completions",
            "responses": "/v1/responses",
            "anthropic": "/v1/messages",
            "ollama": "/api/chat",
        }[protocol]

    def _attach_parser_input_capture(
        self,
        *,
        protocol: str,
        capture_label: str,
        payload: dict[str, Any],
        response: requests.Response,
        started: float,
        started_at: str,
    ) -> DecompressedParserInputCaptureSession | None:
        if self.raw_recorder is None:
            return None
        capture = self.raw_recorder.begin(
            base_label=self.base_label,
            protocol=protocol,
            capture_label=capture_label,
            payload=payload,
            response=response,
            started=started,
            started_at=started_at,
        )
        try:
            response.raw = _CapturingDecompressedStream(response.raw, capture)
        except BaseException as exc:
            capture.finish(
                completed_ms=_milliseconds(started),
                error_type=type(exc).__name__,
            )
            raise
        return capture

    def send(
        self,
        protocol: str,
        payload: dict[str, Any],
        stream: bool,
        *,
        capture_label: str = "request",
    ) -> dict[str, Any]:
        started = time.monotonic()
        started_at = datetime.now(UTC).isoformat()
        response = requests.post(
            self.base_url + self.route(protocol),
            headers=self.headers,
            json=payload,
            stream=stream,
            timeout=(15, self.timeout),
        )
        if not stream:
            elapsed = _milliseconds(started)
            try:
                body = response.json()
            except Exception:
                body = {"error": response.text[:2000]}
            return parse_nonstream(protocol, body, response.status_code, elapsed)

        collector = EventCollector(protocol=protocol, started=started)
        event_name: str | None = None
        capture: DecompressedParserInputCaptureSession | None = None
        capture_error_type: str | None = None
        completed_ms = 0.0
        try:
            capture = self._attach_parser_input_capture(
                protocol=protocol,
                capture_label=capture_label,
                payload=payload,
                response=response,
                started=started,
                started_at=started_at,
            )
            for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
                if raw is None:
                    continue
                line = (
                    raw.decode("utf-8", errors="replace")
                    if isinstance(raw, bytes)
                    else raw
                )
                line = line.strip()
                if not line:
                    event_name = None
                    continue
                at_ms = _milliseconds(started)
                if protocol != "ollama" and line.startswith("event: "):
                    event_name = line[7:]
                    continue
                if protocol != "ollama":
                    if not line.startswith("data: "):
                        continue
                    raw_data = line[6:]
                    if raw_data == "[DONE]":
                        collector.terminal("DONE", at_ms)
                        continue
                else:
                    raw_data = line
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    collector.error("json_parse_error", raw_data, at_ms)
                    continue
                _parse_stream_object(
                    protocol,
                    data,
                    event_name,
                    collector,
                    at_ms,
                )
        except BaseException as exc:
            capture_error_type = type(exc).__name__
            raise
        finally:
            try:
                response.close()
            except BaseException as exc:
                if capture_error_type is None:
                    capture_error_type = type(exc).__name__
                raise
            finally:
                completed_ms = _milliseconds(started)
                if capture is not None:
                    capture.finish(
                        completed_ms=completed_ms,
                        error_type=capture_error_type,
                    )
        return collector.result(response.status_code, completed_ms)


def _request_common(
    model: str, max_tokens: int, enable_thinking: bool
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "enable_thinking": enable_thinking,
        "_max_tokens": max_tokens,
    }


def build_request(
    protocol: str,
    model: str,
    mode: str,
    stage: int,
    *,
    history: list[dict[str, Any]] | str,
    instructions: str,
    previous_response_id: str = "",
    max_tokens: int = 512,
    enable_thinking: bool = True,
    second_tool_choice: str = "auto",
) -> dict[str, Any]:
    stream = mode == "stream"
    common = _request_common(model, max_tokens, enable_thinking)
    common.pop("_max_tokens")
    names = ("file_info", "run_command")
    if protocol == "chat":
        body: dict[str, Any] = {
            **common,
            "messages": history,
            "stream": stream,
            "max_tokens": max_tokens,
            "tools": tool_schemas(protocol, names),
            "tool_choice": tool_choice(protocol, mode, stage, second_tool_choice),
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body
    if protocol == "responses":
        body = {
            **common,
            "input": history,
            "instructions": instructions,
            "stream": stream,
            "store": True,
            "max_output_tokens": max_tokens,
            "tools": tool_schemas(protocol, names),
            "tool_choice": tool_choice(protocol, mode, stage, second_tool_choice),
        }
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        return body
    if protocol == "anthropic":
        return {
            **common,
            "messages": history,
            "stream": stream,
            "max_tokens": max_tokens,
            "tools": tool_schemas(protocol, names),
            "tool_choice": tool_choice(protocol, mode, stage, second_tool_choice),
        }
    if protocol == "ollama":
        body = {
            "model": model,
            "messages": history,
            "stream": stream,
            "think": enable_thinking,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if stage < 3:
            expected = "file_info" if stage == 1 else "run_command"
            body["tools"] = tool_schemas(protocol, (expected,))
        return body
    raise ValueError(f"unknown protocol: {protocol}")


def _sanitized_round(round_result: dict[str, Any]) -> dict[str, Any]:
    reasoning = str(round_result.get("reasoning") or "")
    content = str(round_result.get("content") or "")
    return {
        "status_code": round_result.get("status_code"),
        "elapsed_ms": round_result.get("elapsed_ms"),
        "response_id": round_result.get("response_id"),
        "reasoning_chars": len(reasoning),
        "reasoning_sha256": _sha256(reasoning),
        "reasoning_delta_count": sum(
            1
            for event in round_result.get("events") or []
            if event.get("channel") == "reasoning"
        ),
        "content": content,
        "content_chars": len(content),
        "content_sha256": _sha256(content),
        "content_delta_count": sum(
            1
            for event in round_result.get("events") or []
            if event.get("channel") == "content"
        ),
        "tool_calls": round_result.get("tool_calls") or [],
        "terminals": round_result.get("terminals") or [],
        "errors": round_result.get("errors") or [],
        "events": round_result.get("events") or [],
    }


def _execution_public(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": execution["name"],
        "call_id": execution["call_id"],
        "arguments": execution["arguments"],
        "result": execution["result"],
        "output_chars": len(execution["output"]),
        "output_sha256": _sha256(execution["output"]),
    }


def _contains_control_markup(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in CONTROL_MARKERS)


def final_synthesis_instruction(base_label: str, protocol: str, mode: str) -> str:
    """Ask for result-derived fields without leaking their expected values."""
    # Direct and gateway must receive byte-identical conversations. Keep the
    # base label only in the result artifact; embedding it in the user prompt
    # changes tokenization/cache keys and invalidates a parity comparison.
    _ = base_label
    prefix = f"AGENTIC-{protocol.upper()}-{mode.upper()}-DONE"
    return (
        "Both real tool results are now present. Call no tools. Reply with exactly "
        f"one line in this format: {prefix} SIZE=<copy size_human from the file_info "
        "result> PWD=<copy stdout from the run_command result>. Replace both angle-"
        "bracket placeholders with the real result values; output no other text."
    )


def first_tool_instruction(protocol: str, mode: str) -> str:
    """Return the base-independent first turn used for parity comparison."""
    # Protocol/mode labels belong in the evidence artifact, not the model's
    # instruction. A diagnostic prefix changed Qwen's native tool-call shape
    # and made the supposed parity probe test two artificial conversations.
    _ = (protocol, mode)
    return (
        f"Call the built-in file_info tool exactly once with path {FILE_INFO_PATH}. "
        "You must use the tool and must not answer from memory. Do not call "
        "run_command yet and do not answer yet."
    )


def run_flow(
    client: ProtocolClient,
    *,
    base_label: str,
    protocol: str,
    mode: str,
    model: str,
    repo_root: Path,
    max_tokens: int,
    enable_thinking: bool,
    second_tool_choice: str = "explicit",
) -> dict[str, Any]:
    first_prompt = first_tool_instruction(protocol, mode)
    second_prompt = (
        "The real file_info result is now present. Call run_command exactly once "
        f"with command {PWD_COMMAND}. Do not repeat file_info and do not answer yet."
    )
    initial_history: list[dict[str, Any]] | str
    if protocol == "responses":
        initial_history = first_prompt
    else:
        initial_history = [{"role": "user", "content": first_prompt}]

    request1 = build_request(
        protocol,
        model,
        mode,
        1,
        history=initial_history,
        instructions=first_prompt,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        second_tool_choice=second_tool_choice,
    )
    request_records = [_request_public(1, request1)]
    round1 = client.send(
        protocol,
        request1,
        mode == "stream",
        capture_label=f"{mode}-flow-round1",
    )
    calls1 = round1.get("tool_calls") or []
    check1 = len(calls1) == 1
    error1 = "expected exactly one tool call"
    if check1:
        check1, error1 = validate_allowlisted_call(calls1[0], "file_info")
    if not check1:
        return {
            "pass": False,
            "failure": f"round1: {error1}",
            "requests": request_records,
            "rounds": [_sanitized_round(round1)],
        }
    execution1 = execute_allowlisted_tool(repo_root, calls1[0])

    if protocol == "responses":
        history2 = history_after_tool(protocol, [], round1, execution1, second_prompt)
    else:
        history2 = history_after_tool(
            protocol, list(initial_history), round1, execution1, second_prompt
        )
    request2 = build_request(
        protocol,
        model,
        mode,
        2,
        history=history2,
        instructions=second_prompt,
        previous_response_id=str(round1.get("response_id") or ""),
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        second_tool_choice=second_tool_choice,
    )
    request_records.append(_request_public(2, request2))
    round2 = client.send(
        protocol,
        request2,
        mode == "stream",
        capture_label=f"{mode}-flow-round2",
    )
    calls2 = round2.get("tool_calls") or []
    check2 = len(calls2) == 1
    error2 = "expected exactly one tool call"
    if check2:
        check2, error2 = validate_allowlisted_call(calls2[0], "run_command")
    if not check2:
        return {
            "pass": False,
            "failure": f"round2: {error2}",
            "requests": request_records,
            "rounds": [_sanitized_round(round1), _sanitized_round(round2)],
            "executions": [_execution_public(execution1)],
        }
    execution2 = execute_allowlisted_tool(repo_root, calls2[0])
    final_marker = (
        f"AGENTIC-{protocol.upper()}-{mode.upper()}-DONE "
        f"SIZE={execution1['result']['size_human']} PWD={execution2['result']['stdout']}"
    )
    final_prompt = final_synthesis_instruction(base_label, protocol, mode)
    if protocol == "responses":
        history3 = history_after_tool(protocol, [], round2, execution2, final_prompt)
    else:
        history3 = history_after_tool(
            protocol, history2, round2, execution2, final_prompt
        )
    request3 = build_request(
        protocol,
        model,
        mode,
        3,
        history=history3,
        instructions=final_prompt,
        previous_response_id=str(round2.get("response_id") or ""),
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        second_tool_choice=second_tool_choice,
    )
    request_records.append(_request_public(3, request3))
    round3 = client.send(
        protocol,
        request3,
        mode == "stream",
        capture_label=f"{mode}-flow-round3",
    )

    stream = mode == "stream"
    terminals = [
        classify_terminal(
            protocol, round1.get("terminals") or [], stream=stream, expect_tool=True
        ),
        classify_terminal(
            protocol, round2.get("terminals") or [], stream=stream, expect_tool=True
        ),
        classify_terminal(
            protocol, round3.get("terminals") or [], stream=stream, expect_tool=False
        ),
    ]
    reasoning1 = str(round1.get("reasoning") or "")
    reasoning2 = str(round2.get("reasoning") or "")
    reasoning3 = str(round3.get("reasoning") or "")
    reasoning_values = [
        value for value in (reasoning1, reasoning2, reasoning3) if value
    ]
    response_ids = [
        str(row.get("response_id") or "") for row in (round1, round2, round3)
    ]
    response_tool_lifecycles = []
    if protocol == "responses" and stream:
        for row in (round1, round2):
            event_kinds = [
                str(event.get("kind") or "")
                for event in row.get("events") or []
                if event.get("channel") == "tool"
            ]
            response_tool_lifecycles.append(
                "response.output_item.added" in event_kinds
                and "response.function_call_arguments.done" in event_kinds
                and "response.output_item.done" in event_kinds
            )
    checks = {
        "status_200": all(
            int(row.get("status_code") or 0) == 200 for row in (round1, round2, round3)
        ),
        "round1_exact_tool": check1,
        "round2_exact_tool": check2,
        "final_no_tool": not (round3.get("tool_calls") or []),
        "final_exact": str(round3.get("content") or "").strip() == final_marker,
        "tool_rounds_have_no_visible_prose": all(
            not str(row.get("content") or "").strip() for row in (round1, round2)
        ),
        "no_stream_or_protocol_errors": all(
            not (row.get("errors") or []) for row in (round1, round2, round3)
        ),
        "no_visible_control_markup": all(
            not _contains_control_markup(str(row.get("content") or ""))
            for row in (round1, round2, round3)
        ),
        "terminals_truthful": all(item["pass"] for item in terminals),
        "stream_final_progressive": (
            mode != "stream"
            or sum(
                1
                for event in round3.get("events") or []
                if event.get("channel") == "content"
            )
            > 1
        ),
        "reasoning_present": (
            not enable_thinking
            # Reasoning ON permits, but does not require, a private chain before
            # every individual tool call. Require at least one real reasoning
            # rail across the three-turn flow; the separation/duplication checks
            # below still apply independently to every turn that emits it.
            or bool(reasoning_values)
        ),
        "reasoning_not_stale_when_present": len(reasoning_values)
        == len(set(reasoning_values)),
        "reasoning_not_duplicated_as_content": all(
            not (
                str(row.get("reasoning") or "").strip()
                and str(row.get("reasoning") or "").strip()
                == str(row.get("content") or "").strip()
            )
            for row in (round1, round2, round3)
        ),
        "responses_chain_ids_forwarded": (
            protocol != "responses"
            or (
                all(response_ids)
                and len(set(response_ids)) == 3
                and request2.get("previous_response_id") == response_ids[0]
                and request3.get("previous_response_id") == response_ids[1]
            )
        ),
        "responses_stream_tool_lifecycle_complete": (
            protocol != "responses" or not stream or all(response_tool_lifecycles)
        ),
        "timestamps_monotonic": all(
            all(
                float(events[index].get("at_ms") or 0)
                <= float(events[index + 1].get("at_ms") or 0)
                for index in range(len(events) - 1)
            )
            for events in [row.get("events") or [] for row in (round1, round2, round3)]
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "expected_final": final_marker,
        "requests": request_records,
        "rounds": [
            _sanitized_round(round1),
            _sanitized_round(round2),
            _sanitized_round(round3),
        ],
        "executions": [_execution_public(execution1), _execution_public(execution2)],
        "terminal_classification": terminals,
        "_prefinal_payload": request3,
    }


def _replace_final_instruction(
    protocol: str, payload: dict[str, Any], instruction: str
) -> dict[str, Any]:
    body = copy.deepcopy(payload)
    if protocol == "responses":
        body["instructions"] = instruction
    else:
        messages = body.get("messages") or []
        if not messages:
            raise ValueError("final payload has no messages")
        last = messages[-1]
        if protocol == "anthropic" and isinstance(last.get("content"), list):
            replaced = False
            for part in reversed(last["content"]):
                if part.get("type") == "text":
                    part["text"] = instruction
                    replaced = True
                    break
            if not replaced:
                last["content"].append({"type": "text", "text": instruction})
        else:
            last["content"] = instruction
    if protocol == "ollama":
        body["options"]["num_predict"] = 1024
    elif protocol == "responses":
        body["max_output_tokens"] = 1024
    else:
        body["max_tokens"] = 1024
    return body


def _idle_values(health: dict[str, Any]) -> tuple[int | None, int | None]:
    scheduler = health.get("scheduler") or {}
    cache = health.get("cache") or {}
    scheduler_cache = cache.get("scheduler_cache") or {}
    running = scheduler.get("num_running")
    active = scheduler_cache.get("active_requests")
    return (
        int(running) if isinstance(running, (int, float)) else None,
        int(active) if isinstance(active, (int, float)) else None,
    )


def wait_for_idle(health_url: str, timeout: float = 20.0) -> dict[str, Any]:
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    while time.monotonic() - started < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            health = response.json()
            running, active = _idle_values(health)
            sample = {
                "at_ms": _milliseconds(started),
                "status_code": response.status_code,
                "status": health.get("status"),
                "num_running": running,
                "active_requests": active,
            }
            samples.append(sample)
            if running == 0 and active in (0, None):
                return {"idle": True, "elapsed_ms": sample["at_ms"], "samples": samples}
        except Exception as exc:
            samples.append(
                {
                    "at_ms": _milliseconds(started),
                    "error": type(exc).__name__,
                }
            )
        time.sleep(0.1)
    return {"idle": False, "elapsed_ms": round(timeout * 1000, 3), "samples": samples}


def _cancel_route(protocol: str, response_id: str) -> str | None:
    if protocol == "chat":
        return f"/v1/chat/completions/{response_id}/cancel"
    if protocol == "responses":
        return f"/v1/responses/{response_id}/cancel"
    return None


def abort_stream_after_deltas(
    client: ProtocolClient,
    protocol: str,
    payload: dict[str, Any],
    *,
    health_url: str,
    minimum_deltas: int,
) -> dict[str, Any]:
    body = _replace_final_instruction(
        protocol,
        payload,
        "Using the completed real tool results, output 500 numbered lines beginning "
        "with POST-TOOL-ABORT. Begin immediately and do not call tools.",
    )
    body["stream"] = True
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    response = requests.post(
        client.base_url + client.route(protocol),
        headers=client.headers,
        json=body,
        stream=True,
        timeout=(15, client.timeout),
    )
    collector = EventCollector(protocol=protocol, started=started)
    event_name: str | None = None
    cancel_status: int | None = None
    cancel_body_hash = ""
    capture: DecompressedParserInputCaptureSession | None = None
    capture_error_type: str | None = None
    closed_at_ms = 0.0
    try:
        capture = client._attach_parser_input_capture(
            protocol=protocol,
            capture_label="stream-abort",
            payload=body,
            response=response,
            started=started,
            started_at=started_at,
        )
        for raw in response.iter_lines(decode_unicode=True, chunk_size=1):
            if raw is None:
                continue
            line = (
                raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            )
            line = line.strip()
            if not line:
                event_name = None
                continue
            at_ms = _milliseconds(started)
            if protocol != "ollama" and line.startswith("event: "):
                event_name = line[7:]
                continue
            if protocol != "ollama":
                if not line.startswith("data: "):
                    continue
                raw_data = line[6:]
                if raw_data == "[DONE]":
                    collector.terminal("DONE", at_ms)
                    continue
            else:
                raw_data = line
            data = json.loads(raw_data)
            _parse_stream_object(protocol, data, event_name, collector, at_ms)
            delta_count = len(collector.reasoning_parts) + len(collector.content_parts)
            if delta_count < minimum_deltas:
                continue
            route = _cancel_route(protocol, collector.response_id)
            if route:
                cancelled = requests.post(
                    client.base_url + route,
                    headers=client.headers,
                    json={},
                    timeout=10,
                )
                cancel_status = cancelled.status_code
                cancel_body_hash = _sha256(cancelled.text)
            break
    except BaseException as exc:
        capture_error_type = type(exc).__name__
        raise
    finally:
        try:
            response.close()
        except BaseException as exc:
            if capture_error_type is None:
                capture_error_type = type(exc).__name__
            raise
        finally:
            closed_at_ms = _milliseconds(started)
            if capture is not None:
                capture.finish(
                    completed_ms=closed_at_ms,
                    error_type=capture_error_type,
                )
    return {
        "status_code": response.status_code,
        "closed_at_ms": closed_at_ms,
        "response_id": collector.response_id,
        "delta_events_before_abort": len(collector.reasoning_parts)
        + len(collector.content_parts),
        "cancel_status": cancel_status,
        "cancel_body_sha256": cancel_body_hash,
        "terminals_before_abort": collector.terminals,
        "events": collector.events,
        "idle_after_abort": wait_for_idle(health_url),
    }


def disconnect_nonstream(
    client: ProtocolClient,
    protocol: str,
    payload: dict[str, Any],
    *,
    health_url: str,
    delay_ms: int,
) -> dict[str, Any]:
    body = _replace_final_instruction(
        protocol,
        payload,
        "Using the completed real tool results, output 500 numbered lines beginning "
        "with POST-TOOL-DISCONNECT. Begin immediately and do not call tools.",
    )
    body["stream"] = False
    parsed = urlparse(client.base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError(
            "non-stream disconnect hook currently requires an http base URL"
        )
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port or 80,
        timeout=client.timeout,
    )
    route = (parsed.path.rstrip("/") if parsed.path else "") + client.route(protocol)
    headers = dict(client.headers)
    started = time.monotonic()
    connection.request(
        "POST", route, body=json.dumps(body).encode("utf-8"), headers=headers
    )
    time.sleep(max(delay_ms, 0) / 1000.0)
    connection.close()
    return {
        "disconnected_at_ms": _milliseconds(started),
        "delay_ms": delay_ms,
        "idle_after_disconnect": wait_for_idle(health_url),
    }


def classify_abort(
    protocol: str, mode: str, aborted: dict[str, Any], minimum_deltas: int
) -> dict[str, Any]:
    """Require a real cancel route when available and never accept a false terminal."""
    idle = bool(
        (
            aborted.get("idle_after_abort")
            or aborted.get("idle_after_disconnect")
            or {}
        ).get("idle")
    )
    if mode == "nonstream":
        return {"pass": idle, "idle": idle, "kind": "client_disconnect"}
    deltas = int(aborted.get("delta_events_before_abort") or 0)
    terminals = list(aborted.get("terminals_before_abort") or [])
    cancel_route_ok = (
        aborted.get("cancel_status") in {200, 202}
        if protocol in {"chat", "responses"}
        else True
    )
    passed = idle and deltas >= minimum_deltas and not terminals and cancel_route_ok
    return {
        "pass": passed,
        "idle": idle,
        "delta_events": deltas,
        "no_terminal_before_abort": not terminals,
        "cancel_route_ok": cancel_route_ok,
        "kind": "explicit_cancel"
        if protocol in {"chat", "responses"}
        else "client_disconnect",
    }


def build_recovery_request(
    protocol: str,
    model: str,
    mode: str,
    marker: str,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = f"Call no tools. Reply exactly {marker} and nothing else."
    history: list[dict[str, Any]] | str = (
        prompt if protocol == "responses" else [{"role": "user", "content": prompt}]
    )
    return build_request(
        protocol,
        model,
        mode,
        3,
        history=history,
        instructions=prompt,
        max_tokens=max_tokens,
        enable_thinking=False,
    )


def run_recovery(
    client: ProtocolClient,
    protocol: str,
    mode: str,
    model: str,
    marker: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = build_recovery_request(protocol, model, mode, marker, max_tokens)
    result = client.send(
        protocol,
        payload,
        mode == "stream",
        capture_label=f"{mode}-recovery",
    )
    classification = classify_terminal(
        protocol,
        result.get("terminals") or [],
        stream=mode == "stream",
        expect_tool=False,
    )
    public = _sanitized_round(result)
    public["expected"] = marker
    public["exact"] = str(result.get("content") or "").strip() == marker
    public["terminal_classification"] = classification
    public["pass"] = (
        public["exact"] and classification["pass"] and not result.get("tool_calls")
    )
    return public


def parse_named_urls(values: list[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires NAME=URL, got {value!r}")
        name, url = value.split("=", 1)
        name = name.strip()
        url = url.strip().rstrip("/")
        if not name or not urlparse(url).scheme or not urlparse(url).netloc:
            raise ValueError(f"invalid {option} value: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate {option} name: {name}")
        parsed[name] = url
    return parsed


def expected_parser_input_capture_routes(
    bases: Iterable[str],
    protocols: Iterable[str],
    modes: Iterable[str],
    *,
    skip_cancellation: bool,
) -> list[tuple[str, str, str]]:
    if "stream" not in set(modes):
        return []
    labels = [
        "stream-flow-round1",
        "stream-flow-round2",
        "stream-flow-round3",
    ]
    if not skip_cancellation:
        labels.extend(("stream-abort", "stream-recovery"))
    return [
        (base_label, protocol, capture_label)
        for base_label in bases
        for protocol in protocols
        for capture_label in labels
    ]


def _disabled_capture_summary() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "enabled": False,
        "capture_layer": CAPTURE_LAYER,
        "capture_semantics": CAPTURE_SEMANTICS,
        "reason": "--raw-artifact-dir not supplied",
        "expected": 0,
        "started": 0,
        "finished": 0,
        "errors": 0,
        "complete": True,
        "routes": [],
    }


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    bases = parse_named_urls(args.base_url, "--base-url")
    required_bases = {"direct", "gateway"}
    if not args.allow_single_base and not required_bases.issubset(bases):
        raise ValueError("--base-url must include direct=... and gateway=...")
    health_urls = parse_named_urls(args.health_url or [], "--health-url")
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / FILE_INFO_PATH).is_file():
        raise ValueError(f"repo root does not contain {FILE_INFO_PATH}: {repo_root}")
    protocols = args.protocol or list(PROTOCOLS)
    modes = args.mode or list(MODES)
    raw_artifact_dir = getattr(args, "raw_artifact_dir", None)
    if raw_artifact_dir is not None and "stream" not in modes:
        raise ValueError(
            "--raw-artifact-dir captures streaming parser-input bytes and "
            "requires --mode stream"
        )
    expected_capture_routes = expected_parser_input_capture_routes(
        bases,
        protocols,
        modes,
        skip_cancellation=args.skip_cancellation,
    )
    raw_recorder: DecompressedParserInputCaptureRecorder | None = None
    if raw_artifact_dir is not None:
        raw_recorder = DecompressedParserInputCaptureRecorder(
            Path(raw_artifact_dir),
            _git_worktree_root(repo_root),
        )
        raw_recorder.configure_expected(expected_capture_routes)
    output: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_head": args.source_head,
        "model": args.model,
        "repo_root": str(repo_root),
        "bases": bases,
        "protocols": protocols,
        "modes": modes,
        "second_tool_choice": args.second_tool_choice,
        "raw_capture": (
            {
                **_disabled_capture_summary(),
                "enabled": True,
                "reason": "capture active; manifest pending",
                "expected": len(expected_capture_routes),
                "complete": False,
            }
            if raw_recorder is not None
            else _disabled_capture_summary()
        ),
        "flows": {},
        "abort_recovery": {},
    }
    for base_label, base_url in bases.items():
        client = ProtocolClient(
            base_url,
            args.api_key,
            args.timeout,
            base_label=base_label,
            raw_recorder=raw_recorder,
        )
        output["flows"][base_label] = {}
        output["abort_recovery"][base_label] = {}
        health_url = health_urls.get(base_label, base_url + "/health")
        for protocol in protocols:
            output["flows"][base_label][protocol] = {}
            output["abort_recovery"][base_label][protocol] = {}
            for mode in modes:
                try:
                    flow = run_flow(
                        client,
                        base_label=base_label,
                        protocol=protocol,
                        mode=mode,
                        model=args.model,
                        repo_root=repo_root,
                        max_tokens=args.max_output_tokens,
                        enable_thinking=args.enable_thinking,
                        second_tool_choice=args.second_tool_choice,
                    )
                    prefinal = flow.pop("_prefinal_payload", None)
                    output["flows"][base_label][protocol][mode] = flow
                    if (
                        args.skip_cancellation
                        or not flow.get("pass")
                        or prefinal is None
                    ):
                        continue
                    try:
                        if mode == "stream":
                            aborted = abort_stream_after_deltas(
                                client,
                                protocol,
                                prefinal,
                                health_url=health_url,
                                minimum_deltas=args.minimum_abort_deltas,
                            )
                        else:
                            aborted = disconnect_nonstream(
                                client,
                                protocol,
                                prefinal,
                                health_url=health_url,
                                delay_ms=args.disconnect_delay_ms,
                            )
                        abort_classification = classify_abort(
                            protocol,
                            mode,
                            aborted,
                            args.minimum_abort_deltas,
                        )
                        recovery_marker = f"RECOVERY-{base_label.upper()}-{protocol.upper()}-{mode.upper()}-DONE"
                        recovered = run_recovery(
                            client,
                            protocol,
                            mode,
                            args.model,
                            recovery_marker,
                            args.recovery_max_tokens,
                        )
                        output["abort_recovery"][base_label][protocol][mode] = {
                            "abort": aborted,
                            "abort_classification": abort_classification,
                            "recovery": recovered,
                            "pass": abort_classification["pass"]
                            and recovered.get("pass") is True,
                        }
                    except Exception as exc:
                        output["abort_recovery"][base_label][protocol][mode] = {
                            "pass": False,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                except Exception as exc:
                    output["flows"][base_label][protocol][mode] = {
                        "pass": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
    flow_rows = [
        row
        for base in output["flows"].values()
        for protocol in base.values()
        for row in protocol.values()
    ]
    abort_rows = [
        row
        for base in output["abort_recovery"].values()
        for protocol in base.values()
        for row in protocol.values()
        if row
    ]
    if raw_recorder is not None:
        try:
            output["raw_capture"] = raw_recorder.finalize()
        except Exception as exc:
            output["raw_capture"] = {
                **raw_recorder.current_summary(),
                "complete": False,
                "manifest_error_type": type(exc).__name__,
            }
    output["checks"] = {
        "all_requested_flows_present": len(flow_rows)
        == len(bases) * len(protocols) * len(modes),
        "all_flows_pass": bool(flow_rows)
        and all(row.get("pass") is True for row in flow_rows),
        "abort_recovery_skipped": bool(args.skip_cancellation),
        "all_abort_recovery_pass": (
            True
            if args.skip_cancellation
            else bool(abort_rows) and all(row.get("pass") is True for row in abort_rows)
        ),
        "raw_capture_complete": bool(output["raw_capture"]["complete"]),
    }
    output["pass"] = all(
        value
        for key, value in output["checks"].items()
        if key != "abort_recovery_skipped"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        metavar="NAME=URL",
        help="Endpoint base; provide direct=... and gateway=...",
    )
    parser.add_argument(
        "--health-url",
        action="append",
        metavar="NAME=URL",
        help="Optional backend health URL per base label for idle checks",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-artifact-dir",
        type=Path,
        help=(
            "Optional private root for exact decompressed streaming "
            "parser-input response bytes and safe-allowlisted metadata; "
            "requires --mode stream and must resolve outside every Git worktree"
        ),
    )
    parser.add_argument("--source-head", default="")
    parser.add_argument("--api-key")
    parser.add_argument("--protocol", action="append", choices=PROTOCOLS)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--recovery-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--minimum-abort-deltas", type=int, default=3)
    parser.add_argument("--disconnect-delay-ms", type=int, default=1000)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--second-tool-choice",
        choices=("explicit", "required", "auto"),
        default="explicit",
        help="Tool-choice policy for the second tool turn; run a separate auto row for release proof",
    )
    parser.add_argument("--skip-cancellation", action="store_true")
    parser.add_argument(
        "--allow-single-base",
        action="store_true",
        help="Diagnostic-only override; release proof requires direct and gateway",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_matrix(args)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pass": result.get("pass"),
                "checks": result.get("checks"),
            },
            indent=2,
        )
    )
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
