# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 chat encoder adapter.

DSV4-Flash ships its own ``encoding_dsv4.py`` (tokenizer + chat template
helper) alongside the model bundle. Importing it through a stable path so
server.py can encode chat messages the way the model expects — without
needing every caller to locate the source-model directory and ``sys.path``
hack.

Native reasoning is bundle-owned.  The adapter loads the selected bundle's
``encoding_dsv4.py`` and validates explicit effort names against that encoder
instead of applying a family-wide alias.  The 0731 encoder exposes
``low``, ``high``, and ``max`` (default ``low``); older bundles may expose a
different subset.

  +------------------+---------------------+---------------------+
  |  API input       |  DSV4 encoding call |  Prompt suffix      |
  +------------------+---------------------+---------------------+
  |  enable_thinking |  thinking_mode=     |  ends with          |
  |    = False       |  "chat"             |  "</think>"         |
  +------------------+---------------------+---------------------+
  |  enable_thinking |  thinking_mode=     |  no </think>; model |
  |    = True +      |  "thinking" +       |  opens <think>…</   |
  |  reasoning_      |  reasoning_effort=  |  think> then answer |
  |    effort="high" |  "high"             |                     |
  +------------------+---------------------+---------------------+
  |  reasoning_      |  thinking_mode=     |  extra system hint  |
  |    effort="max"  |  "thinking" +       |  from the canonical |
  |                  |  reasoning_effort=  |  DSV4 encoder       |
  |                  |  "max"              |                     |
  +------------------+---------------------+---------------------+

Multi-turn: ``drop_earlier_reasoning=True`` (default) — DSV4 encoder
strips prior ``<think>…</think>`` blocks from history when building the
next prompt. Honors ``jang_config.chat.reasoning.drop_earlier_reasoning``.

Sampling defaults are pulled from ``jang_config.chat.sampling_defaults``
when the caller doesn't override them. Our chat.ts layer already reads
``generation_config.json``
on new chat creation (chat.ts:441-469) — the jang_config.chat defaults
layer on top as JANG-stamp capabilities.

Tool calls are DSML format (``vmlx_engine/tool_parsers/dsml_tool_parser.py``
— parser key "dsml"). ``jang_config.chat.tool_calling.parser`` = "dsml".

Long-context mode (``DSV4_LONG_CTX``):

  * ``1`` is the supported runtime mode. ``Model.make_cache()`` returns
    ``DeepseekV4Cache`` on ``compress_ratio>0`` layers (CSA/HCA + SWA
    composite) and bounded ``RotatingKVCache`` on ratio-zero local SWA
    layers, preserving the native 128-token ring geometry.
  * Paged prefix cache uses a dedicated ``deepseek_v4`` block record with
    ``deepseek_v4_v10_delta`` metadata; v9 keys DSV4 prompt cache blocks at N-1
    tokens so the last prompt token is re-fed on prefix hits. The loader
    installs the prefill mask-trim patch required for prompts beyond the
    sliding window.
  * ``cache_salt`` / ``skip_prefix_cache`` still bypass all cache layers for
    benchmarks, but DSV4 is no longer force-bypassed by family.

Research refs: DSV4-RUNTIME-ARCHITECTURE.md §17,
DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §12.
"""

from __future__ import annotations

import importlib.util
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def request_explicitly_requests_tool(request_text: str, tool_name: str) -> bool:
    """Return whether the current user positively directs use of ``tool_name``.

    A bare or negated mention is not authorization.  This distinction matters
    for DSV4 because a positive match scopes the native schema catalog and can
    add a request-bound canonical DSML example.
    """
    normalized = tool_name.strip()
    if not request_text or not normalized:
        return False

    name_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    for match in name_pattern.finditer(request_text):
        prefix = request_text[max(0, match.start() - 128) : match.start()]
        if re.search(
            r"\b(?:do\s+not|don['’]t|never|without)\b[^.!?;\n]{0,96}$",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            r"\b(?:call|use|invoke|run|execute)\b[^.!?;\n]{0,80}$",
            prefix,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def select_tools_for_explicit_request(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Keep DSV4's prompt focused when the latest user names a tool.

    The request may still authorize the full tool array for output parsing and
    execution.  This helper only limits what the canonical DSV4 encoder places
    in the prompt.  Without it, one explicit ``run_command`` request expands to
    several thousand tokens of unrelated schemas/examples and live DSV4 copies
    those schemas or emits an empty/no-op argument instead of the requested
    value.
    """
    if not tools:
        return tools

    request_text = ""
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            request_text = content
        elif isinstance(content, list):
            request_text = "\n".join(
                str(part.get("text") or part.get("content") or "")
                for part in content
                if isinstance(part, dict)
            )
        break
    if not request_text:
        return tools

    selected: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = func.get("name") if isinstance(func, dict) else None
        if not isinstance(name, str) or not name:
            continue
        if request_explicitly_requests_tool(request_text, name):
            selected.append(tool)
    if selected:
        return selected

    # A later user turn normally stops naming the tool even though Responses
    # history still carries the assistant function_call and its output. Keep
    # the same schema set in that continuation prompt; expanding back to every
    # enabled coding tool changes the DSV4 prefix and makes the terminal native
    # composite disk block impossible to match after a restart.
    recent_call_names: set[str] = set()
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            func = tool_call.get("function")
            name = func.get("name") if isinstance(func, dict) else None
            if isinstance(name, str) and name:
                recent_call_names.add(name)
        if recent_call_names:
            break
    if recent_call_names:
        historical = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = func.get("name") if isinstance(func, dict) else None
            if name in recent_call_names:
                historical.append(tool)
        if historical:
            return historical
    return tools


def _default_encoding_dirs() -> List[Path]:
    """Return likely local DSV4 encoding directories.

    Production callers should pass ``model_path``. This fallback covers app
    sessions/tests where an older persisted DSV4 path no longer exists but the
    user has the bundle or source checkout in the standard local model roots.
    Keep the search shallow so startup does not walk large model trees.
    """
    dirs: List[Path] = []

    def add(path: Path) -> None:
        if path not in dirs:
            dirs.append(path)

    roots = [Path.home() / "models"]
    # Accept either the canonical name or the historical typo'd one.
    extra_root = (
        os.environ.get("VMLX_MODELS_DIR")
        or os.environ.get("VLLM_MODELS_DIR")
        or os.environ.get("VMLINUX_MODELS_DIR")
    )
    if extra_root:
        roots.insert(0, Path(extra_root).expanduser())
    volumes = Path("/Volumes")
    if volumes.exists():
        try:
            roots.extend(p for p in volumes.iterdir() if p.is_dir())
        except Exception:
            pass

    for root in roots:
        try:
            if not root.exists():
                continue
            add(root / "Sources" / "DeepSeek-V4-Flash" / "encoding")
            for pattern in (
                "DeepSeek-V4-Flash*/encoding",
                "JANGQ/DeepSeek-V4-Flash*/encoding",
                "*/DeepSeek-V4-Flash*/encoding",
                "*/*DeepSeek-V4-Flash*/encoding",
            ):
                for match in root.glob(pattern):
                    add(match)
        except Exception:
            continue
    return dirs


def _load_encoding_dsv4_module(
    encoding_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
):
    """Locate + dynamically import the source model's ``encoding_dsv4.py``.

    Lookup order:
      1. Explicit ``encoding_dir`` argument.
      2. ``DSV4_ENCODING_DIR`` environment variable (session dispatcher
         may set this at session-start).
      3. ``{model_path}/encoding/`` subdir — the standard bundle layout
         as shipped in DeepSeek-V4-Flash-JANGTQ / JANG_2L. This path is
         auto-discovered so no env plumbing is required.
      4. Bundled fallback at ``jang_tools.dsv4.encoding_adapter``.
    """
    candidates: List[Path] = []
    if encoding_dir is not None:
        candidates.append(Path(encoding_dir))
    env = os.environ.get("DSV4_ENCODING_DIR")
    if env:
        candidates.append(Path(env))
    if model_path is not None:
        mp = Path(model_path)
        candidates.append(mp / "encoding")
        candidates.append(mp)
    candidates.extend(_default_encoding_dirs())

    for d in candidates:
        f = d / "encoding_dsv4.py"
        if f.exists():
            spec = importlib.util.spec_from_file_location("encoding_dsv4", str(f))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["encoding_dsv4"] = mod
            spec.loader.exec_module(mod)
            logger.info("DSV4 encoding loaded from %s", f)
            return mod

    # Fall back to the jang_tools adapter which will raise a clear
    # FileNotFoundError if neither an explicit path nor DSV4_ENCODING_DIR is set.
    try:
        from jang_tools.dsv4.encoding_adapter import _load_encoding_module
    except ImportError as e:
        raise RuntimeError(
            "DSV4 chat encoder unavailable: jang_tools.dsv4 not installed "
            "AND no encoding_dsv4.py found via DSV4_ENCODING_DIR or "
            "{model_path}/encoding/. Bring over a DSV4 bundle or set "
            "DSV4_ENCODING_DIR."
        ) from e
    return _load_encoding_module(encoding_dir)


# Per-model-path cache — different bundles may ship different encoder
# revisions (DSV4-Flash vs DSV4-Pro), so key by resolved absolute path.
_encoding_cache: dict[str, Any] = {}


def _get_encoding(model_path: Optional[Path] = None):
    key = str(Path(model_path).resolve()) if model_path else "<default>"
    if key not in _encoding_cache:
        _encoding_cache[key] = _load_encoding_dsv4_module(model_path=model_path)
    return _encoding_cache[key]


def _resolve_mode_and_effort(
    enable_thinking: Optional[bool],
    reasoning_effort: Optional[str],
    *,
    supported_efforts: Optional[set[str]] = None,
) -> tuple[str, Optional[str]]:
    """Map vmlx chat API semantics → DSV4 encoder kwargs.

    DSV4 encoder revisions do not all expose the same effort vocabulary.  An
    explicit effort is therefore preserved only when it is accepted by the
    selected bundle's encoder.  Unsupported values fail clearly; silently
    promoting ``low`` to ``high`` changes the prompt and is not a compatibility
    fix.

    API contract (shared with other reasoning models like Mistral 4):
      - enable_thinking False  → thinking suppressed. DSV4 calls this
        "chat" mode; the encoder appends </think> to the prompt to tell
        the model "skip thinking, go straight to answer".
      - enable_thinking True + no effort → "thinking" mode using the native
        encoder default.
      - an explicit supported effort → "thinking" mode with that exact effort.

    Returns ``(thinking_mode, reasoning_effort)`` for direct passthrough to the
    selected encoder.
    """
    effort = (
        str(reasoning_effort).strip().lower()
        if reasoning_effort is not None
        else None
    )
    if effort:
        allowed = supported_efforts or {"low", "high", "max"}
        if effort not in allowed:
            raise ValueError(
                "Unsupported DSV4 reasoning_effort "
                f"{effort!r}; selected encoder supports: "
                f"{', '.join(sorted(allowed))}"
            )
        return "thinking", effort

    if enable_thinking is False:
        return "chat", None

    if enable_thinking is True:
        return "thinking", None

    # Default when both fields omitted: fall through to "chat" (instruct),
    # matching the conservative default the rest of the engine uses.
    return "chat", None


def apply_chat_template(
    messages: List[Dict[str, Any]],
    *,
    enable_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
    drop_earlier_reasoning: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_default_bos_token: bool = True,
    add_generation_prompt: bool = True,
    context: Optional[List[Dict[str, Any]]] = None,
    model_path: Optional[str] = None,
) -> str:
    """Encode chat messages into a DSV4 prompt string.

    Args:
        messages: OpenAI-format ``[{"role": ..., "content": ...}, ...]``.
            Tools, when used, are declared on the system/developer message
            via its ``tools`` field — DSV4's encoder reads them from there.
        enable_thinking: Whether to allow the model to emit a ``<think>``
            block. None ≡ False (default chat mode).
        reasoning_effort: An effort accepted by the selected bundle encoder,
            or None for its native default. See ``_resolve_mode_and_effort``.
        drop_earlier_reasoning: Multi-turn rule — strip prior
            ``<think>...</think>`` blocks from history. Follows the
            ``jang_config.chat.reasoning.drop_earlier_reasoning`` flag
            when caller doesn't override.
        tools: Optional OpenAI-format tool list. Injected onto the first
            system message's ``tools`` field if absent.
        add_default_bos_token: Whether to prepend ``<｜begin▁of▁sentence｜>``.
            DSV4 prompts need this; turn off only when the caller already
            prepended their own BOS.
        add_generation_prompt: Whether to retain the canonical terminal
            assistant/thinking/task rail. Prefix-cache identity renders with
            this disabled so the rail can be represented by its separate
            generation-prompt discriminator instead of becoming part of the
            reusable message prefix.
        context: Optional cross-turn context injection (DSV4 encoder kwarg;
            pass through unchanged for advanced users).

    Returns:
        The encoded prompt string. Pass directly to ``mlx_lm.generate``
        or the vmlx generation loop.
    """
    enc = _get_encoding(model_path=Path(model_path) if model_path else None)
    effort_prompts = getattr(enc, "REASONING_EFFORT_PROMPTS", None)
    supported_efforts = (
        {str(key).strip().lower() for key in effort_prompts}
        if isinstance(effort_prompts, dict) and effort_prompts
        else None
    )
    thinking_mode, effort = _resolve_mode_and_effort(
        enable_thinking,
        reasoning_effort,
        supported_efforts=supported_efforts,
    )
    messages = copy.deepcopy(messages)
    prompt_tools = select_tools_for_explicit_request(messages, tools)
    if prompt_tools and tools and len(prompt_tools) < len(tools):
        logger.info(
            "DSV4 explicit-tool prompt narrowed from %d schemas to %d",
            len(tools),
            len(prompt_tools),
        )

    # DSV4's bundled encoder predates the strict-template normalization used
    # by the Responses adapter. It expects OpenAI tool-call
    # `function.arguments` to be a JSON string and calls json.loads() itself;
    # if we pass a dict it wraps the whole dict under a single DSML
    # `arguments` parameter. That makes tool-history continuation and DSML
    # exemplars materially wrong for DSV4 while the same dict form is correct
    # for Qwen/Mistral/Llama templates. Normalize only in this adapter.
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)

    # If the caller passed a `tools` list without a system message carrying
    # it, inject onto the first message. DSV4 encoder conventions require
    # `tools` to live on a system/developer message — see test_chat.py in
    # jang_tools.dsv4.
    if prompt_tools and messages:
        # Ensure there's a system message first; tools go on it.
        has_system_with_tools = any(
            m.get("role") == "system" and m.get("tools") for m in messages
        )
        if not has_system_with_tools:
            msgs_out: List[Dict[str, Any]] = []
            if messages and messages[0].get("role") == "system":
                sys_msg = dict(messages[0])
                sys_msg["tools"] = prompt_tools
                msgs_out.append(sys_msg)
                msgs_out.extend(messages[1:])
            else:
                msgs_out.append({"role": "system", "content": "", "tools": prompt_tools})
                msgs_out.extend(messages)
            messages = msgs_out

    prompt = enc.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=drop_earlier_reasoning,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=effort,
    )
    if add_generation_prompt:
        return prompt

    # ``context`` is already-encoded prefix state. The official encoder renders
    # zero context messages when there is no new message, so there is no
    # generation rail owned by this call to remove.
    if not messages:
        return prompt

    # The official 0731 encoder has no ``add_generation_prompt`` argument.
    # Its ``render_message`` appends one exact terminal rail after the final
    # user/developer message (or after a typed task). Remove only that owned
    # suffix; never search/replace an earlier assistant rail in history.
    effective_messages = copy.deepcopy(messages)
    merge_tools = getattr(enc, "merge_tool_messages", None)
    if callable(merge_tools):
        effective_messages = merge_tools(effective_messages)
    full_messages = [*(copy.deepcopy(context) if context else []), *effective_messages]
    if not full_messages:
        return prompt
    last = full_messages[-1]
    suffix = ""
    task = last.get("task") if isinstance(last, dict) else None
    if task is not None:
        task_tokens = getattr(enc, "DS_TASK_SP_TOKENS", {})
        task_token = task_tokens.get(task) if isinstance(task_tokens, dict) else None
        if not isinstance(task_token, str) or not task_token:
            raise RuntimeError(
                f"DSV4 canonical encoder did not expose a suffix for task {task!r}"
            )
        if task == "action":
            suffix = (
                str(enc.ASSISTANT_SP_TOKEN)
                + str(
                    getattr(
                        enc,
                        "thinking_start_token"
                        if thinking_mode == "thinking"
                        else "thinking_end_token",
                    )
                )
                + task_token
            )
        else:
            suffix = task_token
    elif isinstance(last, dict) and last.get("role") in {"user", "developer"}:
        suffix = str(enc.ASSISTANT_SP_TOKEN) + str(
            getattr(
                enc,
                "thinking_start_token"
                if thinking_mode == "thinking"
                else "thinking_end_token",
            )
        )

    # ``latest_reminder`` is rendered *after* the preceding user's assistant
    # rail. That rail is therefore non-terminal and cannot be represented by
    # the cache layer's final-N-token stripping contract. Keep the full prompt
    # as its safe cache identity instead of stripping reminder tokens.
    if not suffix:
        return prompt
    if not prompt.endswith(suffix):
        raise RuntimeError(
            "DSV4 canonical encoder generation suffix did not match its declared rail"
        )
    return prompt[: -len(suffix)]


def parse_completion(
    raw_text: str,
    *,
    enable_thinking: Optional[bool] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a raw DSV4 completion into ``{reasoning_content, content, tool_calls}``.

    Mirror of the server-side streaming parser — exposed here for batch
    use-cases (tests, benchmarks, non-streaming completion endpoints).
    ``enable_thinking`` determines whether to look for ``<think>`` blocks
    in the raw text.
    """
    thinking_mode = "thinking" if enable_thinking else "chat"
    enc = _get_encoding(model_path=Path(model_path) if model_path else None)

    # The generation loop consumes the stop token instead of returning it in
    # decoded completion text.  The official DSV4 parser owns the opposite
    # contract: an ordinary visible completion must terminate with the
    # encoder's EOS token.  Tool-call-only turns happened to parse without EOS
    # because their canonical DSML close is itself terminal, which hid this
    # mismatch until a tool-result continuation produced reasoning + a visible
    # final answer.  Restore only the consumed canonical stop token before
    # invoking the bundle parser, exactly as the production DSML parser does.
    parser_text = raw_text
    eos_token = getattr(enc, "eos_token", None)
    if (
        isinstance(eos_token, str)
        and eos_token
        and not parser_text.endswith(eos_token)
    ):
        parser_text += eos_token

    # Do not turn a rejected official DSV4 grammar into an apparently valid
    # visible answer.  Callers need the parser failure so API/batch harnesses
    # can fail closed instead of leaking reasoning or malformed DSML.
    return enc.parse_message_from_completion_text(
        parser_text,
        thinking_mode=thinking_mode,
    )


def read_chat_config_from_bundle(bundle_path: str) -> Dict[str, Any]:
    """Pull the ``chat`` block from ``jang_config.json`` if present.

    Exposed for the session dispatcher to seed chat defaults (EOS token,
    sampling_defaults temperature/top_p, reasoning modes, tool parser).
    Returns an empty dict when no jang_config exists or the chat block is
    missing — every caller must handle that gracefully.
    """
    try:
        p = Path(bundle_path) / "jang_config.json"
        if not p.exists():
            return {}
        with open(p) as f:
            cfg = json.load(f)
        return cfg.get("chat", {}) or {}
    except Exception as e:
        logger.debug("jang_config.chat read failed for %s: %s", bundle_path, e)
        return {}
