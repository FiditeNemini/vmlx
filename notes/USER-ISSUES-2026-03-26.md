# User-Reported Issues — 2026-03-26

## From Anant (128GB Mac, MiniMax-M2.5-JANG_3L + Qwen3.5-122B)

### Issue 1: 163K context timeout — FIXED
- MiniMax-M2.5 with 163K token context times out at 300s
- User set "No limit" in Server Settings but still got timeout error
- **Root cause**: `timeout` was in `RESTART_REQUIRED_KEYS` — UI told user to restart, but the
  per-request timeout forwarding (chat.ts:818 sends `timeout` in request body) already works
  without restart. User likely changed setting but didn't restart.
- **Fixes applied**:
  - Removed `timeout` from `RESTART_REQUIRED_KEYS` (sessions.ts) — changes take effect immediately
  - Added session lookup fallback by basename (chat.ts:454-470) — handles HF repo ID vs local path mismatch
  - Logs warning when no session found for modelPath
- **Status**: FIXED

### Issue 2: Chat preferences not saved per new chat — FIXED
- User has to manually enable tools/web search every time they start a new chat
- **Root cause**: Default profile auto-apply only works when a profile is starred. Most users
  don't know about profiles, so new chats always start with defaults.
- **Fix applied**: When no starred profile exists, new chats auto-inherit tool/search settings
  from the most recent chat of the same model (chat.ts:324-354). Inherits: builtinToolsEnabled,
  webSearchEnabled, braveSearchEnabled, fetchUrlEnabled, fileToolsEnabled, searchToolsEnabled,
  shellEnabled, gitEnabled, utilityToolsEnabled, maxToolIterations, workingDirectory,
  enableThinking, reasoningEffort, hideToolStatus, systemPrompt, toolResultMaxChars.
- **Status**: FIXED

### Issue 3: Web search crashes and ejects model — FIXED
- When user enables web search tool and triggers a search, the model crashes
- Server process dies (model gets "ejected")
- **Root cause found**: `MCPToolResult.to_message()` in `mcp/types.py` called `json.dumps()`
  on MCP tool response content without try/except. When MCP tools return non-JSON-serializable
  objects (datetime, custom types, nested SDK objects), `json.dumps()` raises `TypeError` that
  propagates uncaught and crashes the entire FastAPI/uvicorn server process.
- **Fixes applied**:
  - `mcp/types.py`: `to_message()` now wraps `json.dumps()` with `default=str` + try/except fallback
  - `mcp/client.py`: `_extract_content()` now always returns strings (str or joined list of str),
    never raw objects. Ensures `to_message()` hits the string branch, bypassing json.dumps entirely.
  - `mcp/executor.py`: `execute_and_format()` now catches per-result exceptions instead of crashing
    the entire list comprehension
- **Status**: FIXED

### Issue 4: "Session Config" vs "Server Settings" naming — FIXED
- Error message said "Increase the Timeout setting in Session Config"
- UI calls it "Server Settings"
- **Fix applied**: chat.ts:1910 now says "Server Settings"
- **Status**: FIXED

### Issue 5: Model name mismatch warning spam — FIXED
- Log shows repeated: `Request model 'dealignai/MiniMax-M2.5-JANG_3L-CRACKED-MLX' differs from served model 'dealignai/MiniMax-M2.5-JANG_3L-CRACK'`
- **Fix applied**: First occurrence logs at INFO, subsequent at DEBUG (server.py:3442-3449, 4053-4060)
- Added `_model_name_mismatch_warned` flag
- **Status**: FIXED

### Issue 6: No UI for managing chat preferences — IMPROVED
- Chat profiles feature exists in ChatSettings.tsx (save/load/star/delete)
- User couldn't find it
- **Fixes applied**:
  - Added helper text explaining auto-inherit behavior (ChatSettings.tsx:259)
  - Auto-inherit from last chat eliminates the need to manually manage profiles for most users
  - Profiles section is already at the top of the Chat settings panel
- **Status**: IMPROVED (auto-inherit reduces need for explicit profile management)

## Priority (Updated)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | 163K timeout still broken | HIGH | FIXED |
| 2 | Chat prefs not saved per new chat | HIGH | FIXED |
| 3 | Web search crashes model | HIGH | FIXED |
| 4 | "Session Config" naming in error msg | LOW | FIXED |
| 5 | Model name warning spam | LOW | FIXED |
| 6 | Chat prefs UI not discoverable | HIGH | IMPROVED |

## Files Modified

| File | Changes |
|------|---------|
| `panel/src/main/sessions.ts` | Removed `timeout` from RESTART_REQUIRED_KEYS |
| `panel/src/main/ipc/chat.ts` | Session fallback by basename, auto-inherit tool settings, error message fix |
| `panel/src/renderer/.../ChatSettings.tsx` | Added profile helper text |
| `vmlx_engine/server.py` | Model name mismatch warning throttle (INFO → DEBUG after first) |
| `vmlx_engine/mcp/types.py` | `to_message()` try/except + `default=str` for json.dumps |
| `vmlx_engine/mcp/client.py` | `_extract_content()` always returns strings, never raw objects |
| `vmlx_engine/mcp/executor.py` | `execute_and_format()` per-result error isolation |
