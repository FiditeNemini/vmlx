/** Returns true if the session config has modelType === 'image' */
export function isImageSession(s: { config?: string }): boolean {
  if (!s.config) return false
  try { return JSON.parse(s.config).modelType === 'image' } catch { return false }
}

/**
 * Normalized model identity: the lowercased final path component.
 *
 * The SAME model is stored under different path prefixes across chats and
 * sessions — a `~/.mlxstudio/models/X` symlink vs the real `/Volumes/…/org/X`,
 * an HF repo id vs its resolved local dir, or different org casing
 * (`jangq-ai` vs `JANGQ-AI`). Raw path-string equality misses those, so match
 * on this identity instead. Remote pseudo-paths (`remote://…@host`) have no
 * filesystem basename; callers should compare those by exact string.
 */
export function modelIdentity(modelPath: string | null | undefined): string {
  if (!modelPath) return ''
  return modelPath.replace(/\/+$/, '').split('/').pop()?.toLowerCase() || ''
}

/**
 * True if a session refers to the same model as `modelPath` — by exact path or,
 * failing that, by normalized model identity. Used to bind a chat to the right
 * session and to re-resolve a stale saved path to a valid one.
 */
export function sessionMatchesModelPath(
  sessionModelPath: string | null | undefined,
  modelPath: string | null | undefined,
): boolean {
  if (!sessionModelPath || !modelPath) return false
  if (sessionModelPath === modelPath) return true
  const want = modelIdentity(modelPath)
  return want !== '' && modelIdentity(sessionModelPath) === want
}
