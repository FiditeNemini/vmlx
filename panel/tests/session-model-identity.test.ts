/**
 * Model-identity matching (LE9 chat model-selection revert, LE10 stale-symlink
 * re-resolution). The same model is stored under different path prefixes; these
 * helpers compare by normalized identity instead of raw path equality.
 */
import { describe, it, expect } from 'vitest'
import { modelIdentity, sessionMatchesModelPath } from '../src/shared/sessionUtils'

describe('modelIdentity', () => {
  it('normalizes to the lowercased final path component', () => {
    expect(modelIdentity('/Volumes/ModelDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK'))
      .toBe('lfm2.5-8b-a1b-mxfp4-crack')
    expect(modelIdentity('/Users/example/.mlxstudio/models/LFM2.5-8B-A1B-MXFP4-CRACK'))
      .toBe('lfm2.5-8b-a1b-mxfp4-crack')
  })

  it('ignores a trailing slash', () => {
    expect(modelIdentity('/a/b/Model-X/')).toBe('model-x')
  })

  it('is empty for null/undefined/empty', () => {
    expect(modelIdentity(null)).toBe('')
    expect(modelIdentity(undefined)).toBe('')
    expect(modelIdentity('')).toBe('')
  })
})

describe('sessionMatchesModelPath', () => {
  it('matches the same model across a symlink alias and its real path (LFM)', () => {
    // The exact bug: chat stored under ~/.mlxstudio symlink, session at /Volumes.
    expect(sessionMatchesModelPath(
      '/Volumes/ModelDrive/dealignai/LFM2.5-8B-A1B-MXFP4-CRACK',
      '/Users/example/.mlxstudio/models/LFM2.5-8B-A1B-MXFP4-CRACK',
    )).toBe(true)
  })

  it('matches across differing org casing / directory layout (Gemma)', () => {
    expect(sessionMatchesModelPath(
      '/Volumes/ModelDrive/jangq-ai/gemma-4-12B-it-qat-JANG_4M',
      '/Users/example/models/JANGQ-AI/gemma-4-12B-it-qat-JANG_4M',
    )).toBe(true)
  })

  it('matches an exact path', () => {
    expect(sessionMatchesModelPath('/x/Zaya-8B-JANG_4M', '/x/Zaya-8B-JANG_4M')).toBe(true)
  })

  it('does NOT match different models (the revert bug: LFM chat must not bind to Qwen)', () => {
    expect(sessionMatchesModelPath(
      '/Volumes/ModelDrive/dealignai/Qwen3.6-27B-MXFP4-CRACK-MTP',
      '/Users/example/.mlxstudio/models/LFM2.5-8B-A1B-MXFP4-CRACK',
    )).toBe(false)
  })

  it('returns false when either side is missing', () => {
    expect(sessionMatchesModelPath(null, '/x/Model')).toBe(false)
    expect(sessionMatchesModelPath('/x/Model', undefined)).toBe(false)
    expect(sessionMatchesModelPath('', '')).toBe(false)
  })

  it('remote pseudo-paths only match by exact string, not identity', () => {
    const a = 'remote://JANGQ-AI/MiniMax-M3-REAP22-Coder@127.0.0.1:8008'
    const b = 'remote://JANGQ-AI/MiniMax-M3-REAP22-Coder@127.0.0.1:9999'
    // Same exact string matches; different host does NOT (identity would be the
    // shared "…@host" tail, but these differ so no false positive here).
    expect(sessionMatchesModelPath(a, a)).toBe(true)
    expect(sessionMatchesModelPath(a, b)).toBe(false)
  })
})
