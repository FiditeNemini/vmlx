import { readFileSync } from 'fs'
import { join } from 'path'
import { describe, expect, it } from 'vitest'

const repo = join(__dirname, '..')

function read(rel: string): string {
  return readFileSync(join(repo, rel), 'utf8')
}

describe('load progress honesty', () => {
  it('does not label phase progress as loading into GPU memory', () => {
    const source = read('src/main/sessions.ts')

    expect(source).not.toContain('Loading model into GPU')
    expect(source).toContain('Resident RAM')
    expect(source).toContain('modelBytes')
  })

  it('scans nested model files instead of only top-level safetensors shards', () => {
    // The recursive counter now lives in the shared modelLaunchMemory module
    // (sessions.ts imports it — the byte-identical private copy was removed to
    // stop the admission preflight and the load bar drifting apart).
    const source = read('src/main/modelLaunchMemory.ts')

    expect(source).toContain('withFileTypes: true')
    expect(source).toContain('entry.isDirectory()')
    expect(source).toContain('estimateModelFileBytes(fullPath)')
    // sessions.ts must consume it, not redefine it.
    const sessions = read('src/main/sessions.ts')
    expect(sessions).toContain("estimateModelFileBytes,\n")
    expect(sessions).not.toContain('function estimateModelFileBytes(')
  })

  it('preserves model-size metadata when later phase updates arrive', () => {
    const source = read('src/renderer/src/contexts/SessionsContext.tsx')

    expect(source).toContain('modelBytes?: number')
    expect(source).toContain('...(next.get(data.sessionId) || {})')
  })

  it('shows model file size separately from phase percent in both session cards', () => {
    const card = read('src/renderer/src/components/sessions/SessionCard.tsx')
    const view = read('src/renderer/src/components/sessions/SessionView.tsx')

    expect(card).toContain('Model files:')
    expect(view).toContain('Model files:')
  })

  it('polls process RSS during loading and renders resident RAM separately', () => {
    const source = read('src/main/sessions.ts')
    const context = read('src/renderer/src/contexts/SessionsContext.tsx')
    const card = read('src/renderer/src/components/sessions/SessionCard.tsx')
    const view = read('src/renderer/src/components/sessions/SessionView.tsx')

    expect(source).toContain('readProcessGroupResidentBytes')
    expect(source).toContain("execFileSync('ps', ['-o', 'rss=', '-g'")
    expect(source).toContain('residentPercent')
    expect(context).toContain('residentPercent?: number')
    expect(card).toContain('Resident RAM:')
    expect(view).toContain('Resident RAM:')
  })
})
