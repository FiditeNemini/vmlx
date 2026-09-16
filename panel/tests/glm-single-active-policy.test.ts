import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import ts from 'typescript'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { finitePositiveInteger } from '../src/shared/launchArgValues'
import { isGlmSingleActiveFamily, resolveGlmConcurrencyControls } from '../src/shared/detectedFamilyNames'

const { rows, db } = vi.hoisted(() => {
  const rows: any[] = []
  const db = {
    getSessions: () => rows,
    getSession: (id: string) => rows.find(row => row.id === id),
    getSessionByModelPath: (path: string) => rows.find(row => row.modelPath === path),
    getSetting: () => undefined,
    createSession: (row: any) => rows.push(row),
    updateSession: (id: string, patch: any) => Object.assign(rows.find(row => row.id === id), patch),
  }
  return { rows, db }
})
vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({
  app: { getAppPath: () => process.cwd(), getPath: () => '/tmp', isPackaged: false },
  powerSaveBlocker: { isStarted: () => false, start: () => 1, stop: () => undefined },
}))
import { SessionManager } from '../src/main/sessions'

const read = (path: string) => readFileSync(path, 'utf8')
const dirs: string[] = []
function bundle(family: string): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-glm-concurrency-'))
  dirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: family }))
  return dir
}
function flag(args: string[], name: string): string | undefined {
  const index = args.indexOf(name)
  return index < 0 ? undefined : args[index + 1]
}

// Execute the actual preview's concurrency slice, not a manually copied mirror.
// Other preview sections are outside this policy's dependency boundary.
function previewConcurrency(config: object, detectedFamily: string): string[] {
  const source = read('src/renderer/src/components/sessions/SessionSettings.tsx')
  const start = source.indexOf('  // Concurrent processing', source.indexOf('function buildCommandPreview'))
  const end = source.indexOf('  if (isVLM)', start)
  expect(start).toBeGreaterThan(0)
  expect(end).toBeGreaterThan(start)
  const js = ts.transpileModule(source.slice(start, end), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText
  return new Function('config', 'detectedFamily', 'dsv4Active', 'finitePositiveInteger',
    'resolveGlmConcurrencyControls', `const parts = []; ${js}; return parts;`)(
    config, detectedFamily, detectedFamily === 'deepseek-v4', finitePositiveInteger, resolveGlmConcurrencyControls,
  )
}

beforeEach(() => { rows.length = 0 })
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })

describe('GLM effective single-active concurrency preserves requested settings', () => {
  it.each(['glm5-next', 'glm5_next', 'glm5_next_text'])('normalizes %s without a route/cache dependency', family => {
    const saved = Object.freeze({ maxNumSeqs: 4, prefillBatchSize: 32, completionBatchSize: 16, prefillStepSize: 768 })
    expect(isGlmSingleActiveFamily(family)).toBe(true)
    expect(resolveGlmConcurrencyControls(family, saved)).toEqual({ maxNumSeqs: 1, prefillBatchSize: 1, completionBatchSize: 1 })
    expect(saved).toEqual({ maxNumSeqs: 4, prefillBatchSize: 32, completionBatchSize: 16, prefillStepSize: 768 })
    expect(previewConcurrency(saved, family)).toEqual([
      '--max-num-seqs', '1', '--prefill-batch-size', '1', '--prefill-step-size', '768', '--completion-batch-size', '1',
    ])
  })

  it.each([undefined, 0, 1, 8])('pins even omitted/default/requested %s values without mutation', value => {
    const saved = Object.freeze({ maxNumSeqs: value, prefillBatchSize: value, completionBatchSize: value })
    expect(resolveGlmConcurrencyControls('glm5_next', saved)).toEqual({ maxNumSeqs: 1, prefillBatchSize: 1, completionBatchSize: 1 })
    expect(saved.maxNumSeqs).toBe(value)
  })

  it.each(['qwen4_exp', 'qwen4-exp', 'qwen3_5', 'qwen3.5', 'glm4', 'unknown', undefined])(
    'does not change unrelated family %s', family => {
      const saved = Object.freeze({ maxNumSeqs: 2, prefillBatchSize: 4, completionBatchSize: 8 })
      expect(isGlmSingleActiveFamily(family)).toBe(false)
      expect(resolveGlmConcurrencyControls(family, saved)).toEqual(saved)
    },
  )

  it.each(['glm5_next', 'glm5_next_text'])('actual launcher and stored config remain distinct for %s', async family => {
    const manager = new SessionManager()
    const row = await manager.createSession(bundle(family), {
      port: 8123, maxNumSeqs: 4, prefillBatchSize: 32, completionBatchSize: 16, prefillStepSize: 768,
    })
    for (const isMultimodal of [true, false]) {
      await manager.updateSessionConfig(row.id, { isMultimodal, maxNumSeqs: 4, prefillBatchSize: 32, completionBatchSize: 16 })
      const before = row.config
      const saved = JSON.parse(before)
      expect(saved).toMatchObject({ maxNumSeqs: 4, prefillBatchSize: 32, completionBatchSize: 16, prefillStepSize: 768 })
      const config = { ...saved, modelPath: row.modelPath,
        additionalArgs: '--max-num-seqs 9 --prefill-batch-size=9 --completion-batch-size 9' }
      const args = (manager as any).buildArgs(config) as string[]
      for (const key of ['--max-num-seqs', '--prefill-batch-size', '--completion-batch-size']) {
        expect(args.filter(value => value === key)).toHaveLength(1)
        expect(flag(args, key)).toBe('1')
        expect(args).not.toContain(`${key}=9`)
        expect(flag(previewConcurrency(config, family), key)).toBe('1')
      }
      expect(flag(args, '--prefill-step-size')).toBe('768')
      expect(row.config).toBe(before)
      expect(config.maxNumSeqs).toBe(4)
    }
  })

  it.each(['qwen4_exp', 'qwen3_5'])('actual %s launcher and preview keep batch2', family => {
    const config = { modelPath: bundle(family), host: '127.0.0.1', port: 8123,
      maxNumSeqs: 2, prefillBatchSize: 4, completionBatchSize: 8, prefillStepSize: 768 }
    const args = (new SessionManager() as any).buildArgs(config) as string[]
    for (const [key, expected] of [['--max-num-seqs', '2'], ['--prefill-batch-size', '4'], ['--completion-batch-size', '8'], ['--prefill-step-size', '768']]) {
      expect(flag(args, key)).toBe(expected)
      expect(flag(previewConcurrency(config, family), key)).toBe(expected)
    }
  })

  it('form binds effective disabled counts, not prefill chunk size or saved values', () => {
    const source = read('src/renderer/src/components/sessions/SessionConfigForm.tsx')
    expect(source).toContain('const glmSingleActive = isGlmSingleActiveFamily(normalizedDetectedFamily)')
    expect(source).toContain('const concurrency = resolveGlmConcurrencyControls(normalizedDetectedFamily, config)')
    expect(source).toContain("{glmSingleActive && <InfoNote text={t('sessions.config.glmSequentialQueueNote')} />}")
    for (const key of ['maxNumSeqs', 'prefillBatchSize', 'completionBatchSize']) {
      const start = source.indexOf(`<SliderField settingKey="${key}"`)
      const field = source.slice(start, source.indexOf('/>', start))
      expect(field).toContain('disabled={singleActiveControls}')
      expect(field).toContain(`onChange={v => onChange('${key}', v)}`)
      expect(source).not.toContain(`onChange('${key}', 1)`)
    }
    const start = source.indexOf('<SliderField settingKey="prefillStepSize"')
    const field = source.slice(start, source.indexOf('/>', start))
    expect(field).toContain('value={config.prefillStepSize}')
    expect(field).toContain('disabled={dsv4Active}')
    expect(field).not.toContain('singleActiveControls')
    for (const language of ['en', 'es', 'zh', 'ko', 'ja']) {
      const locale = JSON.parse(read(`src/renderer/src/i18n/locales/${language}.json`))
      expect(locale.sessions.config.glmSequentialQueueNote).toContain('GLM-5.3')
      expect(locale.sessions.config.glmSequentialQueueNote.length).toBeGreaterThan(50)
    }
  })
})
