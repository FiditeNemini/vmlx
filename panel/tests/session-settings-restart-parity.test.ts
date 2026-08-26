import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const read = (path: string): string => readFileSync(path, 'utf8')

function handlerBlock(source: string, handler: string, nextHandler: string): string {
  const start = source.indexOf(`const ${handler} = async () =>`)
  const end = source.indexOf(`const ${nextHandler} = async () =>`, start + 1)
  expect(start, `${handler} must exist`).toBeGreaterThanOrEqual(0)
  expect(end, `${nextHandler} must follow ${handler}`).toBeGreaterThan(start)
  return source.slice(start, end)
}

describe('session settings restart parity', () => {
  it('preserves explicit stream interval 1 from UI preview through launcher argv', () => {
    const launcher = read('src/main/sessions.ts')
    const settings = read('src/renderer/src/components/sessions/SessionSettings.tsx')
    const form = read('src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const database = read('src/main/database.ts')

    const launchStart = launcher.indexOf('// Performance')
    const launchEnd = launcher.indexOf('// maxTokens:', launchStart)
    const launchBlock = launcher.slice(launchStart, launchEnd)
    const previewStart = settings.indexOf('function buildCommandPreview')
    const previewEnd = settings.indexOf('const SettingsSection', previewStart)
    const previewBlock = settings.slice(previewStart, previewEnd)

    expect(form).toContain('streamInterval: 8,')
    expect(previewBlock).toContain('finitePositiveInteger(config.streamInterval)')
    expect(launchBlock).toContain('finitePositiveInteger(config.streamInterval)')
    expect(launchBlock).toContain("args.push('--stream-interval', streamInterval.toString())")
    expect(launchBlock).not.toMatch(/streamInterval\w*\s*===\s*1/)
    expect(launcher).not.toContain('liftLegacyStreamInterval(')
    expect(database).toContain('migration_lift_legacy_stream_interval_default_1')
    expect(database).toContain('migrateLegacyStreamIntervalDefault(parsed)')
  })

  it('notifies session consumers after an ordinary config save', () => {
    const source = read('src/main/sessions.ts')
    const start = source.indexOf('async updateSessionConfig')
    const end = source.indexOf('repointSessionModelPath(', start)
    const block = source.slice(start, end)

    const persist = block.indexOf('db.updateSession(sessionId, {')
    const readBack = block.indexOf('const updatedSession = db.getSession(sessionId)')
    const notify = block.indexOf("this.emit('session:updated', {")
    expect(persist).toBeGreaterThanOrEqual(0)
    expect(readBack).toBeGreaterThan(persist)
    expect(notify).toBeGreaterThan(readBack)
    expect(block).toContain('session: updatedSession')
    expect(block).toContain('changedKeys')
  })

  it.each([
    ['src/renderer/src/components/sessions/ServerSettingsDrawer.tsx', 'handleReset'],
    ['src/renderer/src/components/sessions/SessionSettings.tsx', 'handleReset'],
  ])('restarts %s in save-stop-start order without renderer sleeps', (path, nextHandler) => {
    const block = handlerBlock(read(path), 'handleSaveAndRestart', nextHandler)
    const save = block.indexOf('window.api.sessions.update(')
    const stop = block.indexOf('window.api.sessions.stop(')
    const start = block.indexOf('window.api.sessions.start(')

    expect(save).toBeGreaterThanOrEqual(0)
    expect(stop).toBeGreaterThan(save)
    expect(start).toBeGreaterThan(stop)
    expect(block).not.toContain('setTimeout')
    expect(block).not.toContain('onStopped')
  })

  it('labels non-disruptive Save clearly while a session is running', () => {
    const drawer = read('src/renderer/src/components/sessions/ServerSettingsDrawer.tsx')
    const settings = read('src/renderer/src/components/sessions/SessionSettings.tsx')
    const locales = ['en', 'es', 'zh', 'ko', 'ja']

    expect(drawer).toContain("t(isRunning ? 'sessions.settings.saveForNextRestart' : 'common.save')")
    expect(settings).toContain("t(isRunning ? 'sessions.settings.saveForNextRestart' : 'sessions.settings.saveSettings')")
    for (const locale of locales) {
      const messages = JSON.parse(read(`src/renderer/src/i18n/locales/${locale}.json`))
      expect(messages.sessions.settings.saveForNextRestart).toBeTruthy()
    }
  })
})
