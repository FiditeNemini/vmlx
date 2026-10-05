import { describe, it, expect, vi } from 'vitest'
import { parsePowerMode } from '../src/shared/powerMode'
import { createPowerModeReader } from '../src/main/powerMode'
import { execFile } from 'node:child_process'
vi.mock('node:child_process', () => ({ execFile: vi.fn() }))
const custom = 'Battery Power:\n powermode 1\nAC Power:\n powermode 2\n'
describe('configured power mode', () => {
  it('selects the active source rather than charging state or first profile', () => {
    expect(parsePowerMode(custom, "Now drawing from 'AC Power'\n80%; not charging")).toEqual({ source: 'ac', mode: 'high' })
    expect(parsePowerMode(custom, "Now drawing from 'Battery Power'\n80%; discharging")).toEqual({ source: 'battery', mode: 'low' })
  })
  it('recognizes Automatic without inventing missing profiles or unknown enums', () => {
    const parse = (text: string) => parsePowerMode(text, "Now drawing from 'AC Power'").mode
    expect(parse('AC Power:\n powermode 0')).toBe('automatic')
    expect(parse('AC Power:\n powermode 3')).toBe('unknown')
    expect(parse('Battery Power:\n powermode 2')).toBe('unknown')
    expect(parsePowerMode(custom, 'unrecognized source').mode).toBe('unknown')
  })
  it('supports legacy flags without inferring missing capabilities', () => {
    const parse = (flags: string) => parsePowerMode(`AC Power:\n${flags}`, "Now drawing from 'AC Power'").mode
    expect(parse(' lowpowermode 0\n highpowermode 1')).toBe('high')
    expect(parse(' lowpowermode 0\n highpowermode 0')).toBe('automatic')
    expect(parse(' lowpowermode 0')).toBe('unknown')
    expect(parse(' lowpowermode 1\n highpowermode 1')).toBe('unknown')
  })
  it('coalesces in-flight events then refreshes changed profiles', async () => {
    let battery = false
    const read = vi.fn(async (arg: string) => arg === 'custom' ? custom : `Now drawing from '${battery ? 'Battery' : 'AC'} Power'`)
    const reader = createPowerModeReader('darwin', read)
    const a = reader(), b = reader()
    expect(a).toBe(b)
    expect((await a).mode).toBe('high')
    expect(read).toHaveBeenCalledTimes(2)
    battery = true
    expect((await reader()).mode).toBe('low')
    expect(read).toHaveBeenCalledTimes(4)
  })
  it('returns honest unsupported and timeout states', async () => {
    const read = vi.fn(async () => { throw new Error('timeout') })
    expect((await createPowerModeReader('linux', read)()).mode).toBe('unsupported')
    expect(read).not.toHaveBeenCalled()
    expect((await createPowerModeReader('darwin', read)()).mode).toBe('unknown')
  })
})


it('uses only bounded read-only pmset commands and reports process timeout honestly', async () => {
  vi.mocked(execFile).mockImplementation((...args: any[]) => {
    args[3](new Error('timed out'), '', '')
    return {} as any
  })
  expect(await createPowerModeReader('darwin')()).toEqual({ mode: 'unknown', source: 'unknown' })
  expect(execFile).toHaveBeenCalledTimes(2)
  for (const [command, args, options] of vi.mocked(execFile).mock.calls) {
    expect(command).toBe('/usr/bin/pmset')
    expect([['-g', 'custom'], ['-g', 'batt']]).toContainEqual(args)
    expect(options).toEqual({ timeout: 2000, maxBuffer: 65536, encoding: 'utf8' })
  }
})
