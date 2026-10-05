import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
vi.mock('electron', () => ({ clipboard: {} }))
vi.mock('../src/main/database', () => ({ db: { getSetting: vi.fn(() => undefined) } }))
import { executeBuiltinTool } from '../src/main/tools/executor'
const roots: string[] = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })
function root() { const dir = mkdtempSync(join(tmpdir(), 'vmlx-lines-')); roots.push(dir); return dir }
describe('built-in file line presentation', () => {
  it.each([
    ['', []], ['a', ['a']], ['a\n', ['a']], ['\n', ['']],
    ['a\n\n', ['a', '']], ['a\n\n\n', ['a', '', '']],
    ['amber=17\nblue=29\n', ['amber=17', 'blue=29']],
    ['a\r\nb\r\n', ['a', 'b']], ['a\r\n\r\n', ['a', '']],
    ['a\rb', ['a\rb']],
  ] as [string, string[]][])('counts and reads %j without inventing an EOF line', async (content, lines) => {
    const dir = root()
    const written = await executeBuiltinTool('write_file', { path: 'test.txt', content }, dir)
    expect(written.is_error).toBe(false)
    expect(written.content).toContain(`(${lines.length} lines, ${Buffer.byteLength(content)} bytes)`)
    expect(readFileSync(join(dir, 'test.txt'), 'utf8')).toBe(content)
    const read = await executeBuiltinTool('read_file', { path: 'test.txt' }, dir)
    expect(read.is_error).toBe(false)
    expect(read.content).toContain(`(${lines.length} lines)`)
    const body = read.content.slice(read.content.indexOf('\n\n') + 2)
    expect(body).toBe(lines.map((line, i) => `${String(i + 1).padStart(5)} | ${line}`).join('\n'))
    if (!lines.length) expect(read.content).toContain('empty file')
  })
  it('applies one-based offset and limit to real lines, including genuine trailing blank lines', async () => {
    const dir = root()
    await executeBuiltinTool('write_file', { path: 'test.txt', content: 'a\nb\n\n' }, dir)
    const page = await executeBuiltinTool('read_file', { path: 'test.txt', offset: 2, limit: 1 }, dir)
    expect(page.content).toContain('showing lines 2–2')
    expect(page.content).toContain('1 more lines. Use offset=3')
    expect(page.content.endsWith('    2 | b')).toBe(true)
    const blank = await executeBuiltinTool('read_file', { path: 'test.txt', offset: 3, limit: 1 }, dir)
    expect(blank.content.endsWith('    3 | ')).toBe(true)
    const eof = await executeBuiltinTool('read_file', { path: 'test.txt', offset: 4, limit: 1 }, dir)
    expect(eof.content).toContain('no lines at offset 4')
    expect(eof.content).not.toContain('showing lines 4–3')
  })
})
