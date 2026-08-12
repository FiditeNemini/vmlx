import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const executor = readFileSync(resolve(__dirname, '../src/main/tools/executor.ts'), 'utf-8')

/** Mirror of decodeSearchEntities, exercised on strings taken from real HTML. */
function decodeSearchEntities(value: string): string {
  return value
    .replace(/<[^>]+>/g, '')
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(parseInt(dec, 10)))
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&rsaquo;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim()
}

describe('web search entity decoding', () => {
  it('decodes the numeric entities that reached the model raw', () => {
    // Both taken verbatim from live Mojeek result titles.
    expect(decodeSearchEntities('Mount Everest Isn&#039;t Really The Tallest'))
      .toBe("Mount Everest Isn't Really The Tallest")
    expect(decodeSearchEntities('Tallest Mountains &#8211; Geology'))
      .toBe('Tallest Mountains – Geology')
  })

  it('decodes hex entities too', () => {
    expect(decodeSearchEntities('it&#x27;s &#x2014; here')).toBe("it's — here")
  })

  it('unescapes &amp; LAST so &amp;lt; does not become a real tag', () => {
    expect(decodeSearchEntities('a &amp;lt; b')).toBe('a &lt; b')
  })
})

describe('web search failure reporting', () => {
  it('never reports a blocked search as "no results found"', () => {
    // DuckDuckGo answers 202 (inside res.ok) with an anti-bot page and zero
    // result markup. Reporting that as an empty result set tells the model the
    // world has no answer.
    //
    // Scoped to the keyless scraping path on purpose: the Brave backend runs
    // against a real API with a key, where zero results genuinely means zero
    // results, and saying so with is_error:false is correct there.
    const start = executor.indexOf('async function ddgSearch(')
    expect(start).toBeGreaterThan(-1)
    const body = executor.slice(start, executor.indexOf('\n}', start))
    // Strip comments: the function's own comment quotes the old wording to
    // explain why it was wrong, and asserting over prose would flag that.
    const code = body
      .split('\n')
      .filter((line) => !line.trim().startsWith('//'))
      .join('\n')
    expect(code).not.toMatch(/No results found for/)
    expect(body).toContain('Web search unavailable for')
    expect(body).toContain('likely blocked or rate-limited')
  })

  it('has a second, independent backend', () => {
    expect(executor).toContain('searchDuckDuckGo')
    expect(executor).toContain('searchMojeek')
    expect(executor).toMatch(/name: 'DuckDuckGo'[\s\S]{0,200}name: 'Mojeek'/)
  })

  it('reports every backend failure with its reason', () => {
    expect(executor).toContain('failures.map((f) => `  - ${f}`)')
    expect(executor).toContain('is_error: true')
  })
})

/**
 * A missing working directory must not disable tools that never use one.
 *
 * MEASURED live: with no valid working directory, a web search returned
 * "The configured working directory does not exist" twice and the model gave
 * up. The precondition exists so FILE tools fail with a nameable cause instead
 * of a raw ENOENT out of resolvePath — applying it to every tool turned one
 * misconfiguration into a dead toolbox.
 */
describe('working-directory precondition scope', () => {
  const dispatch = (() => {
    const start = executor.indexOf('export async function executeBuiltinTool(')
    return executor.slice(start, executor.indexOf('\n}\n', start))
  })()

  /** Tools whose implementation is not handed `workingDir`. */
  const dispatchIndependent = new Set(
    [...dispatch.matchAll(/case '([a-z_0-9]+)':\s*\n\s*result = (?:await )?\w+\(([^;]*)\);/g)]
      .filter(([, , argstr]) => !argstr.includes('workingDir'))
      .map(([, name]) => name),
  )

  /** The set the guard actually consults. */
  const declared = new Set(
    (executor
      .slice(executor.indexOf('WORKING_DIR_INDEPENDENT_TOOLS: ReadonlySet<string> = new Set(['))
      .split('])')[0]
      .match(/'([a-z_0-9]+)'/g) || []).map((s) => s.replace(/'/g, '')),
  )

  it('exempts exactly the tools that never receive workingDir', () => {
    expect([...declared].sort()).toEqual([...dispatchIndependent].sort())
  })

  it('never exempts a filesystem tool', () => {
    for (const fsTool of ['read_file', 'write_file', 'run_command', 'git', 'list_directory']) {
      expect(declared.has(fsTool)).toBe(false)
    }
  })

  it('guards the precondition on that set rather than running it always', () => {
    expect(dispatch).toContain('if (!WORKING_DIR_INDEPENDENT_TOOLS.has(toolName)) {')
  })
})

/**
 * The exemption must apply at BOTH gates.
 *
 * chat.ts rejects every tool when `overrides.workingDirectory` is absent,
 * BEFORE executeBuiltinTool runs — so the executor's exemption was unreachable
 * whenever the directory was UNSET (it only helped a configured-but-broken
 * one). Found by an adversarial re-review of the executor fix, not by the
 * executor's own tests, which never exercised the caller.
 */
describe('working-directory exemption applies at the chat gate too', () => {
  const chat = readFileSync(resolve(__dirname, '../src/main/ipc/chat.ts'), 'utf-8')

  it('the chat gate consults the same shared set', () => {
    expect(chat).toContain('WORKING_DIR_INDEPENDENT_TOOLS')
    expect(chat).toContain('!WORKING_DIR_INDEPENDENT_TOOLS.has(tc.function.name)')
  })

  it('imports the set rather than re-listing the tools', () => {
    expect(chat).toMatch(/import\s*\{[\s\S]*WORKING_DIR_INDEPENDENT_TOOLS[\s\S]*\}\s*from\s*"\.\.\/tools\/executor"/)
    expect(chat).not.toMatch(/'ddg_search',\s*\n\s*'web_search'/)
  })

  it('the executor exports it', () => {
    expect(executor).toContain('export const WORKING_DIR_INDEPENDENT_TOOLS')
  })
})

/**
 * Making the Max Thinking Tokens control RENDER on remote sessions is only
 * half the job — the value has to leave the app. The helper returned early on
 * isRemote, so the newly visible control was decorative.
 */
describe('thinking budget reaches remote sessions', () => {
  const chat = readFileSync(resolve(__dirname, '../src/main/ipc/chat.ts'), 'utf-8')

  it('no longer discards the budget for remote sessions', () => {
    const start = chat.indexOf('const applyThinkingBudget =')
    expect(start).toBeGreaterThan(-1)
    const body = chat.slice(start, start + 2400)
    expect(body).not.toMatch(/if \(isRemote \|\|/)
    expect(body).toContain('resolvedThinkingBudget == null')
  })

  it('still gates on the capability, so an older remote engine sends nothing', () => {
    const start = chat.indexOf('const applyThinkingBudget =')
    const body = chat.slice(start, start + 2400)
    expect(body).toContain('supportsThinkingBudget === true || thinkingBudgetSupported === true')
  })
})
