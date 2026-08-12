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
