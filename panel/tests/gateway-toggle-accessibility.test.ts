import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * The LAN and single-model gateway toggles exist in TWO components. The drawer
 * copy was built with a label and a pressed state; the API-tab copy was not, so
 * assistive tech saw an unlabeled button with no state on the two controls that
 * matter most — one exposes the server on 0.0.0.0, the other decides whether a
 * second model may stay resident in RAM.
 *
 * Found by sweeping the live app: every visible button was checked for an
 * accessible name, and only these two came back empty.
 */
const ROOT = join(__dirname, '..')

const SOURCES = {
  'API tab': 'src/renderer/src/components/api/ApiDashboard.tsx',
  'server drawer': 'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
}

/** Pull each toggle <button ...> open tag out of a source file. */
function toggleOpenTags(source: string): string[] {
  const tags: string[] = []
  const marker = 'h-5 w-9 items-center rounded-full'
  let at = source.indexOf(marker)
  while (at >= 0) {
    const open = source.lastIndexOf('<button', at)
    const close = source.indexOf('>', at)
    if (open >= 0 && close > open) tags.push(source.slice(open, close))
    at = source.indexOf(marker, at + 1)
  }
  return tags
}

describe('gateway toggles are announceable', () => {
  for (const [label, rel] of Object.entries(SOURCES)) {
    it(`${label}: every pill toggle has a name and a pressed state`, () => {
      const source = readFileSync(join(ROOT, rel), 'utf8')
      const tags = toggleOpenTags(source)
      expect(tags.length).toBeGreaterThan(0)
      for (const tag of tags) {
        expect(tag, `missing aria-label:\n${tag}`).toMatch(/aria-label=/)
        expect(tag, `missing aria-pressed:\n${tag}`).toMatch(/aria-pressed=/)
      }
    })
  }

  it('both copies of each control expose the same automation hook', () => {
    const api = readFileSync(join(ROOT, SOURCES['API tab']), 'utf8')
    const drawer = readFileSync(join(ROOT, SOURCES['server drawer']), 'utf8')
    expect(api).toContain('data-vmlx-control="gateway-single-model-mode"')
    expect(drawer).toContain('data-vmlx-control="gateway-single-model-mode"')
    expect(api).toContain('data-vmlx-control="gateway-lan"')
  })
})
