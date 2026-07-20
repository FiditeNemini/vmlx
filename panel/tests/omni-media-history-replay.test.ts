import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const chatSource = readFileSync(resolve(__dirname, '../src/main/ipc/chat.ts'), 'utf8')

describe('Nemotron Omni historical media replay', () => {
  it('keeps prior media bytes on post-media turns for the stateful Omni dispatcher', () => {
    expect(chatSource).toContain('function shouldPreserveHistoricalMediaForOmni(')
    expect(chatSource).toContain('existsSync(join(modelPath, "config_omni.json"))')
    expect(chatSource).toContain('if (preserveHistoricalMediaForOmni) chatIsMultimodal = true')
    expect(chatSource).toContain('!preserveHistoricalMediaForOmni')
  })
})
