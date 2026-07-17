import { describe, expect, it } from 'vitest'
import { stripRedundantNamespacedToolPreview } from '../src/shared/namespacedToolScaffold'

const parsedFileInfo = [
  { function: { name: 'file_info' } },
]

describe('namespaced tool preview cleanup', () => {
  it('removes the exact Qwen3.6 preview when the parsed tool name matches', () => {
    const content = [
      '[Q27M8-D3-T1C]',
      'call:default_api:file_info{path: "panel/package.json"}',
      '>',
    ].join('\n')

    expect(stripRedundantNamespacedToolPreview(content, parsedFileInfo)).toBe('')
  })

  it('preserves legitimate prose surrounding a matching preview', () => {
    const content = [
      'I will inspect the requested file.',
      'call:default_api:file_info{path: "panel/package.json"}',
      '>',
      'This sentence is ordinary prose.',
    ].join('\n')

    expect(stripRedundantNamespacedToolPreview(content, parsedFileInfo)).toBe(
      'I will inspect the requested file.\nThis sentence is ordinary prose.',
    )
  })

  it('does not remove a preview for a different parsed tool', () => {
    const content = 'call:default_api:read_file{path: "README.md"}\n>'

    expect(stripRedundantNamespacedToolPreview(content, parsedFileInfo)).toBe(content)
  })
})
