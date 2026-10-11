import { describe, expect, it } from 'vitest'
import { buildCommandPreview } from '../src/renderer/src/components/sessions/SessionSettings'
import { DEFAULT_CONFIG } from '../src/renderer/src/components/sessions/SessionConfigForm'

describe('actual DFlash2 command preview', () => {
  const detected = { family: 'glm5-next', isMultimodal: true, nativeGlmSsd: true }
  it('shows the bundled drafter, non-batched lane, and saved SSD settings', () => {
    const preview = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG,
      blockDiskCacheDir: '/private/ssd', blockDiskCacheMaxPercent: 7 }, detected,
      '/models/target/dflash2', true)
    expect(preview).toContain('--no-continuous-batching')
    expect(preview).toContain('--speculative-model')
    expect(preview).toContain('/models/target/dflash2')
    expect(preview).toContain('--block-disk-cache-max-percent')
    expect(preview).toContain('/private/ssd')
    expect(preview).not.toContain('--use-paged-cache')
  })
  it('shows the bundled-drafter opt-out rather than silently relying on the CLI default', () => {
    const preview = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG,
      useBundledDflash2: false }, detected, '', true)
    expect(preview).toContain('--no-bundled-dflash2')
    expect(preview).not.toContain('--speculative-model')
    expect(preview).toContain('--continuous-batching')
  })
})
