import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { buildNativeMtpLaunchArgs } from '../src/shared/nativeMtpLaunchArgs'

/**
 * An external draft model and the bundle's own MTP heads are two speculative
 * decoders bidding for the same decode step. The panel used to ship BOTH:
 * `--speculative-model .../DFlash2` alongside `--native-mtp-depth 3`, because
 * the native-MTP block was gated only on `!dsv4Active && nativeMtp?.supported`
 * and never looked at the speculative model.
 *
 * The user-visible symptom was decode rate jumping around on identical
 * requests -- the same code prompt measured 31.6, 44.5 and 45.3 t/s back to
 * back on one such session.
 */
const sessions = readFileSync('src/main/sessions.ts', 'utf-8')

function nativeMtpBlock(): string {
  const start = sessions.indexOf('const nativeMtp = (detected as any).nativeMtp')
  expect(start).toBeGreaterThan(-1)
  return sessions.slice(start, start + 2600)
}

describe('native MTP and an external drafter are mutually exclusive', () => {
  it('the native-MTP branch consults the external speculative model', () => {
    // Without this the two flag sets are emitted together.
    expect(nativeMtpBlock()).toContain('compatibleExternalSpeculative')
  })

  it('an active external drafter disables native MTP explicitly', () => {
    const block = nativeMtpBlock()
    expect(block).toContain('args.push(...buildNativeMtpLaunchArgs({')
    expect(block).toContain('externalSpeculativeActive: compatibleExternalSpeculative')
    expect(buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 3,
      mode: 'auto',
      externalSpeculativeActive: true,
    })).toEqual(['--disable-native-mtp'])
  })

  it('the depth flag is NOT emitted on that branch', () => {
    const args = buildNativeMtpLaunchArgs({
      supported: true,
      detectedDepth: 3,
      mode: 'auto',
      externalSpeculativeActive: true,
    })
    expect(args).not.toContain('--native-mtp-depth')
  })

  it('the drafter is still pushed when it is compatible', () => {
    expect(sessions).toContain("args.push('--speculative-model', externalSpeculativeModel)")
  })
})
