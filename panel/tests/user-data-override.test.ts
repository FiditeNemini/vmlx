import { describe, expect, it } from 'vitest'
import {
  resolveProofOwnedGatewayPort,
  resolveUserDataOverride,
  shouldAllowSecondaryInstance,
  shouldUseProofOwnedEngineLifecycle,
} from '../src/shared/userDataOverride'

describe('local proof user-data override', () => {
  it('prefers explicit launch args over env vars', () => {
    expect(
      resolveUserDataOverride(
        ['vMLX', '--vmlx-user-data-dir=/tmp/vmlx-proof-arg'],
        {
          VMLINUX_USER_DATA_DIR: '/tmp/vmlx-proof-env',
          VMLX_USER_DATA_DIR: '/tmp/vmlx-short-env',
        },
      ),
    ).toBe('/tmp/vmlx-proof-arg')
  })

  it('supports split args and the shorter env override', () => {
    expect(resolveUserDataOverride(['vMLX', '--user-data-dir', '/tmp/split'], {})).toBe('/tmp/split')
    expect(resolveUserDataOverride(['vMLX'], { VMLX_USER_DATA_DIR: '/tmp/env' })).toBe('/tmp/env')
  })

  it('ignores blank values', () => {
    expect(resolveUserDataOverride(['vMLX', '--vmlx-user-data-dir=  '], {})).toBe('')
    expect(resolveUserDataOverride(['vMLX'], { VMLINUX_USER_DATA_DIR: ' ' })).toBe('')
  })

  it('allows secondary instances only for explicit isolated proof launches', () => {
    expect(shouldAllowSecondaryInstance(['vMLX'], {})).toBe(false)
    expect(
      shouldAllowSecondaryInstance(['vMLX', '--vmlx-user-data-dir=/tmp/proof'], {}),
    ).toBe(false)
    expect(
      shouldAllowSecondaryInstance(['vMLX', '--vmlx-allow-secondary-instance'], {}),
    ).toBe(false)
    expect(
      shouldAllowSecondaryInstance(
        ['vMLX', '--vmlx-user-data-dir=/tmp/proof', '--vmlx-allow-secondary-instance'],
        {},
      ),
    ).toBe(true)
    expect(
      shouldAllowSecondaryInstance(['vMLX'], {
        VMLX_USER_DATA_DIR: '/tmp/proof',
        VMLX_ALLOW_SECONDARY_INSTANCE: '1',
      }),
    ).toBe(true)
  })

  it('enables proof-owned engine lifecycle only with all three isolation gates', () => {
    const argv = [
      'vMLX',
      '--vmlx-user-data-dir=/tmp/proof',
      '--vmlx-allow-secondary-instance',
    ]
    expect(shouldUseProofOwnedEngineLifecycle(['vMLX'], {
      VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
    })).toBe(false)
    expect(shouldUseProofOwnedEngineLifecycle(
      ['vMLX', '--vmlx-user-data-dir=/tmp/proof'],
      { VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1' },
    )).toBe(false)
    expect(shouldUseProofOwnedEngineLifecycle(argv, {})).toBe(false)
    expect(shouldUseProofOwnedEngineLifecycle(argv, {
      VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
    })).toBe(true)
    expect(shouldUseProofOwnedEngineLifecycle(['vMLX'], {
      VMLINUX_USER_DATA_DIR: '/tmp/proof',
      VMLINUX_ALLOW_SECONDARY_INSTANCE: '1',
      VMLINUX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
    })).toBe(true)
  })

  it('accepts a proof gateway port only inside proof-owned lifecycle', () => {
    const argv = [
      'vMLX',
      '--vmlx-user-data-dir=/tmp/proof',
      '--vmlx-allow-secondary-instance',
    ]
    expect(resolveProofOwnedGatewayPort(argv, {
      VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
      VMLX_PROOF_GATEWAY_PORT: '18081',
    })).toBe(18081)
    expect(resolveProofOwnedGatewayPort(['vMLX'], {
      VMLX_PROOF_GATEWAY_PORT: '18081',
    })).toBeUndefined()
    expect(() => resolveProofOwnedGatewayPort(argv, {
      VMLX_PROOF_OWNED_ENGINE_LIFECYCLE: '1',
      VMLX_PROOF_GATEWAY_PORT: '65536',
    })).toThrow(/integer from 1 to 65535/)
  })
})
