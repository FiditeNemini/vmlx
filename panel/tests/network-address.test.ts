import { describe, expect, it } from 'vitest'
import { selectLanAddress } from '../src/main/network-address'

describe('gateway LAN display address selection', () => {
  it('skips APIPA interfaces that macOS enumerates before the active LAN', () => {
    expect(
      selectLanAddress({
        en4: [
          { address: '169.254.62.28', family: 'IPv4', internal: false },
        ],
        bridge0: [
          { address: '169.254.143.227', family: 'IPv4', internal: false },
        ],
        en9: [
          { address: '192.168.1.110', family: 'IPv4', internal: false },
        ],
      }),
    ).toBe('192.168.1.110')
  })

  it('prefers an RFC1918 LAN address over public and overlay candidates', () => {
    expect(
      selectLanAddress({
        public0: [{ address: '203.0.113.8', family: 'IPv4', internal: false }],
        overlay0: [{ address: '100.100.10.5', family: 'IPv4', internal: false }],
        wifi0: [{ address: '10.0.0.42', family: 'IPv4', internal: false }],
      }),
    ).toBe('10.0.0.42')
  })

  it('returns no advertised LAN URL when only loopback or link-local exists', () => {
    expect(
      selectLanAddress({
        lo0: [{ address: '127.0.0.1', family: 'IPv4', internal: true }],
        en4: [{ address: '169.254.1.9', family: 'IPv4', internal: false }],
      }),
    ).toBeUndefined()
  })
})
