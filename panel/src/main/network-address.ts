export interface NetworkAddressEntry {
  address?: string
  family?: string
  internal?: boolean
}

export type NetworkInterfaceMap = Record<
  string,
  NetworkAddressEntry[] | undefined
>

function ipv4Rank(address: string): number {
  const parts = address.split('.').map(Number)
  if (
    parts.length !== 4 ||
    parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)
  ) {
    return 0
  }

  const [a, b] = parts
  // Loopback, unspecified, multicast, and APIPA/link-local addresses are not
  // useful dashboard URLs for peers on the user's LAN.
  if (a === 0 || a === 127 || a >= 224 || (a === 169 && b === 254)) return 0

  // Prefer RFC1918 interfaces. macOS commonly enumerates inactive USB or
  // Thunderbolt APIPA interfaces before the real Ethernet/Wi-Fi route.
  if (a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168)) {
    return 3
  }
  // CGNAT addresses (often a private overlay) are preferable to a public
  // address when no RFC1918 LAN interface is present.
  if (a === 100 && b >= 64 && b <= 127) return 2
  return 1
}

export function selectLanAddress(nets: NetworkInterfaceMap): string | undefined {
  let selected: string | undefined
  let selectedRank = 0

  for (const addrs of Object.values(nets)) {
    for (const addr of addrs || []) {
      if (addr.family !== 'IPv4' || addr.internal || !addr.address) continue
      const rank = ipv4Rank(addr.address)
      if (rank > selectedRank) {
        selected = addr.address
        selectedRank = rank
      }
    }
  }

  return selected
}
