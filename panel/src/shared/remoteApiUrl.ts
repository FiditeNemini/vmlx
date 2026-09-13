/**
 * The transport appends /v1/<route>. Accept either a server root or the API
 * base URL commonly copied from SDK configuration, without doubling /v1.
 * Preserve proxy/tenant path prefixes and the user's persisted URL verbatim.
 */
export function remoteServerBaseUrl(value: string): string {
  return value.trim().replace(/\/+$/, '').replace(/\/v1$/, '')
}
