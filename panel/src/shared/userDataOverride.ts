type UserDataEnv = Record<string, string | undefined>

function hasSecondaryInstanceFlag(argv: readonly string[], env: UserDataEnv): boolean {
  if (env.VMLX_ALLOW_SECONDARY_INSTANCE === '1' || env.VMLINUX_ALLOW_SECONDARY_INSTANCE === '1') {
    return true
  }
  return argv.includes('--vmlx-allow-secondary-instance')
}

export function resolveUserDataOverride(argv: readonly string[], env: UserDataEnv): string {
  const argPrefixes = ['--vmlx-user-data-dir=', '--user-data-dir=']

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i] || ''
    const prefix = argPrefixes.find((value) => arg.startsWith(value))
    if (prefix) {
      const value = arg.slice(prefix.length).trim()
      if (value) return value
      continue
    }
    if ((arg === '--vmlx-user-data-dir' || arg === '--user-data-dir') && argv[i + 1]) {
      const value = String(argv[i + 1]).trim()
      if (value) return value
    }
  }

  return (
    env.VMLINUX_USER_DATA_DIR ||
    env.VMLX_USER_DATA_DIR ||
    ''
  ).trim()
}

export function shouldAllowSecondaryInstance(argv: readonly string[], env: UserDataEnv): boolean {
  return !!resolveUserDataOverride(argv, env) && hasSecondaryInstanceFlag(argv, env)
}

export function shouldUseProofOwnedEngineLifecycle(
  argv: readonly string[],
  env: UserDataEnv,
): boolean {
  const requested =
    env.VMLINUX_PROOF_OWNED_ENGINE_LIFECYCLE === '1' ||
    env.VMLX_PROOF_OWNED_ENGINE_LIFECYCLE === '1'
  return requested && shouldAllowSecondaryInstance(argv, env)
}

export function resolveProofOwnedGatewayPort(
  argv: readonly string[],
  env: UserDataEnv,
): number | undefined {
  if (!shouldUseProofOwnedEngineLifecycle(argv, env)) return undefined
  const raw = (
    env.VMLINUX_PROOF_GATEWAY_PORT ||
    env.VMLX_PROOF_GATEWAY_PORT ||
    ''
  ).trim()
  if (!raw) return undefined
  const port = Number(raw)
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('proof-owned gateway port must be an integer from 1 to 65535')
  }
  return port
}
