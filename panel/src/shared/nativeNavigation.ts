export const NATIVE_NAVIGATION_ACTIONS = ['new-chat', 'chat', 'servers', 'models', 'api', 'preferences'] as const
export type NativeNavigationAction = typeof NATIVE_NAVIGATION_ACTIONS[number]

export function isNativeNavigationAction(value: unknown): value is NativeNavigationAction {
  return typeof value === 'string' && (NATIVE_NAVIGATION_ACTIONS as readonly string[]).includes(value)
}
