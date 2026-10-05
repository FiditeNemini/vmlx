export type PowerModeStatus = {
  mode: 'high' | 'automatic' | 'low' | 'unknown' | 'unsupported'
  source: 'ac' | 'battery' | 'unknown'
}

/** Reports configuration for the active power source, never GPU clock speed. */
export function parsePowerMode(custom: string, battery: string): PowerModeStatus {
  const source = /Now drawing from 'AC Power'/.test(battery) ? 'ac'
    : /Now drawing from 'Battery Power'/.test(battery) ? 'battery' : 'unknown'
  if (source === 'unknown') return { source, mode: 'unknown' }
  let active = false
  const values: Record<string, number> = {}
  for (const line of custom.split(/\r?\n/)) {
    const section = line.match(/^\s*(.+ Power):\s*$/)
    if (section) {
      active = section[1] === (source === 'ac' ? 'AC Power' : 'Battery Power')
      continue
    }
    const value = active && line.match(/^\s*(powermode|lowpowermode|highpowermode)\s+(\d+)\s*$/)
    if (value) values[value[1]] = Number(value[2])
  }
  if ('powermode' in values) {
    return { source, mode: ({ 0: 'automatic', 1: 'low', 2: 'high' } as const)[values.powermode] ?? 'unknown' }
  }
  // Legacy releases expose separate switches. Missing switches do not prove
  // Automatic or High Power support; retain an honest unknown state.
  if (values.lowpowermode === 1 && values.highpowermode === 1) return { source, mode: 'unknown' }
  if (values.highpowermode === 1) return { source, mode: 'high' }
  if (values.lowpowermode === 1) return { source, mode: 'low' }
  if (values.highpowermode === 0 && values.lowpowermode === 0) return { source, mode: 'automatic' }
  return { source, mode: 'unknown' }
}
