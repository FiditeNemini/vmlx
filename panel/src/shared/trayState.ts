type ProcessRow = { port: number; status: string; gpuMemoryMB?: number }
type SessionRow = { id: string; port: number; host?: string; status: string; type?: string }

export function localApiUrl(host: string, port: number, openAI = false): string {
  let address = host || '127.0.0.1'
  if (address === '0.0.0.0') address = '127.0.0.1'
  if (address === '::' || address === '[::]') address = '::1'
  if (address.includes(':') && !address.startsWith('[')) address = `[${address}]`
  return `http://${address}:${port}${openAI ? '/v1' : ''}`
}

export function sessionShownByProcess(session: SessionRow, processes: ProcessRow[]): boolean {
  const host = session.host || '127.0.0.1'
  return session.type !== 'remote' && ['localhost', '127.0.0.1', '0.0.0.0', '::', '[::]', '::1', '[::1]'].includes(host)
    && processes.some(p => p.port === session.port && ['running', 'starting'].includes(p.status))
}

export function summarizeTrayState(processes: ProcessRow[], sessions: SessionRow[], memory: ReadonlyMap<string, number>) {
  const uniqueSessions = sessions.filter(s => !sessionShownByProcess(s, processes))
  const running = processes.filter(p => p.status === 'running').length + uniqueSessions.filter(s => s.status === 'running').length
  const loading = processes.filter(p => p.status === 'starting').length + uniqueSessions.filter(s => s.status === 'loading').length
  const standby = uniqueSessions.filter(s => s.status === 'standby').length
  // Sleeping/stopped sessions have released the model. Ignore stale health
  // samples, and never add the same server's ProcessManager and session data.
  const memoryMB = processes.filter(p => ['running', 'starting'].includes(p.status))
    .reduce((n, p) => n + Math.max(0, p.gpuMemoryMB || 0), 0)
    + uniqueSessions.filter(s => ['running', 'loading'].includes(s.status))
      .reduce((n, s) => n + Math.max(0, memory.get(s.id) || 0), 0)
  return { running, loading, standby, memoryMB }
}
