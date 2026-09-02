import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('manual session single-model enforcement', () => {
  it('fails closed at the global detector for proof-owned Electron lifecycles', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('async detect(): Promise<DetectedProcess[]>')
    const end = source.indexOf('private parsePsLine', start)
    const block = source.slice(start, end)
    const proofGuard = block.indexOf(
      'shouldUseProofOwnedEngineLifecycle(process.argv, process.env)',
    )
    const processScan = block.indexOf("execSync('ps aux'")

    expect(source).toContain(
      "import { shouldUseProofOwnedEngineLifecycle } from '../shared/userDataOverride'",
    )
    expect(proofGuard).toBeGreaterThanOrEqual(0)
    expect(processScan).toBeGreaterThan(proofGuard)
    expect(block.slice(proofGuard, processScan)).toContain('return detected')

    const stopStart = source.indexOf('async stopAll(): Promise<void>')
    const stopEnd = source.indexOf('// ─── Queries', stopStart)
    const stopBlock = source.slice(stopStart, stopEnd)
    expect(stopBlock).toContain('const processes = await this.detect()')
    expect(stopBlock).toContain("managed.process.kill('SIGTERM')")
    expect(stopBlock).toContain("managed.process.kill('SIGKILL')")
  })

  it('serializes UI starts and unloads every other local engine before launch', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('async startSession(')
    const end = source.indexOf('private async _startSessionInner', start)
    const block = source.slice(start, end)

    expect(source).toContain('private singleModelStartTransitionPending: Promise<void>')
    // The literal moved to src/shared/gatewaySettingsKeys.ts so a typo cannot
    // silently disable enforcement at one call site. Assert the gate still reads
    // the setting HERE, through the shared key.
    expect(block).toContain('isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))')
    expect(block).toContain('await this.stopDetectedLocalEnginesForSingleModel(sessionId)')
    expect(block).toContain('await this.adoptDetectedTargetProcessForStart(sessionId)')
    expect(block).toContain("other.type !== 'remote'")
    expect(block).toContain("['running', 'loading', 'standby'].includes(other.status)")
    expect(block).toContain('await this.stopSession(other.id)')
    expect(block.indexOf('await this.stopDetectedLocalEnginesForSingleModel(sessionId)')).toBeLessThan(
      block.indexOf('this._startSessionInner(sessionId, options)'),
    )
    expect(block.indexOf('await this.stopSession(other.id)')).toBeLessThan(
      block.indexOf('this._startSessionInner(sessionId, options)'),
    )
    expect(block).toContain('const previous = this.singleModelStartTransitionPending.catch')
    expect(block).toContain('this.singleModelStartTransitionPending = previous.then')
  })

  it('shows the wired-limit dialog for UI starts without gating the spawn, and skips it for gateway JIT', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('private async _startSessionInner(')
    const end = source.indexOf('// Never kill arbitrary processes by port', start)
    const block = source.slice(start, end)

    expect(block).toContain("options?.launchOrigin !== 'gateway'")
    // Advisory only: the modal must be fire-and-forget. Awaiting it made the
    // application-modal sheet gate the engine spawn until dismissed.
    expect(block).toContain('void dialog.showMessageBox({')
    expect(block).not.toContain('await dialog.showMessageBox({')
    expect(block.indexOf("options?.launchOrigin !== 'gateway'")).toBeLessThan(
      block.indexOf('void dialog.showMessageBox({'),
    )
    expect(block).toContain('this.emit(\'session:log\'')
  })

  it('kills detected stale engines, not only sessions whose DB status is running', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    expect(source).toContain('async enforceSingleModelLocalProcessContract(')
    expect(source).toContain('await this.stopDetectedLocalEnginesForSingleModel(targetSessionId)')
    expect(source).toContain('return this.adoptDetectedTargetProcessForStart(targetSessionId)')
    const start = source.indexOf('private async stopDetectedLocalEnginesForSingleModel')
    const end = source.indexOf('private async adoptDetectedTargetProcessForStart', start)
    const block = source.slice(start, end)

    expect(start).toBeGreaterThanOrEqual(0)
    expect(block).toContain('const detected = await this.detect()')
    expect(block).toContain("s.type !== 'remote'")
    expect(block).toContain('normalizePath(s.modelPath) === livePath || s.port === proc.port || s.pid === proc.pid')
    expect(block).toContain('await this.terminateDetectedLocalEngine(proc, allSessions)')
  })

  it('single-model adoption prunes extra healthy engines instead of adopting all of them', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const detectStart = source.indexOf('async detectAndAdoptAll()')
    const detectBlock = source.slice(detectStart, source.indexOf('const adopted: Session[]', detectStart))
    const pruneStart = source.indexOf('private async pruneDetectedProcessesForSingleModel')
    const pruneEnd = source.indexOf('// ─── Global Health Monitor', pruneStart)
    const pruneBlock = source.slice(pruneStart, pruneEnd)

    expect(detectBlock).toContain('processes = await this.pruneDetectedProcessesForSingleModel(processes)')
    expect(pruneBlock).toContain('!isGatewaySettingEnabled(db.getSetting(GATEWAY_SINGLE_MODEL_MODE_KEY))')
    expect(pruneBlock).toContain('const healthy = processes.filter(proc => proc.healthy)')
    expect(pruneBlock).toContain("['running', 'loading', 'standby'].includes(owner.status)")
    expect(pruneBlock).toContain('keeping pid=${keep.pid} port=${keep.port}')
    expect(pruneBlock).toContain('await this.terminateDetectedLocalEngine(proc, sessions)')
    expect(pruneBlock).toContain('return processes.filter(proc => proc === keep || !healthy.includes(proc))')
  })

  it('detected process termination kills the live PID and marks any owning row stopped', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('private async terminateDetectedLocalEngine')
    const end = source.indexOf('private async adoptDetectedTargetProcessForStart', start)
    const block = source.slice(start, end)

    expect(start).toBeGreaterThanOrEqual(0)
    expect(block).toContain('this.killPid(proc.pid)')
    expect(block).toContain("this.killPid(proc.pid, 'SIGKILL')")
    expect(block).toContain("status: 'stopped'")
    expect(block).toContain("pid: undefined")
  })

  it('adopts a healthy already-running target instead of spawning a duplicate on the same port', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('private async adoptDetectedTargetProcessForStart')
    const end = source.indexOf('private async _startSessionInner', start)
    const block = source.slice(start, end)

    expect(start).toBeGreaterThanOrEqual(0)
    expect(block).toContain('const detected = await this.detect()')
    expect(block).toContain('p.healthy')
    expect(block).toContain('normalizePath(p.modelPath) === targetPath')
    expect(block).toContain('p.port === session.port')
    expect(block).toContain("this.processes.set(session.id, { process: null, adoptedPid: proc.pid })")
    expect(block).toContain("this.emit('session:ready'")
    expect(block).toContain('return true')
  })

  it('re-adopts a healthy replacement process before marking a stale pid down', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const adoptStart = source.indexOf('private async adoptHealthyReplacementForSession')
    const adoptEnd = source.indexOf('private async incrementFailAndCheck', adoptStart)
    const adoptBlock = source.slice(adoptStart, adoptEnd)
    const failStart = source.indexOf('private async incrementFailAndCheck')
    const failEnd = source.indexOf('private handleSessionDown', failStart)
    const failBlock = source.slice(failStart, failEnd)

    expect(adoptStart).toBeGreaterThanOrEqual(0)
    expect(adoptBlock).toContain('const detected = await this.detect()')
    expect(adoptBlock).toContain('p.healthy')
    expect(adoptBlock).toContain('p.port === session.port')
    expect(adoptBlock).toContain('sessionMatchesModelPath(p.modelPath, targetPath)')
    expect(adoptBlock).toContain("this.processes.set(session.id, { process: null, adoptedPid: proc.pid })")
    expect(adoptBlock).toContain("this.emit('session:ready'")
    expect(failBlock).toContain('if (await this.adoptHealthyReplacementForSession(session)) return')
    expect(failBlock.indexOf('adoptHealthyReplacementForSession')).toBeLessThan(
      failBlock.indexOf('this.handleSessionDown(sessionId)'),
    )
  })

  it('does not let stale managed child state override a fresh db pid or adopted pid', () => {
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const aliveStart = source.indexOf('private isProcessAlive')
    const aliveEnd = source.indexOf('private getSessionByPort', aliveStart)
    const aliveBlock = source.slice(aliveStart, aliveEnd)
    const exitStart = source.indexOf("proc.on('exit'")
    const exitEnd = source.indexOf("proc.on('error'", exitStart)
    const exitBlock = source.slice(exitStart, exitEnd)

    expect(aliveStart).toBeGreaterThanOrEqual(0)
    expect(aliveBlock).toContain('const candidates = [')
    expect(aliveBlock).toContain('dbPid,')
    expect(aliveBlock).toContain('managed?.adoptedPid')
    expect(aliveBlock).toContain('managed?.process?.pid')
    expect(aliveBlock).toContain('for (const pid of [...new Set(candidates)])')

    expect(exitStart).toBeGreaterThanOrEqual(0)
    expect(exitBlock).toContain('currentSession?.pid && currentSession.pid !== proc.pid')
    expect(exitBlock).toContain('db now owns pid=${currentSession.pid}')
    expect(exitBlock).toContain('if (managed && managed.process !== proc)')
    expect(exitBlock).toContain('Ignoring stale child exit')
    expect(exitBlock.indexOf('currentSession?.pid && currentSession.pid !== proc.pid')).toBeLessThan(
      exitBlock.indexOf('db.updateSession(sessionId, {'),
    )
  })
})
