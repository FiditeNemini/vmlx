import { describe, expect, it } from 'vitest'

import { CachePanelRequestGuard } from '../src/renderer/src/components/sessions/CachePanel'

describe('CachePanel request lifecycle', () => {
  it('allows only the latest stats request to update the panel', () => {
    const guard = new CachePanelRequestGuard()
    const first = guard.beginLatest()
    const second = guard.beginLatest()

    expect(first).not.toBeNull()
    expect(second).not.toBeNull()
    expect(guard.isCurrent(first!)).toBe(false)
    expect(guard.isCurrent(second!)).toBe(true)
  })

  it('invalidates an in-flight request during effect cleanup', () => {
    const guard = new CachePanelRequestGuard()
    const request = guard.beginLatest()

    guard.invalidateRequests()

    expect(request).not.toBeNull()
    expect(guard.isCurrent(request!)).toBe(false)
  })

  it('rejects work captured for a previous session identity', () => {
    const guard = new CachePanelRequestGuard()
    const oldIdentity = guard.captureIdentity()
    const oldRequest = guard.beginLatest(oldIdentity)

    guard.resetIdentity()

    expect(guard.isIdentityCurrent(oldIdentity)).toBe(false)
    expect(guard.beginLatest(oldIdentity)).toBeNull()
    expect(oldRequest).not.toBeNull()
    expect(guard.isCurrent(oldRequest!)).toBe(false)
  })

  it('serializes cache actions and suspends background polling until completion', () => {
    const guard = new CachePanelRequestGuard()
    const stalePoll = guard.beginLatest()
    const action = guard.beginAction()

    expect(stalePoll).not.toBeNull()
    expect(action).not.toBeNull()
    expect(guard.isCurrent(stalePoll!)).toBe(false)
    expect(guard.beginLatest()).toBeNull()
    expect(guard.beginAction()).toBeNull()

    guard.finishAction(action!)

    const resumedPoll = guard.beginLatest()
    expect(resumedPoll).not.toBeNull()
    expect(guard.isCurrent(resumedPoll!)).toBe(true)
  })

  it('invalidates an active action on session cleanup', () => {
    const guard = new CachePanelRequestGuard()
    const action = guard.beginAction()

    guard.invalidateRequests()

    expect(action).not.toBeNull()
    expect(guard.isCurrent(action!)).toBe(false)
    expect(guard.beginLatest()).not.toBeNull()
  })
})
