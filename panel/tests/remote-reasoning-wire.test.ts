import { describe, expect, it } from 'vitest'
import { applyReasoningRequestFields, remoteReasoningFormatForUrl } from '../src/shared/reasoningEffortPolicy'
import { readFileSync } from 'node:fs'
import { buildChatSettingsCompatibilityWarnings } from '../src/renderer/src/components/chat/chatSettingsCompatibility'

function request(overrides: Record<string, unknown>) {
  const body: Record<string, any> = {}
  applyReasoningRequestFields(body, {
    isRemote: true,
    sessionHasReasoningParser: false,
    ...overrides,
  })
  return body
}

describe('remote reasoning request contract', () => {
  it('matches provider host identity, never a substring in a proxy path or hostile host', () => {
    expect(remoteReasoningFormatForUrl('https://openrouter.ai/api/v1')).toBe('openrouter')
    expect(remoteReasoningFormatForUrl('https://openrouter.ai.invalid/api/v1')).toBe('openai')
    expect(remoteReasoningFormatForUrl('https://proxy.test/openrouter.ai/v1')).toBe('openai')
  })
  it('uses native Responses reasoning instead of a Chat-only field', () => {
    expect(request({ remoteReasoningFormat: 'openai', wireApi: 'responses',
      enableThinking: true, reasoningEffort: 'high' })).toEqual({ reasoning: { effort: 'high' } })
  })
  it('does not require a local text parser for provider-native Chat reasoning', () => {
    expect(request({ remoteReasoningFormat: 'openai', wireApi: 'completions',
      enableThinking: true, reasoningEffort: 'low' })).toEqual({ reasoning_effort: 'low' })
  })
  it.each(['completions', 'responses'])('omits Auto on %s', wireApi => {
    expect(request({ remoteReasoningFormat: 'openai', wireApi })).toEqual({})
  })
  it.each(['completions', 'responses'])('does not silently treat explicit On as provider Auto on %s', wireApi => {
    expect(() => request({ remoteReasoningFormat: 'openai', wireApi, enableThinking: true }))
      .toThrow(/explicit reasoning effort/i)
  })
  it.each(['completions', 'responses'])('sends explicit Off on %s without vMLX extensions', wireApi => {
    expect(request({ remoteReasoningFormat: 'openai', wireApi, enableThinking: false,
      reasoningEffort: 'high' })).toEqual(wireApi === 'responses'
      ? { reasoning: { effort: 'none' } } : { reasoning_effort: 'none' })
  })
  it('uses OpenRouter Chat enabled/effort fields, not enable_thinking', () => {
    expect(request({ remoteReasoningFormat: 'openrouter', wireApi: 'completions',
      enableThinking: true, reasoningEffort: 'xhigh' }))
      .toEqual({ reasoning: { enabled: true, effort: 'xhigh' } })
  })
  it('validates an explicitly advertised remote effort list before mutating', () => {
    expect(() => request({ remoteReasoningFormat: 'vmlx', sessionHasReasoningParser: true,
      reasoningEffort: 'high', supportedReasoningEfforts: ['low', 'medium', 'xhigh'] }))
      .toThrow(/not supported/)
  })
  it('does not turn an advertised empty effort list into generic support', () => {
    expect(() => request({ remoteReasoningFormat: 'openrouter', wireApi: 'completions',
      enableThinking: true, reasoningEffort: 'high', supportedReasoningEfforts: [] }))
      .toThrow(/not supported/)
  })
  it('keeps the declared vMLX toggle and native Responses effort together', () => {
    expect(request({ remoteReasoningFormat: 'vmlx', wireApi: 'responses',
      sessionHasReasoningParser: true, enableThinking: true, reasoningEffort: 'medium' }))
      .toEqual({ enable_thinking: true, reasoning: { effort: 'medium' } })
  })
  it('keeps declared vMLX Chat On without inventing an effort', () => {
    expect(request({ remoteReasoningFormat: 'vmlx', wireApi: 'completions', enableThinking: true }))
      .toEqual({ enable_thinking: true })
  })
  it('uses an advertised default effort for native On, not a guessed tier', () => {
    expect(request({ remoteReasoningFormat: 'openai', wireApi: 'responses', enableThinking: true,
      defaultReasoningEffort: 'low' })).toEqual({ reasoning: { effort: 'low' } })
  })
  it('does not falsely claim remote native tool/reasoning transport needs local parsers', () => {
    expect(buildChatSettingsCompatibilityWarnings({ messageCount: 2, isRemote: true,
      supportedReasoningEfforts: ['low'],
      overrides: { enableThinking: true, reasoningEffort: 'low', builtinToolsEnabled: true } })).toEqual([])
  })
  it('keeps both production request lanes on the same protocol-aware builder', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    expect(source.match(/remoteReasoningFormat: isRemote \?/g)).toHaveLength(2)
    expect(source).toContain("wireApi: 'responses'")
    expect(source).toContain("wireApi: 'completions'")
    expect(source).not.toContain('allowRequestControls: !isStrictApi')
  })
})
