import { readFileSync } from 'fs'
import { describe, expect, it } from 'vitest'
import {
  buildChatSettingsCompatibilityWarnings,
  type ChatSettingsCompatibilityInput,
} from '../src/renderer/src/components/chat/chatSettingsCompatibility'

function warnings(input: Partial<ChatSettingsCompatibilityInput>): string[] {
  return buildChatSettingsCompatibilityWarnings({
    messageCount: 3,
    currentModelPath: '/models/qwen',
    overrides: {},
    ...input,
  })
}

describe('chat settings cross-family compatibility warnings', () => {
  it('does not warn for empty chats', () => {
    expect(warnings({
      messageCount: 0,
      savedChatModelPath: '/models/old',
      currentModelPath: '/models/new',
      overrides: { enableThinking: true, reasoningEffort: 'medium' },
    })).toEqual([])
  })

  it('warns when a chat with history is opened against a different model path', () => {
    expect(warnings({
      savedChatModelPath: '/models/qwen36',
      currentModelPath: '/models/nemotron',
    })).toContain('This chat was started on qwen36 but is now attached to nemotron. Review saved per-chat settings before continuing.')
  })

  it('warns when saved Thinking On reaches a model with no reasoning parser', () => {
    expect(warnings({
      reasoningParser: undefined,
      overrides: { enableThinking: true },
    })).toContain('Saved Thinking On cannot take effect because this model has no detected reasoning parser.')
  })

  it('warns when stale reasoning effort reaches a parser that does not use effort levels', () => {
    expect(warnings({
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not used by qwen3. Reset the chat setting or switch to Auto.')
  })

  it('allows Hy3 low/high effort even though it reuses the qwen3 text parser', () => {
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'low' },
    })).toEqual([])
  })

  it('warns when Mistral carries a non-high effort from another model family', () => {
    expect(warnings({
      reasoningParser: 'mistral',
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not supported by Mistral. Use Auto or High.')
  })

  it('allows Hy3 low/high reasoning effort but warns on medium', () => {
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'low' },
    })).toEqual([])
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'high' },
    })).toEqual([])
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not supported by Hy3. Use Auto or High.')
  })

  it('warns when built-in tools are enabled without a detected tool parser', () => {
    expect(warnings({
      toolParser: undefined,
      overrides: { builtinToolsEnabled: true },
    })).toContain('Built-in tools are enabled, but this model has no detected tool parser. Tool calls may not round-trip.')
  })

  it('disables Thinking buttons when no reasoning parser is detected', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('const [detectedSupportsThinking, setDetectedSupportsThinking]')
    expect(source).toContain('const resolvedReasoningParser = resolveEffectiveReasoningParser({')
    expect(source).toContain("const thinkingSupported = resolvedReasoningParser !== 'none' && (")
    expect(source).toContain('reasoningParserIsEnabled(resolvedReasoningParser)')
    expect(source).toContain("const showReasoningEffort = (detectedReasoningEfforts?.length ?? 0) > 0")
    expect(source).toContain('const displayedEnableThinking = thinkingSupported ? overrides.enableThinking : undefined')
    expect(source).toContain('disabled={!thinkingSupported}')
  })

  it('shows Hy3 low/high effort controls without exposing medium', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain("detectedFamily === 'hy3' || effectiveReasoningParser === 'openai_gptoss' || effectiveReasoningParser === 'mistral'")
    expect(source).toContain("detectedReasoningEfforts.includes('medium')")
    expect(source).toContain("detectedFamily !== 'hy3'")
  })

  it('hides Thinking Off and exposes native effort levels when instruct mode is unsupported', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
    const ipc = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const thinkingOffSupported = detectedSupportsInstructMode !== false')
    expect(source).toContain('{thinkingOffSupported && (')
    expect(source).toContain("'chat.settings.thinkingNativeOnlyHelp'")
    expect(source).toContain('nextDetectedReasoningEfforts = detected?.supportedReasoningEfforts')
    expect(source).toContain('setDetectedReasoningEfforts(nextDetectedReasoningEfforts)')
    expect(ipc).toContain('supportsInstructMode === false && overrides?.enableThinking === false')
    expect(ipc).toContain('if (supportsInstructMode === false) return;')
  })

  it('exposes DSV4 Max without consulting legacy force-direct session state', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('const dsv4MaxEnabled =')
    expect(source).not.toContain("sessionConfig?.dsv4ForceDirect")
    expect(source).not.toContain("sessionConfig?.dsv4RawMax === true")
    expect(source).toContain("disabled={!dsv4MaxEnabled}")
    expect(source).toContain("overrides.reasoningEffort !== 'max' || !dsv4MaxEnabled")
  })

  it('keeps DSV4 model-default reasoning visible as a real Auto state', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('onClick={() => updateThinkingMode(undefined, undefined)}')
    expect(source).toContain('overrides.enableThinking == null')
    expect(source).toContain("{t('chat.settings.thinkingAuto')}")
  })

  it('does not silently mutate DSV4 output budgets when the user changes reasoning mode', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).not.toContain('DSV4_THINKING_MIN_TOKENS')
    expect(source).not.toContain('DSV4_MAX_MIN_TOKENS')
    expect(source).not.toContain('next.maxTokens = Math.max')
  })

  it('main IPC refuses stale local Thinking On when fresh detection has no reasoning parser', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const effectiveReasoningParser = resolveEffectiveReasoningParser({')
    expect(source).toContain('sessionHasReasoningParser = reasoningParserIsEnabled(')
    expect(source).toContain('supportsThinking: detected.supportsThinking,')
    expect(source).toContain('const effectiveEnableThinkingOverride =')
    expect(source).toContain('!sessionHasReasoningParser')
    expect(source).toContain('chatDetectedFamily !== "deepseek-v4"')
  })

  it('gates the Max Thinking Tokens field on engine-honoring or template budget support', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('const [supportsThinkingBudget, setSupportsThinkingBudget]')
    expect(source).toContain('nextSupportsThinkingBudget = detected?.supportsThinkingBudget')
    expect(source).toContain('setSupportsThinkingBudget(nextSupportsThinkingBudget)')
    expect(source).toContain('{(supportsThinkingBudget === true || thinkingBudgetSupported === true) && displayedEnableThinking !== false && (')
  })

  it('main IPC sends top-level max_thinking_tokens only for engine-honoring families', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    // The whole-block gate now keys off the registry/template budget capability,
    // and the deepseek-v4 special-case is gone from applyLocalThinkingBudget.
    expect(source).toContain('if (!(supportsThinkingBudget === true || thinkingBudgetSupported === true)) {')
    expect(source).toContain('supportsThinkingBudget = detected.supportsThinkingBudget;')
    expect(source).not.toContain('if (!sessionHasReasoningParser && chatDetectedFamily !== "deepseek-v4") {')
    // Template kwarg stays TEMPLATE-side only.
    expect(source).toContain('if (thinkingBudgetSupported !== false) {')
  })
})
