/**
 * Tests for chat request body building — verifies that all sampling parameters
 * (temperature, top_p, top_k, min_p, repeat_penalty, stop, max_tokens, etc.)
 * are correctly forwarded for both Chat Completions and Responses API wire formats,
 * and that remote-only gating (chat_template_kwargs exclusion) works correctly.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { dsv4OutputBudget } from '../src/shared/dsv4RequestBudget'
import {
    applyReasoningRequestFields,
    type ReasoningEffort,
} from '../src/shared/reasoningEffortPolicy'
import { splitResponsesSystemMessages } from '../src/shared/responsesSystemMessages'

// ─── buildRequestBody logic (extracted from chat.ts) ─────────────────────────

interface ChatOverrides {
    temperature?: number
    topP?: number
    topK?: number
    minP?: number
    maxTokens?: number
    maxThinkingTokens?: number
    repeatPenalty?: number
    systemPrompt?: string
    stopSequences?: string
    wireApi?: 'completions' | 'responses'
    builtinToolsEnabled?: boolean
    enableThinking?: boolean
    reasoningEffort?: string
}

function buildRequestBody(
    wireApi: 'completions' | 'responses',
    modelName: string,
    requestMessages: any[],
    overrides: ChatOverrides | undefined,
    isRemote: boolean,
    sessionHasReasoningParser: boolean,
    tools?: any[],
    detectedFamily?: string,
    thinkingBudgetSupported?: boolean,
    supportsThinkingBudget?: boolean,
    supportedReasoningEfforts?: ReasoningEffort[],
): Record<string, any> {
    const stopSequences = overrides?.stopSequences
        ? overrides.stopSequences.split(',').map(s => s.trim()).filter(Boolean)
        : undefined
    const outputBudget = dsv4OutputBudget(
        overrides?.maxTokens,
        overrides?.enableThinking,
        detectedFamily,
        overrides?.reasoningEffort,
    )
    const thinkingBudget = typeof overrides?.maxThinkingTokens === 'number' &&
        Number.isFinite(overrides.maxThinkingTokens) &&
        overrides.maxThinkingTokens > 0
        ? Math.floor(overrides.maxThinkingTokens)
        : undefined
    const effectiveEnableThinkingOverride =
        !isRemote &&
        !sessionHasReasoningParser &&
        detectedFamily !== 'deepseek-v4'
            ? undefined
            : overrides?.enableThinking
    const applyThinkingBudget = (obj: Record<string, any>) => {
        // No isRemote gate: the engine now emits supports_thinking_budget in
        // /v1/capabilities, so remote sessions know it too. Keeping the gate
        // here would make the newly-rendered remote control decorative — the
        // capability check below is what keeps an older remote engine silent.
        if (thinkingBudget == null || obj.enable_thinking === false) return
        // Only families whose engine honors a top-level max_thinking_tokens cap
        // (registry supportsThinkingBudget) or whose chat TEMPLATE declares a budget
        // marker (thinkingBudgetSupported) get the field. Families with a reasoning
        // parser but no engine-side thinking-budget behavior (deepseek-v4,
        // step-3.7-flash) ignore it — dsv4 uses reasoning_effort instead.
        if (!(supportsThinkingBudget === true || thinkingBudgetSupported === true)) return
        obj.max_thinking_tokens = thinkingBudget
        // Template kwarg is TEMPLATE-side only; suppress it when the template has no
        // budget marker even though the engine-level field is still sent.
        if (thinkingBudgetSupported !== false) {
            obj.chat_template_kwargs = {
                ...(obj.chat_template_kwargs || {}),
                thinking_budget: thinkingBudget,
            }
        }
    }

    if (wireApi === 'responses') {
        const { systemMessages, inputMessages } = splitResponsesSystemMessages(
            requestMessages,
            detectedFamily === 'deepseek-v4',
        )
        const instructions =
            overrides?.builtinToolsEnabled && systemMessages.length > 0
                ? systemMessages.map(m => m.content).join('\n')
                : overrides?.systemPrompt || (systemMessages.length > 0 ? systemMessages.map(m => m.content).join('\n') : undefined)
        const obj: Record<string, any> = {
            model: modelName,
            input: inputMessages,
            instructions,
            stream: true
        }
        if (overrides?.temperature != null) obj.temperature = overrides.temperature
        if (overrides?.topP != null) obj.top_p = overrides.topP
        if (outputBudget) obj.max_output_tokens = outputBudget
        if (stopSequences) obj.stop = stopSequences
        const effectiveTopK = overrides?.topK
        if (effectiveTopK != null) obj.top_k = effectiveTopK
        if (overrides?.minP != null) obj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null) obj.repetition_penalty = overrides.repeatPenalty
        if (tools) {
            obj.tools = tools.map(t => ({
                type: 'function',
                name: t.function.name,
                description: t.function.description,
                parameters: t.function.parameters
            }))
        }
        applyReasoningRequestFields(obj, {
            enableThinking: effectiveEnableThinkingOverride,
            reasoningEffort: overrides?.reasoningEffort,
            isRemote,
            sessionHasReasoningParser,
            detectedFamily,
            supportedReasoningEfforts,
        })
        applyThinkingBudget(obj)
        return obj
    } else {
        const obj: Record<string, any> = {
            model: modelName,
            messages: requestMessages,
            stream: true,
            stream_options: { include_usage: true }
        }
        if (overrides?.temperature != null) obj.temperature = overrides.temperature
        if (overrides?.topP != null) obj.top_p = overrides.topP
        if (outputBudget) obj.max_tokens = outputBudget
        if (stopSequences) obj.stop = stopSequences
        const effectiveTopK = overrides?.topK
        if (effectiveTopK != null) obj.top_k = effectiveTopK
        if (overrides?.minP != null) obj.min_p = overrides.minP
        if (overrides?.repeatPenalty != null) obj.repetition_penalty = overrides.repeatPenalty
        if (tools) {
            obj.tools = tools
        }
        applyReasoningRequestFields(obj, {
            enableThinking: effectiveEnableThinkingOverride,
            reasoningEffort: overrides?.reasoningEffort,
            isRemote,
            sessionHasReasoningParser,
            detectedFamily,
            supportedReasoningEfforts,
        })
        applyThinkingBudget(obj)
        return obj
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('buildRequestBody — Chat Completions API', () => {
    const messages = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello' }
    ]

    it('omits sampling and token defaults when unset so the engine resolves bundle metadata', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, undefined, false, false)
        expect(body.model).toBe('gpt-4')
        expect(body.messages).toBe(messages)
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.max_tokens).toBeUndefined()
        expect(body.stream).toBe(true)
        expect(body.stream_options).toEqual({ include_usage: true })
    })

    it('does not invent sampler or output-budget values when chat overrides are absent', () => {
        const body = buildRequestBody('completions', 'qwen-mtp', messages, undefined, false, true)
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.max_tokens).toBeUndefined()
        expect(body.repetition_penalty).toBeUndefined()
    })

    it('forwards explicit greedy MTP-compatible sampler values including neutral repeat penalty', () => {
        const body = buildRequestBody(
            'completions',
            'qwen-mtp',
            messages,
            { temperature: 0, topP: 1, repeatPenalty: 1.0 },
            false,
            true,
        )
        expect(body.temperature).toBe(0)
        expect(body.top_p).toBe(1)
        expect(body.repetition_penalty).toBe(1.0)
    })

    it('forwards custom temperature and top_p', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { temperature: 0.3, topP: 0.5 }, false, false)
        expect(body.temperature).toBe(0.3)
        expect(body.top_p).toBe(0.5)
    })

    it('includes a positive top_k override', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { topK: 40 }, false, false)
        expect(body.top_k).toBe(40)
    })

    it.each(['completions', 'responses'] as const)('forwards explicit top_k=0 for %s so bundle defaults are disabled', wireApi => {
        const body = buildRequestBody(wireApi, 'gpt-4', messages, { topK: 0 }, false, false)
        expect(body.top_k).toBe(0)
    })

    it('omits top_k when unset so the engine uses bundle metadata or disabled fallback', () => {
        const body = buildRequestBody('completions', 'local-model', messages, {}, false, false)
        expect(body.top_k).toBeUndefined()
    })

    it('does not default remote requests to top_k', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, {}, true, false)
        expect(body.top_k).toBeUndefined()
    })

    it('includes min_p when > 0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { minP: 0.05 }, false, false)
        expect(body.min_p).toBe(0.05)
    })

    it.each(['completions', 'responses'] as const)('includes explicit min_p=0 for %s so bundle defaults are disabled', wireApi => {
        const body = buildRequestBody(wireApi, 'local-model', messages, { minP: 0 }, false, false)
        expect(body.min_p).toBe(0)
    })

    it('persists zero from the chat settings slider instead of converting it to inherit', () => {
        const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
        expect(source).toContain("update('minP', sanitized)")
        expect(source).not.toContain("update('minP', v === 0 ? undefined : v)")
    })

    it('includes repetition_penalty when != 1.0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { repeatPenalty: 1.2 }, false, false)
        expect(body.repetition_penalty).toBe(1.2)
    })

    it('forwards repetition_penalty when explicitly set to neutral 1.0', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { repeatPenalty: 1.0 }, false, false)
        expect(body.repetition_penalty).toBe(1.0)
    })

    it('includes stop sequences when provided', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { stopSequences: '<stop>, END' }, false, false)
        expect(body.stop).toEqual(['<stop>', 'END'])
    })

    it('omits stop when stopSequences is empty', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { stopSequences: '' }, false, false)
        expect(body.stop).toBeUndefined()
    })

    it('forwards max_tokens override', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 8192 }, false, false)
        expect(body.max_tokens).toBe(8192)
    })

    it('keeps per-chat maxTokens as output budget only, never prompt context', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 2048 }, false, false)
        expect(body.max_tokens).toBe(2048)
        expect(body.max_output_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('per-chat maxTokens below or above the server startup default remain request scoped', () => {
        const serverStartupDefaultMaxTokens = 4096
        const below = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const above = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(below.max_tokens).toBe(512)
        expect(above.max_tokens).toBe(8192)
        expect(below.max_tokens).toBeLessThan(serverStartupDefaultMaxTokens)
        expect(above.max_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)
        for (const body of [below, above]) {
            expect(body.max_output_tokens).toBeUndefined()
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
        }
    })

    it('omits invalid persisted maxTokens values instead of poisoning Chat Completions', () => {
        const badValues = [0, -5, Number.NaN, Number.POSITIVE_INFINITY, '1024']
        for (const value of badValues) {
            const body = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: value as any }, false, false)
            expect(body.max_tokens).toBeUndefined()
        }
    })

    it('cleared persisted chat maxTokens null stays Auto for Chat Completions and Responses', () => {
        const completions = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: null as any }, false, false)
        const responses = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: null as any }, false, false)

        expect(completions.max_tokens).toBeUndefined()
        expect(completions.max_output_tokens).toBeUndefined()
        expect(completions.max_prompt_tokens).toBeUndefined()
        expect(responses.max_output_tokens).toBeUndefined()
        expect(responses.max_tokens).toBeUndefined()
        expect(responses.max_prompt_tokens).toBeUndefined()
    })

    it('Auto chat maxTokens omits per-request output caps so server default can apply', () => {
        const autoCompletions = buildRequestBody('completions', 'gpt-4', messages, {}, false, false)
        const disabledCompletions = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 0 }, false, false)
        const dsv4MaxThinkingAuto = buildRequestBody(
            'completions',
            'dsv4',
            messages,
            { reasoningEffort: 'max' },
            false,
            true,
            undefined,
            'deepseek-v4',
        )
        const autoResponses = buildRequestBody('responses', 'gpt-4', messages, {}, false, false)
        const disabledResponses = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 0 }, false, false)

        for (const body of [
            autoCompletions,
            disabledCompletions,
            dsv4MaxThinkingAuto,
            autoResponses,
            disabledResponses,
        ]) {
            expect(body.max_tokens).toBeUndefined()
            expect(body.max_output_tokens).toBeUndefined()
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
            expect(body.max_thinking_tokens).toBeUndefined()
            expect(body.chat_template_kwargs?.thinking_budget).toBeUndefined()
        }
        expect(dsv4MaxThinkingAuto.reasoning_effort).toBe('max')
    })

    it('does NOT send max_thinking_tokens for deepseek-v4 (no engine thinking-budget; uses reasoning_effort)', () => {
        const body = buildRequestBody(
            'completions',
            'dsv4',
            messages,
            { enableThinking: true, reasoningEffort: 'max', maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'deepseek-v4',
        )

        expect(body.max_thinking_tokens).toBeUndefined()
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
        expect(body.reasoning_effort).toBe('max')
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_output_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('sends top-level max_thinking_tokens for an engine-honoring family (supportsThinkingBudget)', () => {
        const body = buildRequestBody(
            'completions',
            'qwen35',
            messages,
            { enableThinking: true, maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'qwen3.5',
            undefined,
            true,
        )

        expect(body.max_thinking_tokens).toBe(4096)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true, thinking_budget: 4096 })
        expect(body.max_tokens).toBeUndefined()
    })

    it('sends top-level max_thinking_tokens but suppresses the template kwarg when the template declares no budget marker (engine-honoring family)', () => {
        const body = buildRequestBody(
            'completions',
            'gemma4',
            messages,
            { enableThinking: true, maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'gemma4',
            false,
            true,
        )

        expect(body.enable_thinking).toBe(true)
        // Engine honors top-level max_thinking_tokens for gemma4 even though the
        // template has no budget marker (thinkingBudgetSupported === false).
        expect(body.max_thinking_tokens).toBe(4096)
        // Template kwarg is TEMPLATE-side only and stays suppressed.
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
    })

    it('suppresses max_thinking_tokens for a reasoning family with neither engine nor template budget support', () => {
        const body = buildRequestBody(
            'completions',
            'zaya',
            messages,
            { enableThinking: true, maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'zaya',
        )

        expect(body.enable_thinking).toBe(true)
        expect(body.max_thinking_tokens).toBeUndefined()
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
    })

    it('switching chats never carries a previous chat maxTokens into Auto Chat Completions', () => {
        const cappedChat = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const autoChat = buildRequestBody('completions', 'gpt-4', messages, {}, false, false)
        const raisedChat = buildRequestBody('completions', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(cappedChat.max_tokens).toBe(512)
        expect(autoChat.max_tokens).toBeUndefined()
        expect(autoChat.max_output_tokens).toBeUndefined()
        expect(autoChat.max_prompt_tokens).toBeUndefined()
        expect(autoChat.max_context_tokens).toBeUndefined()
        expect(autoChat.max_context).toBeUndefined()
        expect(raisedChat.max_tokens).toBe(8192)
    })

    it('profile-loaded chat maxTokens remains request-scoped and later Auto still uses the server default', () => {
        const serverStartupDefaultMaxTokens = 4096
        const profileLoadedCompletions = buildRequestBody(
            'completions',
            'gpt-4',
            messages,
            { maxTokens: 12000, builtinToolsEnabled: true },
            false,
            false,
        )
        const profileLoadedResponses = buildRequestBody(
            'responses',
            'gpt-4',
            messages,
            { maxTokens: 12000, builtinToolsEnabled: true },
            false,
            false,
        )
        const laterAutoCompletions = buildRequestBody('completions', 'gpt-4', messages, {}, false, false)
        const laterAutoResponses = buildRequestBody('responses', 'gpt-4', messages, {}, false, false)

        expect(profileLoadedCompletions.max_tokens).toBe(12000)
        expect(profileLoadedResponses.max_output_tokens).toBe(12000)
        expect(profileLoadedCompletions.max_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)
        expect(profileLoadedResponses.max_output_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)

        for (const body of [profileLoadedCompletions, profileLoadedResponses]) {
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
        }
        expect(laterAutoCompletions.max_tokens).toBeUndefined()
        expect(laterAutoResponses.max_output_tokens).toBeUndefined()
        expect(laterAutoCompletions.max_prompt_tokens).toBeUndefined()
        expect(laterAutoResponses.max_prompt_tokens).toBeUndefined()
    })

    it('forwards reasoning_effort', () => {
        const body = buildRequestBody('completions', 'gpt-4', messages, { reasoningEffort: 'high' }, false, true)
        expect(body.reasoning_effort).toBe('high')
    })

    it('suppresses stale reasoning_effort when the model has no reasoning parser', () => {
        const body = buildRequestBody('completions', 'plain-model', messages, { reasoningEffort: 'max' }, false, false)
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('suppresses reasoning_effort when thinking is explicitly off', () => {
        const body = buildRequestBody('completions', 'qwen', messages, { enableThinking: false, reasoningEffort: 'high' }, false, true)
        expect(body.enable_thinking).toBe(false)
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('preserves explicit DSV4 standard thinking max_tokens', () => {
        const body = buildRequestBody('completions', 'dsv4', messages, { maxTokens: 128 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_tokens).toBe(128)
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('preserves explicit DSV4 Max thinking max_tokens', () => {
        const body = buildRequestBody('completions', 'dsv4', messages, { maxTokens: 128, reasoningEffort: 'max' }, false, true, undefined, 'deepseek-v4')
        expect(body.max_tokens).toBe(128)
        expect(body.reasoning_effort).toBe('max')
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('does not floor DSV4 max_tokens when thinking is explicitly off', () => {
        const body = buildRequestBody('completions', 'dsv4', messages, { enableThinking: false, maxTokens: 128 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_tokens).toBe(128)
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })
})

describe('buildRequestBody — Remote vs Local gating', () => {
    const messages = [{ role: 'user', content: 'Hello' }]

    it('omits enable_thinking and chat_template_kwargs for plain local Auto sessions without a reasoning parser', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('omits thinking for local Auto sessions with a reasoning parser', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, false, true)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('includes chat_template_kwargs for explicit local thinking override on reasoning-capable models', () => {
        const body = buildRequestBody('completions', 'model', messages, { enableThinking: true }, false, true)
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('suppresses explicit local thinking override when a plain model has no reasoning parser', () => {
        const body = buildRequestBody('completions', 'plain-model', messages, { enableThinking: true }, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('forwards explicit ZAYA thinking override when registry detection provides qwen3 parser', () => {
        const body = buildRequestBody('completions', 'zaya-k', messages, { enableThinking: true }, false, true, undefined, 'zaya')
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('EXCLUDES chat_template_kwargs for remote sessions', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, true, false)
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('remote Auto stays omitted when session has reasoning parser', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, true, true)
        expect(body.enable_thinking).toBeUndefined()
    })

    it('enable_thinking can be explicitly set via overrides', () => {
        const body = buildRequestBody('completions', 'model', messages, { enableThinking: false }, true, true)
        expect(body.enable_thinking).toBe(false)
    })
})

describe('buildRequestBody — Responses API', () => {
    const messages = [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: 'Hello' }
    ]

    it('uses max_output_tokens instead of max_tokens', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 4096 }, false, false)
        expect(body.max_output_tokens).toBe(4096)
        expect(body.max_tokens).toBeUndefined()
    })

    it('keeps Responses maxTokens as output budget only, never prompt context', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 2048 }, false, false)
        expect(body.max_output_tokens).toBe(2048)
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('per-chat maxTokens below or above the server startup default remain request scoped for Responses', () => {
        const serverStartupDefaultMaxTokens = 4096
        const below = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const above = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(below.max_output_tokens).toBe(512)
        expect(above.max_output_tokens).toBe(8192)
        expect(below.max_output_tokens).toBeLessThan(serverStartupDefaultMaxTokens)
        expect(above.max_output_tokens).toBeGreaterThan(serverStartupDefaultMaxTokens)
        for (const body of [below, above]) {
            expect(body.max_tokens).toBeUndefined()
            expect(body.max_prompt_tokens).toBeUndefined()
            expect(body.max_context_tokens).toBeUndefined()
            expect(body.max_context).toBeUndefined()
        }
    })

    it('omits invalid persisted maxTokens values instead of poisoning Responses', () => {
        const badValues = [0, -5, Number.NaN, Number.POSITIVE_INFINITY, '1024']
        for (const value of badValues) {
            const body = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: value as any }, false, false)
            expect(body.max_output_tokens).toBeUndefined()
            expect(body.max_tokens).toBeUndefined()
        }
    })

    it('does not invent Responses sampler or output-budget values when chat overrides are absent', () => {
        const body = buildRequestBody('responses', 'qwen-mtp', messages, undefined, false, true)
        expect(body.temperature).toBeUndefined()
        expect(body.top_p).toBeUndefined()
        expect(body.max_output_tokens).toBeUndefined()
        expect(body.repetition_penalty).toBeUndefined()
    })

    it('does not put Chat include_usage in a standard Responses request body', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, true, true)
        expect(body.stream).toBe(true)
        expect(body.stream_options).toBeUndefined()
    })

    it('switching chats never carries a previous chat maxTokens into Auto Responses', () => {
        const cappedChat = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 512 }, false, false)
        const autoChat = buildRequestBody('responses', 'gpt-4', messages, {}, false, false)
        const raisedChat = buildRequestBody('responses', 'gpt-4', messages, { maxTokens: 8192 }, false, false)

        expect(cappedChat.max_output_tokens).toBe(512)
        expect(autoChat.max_output_tokens).toBeUndefined()
        expect(autoChat.max_tokens).toBeUndefined()
        expect(autoChat.max_prompt_tokens).toBeUndefined()
        expect(autoChat.max_context_tokens).toBeUndefined()
        expect(autoChat.max_context).toBeUndefined()
        expect(raisedChat.max_output_tokens).toBe(8192)
    })

    it('uses input instead of messages', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, false, false)
        expect(body.input).toEqual([{ role: 'user', content: 'Hello' }])
        expect(body.messages).toBeUndefined()
    })

    it('extracts system messages as instructions', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, false, false)
        expect(body.instructions).toBe('You are helpful.')
    })

    it('prefers systemPrompt override for instructions', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, { systemPrompt: 'Custom system prompt' }, false, false)
        expect(body.instructions).toBe('Custom system prompt')
    })

    it('EXCLUDES chat_template_kwargs for remote', () => {
        const body = buildRequestBody('responses', 'gpt-4', messages, undefined, true, false)
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('preserves DSV4 Responses max_output_tokens for Auto/On thinking', () => {
        const body = buildRequestBody('responses', 'dsv4', messages, { maxTokens: 300 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_output_tokens).toBe(300)
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('preserves DSV4 Responses max_output_tokens for Max thinking', () => {
        const body = buildRequestBody('responses', 'dsv4', messages, { maxTokens: 300, reasoningEffort: 'max' }, false, true, undefined, 'deepseek-v4')
        expect(body.max_output_tokens).toBe(300)
        expect(body.reasoning_effort).toBe('max')
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('does NOT send Responses max_thinking_tokens for deepseek-v4 (uses reasoning_effort, output cap preserved)', () => {
        const body = buildRequestBody(
            'responses',
            'dsv4',
            messages,
            { enableThinking: true, maxTokens: 300, reasoningEffort: 'max', maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'deepseek-v4',
        )
        expect(body.max_output_tokens).toBe(300)
        expect(body.max_thinking_tokens).toBeUndefined()
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
        expect(body.reasoning_effort).toBe('max')
        expect(body.max_tokens).toBeUndefined()
        expect(body.max_prompt_tokens).toBeUndefined()
        expect(body.max_context_tokens).toBeUndefined()
        expect(body.max_context).toBeUndefined()
    })

    it('sends Responses max_thinking_tokens for an engine-honoring family (supportsThinkingBudget)', () => {
        const body = buildRequestBody(
            'responses',
            'minimax',
            messages,
            { enableThinking: true, maxTokens: 300, maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'minimax',
            undefined,
            true,
        )
        expect(body.max_output_tokens).toBe(300)
        expect(body.max_thinking_tokens).toBe(4096)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true, thinking_budget: 4096 })
    })

    it('suppresses stale maxThinkingTokens when thinking is explicitly off', () => {
        const body = buildRequestBody(
            'responses',
            'dsv4',
            messages,
            { enableThinking: false, maxTokens: 300, reasoningEffort: 'max', maxThinkingTokens: 4096 },
            false,
            true,
            undefined,
            'deepseek-v4',
        )
        expect(body.max_output_tokens).toBe(300)
        expect(body.max_thinking_tokens).toBeUndefined()
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('suppresses stale Responses reasoning_effort when thinking is explicitly off', () => {
        const body = buildRequestBody('responses', 'dsv4', messages, { enableThinking: false, reasoningEffort: 'max', maxTokens: 300 }, false, true, undefined, 'deepseek-v4')
        expect(body.max_output_tokens).toBe(300)
        expect(body.reasoning_effort).toBeUndefined()
        expect(body.dsv4_finalizer_tokens).toBeUndefined()
    })

    it('suppresses Responses enable_thinking when a plain local model has no reasoning parser', () => {
        const body = buildRequestBody('responses', 'plain-model', messages, { enableThinking: true }, false, false)
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
    })

    it('forwards explicit ZAYA-VL thinking override when registry detection provides qwen3 parser', () => {
        const body = buildRequestBody('responses', 'zaya-vl-k', messages, { enableThinking: true }, false, true, undefined, 'zaya1-vl')
        expect(body.enable_thinking).toBe(true)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
    })

    it('Hy3 local Responses Auto omits the toggle and reasoning_effort', () => {
        const body = buildRequestBody('responses', 'hy3', messages, undefined, false, true, undefined, 'hy3')
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('Hy3 local Responses Auto suppresses stale High effort', () => {
        const body = buildRequestBody(
            'responses',
            'hy3',
            messages,
            { enableThinking: undefined, reasoningEffort: 'high' },
            false,
            true,
            undefined,
            'hy3',
        )
        expect(body.enable_thinking).toBeUndefined()
        expect(body.chat_template_kwargs).toBeUndefined()
        expect(body.reasoning_effort).toBeUndefined()
    })

    it('Hy3 local Responses Thinking Off sends explicit off and no High effort', () => {
        const body = buildRequestBody(
            'responses',
            'hy3',
            messages,
            { enableThinking: false, reasoningEffort: 'high' },
            false,
            true,
            undefined,
            'hy3',
        )
        expect(body.enable_thinking).toBe(false)
        expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
        expect(body.reasoning_effort).toBeUndefined()
        expect(body.chat_template_kwargs.reasoning_effort).toBeUndefined()
        expect(body.chat_template_kwargs.thinking_budget).toBeUndefined()
    })
})

describe('DSV4-0731 bundle reasoning payload', () => {
    const levels: ReasoningEffort[] = ['low', 'high', 'max']
    const messages = [{ role: 'user', content: 'Use the requested reasoning mode.' }]
    const build = (
        wireApi: 'completions' | 'responses',
        overrides: ChatOverrides,
        requestMessages = messages,
    ) => buildRequestBody(
        wireApi,
        'dsv4-0731',
        requestMessages,
        overrides,
        false,
        true,
        undefined,
        'deepseek-v4',
        undefined,
        undefined,
        levels,
    )

    it.each(['completions', 'responses'] as const)(
        '%s Auto omits all reasoning kwargs so the bundle owns native Low',
        wireApi => {
            const body = build(wireApi, {})
            expect(body.enable_thinking).toBeUndefined()
            expect(body.reasoning_effort).toBeUndefined()
            expect(body.thinking_mode).toBeUndefined()
            expect(body.chat_template_kwargs).toBeUndefined()
        },
    )

    it.each(['completions', 'responses'] as const)(
        '%s Off selects the native Chat rail without an effort',
        wireApi => {
            const body = build(wireApi, { enableThinking: false })
            expect(body.enable_thinking).toBe(false)
            expect(body.chat_template_kwargs).toEqual({ enable_thinking: false })
            expect(body.thinking_mode).toBeUndefined()
            expect(body.reasoning_effort).toBeUndefined()
        },
    )

    it.each(['low', 'high', 'max'] as const)(
        'preserves explicit %s exactly on Chat and Responses payloads',
        effort => {
            for (const wireApi of ['completions', 'responses'] as const) {
                const body = build(wireApi, { enableThinking: true, reasoningEffort: effort })
                expect(body.enable_thinking).toBe(true)
                expect(body.chat_template_kwargs).toEqual({ enable_thinking: true })
                expect(body.reasoning_effort).toBe(effort)
                expect(body.thinking_mode).toBeUndefined()
            }
        },
    )

    it('rejects stale Medium instead of mapping or dropping it to native Low', () => {
        expect(() => build('responses', {
            enableThinking: true,
            reasoningEffort: 'medium',
        })).toThrow('Reasoning effort "medium" is not supported by the loaded bundle. Choose Auto or Low/High/Max.')
    })

    it('keeps explicit High unchanged on the built-in-tool continuation body', () => {
        const overrides = { enableThinking: true, reasoningEffort: 'high' as const }
        const initial = build('responses', overrides)
        const continuation = build('responses', overrides, [
            ...messages,
            { type: 'function_call', call_id: 'call_1', name: 'read_file', arguments: '{}' },
            { type: 'function_call_output', call_id: 'call_1', output: 'result' },
        ])

        for (const body of [initial, continuation]) {
            expect(body.enable_thinking).toBe(true)
            expect(body.reasoning_effort).toBe('high')
            expect(body.thinking_mode).toBeUndefined()
        }
    })

    it('Responses moves only the contiguous leading system run into instructions', () => {
        const ordered = [
            { role: 'system', content: 'base policy' },
            { role: 'system', content: 'leading tool policy' },
            { role: 'latest_reminder', content: '2026-08-02,US,App,en' },
            { role: 'user', content: 'first turn' },
            { role: 'assistant', content: 'first answer' },
            { role: 'system', content: 'tail system' },
            { role: 'user', content: 'second turn' },
        ]

        const body = build('responses', {}, ordered)

        expect(body.instructions).toBe('base policy\nleading tool policy')
        expect(body.input).toEqual(ordered.slice(2))
        expect(body.input[3]).toEqual({ role: 'system', content: 'tail system' })
    })

    it('Responses preserves every row when the DSV4 conversation has no leading system run', () => {
        const ordered = [
            { role: 'latest_reminder', content: '2026-08-02,US,App,en' },
            { role: 'user', content: 'first turn' },
            { role: 'system', content: 'tail system' },
        ]

        const body = build('responses', {}, ordered)

        expect(body.instructions).toBeUndefined()
        expect(body.input).toEqual(ordered)
    })

    it('Chat wire keeps the same DSV4 message order unchanged', () => {
        const ordered = [
            { role: 'system', content: 'base policy' },
            { role: 'user', content: 'first turn' },
            { role: 'system', content: 'tail system' },
            { role: 'latest_reminder', content: '2026-08-02,US,App,en' },
            { role: 'user', content: 'second turn' },
        ]

        const body = build('completions', {}, ordered)

        expect(body.messages).toEqual(ordered)
    })
})

describe('Responses system-message ordering policy', () => {
    it('keeps generic-family behavior unchanged by hoisting every system row', () => {
        const messages = [
            { role: 'system', content: 'base policy' },
            { role: 'user', content: 'first turn' },
            { role: 'assistant', content: 'first answer' },
            { role: 'system', content: 'tail system' },
            { role: 'user', content: 'second turn' },
        ]

        const body = buildRequestBody(
            'responses',
            'generic-model',
            messages,
            undefined,
            false,
            false,
            undefined,
            'qwen3',
        )

        expect(body.instructions).toBe('base policy\ntail system')
        expect(body.input).toEqual([
            { role: 'user', content: 'first turn' },
            { role: 'assistant', content: 'first answer' },
            { role: 'user', content: 'second turn' },
        ])
    })
})

describe('buildRequestBody source parity', () => {
    it('actual chat IPC uses the shared reasoning policy for Responses payloads', () => {
        const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
        const branchStart = source.indexOf('if (useResponsesApi) {')
        expect(branchStart).toBeGreaterThanOrEqual(0)
        const branchReturn = source.indexOf('return obj;', branchStart)
        expect(branchReturn).toBeGreaterThan(branchStart)
        const responsesBranch = source.slice(branchStart, branchReturn)

        expect(responsesBranch).toContain('applyReasoningRequestFields(obj, {')
        expect(responsesBranch).toContain('enableThinking: effectiveEnableThinkingOverride')
        expect(responsesBranch).toContain('supportedReasoningEfforts,')
        expect(responsesBranch).toContain('applyThinkingBudget(obj);')
        expect(responsesBranch.match(/applyReasoningRequestFields\(obj, \{/g)?.length).toBe(1)
        expect(responsesBranch.match(/applyThinkingBudget\(obj\);/g)?.length).toBe(1)
        expect(responsesBranch).toContain('splitResponsesSystemMessages(')
        expect(responsesBranch).toContain('chatDetectedFamily === "deepseek-v4"')
    })

    it('serializes both the initial request and built-in-tool continuation through the same builder', () => {
        const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
        expect(source.match(/JSON\.stringify\(buildRequestBody\(\)\)/g)).toHaveLength(2)
        expect(source).toContain('const requestBody = JSON.stringify(buildRequestBody());')
        expect(source).toContain('const followUpBody = JSON.stringify(buildRequestBody());')
    })
})

describe('buildRequestBody — Tool format', () => {
    const messages = [{ role: 'user', content: 'Hello' }]
    const sampleTools = [
        {
            type: 'function',
            function: {
                name: 'read_file',
                description: 'Read a file',
                parameters: { type: 'object', properties: { path: { type: 'string' } } }
            }
        }
    ]

    it('Completions API: tools use OpenAI format with function wrapper', () => {
        const body = buildRequestBody('completions', 'model', messages, undefined, false, false, sampleTools)
        expect(body.tools).toEqual(sampleTools)
        expect(body.tools[0].function).toBeDefined()
        expect(body.tools[0].function.name).toBe('read_file')
    })

    it('Responses API: tools use flat format WITHOUT function wrapper', () => {
        const body = buildRequestBody('responses', 'model', messages, undefined, false, false, sampleTools)
        expect(body.tools[0].name).toBe('read_file')
        expect(body.tools[0].function).toBeUndefined()
    })

    it('Responses API preserves the augmented custom system prompt when built-in tools are enabled', () => {
        const augmentedSystemPrompt =
            'Custom coding instructions\n\nIMPORTANT: After using any tools, provide a final response. If the user explicitly requested exact final wording or a strict output format, follow that format exactly; otherwise provide a substantive response explaining what you found or did. Never stop after just executing tools.'
        const body = buildRequestBody(
            'responses',
            'model',
            [
                { role: 'system', content: augmentedSystemPrompt },
                { role: 'user', content: 'Create index.html' },
            ],
            { systemPrompt: 'Custom coding instructions', builtinToolsEnabled: true },
            false,
            false,
            sampleTools,
        )

        expect(body.instructions).toBe(augmentedSystemPrompt)
        expect(body.input).toEqual([{ role: 'user', content: 'Create index.html' }])
        expect(body.tools[0].name).toBe('read_file')
    })

    it('agentic tool prompts do not override explicit exact-output user formats', () => {
        const registry = readFileSync('src/main/tools/registry.ts', 'utf8')
        const chat = readFileSync('src/main/ipc/chat.ts', 'utf8')

        expect(registry).toContain('If the user explicitly requested exact final wording or a strict output format, follow that format exactly')
        expect(registry).not.toContain('MUST ALWAYS provide a substantive response')
        expect(registry).not.toContain('After ALL tool calls are complete')
        expect(chat).toContain('If the user explicitly requested exact final wording or a strict output format, follow that format exactly')
        expect(chat).toContain('suppressAgenticToolPromptForExactOutput')
        expect(chat).toContain('/\\breply exactly\\b/i.test(latestUserText)')
        expect(chat).toContain('!suppressAgenticToolPromptForExactOutput')
    })

    it('restricts AppleScript-specialized ZAYA bundles to their trained native tool schema', () => {
        const chat = readFileSync('src/main/ipc/chat.ts', 'utf8')

        expect(chat).toContain('function isZayaAppleScriptToolBundle(')
        expect(chat).toContain('readFileSync(join(modelPath, "README.md"), "utf8")')
        expect(chat).toContain('/\\brun_applescript\\b/i.test(readme)')
        expect(chat).toContain('zayaAppleScriptToolBundle: chatUsesZayaAppleScriptToolBundle')
        expect(chat).toContain('BUILTIN_TOOLS.filter((t: any) => t.function.name === "run_applescript")')
    })
})

// ─── filterTools logic ───────────────────────────────────────────────────────

describe('filterTools', () => {
    const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file', 'patch_file', 'batch_edit', 'copy_file', 'move_file', 'delete_file', 'create_directory', 'list_directory', 'insert_text', 'replace_lines', 'apply_regex', 'read_image'])
    const SEARCH_TOOLS = new Set(['search_files', 'find_files', 'file_info', 'get_diagnostics', 'get_tree', 'diff_files'])
    const SHELL_TOOLS = new Set(['run_command', 'spawn_process', 'get_process_output'])

    function filterTools(allTools: any[], overrides: any): any[] {
        const disabled = new Set<string>()
        if (overrides.fileToolsEnabled === false) FILE_TOOLS.forEach(t => disabled.add(t))
        if (overrides.searchToolsEnabled === false) SEARCH_TOOLS.forEach(t => disabled.add(t))
        if (overrides.shellEnabled === false) SHELL_TOOLS.forEach(t => disabled.add(t))
        if (disabled.size === 0) return allTools
        return allTools.filter((t: any) => !disabled.has(t.function.name))
    }

    const allTools = [
        { function: { name: 'read_file' } },
        { function: { name: 'search_files' } },
        { function: { name: 'run_command' } },
        { function: { name: 'ask_user' } },
    ]

    it('returns all tools when no toggles disabled', () => {
        expect(filterTools(allTools, {})).toEqual(allTools)
    })

    it('disables file tools when fileToolsEnabled=false', () => {
        const result = filterTools(allTools, { fileToolsEnabled: false })
        expect(result.find(t => t.function.name === 'read_file')).toBeUndefined()
        expect(result.find(t => t.function.name === 'search_files')).toBeDefined()
    })

    it('disables shell tools when shellEnabled=false', () => {
        const result = filterTools(allTools, { shellEnabled: false })
        expect(result.find(t => t.function.name === 'run_command')).toBeUndefined()
        expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
    })

    it('ask_user is never disabled by any toggle', () => {
        const result = filterTools(allTools, {
            fileToolsEnabled: false,
            searchToolsEnabled: false,
            shellEnabled: false
        })
        expect(result.find(t => t.function.name === 'ask_user')).toBeDefined()
    })
})
