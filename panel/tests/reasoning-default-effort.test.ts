import { describe, expect, it } from 'vitest'
import { applyReasoningRequestFields, type ReasoningEffort } from '../src/shared/reasoningEffortPolicy'

describe('local model-default reasoning effort', () => {
  for (const wireApi of ['completions', 'responses'] as const) {
    it(`${wireApi}: Thinking On does not replace the bundle default with an alias tier`, () => {
      const body: Record<string, any> = {}
      applyReasoningRequestFields(body, {
        enableThinking: true,
        isRemote: false,
        sessionHasReasoningParser: true,
        wireApi,
        supportedReasoningEfforts: ['low', 'medium', 'xhigh'],
        defaultReasoningEffort: 'xhigh',
      })
      expect(body).toEqual({ enable_thinking: true, chat_template_kwargs: { enable_thinking: true } })
    })

    for (const effort of ['low', 'medium', 'high', 'xhigh', 'max'] as ReasoningEffort[]) {
      it(`${wireApi}: preserves explicit ${effort}`, () => {
        const body: Record<string, any> = {}
        applyReasoningRequestFields(body, {
          enableThinking: true, reasoningEffort: effort, isRemote: false,
          sessionHasReasoningParser: true, wireApi, supportedReasoningEfforts: [effort],
        })
        expect(body.reasoning_effort).toBe(effort)
        expect(body.thinking_mode).toBe(effort === 'max' ? 'max' : 'reasoning')
      })
    }

    it(`${wireApi}: preserves Off and Auto independently of model-default effort`, () => {
      const off: Record<string, any> = {}
      applyReasoningRequestFields(off, {
        enableThinking: false, reasoningEffort: 'xhigh', isRemote: false,
        sessionHasReasoningParser: true, wireApi,
      })
      expect(off).toEqual({ enable_thinking: false, chat_template_kwargs: { enable_thinking: false }, thinking_mode: 'instruct' })
      const auto = {}
      applyReasoningRequestFields(auto, { isRemote: false, sessionHasReasoningParser: true, wireApi })
      expect(auto).toEqual({})
    })
  }
})
