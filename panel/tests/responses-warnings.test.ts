import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  appendOutputTruncationWarning,
  extractResponsesWarnings,
  dropSupersededRecoveryWarnings,
  OUTPUT_TRUNCATED_WARNING,
  categorizeResponsesWarning,
  responsesTerminalFinishReason,
} from '../src/renderer/src/lib/responsesWarnings'

describe('extractResponsesWarnings', () => {
  it('returns null for null/undefined payload', () => {
    expect(extractResponsesWarnings(null)).toBeNull()
    expect(extractResponsesWarnings(undefined)).toBeNull()
  })

  it('returns null when warnings field absent', () => {
    expect(extractResponsesWarnings({})).toBeNull()
  })

  it('returns null when warnings is not an array', () => {
    expect(extractResponsesWarnings({ warnings: 'not an array' as any })).toBeNull()
    expect(extractResponsesWarnings({ warnings: 42 as any })).toBeNull()
    expect(extractResponsesWarnings({ warnings: null })).toBeNull()
  })

  it('returns null when warnings array is empty', () => {
    expect(extractResponsesWarnings({ warnings: [] })).toBeNull()
  })

  it('returns null when warnings contain only empty/whitespace strings', () => {
    expect(extractResponsesWarnings({ warnings: ['', '   ', '\n\t'] })).toBeNull()
  })

  it('returns null when warnings contain only non-strings', () => {
    expect(extractResponsesWarnings({ warnings: [1, 2, {}, null] as any })).toBeNull()
  })

  it('extracts a single valid warning', () => {
    const w = extractResponsesWarnings({ warnings: ['something happened'] })
    expect(w).toEqual(['something happened'])
  })

  it('trims whitespace from warnings', () => {
    expect(extractResponsesWarnings({ warnings: ['  hello  '] })).toEqual(['hello'])
  })

  it('drops non-string entries but keeps strings', () => {
    expect(
      extractResponsesWarnings({ warnings: ['real', 42, 'also real', null] as any }),
    ).toEqual(['real', 'also real'])
  })

  it('deduplicates identical strings', () => {
    expect(
      extractResponsesWarnings({
        warnings: ['same', 'same', '  same  ', 'different'],
      }),
    ).toEqual(['same', 'different'])
  })

  it('extracts response.warning SSE event messages', () => {
    expect(
      extractResponsesWarnings({
        type: 'response.warning',
        code: 'empty_model_response',
        message: 'The model produced no visible response.',
      } as any),
    ).toEqual(['The model produced no visible response.'])
  })

  it('matches the exact server-emitted reasoning-only warning shape', () => {
    // This is the literal string emitted by
    // vmlx_engine/server.py:_chain_warnings_for_previous_response_id.
    // Pinning it here means the frontend test file is a contract surface
    // — if the server changes wording, this fails and forces UI review.
    const SERVER_MSG =
      'previous_response_id chained a response that produced reasoning only ' +
      '(no visible message, no tool calls). Chat continuity may be impaired ' +
      'and prefix-cache reuse may be lower than expected. Inspect the prior ' +
      'response terminal status before deciding whether to retry it.'
    const w = extractResponsesWarnings({ warnings: [SERVER_MSG] })
    expect(w).toEqual([SERVER_MSG])
  })
})

describe('dropSupersededRecoveryWarnings', () => {
  const INTERMEDIATE_EMPTY =
    'This response produced reasoning only (no visible message, no tool calls). ' +
    'The reasoning was preserved separately, but the visible answer is empty. ' +
    'Consider raising max_output_tokens or sending enable_thinking=false for the final synthesis turn.'

  it('drops current-response empty diagnostics after visible recovery', () => {
    expect(dropSupersededRecoveryWarnings([INTERMEDIATE_EMPTY])).toBeNull()
    expect(
      dropSupersededRecoveryWarnings(['The model produced no visible response.']),
    ).toBeNull()
  })

  it('preserves unrelated and previous-response warnings', () => {
    const previous =
      'previous_response_id chained a response that produced reasoning only (no visible message, no tool calls).'
    expect(
      dropSupersededRecoveryWarnings([
        INTERMEDIATE_EMPTY,
        'Schema coercion was required.',
        previous,
      ]),
    ).toEqual(['Schema coercion was required.', previous])
  })
})

describe('appendOutputTruncationWarning', () => {
  it('adds one persisted warning for a length terminal state', () => {
    expect(appendOutputTruncationWarning(null, 'length')).toEqual([
      OUTPUT_TRUNCATED_WARNING,
    ])
  })

  it('preserves existing warnings and deduplicates the truncation warning', () => {
    expect(
      appendOutputTruncationWarning(
        ['Parser warning.', OUTPUT_TRUNCATED_WARNING],
        'length',
      ),
    ).toEqual(['Parser warning.', OUTPUT_TRUNCATED_WARNING])
  })

  it('does not alter warnings for a completed response', () => {
    expect(appendOutputTruncationWarning(['Parser warning.'], 'stop')).toEqual([
      'Parser warning.',
    ])
    expect(appendOutputTruncationWarning(null, undefined)).toBeNull()
  })
})

describe('responsesTerminalFinishReason', () => {
  it('maps only the output-limit incomplete cause to length', () => {
    expect(
      responsesTerminalFinishReason('incomplete', {
        reason: 'max_output_tokens',
      }),
    ).toBe('length')
  })

  it('preserves cancellation without synthesizing truncation', () => {
    expect(
      responsesTerminalFinishReason('incomplete', {
        reason: 'cancelled',
      }),
    ).toBe('cancelled')
    expect(
      responsesTerminalFinishReason('incomplete', {
        reason: 'abort',
      }),
    ).toBe('cancelled')
    expect(
      appendOutputTruncationWarning(
        null,
        responsesTerminalFinishReason('incomplete', {
          reason: 'cancelled',
        }),
      ),
    ).toBeNull()
  })

  it('keeps completed and unknown incomplete terminals distinct', () => {
    expect(responsesTerminalFinishReason('completed', null)).toBe('stop')
    expect(responsesTerminalFinishReason('failed', null)).toBe('error')
    expect(responsesTerminalFinishReason('incomplete', null)).toBe('incomplete')
    expect(responsesTerminalFinishReason(undefined, null)).toBeUndefined()
  })
})

describe('categorizeResponsesWarning', () => {
  it('categorizes the reasoning-only chain warning', () => {
    expect(
      categorizeResponsesWarning(
        'previous_response_id chained a response that produced reasoning only',
      ),
    ).toBe('chain_reasoning_only')
  })

  it('categorizes cache/prefix warnings', () => {
    expect(categorizeResponsesWarning('prefix-cache reuse may be lower')).toBe(
      'cache_alignment',
    )
    expect(categorizeResponsesWarning('cache may be invalidated')).toBe(
      'cache_alignment',
    )
  })

  it('categorizes no-visible-response warnings separately from generic warnings', () => {
    expect(categorizeResponsesWarning('The model produced no visible response.')).toBe(
      'empty_visible_response',
    )
  })

  it('falls back to "other" for unrecognized warnings', () => {
    expect(categorizeResponsesWarning('something completely unrelated')).toBe('other')
  })

  it('is case-insensitive', () => {
    expect(categorizeResponsesWarning('PREVIOUS_RESPONSE_ID was bad')).toBe(
      'chain_reasoning_only',
    )
  })
})

describe('Responses warnings panel wiring', () => {
  it('main chat IPC handles completed and incomplete terminal warnings and usage', () => {
    const source = readFileSync(new URL('../src/main/ipc/chat.ts', import.meta.url), 'utf8')
    expect(source).toContain('dropSupersededRecoveryWarnings,')
    expect(source).toContain('appendOutputTruncationWarning,')
    expect(source).toContain('responsesTerminalFinishReason,')
    expect(source).toContain('extractResponsesWarnings,')
    expect(source).toContain('from "../../shared/responsesWarnings"')
    expect(source).toContain('responsesEventType === "response.warning"')
    expect(source).toContain('responsesEventType === "response.completed" ||')
    expect(source).toContain('responsesEventType === "response.incomplete"')
    expect(source).toContain('if (isResponsesTerminalEvent)')
    expect(source).toContain('if (isResponsesTerminalEvent && respUsage)')
    expect(source).toContain('const eventWarnings = extractResponsesWarnings(parsed)')
    expect(source).toContain('const completedWarnings = extractResponsesWarnings(')
    expect(source).toContain('const chatWarnings = extractResponsesWarnings(parsed)')
    expect(source).toContain('const finalResponseWarnings = appendOutputTruncationWarning(')
    expect(source).toContain('responsesTerminalFinishReason(')
    expect(source).toContain('assistantMessage.warningsJson = JSON.stringify(finalResponseWarnings)')
    expect(source).toContain('warnings: finalResponseWarnings || undefined')
  })

  it('database persists warnings_json with a safe additive migration', () => {
    const source = readFileSync(new URL('../src/main/database.ts', import.meta.url), 'utf8')
    expect(source).toContain('warningsJson?: string')
    expect(source).toContain('warnings_json TEXT')
    expect(source).toContain('ALTER TABLE messages ADD COLUMN warnings_json TEXT')
    expect(source).toContain('message.warningsJson')
    expect(source).toContain('warningsJson: row.warnings_json')
  })

  it('renderer keeps warnings separate from assistant content', () => {
    const chatInterface = readFileSync(new URL('../src/renderer/src/components/chat/ChatInterface.tsx', import.meta.url), 'utf8')
    const messageBubble = readFileSync(new URL('../src/renderer/src/components/chat/MessageBubble.tsx', import.meta.url), 'utf8')
    expect(chatInterface).toContain('const responseWarnings = extractResponsesWarnings({ warnings: data.warnings })')
    expect(chatInterface).toContain('warnings: responseWarnings ?? m.warnings')
    expect(chatInterface).not.toContain("finalContent += '\\n\\n---\\n*'")
    expect(messageBubble).toContain('{warnings && warnings.length > 0 && (')
    expect(messageBubble).toContain('warnings.map((warning, index) => (')
    expect(messageBubble).toContain('{warning}</span>')
  })
})
