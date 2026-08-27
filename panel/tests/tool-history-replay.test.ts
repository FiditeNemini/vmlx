import { describe, expect, it } from 'vitest'
import { replayPersistedAssistantHistory } from '../src/shared/toolHistoryReplay'

describe('persisted assistant tool-history replay', () => {
  const row = {
    content: 'D4-CURRENT-TO1-DONE',
    reasoningContent: 'inspect\n\nreport',
    reasoningSegmentsJson: JSON.stringify(['inspect', 'report']),
    toolCallsJson: JSON.stringify([
      {
        phase: 'calling',
        toolName: 'file_info',
        toolCallId: 'call_1',
        iteration: 0,
      },
    ]),
    toolCallsOaiJson: JSON.stringify([
      {
        id: 'call_1',
        type: 'function',
        function: {
          name: 'file_info',
          arguments: '{"path":"panel/package.json"}',
        },
      },
    ]),
    toolResultsOaiJson: JSON.stringify([
      { tool_call_id: 'call_1', content: 'Size: 5.2 KB' },
    ]),
  }

  it('replays Responses reasoning around the real call and result', () => {
    expect(replayPersistedAssistantHistory(row, true)).toEqual([
      {
        type: 'reasoning',
        content: [{ type: 'reasoning', text: 'inspect' }],
      },
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'file_info',
        arguments: '{"path":"panel/package.json"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: 'Size: 5.2 KB',
      },
      {
        type: 'reasoning',
        content: [{ type: 'reasoning', text: 'report' }],
      },
      { type: 'output_text', text: 'D4-CURRENT-TO1-DONE' },
    ])
  })

  it('replays Chat Completions reasoning_content on each assistant turn', () => {
    expect(replayPersistedAssistantHistory(row, false)).toEqual([
      {
        role: 'assistant',
        content: null,
        tool_calls: [JSON.parse(row.toolCallsOaiJson)[0]],
        reasoning_content: 'inspect',
      },
      {
        role: 'tool',
        tool_call_id: 'call_1',
        content: 'Size: 5.2 KB',
      },
      {
        role: 'assistant',
        content: 'D4-CURRENT-TO1-DONE',
        reasoning_content: 'report',
      },
    ])
  })

  it('uses call iteration metadata for sequential multi-tool ordering', () => {
    const multi = {
      ...row,
      content: 'DONE',
      reasoningSegmentsJson: JSON.stringify(['first', 'second', '']),
      toolCallsJson: JSON.stringify([
        { phase: 'calling', toolCallId: 'call_1', iteration: 0 },
        { phase: 'calling', toolCallId: 'call_2', iteration: 1 },
      ]),
      toolCallsOaiJson: JSON.stringify([
        JSON.parse(row.toolCallsOaiJson)[0],
        {
          id: 'call_2',
          type: 'function',
          function: { name: 'run_command', arguments: '{"command":"pwd"}' },
        },
      ]),
      toolResultsOaiJson: JSON.stringify([
        { tool_call_id: 'call_1', content: 'Size: 5.2 KB' },
        { tool_call_id: 'call_2', content: '/repo' },
      ]),
    }
    const replay = replayPersistedAssistantHistory(multi, true)

    expect(replay.map((item) => item.type)).toEqual([
      'reasoning',
      'function_call',
      'function_call_output',
      'reasoning',
      'function_call',
      'function_call_output',
      'output_text',
    ])
    expect(replay[4].call_id).toBe('call_2')
  })

  it('keeps a Responses reasoning-only assistant row as a boundary without replaying stale hidden text', () => {
    expect(
      replayPersistedAssistantHistory(
        {
          content: '',
          reasoningSegmentsJson: JSON.stringify([
            'stale private reasoning about a previous user prompt',
          ]),
        },
        true,
      ),
    ).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: '',
      },
    ])
  })

  it('omits stale private reasoning for an explicit no-tool turn while preserving real history', () => {
    expect(
      replayPersistedAssistantHistory(row, true, { includeReasoning: false }),
    ).toEqual([
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'file_info',
        arguments: '{"path":"panel/package.json"}',
      },
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: 'Size: 5.2 KB',
      },
      { type: 'output_text', text: 'D4-CURRENT-TO1-DONE' },
    ])

    expect(
      replayPersistedAssistantHistory(
        {
          content: '',
          reasoningContent: 'stale requirement to call a tool',
        },
        true,
        { includeReasoning: false },
      ),
    ).toEqual([{ type: 'message', role: 'assistant', content: '' }])
  })
})
