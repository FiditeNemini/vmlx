import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { MessageBubble } from '../src/renderer/src/components/chat/MessageBubble'
import { ReasoningBox } from '../src/renderer/src/components/chat/ReasoningBox'
import { AssistantDisplayTimelineRecorder } from '../src/shared/assistantDisplayTimeline'

vi.mock('dompurify', () => ({
  default: {
    sanitize: (html: string) => html,
  },
}))

const baseMessage = {
  id: 'assistant-1',
  role: 'assistant' as const,
  content: '',
  timestamp: Date.now(),
}

function renderBubble(props: Record<string, unknown>): string {
  return renderToStaticMarkup(React.createElement(MessageBubble as any, props))
}

describe('interleaved reasoning rendered display', () => {
  it.each([true, false])('renders recorded reasoning/text/tool/result phases in order with streaming=%s', isStreaming => {
    const record = new AssistantDisplayTimelineRecorder()
    const segments = ['PLAN-WRITE', 'PLAN-READ', 'CHECKED-BOTH']
    record.observeReasoning([segments[0]])
    record.observeContent('BEFORE-WRITE\n')
    record.tool('write-1')
    record.observeReasoning(segments.slice(0, 2))
    record.observeContent('BEFORE-WRITE\nBETWEEN-TOOLS\n')
    record.tool('read-2')
    record.observeReasoning(segments)
    const content = 'BEFORE-WRITE\nBETWEEN-TOOLS\nFINAL-ANSWER'
    record.observeContent(content)
    const html = renderBubble({
      message: { ...baseMessage, content, displayTimeline: record.finalize(content, segments) },
      reasoningSegments: segments, reasoningDone: !isStreaming, isStreaming,
      toolStatuses: [
        { phase: 'calling', toolName: 'write_file', toolCallId: 'write-1', contentOffset: 13, detail: '{"path":"/tmp/write","content":"value"}' },
        { phase: 'result', toolName: 'write_file', toolCallId: 'write-1', detail: 'write completed' },
        { phase: 'calling', toolName: 'read_file', toolCallId: 'read-2', contentOffset: 27, detail: '{"path":"/tmp/read"}' },
        { phase: 'result', toolName: 'read_file', toolCallId: 'read-2', detail: 'read completed' },
      ],
    })
    const positions = ['PLAN-WRITE', 'BEFORE-WRITE', 'data-vmlx-proof-tool-call-id="write-1"', 'PLAN-READ', 'BETWEEN-TOOLS', 'data-vmlx-proof-tool-call-id="read-2"', 'CHECKED-BOTH', 'FINAL-ANSWER'].map(text => html.indexOf(text))
    expect(positions.every(position => position >= 0)).toBe(true)
    expect(positions).toEqual([...positions].sort((a, b) => a - b))
    expect(html.match(/data-vmlx-proof-tool-phase="result"/g)).toHaveLength(2)
  })

  it('keeps the live processing/generating status row after the last recorded item while streaming', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.observeReasoning(['PLAN'])
    record.tool('write-1')
    const toolStatuses = [
      { phase: 'calling', toolName: 'write_file', toolCallId: 'write-1', contentOffset: 0, detail: '{}' },
      { phase: 'result', toolName: 'write_file', toolCallId: 'write-1', detail: 'done' },
      { phase: 'processing' },
    ]
    const streaming = renderBubble({
      message: { ...baseMessage, content: '', displayTimeline: record.snapshot() },
      reasoningSegments: ['PLAN'], reasoningDone: false, isStreaming: true, toolStatuses,
    })
    const toolAt = streaming.indexOf('data-vmlx-proof-tool-call-id="write-1"')
    const rowAt = streaming.indexOf('data-vmlx-proof-tool-progress="processing"')
    expect(toolAt).toBeGreaterThan(-1)
    expect(rowAt).toBeGreaterThan(toolAt)
    const generating = renderBubble({
      message: { ...baseMessage, content: '', displayTimeline: record.snapshot() },
      reasoningSegments: ['PLAN'], reasoningDone: false, isStreaming: true,
      toolStatuses: [...toolStatuses.slice(0, 2), { phase: 'generating' }],
    })
    expect(generating.indexOf('data-vmlx-proof-tool-progress="generating"')).toBeGreaterThan(generating.indexOf('data-vmlx-proof-tool-call-id="write-1"'))
    // Not streaming: no synthetic progress row survives completion.
    const done = renderBubble({
      message: { ...baseMessage, content: 'ANSWER', displayTimeline: (() => { record.observeContent('ANSWER'); return record.snapshot() })() },
      reasoningSegments: ['PLAN'], reasoningDone: true, isStreaming: false, toolStatuses,
    })
    expect(done).not.toContain('data-vmlx-proof-tool-progress')
  })

  it('shows real elapsed time on an open tool call and the measured duration once it completed', () => {
    const t0 = Date.now() - 7_400
    const record = new AssistantDisplayTimelineRecorder()
    record.observeReasoning(['PLAN'])
    record.tool('cmd-1')
    const open = renderBubble({
      message: { ...baseMessage, content: '', displayTimeline: record.snapshot() },
      reasoningSegments: ['PLAN'], reasoningDone: false, isStreaming: true,
      toolStatuses: [
        { phase: 'calling', toolName: 'run_command', toolCallId: 'cmd-1', contentOffset: 0, detail: '{"command":"sleep 6"}', timestamp: t0 - 300 },
        { phase: 'executing', toolName: 'run_command', toolCallId: 'cmd-1', timestamp: t0 },
      ],
    })
    expect(open).toMatch(/data-vmlx-proof-tool-elapsed-s="7"/)
    expect(open).toContain('running… 7s')
    expect(open).not.toMatch(/\d+%<\/span>/)   // no fabricated percentage in any status label
    const done = renderBubble({
      message: { ...baseMessage, content: 'ANSWER', displayTimeline: (() => { record.observeContent('ANSWER'); return record.snapshot() })() },
      reasoningSegments: ['PLAN'], reasoningDone: true, isStreaming: false,
      toolStatuses: [
        { phase: 'calling', toolName: 'run_command', toolCallId: 'cmd-1', contentOffset: 0, detail: '{"command":"sleep 6"}', timestamp: t0 - 300 },
        { phase: 'executing', toolName: 'run_command', toolCallId: 'cmd-1', timestamp: t0 },
        { phase: 'result', toolName: 'run_command', toolCallId: 'cmd-1', detail: 'done', timestamp: t0 + 6_250 },
      ],
    })
    expect(done).toMatch(/data-vmlx-proof-tool-duration-ms="6250"/)
    expect(done).toContain('6.3s')
    expect(done).not.toContain('data-vmlx-proof-tool-elapsed-s')
    // Legacy rows without timestamps keep the indicator only, no invented number.
    const legacy = renderBubble({
      message: { ...baseMessage, content: '', displayTimeline: record.snapshot() },
      reasoningSegments: ['PLAN'], reasoningDone: false, isStreaming: true,
      toolStatuses: [
        { phase: 'calling', toolName: 'run_command', toolCallId: 'cmd-1', contentOffset: 0, detail: '{}' },
        { phase: 'executing', toolName: 'run_command', toolCallId: 'cmd-1' },
      ],
    })
    expect(legacy).toContain('running…')
    expect(legacy).not.toContain('data-vmlx-proof-tool-elapsed-s')
    // The processing row carries its own real elapsed seconds.
    const processing = renderBubble({
      message: { ...baseMessage, content: '', displayTimeline: record.snapshot() },
      reasoningSegments: ['PLAN'], reasoningDone: false, isStreaming: true,
      toolStatuses: [
        { phase: 'calling', toolName: 'run_command', toolCallId: 'cmd-1', contentOffset: 0, detail: '{}', timestamp: t0 - 300 },
        { phase: 'result', toolName: 'run_command', toolCallId: 'cmd-1', detail: 'done', timestamp: t0 },
        { phase: 'processing', timestamp: Date.now() - 3_100 },
      ],
    })
    expect(processing).toMatch(/data-vmlx-proof-tool-progress-elapsed-s="3"/)
  })

  it('keeps final reasoning after multiple calls from an empty-reasoning pass', () => {
    const record = new AssistantDisplayTimelineRecorder()
    record.tool('a'); record.tool('b'); record.observeReasoning(['', 'AFTER-RESULTS'])
    record.observeContent('VISIBLE-FINAL')
    const html = renderBubble({
      message: { ...baseMessage, content: 'VISIBLE-FINAL', displayTimeline: record.snapshot() },
      reasoningSegments: ['', 'AFTER-RESULTS'], reasoningDone: true,
      toolStatuses: ['a', 'b'].flatMap(toolCallId => [
        { phase: 'calling', toolName: 'read_file', toolCallId, iteration: 1, detail: '{"path":"/tmp/value"}' },
        { phase: 'result', toolName: 'read_file', toolCallId, detail: 'result' },
      ]),
    })
    expect(html.indexOf('data-vmlx-proof-tool-call-id="b"')).toBeLessThan(html.indexOf('AFTER-RESULTS'))
    expect(html.indexOf('AFTER-RESULTS')).toBeLessThan(html.indexOf('VISIBLE-FINAL'))
  })
  it('renders user-message TeX through the same sanitized KaTeX path as assistant messages', () => {
    const html = renderBubble({
      message: {
        id: 'user-math-1',
        role: 'user',
        content: 'The literal currency string is $43 and \\(47 \\times 19 = 893 < 920 = 46 \\times 20\\).',
        timestamp: Date.now(),
      },
      isStreaming: false,
    })

    expect(html).toContain('class="katex"')
    expect(html).toContain('47')
    expect(html).toContain('×')
    expect(html).toContain('$43')
    expect(html).not.toContain('\\times')
    expect(html).not.toContain('\\(')
  })

  it('renders single-dollar math immediately after literal currency', () => {
    const html = renderBubble({
      message: {
        ...baseMessage,
        content:
          'The literal currency string is $43 and $47 \\times 19 = 893 < 920 = 46 \\times 20$.',
      },
      isStreaming: false,
    })

    expect(html).toContain('$43')
    expect(html).toContain('class="katex"')
    expect(html).toContain('47')
    expect(html).toContain('×')
    expect(html).not.toContain('$47')
    expect(html).not.toContain('\\times')
  })

  it('shows angle-bracket placeholders in user prompts instead of parsing them as HTML', () => {
    const html = renderBubble({
      message: {
        id: 'user-placeholder-1',
        role: 'user',
        content:
          'PATH=panel/package.json SIZE=<human-readable size>; MATH: \\(893 < 920\\).',
        timestamp: Date.now(),
      },
      isStreaming: false,
    })

    expect(html).toContain('SIZE=&lt;human-readable size&gt;')
    expect(html).not.toContain('<human-readable')
    expect(html).toContain('class="katex"')
  })

  it('shows structured assistant XML literally instead of dropping its tags', () => {
    const html = renderBubble({
      message: {
        ...baseMessage,
        content:
          'XML=<result status="ok">5.2 KB</result>\nMATH: \\(893 < 920\\)',
      },
      isStreaming: false,
    })

    expect(html).toContain(
      'XML=&lt;result status=&quot;ok&quot;&gt;5.2 KB&lt;/result&gt;',
    )
    expect(html).not.toContain('<result')
    expect(html).toContain('class="katex"')
  })

  it('shows structured XML literally in a completed reasoning rail', () => {
    const html = renderToStaticMarkup(
      React.createElement(ReasoningBox, {
        content:
          'Check <constraint status="active">x < 4</constraint> and \\(2 + 2 = 4\\).',
        isStreaming: false,
        isDone: false,
      }),
    )

    expect(html).toContain(
      'Check &lt;constraint status=&quot;active&quot;&gt;x &lt; 4&lt;/constraint&gt;',
    )
    expect(html).not.toContain('<constraint')
    expect(html).toContain('class="katex"')
  })

  it('renders multimodal user text as math without rewriting currency or code', () => {
    const html = renderBubble({
      message: {
        id: 'user-math-2',
        role: 'user',
        content: JSON.stringify([
          {
            type: 'text',
            text: 'Cost $43; calculate \\(6 \\times 7 = 42\\); keep `\\times` literal in code.',
          },
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,AA==' },
          },
        ]),
        timestamp: Date.now(),
      },
      isStreaming: false,
    })

    expect(html).toContain('class="katex"')
    expect(html).toContain('6')
    expect(html).toContain('×')
    expect(html).toContain('$43')
    expect(html).toContain('<code>\\times</code>')
    expect(html).toContain('<img')
  })

  it('preserves TeX-looking source inside GFM tilde-fenced code', () => {
    const html = renderBubble({
      message: {
        ...baseMessage,
        content: [
          'Rendered math: \\(6 \\times 7 = 42\\).',
          '~~~python',
          'raw = r"\\frac{94}{90} and $x_1$ and 2 * 3"',
          '~~~',
        ].join('\n'),
      },
      isStreaming: false,
    })

    expect(html).toContain('class="katex"')
    expect(html).toContain('class="hljs language-python"')
    expect(html).toContain('\\frac{94}{90}')
    expect(html).toContain('$x_1$')
    expect(html).toContain('2 * 3')
  })

  it('live-replaces previous reasoning segments while streaming and shows all after completion', () => {
    const segments = [
      'First reasoning segment before tool.',
      'Second reasoning segment after tool.',
    ]

    const streaming = renderBubble({
      message: baseMessage,
      isStreaming: true,
      reasoningSegments: segments,
      reasoningDone: false,
      isLastAssistant: true,
    })

    expect(streaming).not.toContain('First reasoning segment before tool.')
    expect(streaming).toContain('Second reasoning segment after tool.')

    const completed = renderBubble({
      message: baseMessage,
      isStreaming: false,
      reasoningSegments: segments,
      reasoningDone: true,
      isLastAssistant: true,
    })

    expect(completed).toContain('First reasoning segment before tool.')
    expect(completed).toContain('Second reasoning segment after tool.')
  })

  it('renders reasoning and structured tool status without leaking raw tool parser markup', () => {
    const html = renderBubble({
      message: {
        ...baseMessage,
        content: [
          'I will inspect the file.',
          '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/a.txt"}}</tool_call>',
          'The file says hello.',
        ].join('\n'),
      },
      isStreaming: false,
      reasoningSegments: [
        'Need to inspect the file before answering.',
        'Tool returned the relevant text.',
      ],
      reasoningDone: true,
      toolStatuses: [
        {
          phase: 'calling',
          toolName: 'read_file',
          toolCallId: 'call-read-1',
          detail: '{"path":"/tmp/a.txt"}',
          contentOffset: 25,
          timestamp: 1,
        },
        {
          phase: 'result',
          toolName: 'read_file',
          toolCallId: 'call-read-1',
          detail: 'hello',
          timestamp: 2,
        },
      ],
      isLastAssistant: true,
    })

    expect(html).toContain('Need to inspect the file before answering.')
    expect(html).toContain('Tool returned the relevant text.')
    expect(html).toContain('Read')
    expect(html).toContain('/tmp/a.txt')
    expect(html).toContain('The file says hello.')
    expect(html).not.toContain('<tool_call')
    expect(html).not.toContain('</tool_call>')
    expect(html).not.toContain('zyphra_tool_call')
    expect(html).not.toContain('&lt;tool_call')
  })
})
