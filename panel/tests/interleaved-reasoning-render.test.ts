import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { MessageBubble } from '../src/renderer/src/components/chat/MessageBubble'

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
