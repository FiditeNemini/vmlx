import { describe, expect, it } from 'vitest'
import { marked } from 'marked'
import {
  prepareMarkdownWithMath,
  prepareAssistantMarkdownWithMath,
  prepareStreamingPlainTextMath,
  prepareUserMarkdownWithMath,
} from '../src/renderer/src/components/chat/mathMarkdown'

describe('prepareMarkdownWithMath', () => {
  it('renders inline TeX delimiters as readable math text', () => {
    const rendered = prepareMarkdownWithMath('Multiply first: \\(47 \\times 2 = 94\\)')

    expect(rendered).toContain('math-inline')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('47')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('\\times')
    expect(rendered).not.toContain('\\(')
  })

  it('renders TeX fractions without exposing raw commands', () => {
    const rendered = prepareMarkdownWithMath('Exact answer: \\(\\frac{47}{45}\\)')

    expect(rendered).toContain('math-inline')
    expect(rendered).toContain('class="mfrac"')
    expect(rendered).toContain('47')
    expect(rendered).toContain('45')
    expect(rendered).not.toContain('\\frac')
  })

  it('renders display math blocks without raw dollar delimiters', () => {
    const rendered = prepareMarkdownWithMath('Compare:\n$$43 \\times 17 = 731$$')

    expect(rendered).toContain('math-block')
    expect(rendered).toContain('class="katex-display"')
    expect(rendered).toContain('43')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('$$')
  })

  it('normalizes bare TeX commands emitted by models outside delimiters', () => {
    const rendered = prepareMarkdownWithMath(
      'Divide: (94 \\div 90 = \\frac{94}{90}); \\approx 1.4 \\times 10^{-6}',
    )

    expect(rendered).toContain('÷')
    expect(rendered).toContain('94/90')
    expect(rendered).toContain('≈ 1.4 × 10⁻⁶')
    expect(rendered).not.toContain('\\div')
    expect(rendered).not.toContain('\\frac')
  })

  it('keeps the active reasoning rail readable before TeX delimiters close', () => {
    const rendered = prepareStreamingPlainTextMath(
      'Work: \\(47 \\times 2 = 94 and \\frac{47}{45}',
    )

    expect(rendered).toBe('Work: 47 × 2 = 94 and 47/45')
    expect(rendered).not.toContain('\\(')
    expect(rendered).not.toContain('\\times')
  })

  it('does not let an unmatched inline opener consume later paragraphs', () => {
    const rendered = prepareMarkdownWithMath(
      'Draft \\(47/45\nLater valid math: \\(2 + 2 = 4\\)',
    )

    expect(rendered).toContain('Draft (47/45')
    expect(rendered).toContain('Later valid math:')
    expect(rendered.match(/class="katex"/g)).toHaveLength(1)
  })

  it('uses readable escaped fallback instead of a visible KaTeX error node', () => {
    const rendered = prepareMarkdownWithMath('Malformed: \\(\\frac{47}{\\)')

    expect(rendered).toContain('math-fallback')
    expect(rendered).not.toContain('katex-error')
    expect(rendered).toContain('\\frac{47}{')
  })

  it('repairs duplicated adjacent model delimiters before KaTeX rendering', () => {
    const malformed =
      'Path: panel/package.json — Size: 5.2 KB\n$43 and inline TeX: \\(\\(47 \\times 19 = 893 < 920 = 46 \\times 20\\)'
    const rendered = prepareMarkdownWithMath(malformed)

    expect(rendered).toContain('$43')
    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('math-fallback')
    expect(rendered).not.toContain('\\(')
    expect(rendered).not.toContain('\\times')
  })

  it('renders a Unicode operator emitted after a TeX escape', () => {
    const malformed = 'CURRENCY=$43 TEX=\\(47 \\× 19 = 893\\)'
    const rendered = prepareMarkdownWithMath(malformed)

    expect(rendered).toContain('$43')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('math-fallback')
    expect(rendered).not.toContain('\\×')
  })

  it('retains canonical TeX identity in inert source, delimiter, and mode attributes', () => {
    const rawSource = '47 \\× 19 = 893 < 920'
    const rendered = prepareMarkdownWithMath(
      `CURRENCY=$43 TEX=\\(${rawSource}\\)`,
    )
    const encodedSource = Array.from(
      rawSource,
      (char) => char.codePointAt(0)!.toString(16),
    ).join('-')

    expect(rendered).toContain(
      `data-vmlx-math-source-codepoints="${encodedSource}"`,
    )
    expect(rendered).toContain('data-vmlx-math-delimiter="paren"')
    expect(rendered).toContain('data-vmlx-math-display-mode="inline"')
    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('math-fallback')

    const hostile = prepareMarkdownWithMath(
      '\\(x" data-vmlx-injected="yes < y\\)',
    )
    expect(hostile).toMatch(/data-vmlx-math-source-codepoints="[0-9a-f-]+"/)
    expect(hostile).not.toContain('data-vmlx-injected="yes')
  })

  it('retains distinct display delimiter identities for bracket and dollar math', () => {
    const rendered = prepareMarkdownWithMath(
      '\\[x + 1\\]\n$$y + 2$$',
    )

    expect(rendered).toContain('data-vmlx-math-delimiter="bracket"')
    expect(rendered).toContain('data-vmlx-math-delimiter="double-dollar"')
    expect(rendered.match(/data-vmlx-math-display-mode="display"/g)).toHaveLength(2)
    expect(rendered.match(/class="katex-display"/g)).toHaveLength(2)
  })

  it('keeps duplicated adjacent delimiters readable during reasoning streaming', () => {
    const rendered = prepareStreamingPlainTextMath(
      'Draft: \\(\\(47 \\times 19 = 893 < 920',
    )

    expect(rendered).toBe('Draft: 47 × 19 = 893 < 920')
  })

  it('keeps an escaped Unicode operator readable during reasoning streaming', () => {
    const rendered = prepareStreamingPlainTextMath('Draft: \\(47 \\× 19 = 893')

    expect(rendered).toBe('Draft: 47 × 19 = 893')
  })

  it('does not treat plain dollar amounts as math', () => {
    const rendered = prepareMarkdownWithMath('The cost is $43 today.')

    expect(rendered).toBe('The cost is $43 today.')
  })

  it('does not consume dollars when comparing currency amounts', () => {
    const rendered = prepareMarkdownWithMath('Compare $5<$10 and $5 < $10.')

    expect(rendered).toBe('Compare $5<$10 and $5 < $10.')
    expect(rendered).not.toContain('math-inline')
  })

  it('retries a rejected currency closer as the next valid math opener', () => {
    const rendered = prepareMarkdownWithMath(
      'The literal currency string is $43 and $47 \\times 19 = 893 < 920 = 46 \\times 20$.',
    )

    expect(rendered).toContain('$43 and ')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('47')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('$47')
    expect(rendered).not.toContain('\\times')
  })

  it('does not pair an unclosed math opener with a later currency amount through prose', () => {
    const rendered = prepareMarkdownWithMath(
      'The literal currency string is $43 and $47 \\times 19 = 893 < 920 = 46 \\times 20. This seems to be the second line. There is another literal currency string $43.',
    )

    expect(rendered).not.toContain('class="katex"')
    expect(rendered).toContain('$43 and $47 × 19 = 893 < 920 = 46 × 20')
    expect(rendered).toContain('literal currency string $43')
  })

  it('preserves escaped currency before a later valid single-dollar span', () => {
    const rendered = prepareMarkdownWithMath(
      'Escaped currency is \\$43; math is $6 \\times 7 = 42$.',
    )

    expect(rendered).toContain('\\$43')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('$6')
  })

  it('keeps the currency/math overlap readable during reasoning streaming', () => {
    const rendered = prepareStreamingPlainTextMath(
      'The literal currency string is $43 and $47 \\times 19 = 893$.',
    )

    expect(rendered).toBe(
      'The literal currency string is $43 and 47 × 19 = 893.',
    )
  })

  it('renders adjacent inline spans without mistaking their boundary for display math', () => {
    const rendered = marked.parse(
      prepareAssistantMarkdownWithMath('$x$$y$'),
    ) as string

    expect(rendered.match(/class="katex"/g)).toHaveLength(2)
    expect(rendered).not.toContain('$x')
    expect(rendered).not.toContain('$y')
  })

  it('renders multiplication without letting Markdown create emphasis', () => {
    const rendered = prepareMarkdownWithMath('Product: \\(2 * 3 * 4\\)')

    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('<em>')
    expect(rendered).not.toContain('*')
  })

  it('preserves bare arithmetic asterisks across repeated model calculations', () => {
    const prepared = prepareMarkdownWithMath(
      '37*28=1036 (sum 10)\n37*29=1073 (sum 11)\n37*30=1110 (sum 3)',
    )
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain('37*28=1036')
    expect(rendered).toContain('37*29=1073')
    expect(rendered).toContain('37*30=1110')
    expect(rendered).not.toContain('<em>')
    expect(rendered).not.toContain('<strong>')
  })

  it('keeps ordinary Markdown emphasis while escaping only arithmetic operators', () => {
    const prepared = prepareMarkdownWithMath('This is *important*, and x*y stays multiplication.')
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain('<em>important</em>')
    expect(rendered).toContain('x*y stays multiplication')
  })

  it('preserves raw TeX inside inline code and fenced code', () => {
    const rendered = prepareMarkdownWithMath('`\\frac{1}{2}`\n```txt\n47 \\times 2\n```')

    expect(rendered).toContain('`\\frac{1}{2}`')
    expect(rendered).toContain('47 \\times 2')
    expect(rendered).not.toContain('math-inline')
    expect(rendered).not.toContain('×')
  })

  it('escapes raw user HTML without corrupting protected TeX or code', () => {
    const prepared = prepareUserMarkdownWithMath(
      'SIZE=<human-readable size>; compare \\(893 < 920\\); keep `<tag>` literal.',
    )
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain('SIZE=&lt;human-readable size&gt;')
    expect(rendered).not.toContain('<human-readable')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('893')
    expect(rendered).toContain('<code>&lt;tag&gt;</code>')
  })

  it('preserves structured assistant XML as visible text alongside KaTeX', () => {
    const prepared = prepareAssistantMarkdownWithMath(
      'XML=<result status="ok">5.2 KB</result>; compare \\(893 < 920\\).',
    )
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain(
      'XML=&lt;result status=&quot;ok&quot;&gt;5.2 KB&lt;/result&gt;',
    )
    expect(rendered).not.toContain('<result')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('893')
  })
})
