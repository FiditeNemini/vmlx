import { describe, expect, it } from 'vitest'
import { marked } from 'marked'
import {
  prepareMarkdownWithMath,
  prepareStreamingPlainTextMath,
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

  it('does not treat plain dollar amounts as math', () => {
    const rendered = prepareMarkdownWithMath('The cost is $43 today.')

    expect(rendered).toBe('The cost is $43 today.')
  })

  it('does not consume dollars when comparing currency amounts', () => {
    const rendered = prepareMarkdownWithMath('Compare $5<$10 and $5 < $10.')

    expect(rendered).toBe('Compare $5<$10 and $5 < $10.')
    expect(rendered).not.toContain('math-inline')
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
})
