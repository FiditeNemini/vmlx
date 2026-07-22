import { describe, expect, it } from 'vitest'
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

  it('preserves raw TeX inside inline code and fenced code', () => {
    const rendered = prepareMarkdownWithMath('`\\frac{1}{2}`\n```txt\n47 \\times 2\n```')

    expect(rendered).toContain('`\\frac{1}{2}`')
    expect(rendered).toContain('47 \\times 2')
    expect(rendered).not.toContain('math-inline')
    expect(rendered).not.toContain('×')
  })
})
