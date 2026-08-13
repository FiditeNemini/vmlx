import { describe, expect, it } from 'vitest'
import {
  prepareAssistantMarkdownWithMath,
} from '../src/renderer/src/components/chat/mathMarkdown'

/**
 * Chat text is HTML-escaped before Markdown parsing so model output like
 * `<result status="ok">` survives as visible text instead of being eaten as
 * markup. Code and TeX spans are pulled out first so they stay literal.
 *
 * Only CLOSED fences were pulled out. A fence is unclosed for the entire time
 * it streams, so every code block rendered `print(&quot;hello&quot;)` while it
 * was arriving, and stayed that way whenever the model finished inside the
 * block or was cut off by max_tokens — which is the normal shape of a reply
 * that ends with code. Live in the app, DSV4 rendered exactly that.
 *
 * Escaping is not the XSS boundary here — DOMPurify (sanitizeChatHtml) is, and
 * it runs after Markdown. Protecting an unclosed fence therefore extends what
 * closed fences already had rather than removing a defence.
 */
describe('code fences keep their literal text', () => {
  it('keeps quotes in a closed fence', () => {
    expect(prepareAssistantMarkdownWithMath('```python\nprint("hello")\n```\n'))
      .toContain('print("hello")')
  })

  it('keeps quotes while the fence is still streaming', () => {
    // No closing fence yet — the regression.
    expect(prepareAssistantMarkdownWithMath('```python\nprint("hello")\n'))
      .toContain('print("hello")')
  })

  it('keeps quotes in an unclosed tilde fence', () => {
    expect(prepareAssistantMarkdownWithMath('~~~python\nprint("hello")\n'))
      .toContain('print("hello")')
  })

  it('keeps quotes in an inline code span', () => {
    expect(prepareAssistantMarkdownWithMath('Use `print("hi")` now.'))
      .toContain('print("hi")')
  })

  it('still escapes raw HTML OUTSIDE code', () => {
    // The reason escaping exists: this must not become markup.
    const out = prepareAssistantMarkdownWithMath('a <result status="ok"> b')
    expect(out).toContain('&lt;result')
    expect(out).not.toContain('<result')
  })

  it('a closed fence still wins over the unclosed alternative', () => {
    // Ordering: if the unclosed form matched first it would swallow the text
    // after the fence and protect it from escaping too.
    const out = prepareAssistantMarkdownWithMath(
      '```py\ncode("x")\n```\nthen <b>bold</b> after\n',
    )
    expect(out).toContain('code("x")')
    expect(out).toContain('&lt;b&gt;')
  })

  it('an unclosed fence protects its body, not a stray inline span', () => {
    // The inline-backtick alternative would match the first two backticks of an
    // unclosed ``` as an empty span and leave the body exposed.
    const out = prepareAssistantMarkdownWithMath('```js\nconst a = "q";\n')
    expect(out).toContain('const a = "q";')
    expect(out).not.toContain('&quot;')
  })
})

/**
 * Math delimiters are deliberately NOT given an unclosed form. Extending the
 * fence fix to `$$`/`\[` does stop `<` escaping mid-stream, but it swallows the
 * rest of any message with a stray opener and real markup then stops being
 * escaped — dropped from view instead of shown as text. A lone `$` is ordinary
 * prose. The fence case differs: its body is literal code, so swallowing an
 * unterminated block is correct, and its corruption was permanent, not
 * transient.
 */
describe('math delimiters decline the unclosed-span trick', () => {
  it('a stray dollar never disables escaping for the rest of the message', () => {
    const out = prepareAssistantMarkdownWithMath(
      'It costs $5 and then <result status="ok"> appears',
    )
    expect(out).toContain('&lt;result')
  })

  it('a stray double-dollar never disables escaping either', () => {
    // This is the regression that an unclosed-$$ alternative introduces.
    const out = prepareAssistantMarkdownWithMath('cost $$5 then <result> here')
    expect(out).toContain('&lt;result')
  })

  it('paired math still renders', () => {
    expect(prepareAssistantMarkdownWithMath('x $a < b$ y')).toContain('math-inline')
    expect(prepareAssistantMarkdownWithMath('x\n\n$$a < b$$\n')).toContain('math-block')
  })
})
