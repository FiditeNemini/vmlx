import { renderToString } from 'katex'

// Preserve both CommonMark backtick fences and GFM tilde fences before any
// TeX normalization. Model-generated code frequently uses either spelling;
// rewriting `\times`, `$...$`, or `*` inside a tilde fence corrupts source.
const CODE_RE = /```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`/g
const LITERAL_MARKDOWN_PROTECTED_RE =
  /```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)|\$\$[\s\S]*?\$\$/g

const KATEX_OPTIONS = {
  output: 'html' as const,
  // A failed parse must take the escaped text fallback below. KaTeX's
  // throwOnError=false path emits a visible `.katex-error` node, which makes
  // malformed model scratch math look like renderer corruption.
  throwOnError: true,
  strict: 'ignore' as const,
  trust: false,
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function looksLikeSingleDollarMath(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed || trimmed !== text) return false
  // Two currency amounts can otherwise be mistaken for one math span while
  // streaming (for example "$5<$10" temporarily matches "$5<$").
  if (/^[+\-*/=<>]/.test(trimmed) || /[+\-*/=<>]$/.test(trimmed)) return false
  if (/^\d+(?:[.,]\d{2})?$/.test(trimmed)) return false
  if (/\\[A-Za-z]+/.test(trimmed)) return true
  if (/[{}_^=<>]/.test(trimmed)) return true
  if (/(?:[\dA-Za-z])\s*[+\-*/]\s*(?:[\dA-Za-z])/.test(trimmed)) return true
  if (/^[A-Za-z]$/.test(trimmed)) return true
  return false
}

function isEscapedAt(text: string, index: number): boolean {
  let precedingBackslashes = 0
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor--) {
    precedingBackslashes += 1
  }
  return precedingBackslashes % 2 === 1
}

function findNextSingleDollar(text: string, start: number): number {
  for (let index = start; index < text.length; index++) {
    if (text[index] !== '$' || isEscapedAt(text, index)) continue
    return index
  }
  return -1
}

function replaceSingleDollarMathLine(
  line: string,
  renderBody: (body: string) => string,
): string {
  let output = ''
  let unchangedStart = 0
  let opener = findNextSingleDollar(line, 0)

  while (opener >= 0) {
    let closer = findNextSingleDollar(line, opener + 1)

    while (closer >= 0) {
      const body = line.slice(opener + 1, closer)
      if (looksLikeSingleDollarMath(body)) {
        output += line.slice(unchangedStart, opener)
        output += renderBody(body)
        unchangedStart = closer + 1
        opener = findNextSingleDollar(line, unchangedStart)
        break
      }

      // A rejected pair must not consume the candidate closer. It may be the
      // valid opener that follows literal currency, as in
      // `$43 and $47 \times 19 = 893$`.
      output += line.slice(unchangedStart, opener + 1)
      unchangedStart = opener + 1
      opener = closer
      closer = findNextSingleDollar(line, opener + 1)
    }

    if (closer < 0) {
      return output + line.slice(unchangedStart)
    }
  }

  return output + line.slice(unchangedStart)
}

function replaceSingleDollarMath(
  text: string,
  renderBody: (body: string) => string,
): string {
  return text
    .split('\n')
    .map((line) => replaceSingleDollarMathLine(line, renderBody))
    .join('\n')
}

function normalizeEscapedUnicodeMath(text: string): string {
  // A model can begin a TeX command with `\` and then emit the equivalent
  // Unicode operator token instead of the command letters (`\×` rather than
  // `\times`). That byte sequence is invalid TeX but has one unambiguous
  // display meaning. Remove only the stray slash before known math glyphs;
  // ordinary backslashes and API payloads remain untouched.
  return text.replace(/\\([×÷·≈≤≥≠±→←∞π])/g, '$1')
}

function renderMath(raw: string, displayMode: boolean): string {
  const source = normalizeEscapedUnicodeMath(raw.trim())
  if (!source) return ''

  try {
    const html = renderToString(source, {
      ...KATEX_OPTIONS,
      displayMode,
    })
    return displayMode
      ? `<div class="math-block">${html}</div>`
      : `<span class="math-inline">${html}</span>`
  } catch (_error) {
    const fallback = escapeHtml(source)
    return displayMode
      ? `<div class="math-block math-fallback">${fallback}</div>`
      : `<span class="math-inline math-fallback">${fallback}</span>`
  }
}

function normalizeBareLatexCommands(markdown: string): string {
  // Models sometimes emit a few TeX commands without delimiters. Do not try to
  // parse arbitrary surrounding prose as math here; just make common commands
  // readable so the UI never shows broken-looking backslash words in normal text.
  let out = normalizeEscapedUnicodeMath(markdown)
  for (let i = 0; i < 8; i++) {
    const next = out.replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, '$1/$2')
    if (next === out) break
    out = next
  }
  out = out
    .replace(/\\sqrt\s*\{([^{}]+)\}/g, '√($1)')
    .replace(/\\overline\s*\{([^{}]+)\}/g, (_match, body: string) =>
      [...body].map((char) => `${char}\u0305`).join('')
    )
    .replace(/\\text\s*\{([^{}]+)\}/g, '$1')
    .replace(/([A-Za-z0-9)])\^\{([+\-]?\d+)\}/g, (_match, base: string, exponent: string) => {
      const superscript: Record<string, string> = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻',
      }
      return `${base}${[...exponent].map((char) => superscript[char] || char).join('')}`
    })
  return out
    .replace(/\\\(/g, '(')
    .replace(/\\\)/g, ')')
    .replace(/\\times\b/g, '×')
    .replace(/\\div\b/g, '÷')
    .replace(/\\cdot\b/g, '·')
    .replace(/\\approx\b/g, '≈')
    .replace(/\\leq?\b/g, '≤')
    .replace(/\\geq?\b/g, '≥')
    .replace(/\\neq\b/g, '≠')
    .replace(/\\pm\b/g, '±')
    .replace(/\\rightarrow\b/g, '→')
    .replace(/\\leftarrow\b/g, '←')
    .replace(/\\infty\b/g, '∞')
    .replace(/\\ldots\b/g, '…')
    .replace(/\\pi\b/g, 'π')
    .replace(/\\%/g, '%')
    .replace(/\\,/g, ' ')
    .replace(/\\;/g, ' ')
    .replace(/\\!/g, '')
    .replace(/\\left\b/g, '')
    .replace(/\\right\b/g, '')
}

function escapeBareArithmeticAsterisks(markdown: string): string {
  // CommonMark may pair multiplication operators from separate expressions as
  // emphasis delimiters. A model list such as `37*28=...\n37*29=...` then
  // renders as `3728=...3729=...`, even though the API/SQLite bytes are intact.
  // Escape only operator runs directly between operands. Normal prose emphasis
  // (`*important*`) and code spans/fences remain untouched.
  return markdown.replace(
    /([\p{L}\p{N})\]])(\*{1,2})(?=[\p{L}\p{N}(\[])/gu,
    (_match, left: string, operator: string) =>
      `${left}${operator.replace(/\*/g, '\\*')}`,
  )
}

function normalizeRepeatedMathDelimiters(markdown: string): string {
  // Some model streams duplicate an adjacent opener while producing only one
  // closer (for example `\(\(47 \times 19\)`). Nested TeX math delimiters are
  // invalid, so collapse only immediately repeated delimiter tokens. This
  // keeps currency and ordinary parentheses untouched and gives KaTeX the
  // valid span the model clearly intended.
  return markdown
    .replace(/(?:\\\(\s*){2,}/g, '\\(')
    .replace(/(?:\\\)\s*){2,}/g, '\\)')
    .replace(/(?:\\\[\s*){2,}/g, '\\[')
    .replace(/(?:\\\]\s*){2,}/g, '\\]')
}

/**
 * Readable, allocation-light math view for the actively streaming reasoning
 * rail. The completed rail is rendered with KaTeX; this path only prevents
 * transient raw delimiters and common TeX commands from flashing while tokens
 * are still arriving.
 */
export function prepareStreamingPlainTextMath(markdown: string): string {
  if (!markdown) return ''
  const normalized = normalizeRepeatedMathDelimiters(markdown)
  return normalizeBareLatexCommands(
    replaceSingleDollarMath(
      normalized
      .replace(/\\\[([\s\S]*?)(?:\\\]|$)/g, '$1')
      .replace(/\\\(([^\n]*?)(?:\\\)|$)/g, '$1')
      .replace(/\$\$([\s\S]*?)(?:\$\$|$)/g, '$1'),
      (body) => body,
    )
  )
}

function transformMath(markdown: string): string {
  let out = markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, body) => renderMath(body, true))
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, body) => renderMath(body, true))
    // Inline math must not consume later paragraphs when a model leaves one
    // opener unmatched in a reasoning stream.
    .replace(/\\\(([^\n]*?)\\\)/g, (_match, body) => renderMath(body, false))

  out = replaceSingleDollarMath(out, (body) => renderMath(body, false))

  out = normalizeBareLatexCommands(out)
  return escapeBareArithmeticAsterisks(out)
}

export function prepareMarkdownWithMath(markdown: string): string {
  if (!markdown) return ''

  const protectedSegments: string[] = []
  const protectedMarkdown = markdown.replace(CODE_RE, (segment) => {
    const index = protectedSegments.push(segment) - 1
    return `\u0000CODE${index}\u0000`
  })

  const transformed = transformMath(
    normalizeRepeatedMathDelimiters(protectedMarkdown)
  )

  return transformed.replace(/\u0000CODE(\d+)\u0000/g, (_match, indexText) => {
    const index = Number(indexText)
    return protectedSegments[index] || ''
  })
}

/**
 * Chat text is Markdown, not trusted/raw HTML. Marked otherwise treats a
 * placeholder such as `<human-readable size>` or structured model output such
 * as `<result status="ok">` as HTML; DOMPurify then removes unknown tags and
 * the UI visibly drops bytes that remain intact in SQLite and the API.
 *
 * Escape raw HTML before Markdown parsing while protecting code and TeX spans
 * so code fences stay literal and KaTeX still receives the original
 * comparison operators. Renderer-owned KaTeX HTML is injected only after the
 * raw chat text has been escaped.
 */
export function prepareLiteralMarkdownWithMath(markdown: string): string {
  if (!markdown) return ''

  const protectedSegments: string[] = []
  let protectedMarkdown = markdown.replace(LITERAL_MARKDOWN_PROTECTED_RE, (segment) => {
    const index = protectedSegments.push(segment) - 1
    return `\u0000CHATLITERAL${index}\u0000`
  })
  protectedMarkdown = replaceSingleDollarMath(protectedMarkdown, (body) => {
    const index = protectedSegments.push(`$${body}$`) - 1
    return `\u0000CHATLITERAL${index}\u0000`
  })
  const escapedMarkdown = escapeHtml(protectedMarkdown)
  const restoredMarkdown = escapedMarkdown.replace(
    /\u0000CHATLITERAL(\d+)\u0000/g,
    (_match, indexText) => protectedSegments[Number(indexText)] || '',
  )
  return prepareMarkdownWithMath(restoredMarkdown)
}

export function prepareUserMarkdownWithMath(markdown: string): string {
  return prepareLiteralMarkdownWithMath(markdown)
}

export function prepareAssistantMarkdownWithMath(markdown: string): string {
  return prepareLiteralMarkdownWithMath(markdown)
}
