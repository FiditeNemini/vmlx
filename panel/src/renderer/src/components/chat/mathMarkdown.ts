import { renderToString } from 'katex'

// Preserve both CommonMark backtick fences and GFM tilde fences before any
// TeX normalization. Model-generated code frequently uses either spelling;
// rewriting `\times`, `$...$`, or `*` inside a tilde fence corrupts source.
const CODE_RE = /```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`/g

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

function renderMath(raw: string, displayMode: boolean): string {
  const source = raw.trim()
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
  let out = markdown
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
    normalized
      .replace(/\\\[([\s\S]*?)(?:\\\]|$)/g, '$1')
      .replace(/\\\(([^\n]*?)(?:\\\)|$)/g, '$1')
      .replace(/\$\$([\s\S]*?)(?:\$\$|$)/g, '$1')
      .replace(/(^|[^\\])\$([^$\n]+)\$/g, (match, prefix, body) => {
        if (!looksLikeSingleDollarMath(body)) return match
        return `${prefix}${body}`
      })
  )
}

function transformMath(markdown: string): string {
  let out = markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, body) => renderMath(body, true))
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, body) => renderMath(body, true))
    // Inline math must not consume later paragraphs when a model leaves one
    // opener unmatched in a reasoning stream.
    .replace(/\\\(([^\n]*?)\\\)/g, (_match, body) => renderMath(body, false))
    .replace(/(^|[^\\])\$([^$\n]+)\$/g, (match, prefix, body) => {
      if (!looksLikeSingleDollarMath(body)) return match
      return `${prefix}${renderMath(body, false)}`
    })

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
