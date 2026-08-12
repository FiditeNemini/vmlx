import DOMPurify from 'dompurify'

/**
 * ONE sanitizer for every stretch of model-authored HTML the chat renders.
 *
 * MessageBubble (the answer) and ReasoningBox (the thinking rail) both render
 * markdown that the MODEL wrote, and each carried its own byte-equivalent
 * DOMPurify call. They had not diverged yet — but this is the worst kind of
 * duplicate to leave lying around: a future hardening (a FORBID_TAGS entry, a
 * narrowed ADD_ATTR) applied to one copy leaves the other permissive, and the
 * gap is invisible until someone finds it. A sanitizer is exactly the rule that
 * must exist once.
 *
 * The allowlist additions are for KaTeX, whose HTML renderer emits inline
 * layout styles, aria-hidden spans, and the data-vmlx-math-* attributes the
 * math pipeline round-trips. DOMPurify still owns the allowlist; scripts and
 * event handlers remain forbidden.
 */
export function sanitizeChatHtml(html: string): string {
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ADD_TAGS: ['pre', 'code'],
    ADD_ATTR: [
      'class',
      'style',
      'aria-hidden',
      'aria-label',
      'role',
      'data-vmlx-math-source-codepoints',
      'data-vmlx-math-delimiter',
      'data-vmlx-math-display-mode'
    ]
  })
}
