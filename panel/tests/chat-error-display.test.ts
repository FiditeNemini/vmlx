import { describe, expect, it } from 'vitest'
import {
  projectedMetalHeadroomChatErrorContent,
  promptTooLongChatErrorContent,
} from '../src/shared/chatErrorDisplay'

describe('chat error display policy', () => {
  it('turns projected Metal headroom rejects into visible assistant content', () => {
    const content = projectedMetalHeadroomChatErrorContent(
      'API error: 413 - Requested max output tokens exceed projected safe Metal headroom: requested=8192, safe_cap=1. Reduce max_tokens/max_output_tokens, context length, paged cache blocks, or load a smaller model. Set VMLX_METAL_PROJECTED_OUTPUT_GUARD=0 only for explicit developer diagnostics; disabling it accepts Metal OOM / kernel-panic risk.',
    )

    expect(content).toContain('Generation blocked')
    expect(content).toContain('requested=8192')
    expect(content).toContain('safe_cap=1')
    expect(content).toContain('Metal OOM / kernel-panic risk')
    expect(content).toContain('sudo sysctl iogpu.wired_limit_mb=120000')
  })

  it('does not convert ordinary connection errors into assistant content', () => {
    expect(projectedMetalHeadroomChatErrorContent('fetch failed')).toBeNull()
    expect(projectedMetalHeadroomChatErrorContent('Server connection lost')).toBeNull()
  })

  it('turns engine prompt_too_long 413 into visible assistant content (GH #253)', () => {
    const content = promptTooLongChatErrorContent(
      'Failed to send message: API error: 413 - Prompt too long (prompt: ~20,767 tokens), ' +
        'exceeding the configured prompt/context limit of ~20,119 tokens. Shorten the prompt, ' +
        'raise --max-prompt-tokens, or use a model/session with a larger prompt/context cap.',
    )

    expect(content).toContain('Message not sent')
    expect(content).toContain('~20,767 tokens')
    expect(content).toContain('~20,119 tokens')
    expect(content).toContain('start a new chat')
  })

  it('unwraps a raw OpenAI-style JSON error body for prompt_too_long', () => {
    const body = JSON.stringify({
      error: {
        message:
          'Prompt too long (prompt: ~34,354 tokens), exceeding the configured ' +
          'prompt/context limit of ~20,119 tokens. Shorten the prompt.',
        type: 'invalid_request_error',
        code: 'prompt_too_long',
      },
    })
    const content = promptTooLongChatErrorContent(`API error: 413 - ${body}`)

    expect(content).toContain('Message not sent')
    expect(content).toContain('~34,354 tokens')
    expect(content).not.toContain('invalid_request_error')
    expect(content).not.toContain('{')
  })

  it('matches the PromptTooLongError exception wording too', () => {
    const content = promptTooLongChatErrorContent(
      'prompt_too_long: prompt has 30000 tokens, max prompt/context tokens is 20119',
    )
    expect(content).toContain('Message not sent')
    expect(content).toContain('30000 tokens')
  })

  it('does not convert unrelated errors into prompt_too_long bubbles', () => {
    expect(promptTooLongChatErrorContent('fetch failed')).toBeNull()
    expect(promptTooLongChatErrorContent('Server connection lost')).toBeNull()
    expect(
      promptTooLongChatErrorContent(
        'API error: 413 - Requested max output tokens exceed projected safe Metal headroom',
      ),
    ).toBeNull()
    expect(promptTooLongChatErrorContent('')).toBeNull()
    expect(promptTooLongChatErrorContent(null)).toBeNull()
  })
})
