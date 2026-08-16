import { readFileSync } from 'fs'
import { resolve } from 'path'
import { describe, expect, it } from 'vitest'
import {
  isMetalHeadroomBubbleContent,
  isPromptTooLongBubbleContent,
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

  it('strips the streaming "Server error:" wrapper (live-observed shape)', () => {
    const content = promptTooLongChatErrorContent(
      'Server error: prompt_too_long: tokenized prompt has 6251 tokens, max prompt/context tokens is 2048',
    )
    expect(content).toContain('Message not sent — prompt_too_long: tokenized prompt has 6251 tokens')
    expect(content).not.toContain('Server error:')
  })

  it('recognizes its own bubbles so history replay can exclude them', () => {
    const ptl = promptTooLongChatErrorContent(
      'API error: 413 - Prompt too long (prompt: ~30,000 tokens), exceeding the configured prompt/context limit of ~20,119 tokens.',
    )
    const metal = projectedMetalHeadroomChatErrorContent(
      'API error: 413 - Requested max output tokens exceed projected safe Metal headroom: requested=8192, safe_cap=1.',
    )
    expect(isPromptTooLongBubbleContent(ptl)).toBe(true)
    expect(isMetalHeadroomBubbleContent(metal)).toBe(true)
    // classes stay distinct and ordinary content stays replayable
    expect(isMetalHeadroomBubbleContent(ptl)).toBe(false)
    expect(isPromptTooLongBubbleContent(metal)).toBe(false)
    expect(isPromptTooLongBubbleContent('Here is my answer about tokens')).toBe(false)
    expect(isMetalHeadroomBubbleContent(undefined)).toBe(false)
  })

  it('chat.ts history replay skips error bubbles and pops the poisoned user message', () => {
    const chatSource = readFileSync(
      resolve(process.cwd(), 'src/main/ipc/chat.ts'),
      'utf8',
    )
    expect(chatSource).toContain(
      'if (m.role === "assistant" && isPromptTooLongBubbleContent(m.content)) {',
    )
    expect(chatSource).toContain(
      'if (prev && prev.role === "user") requestMessages.pop();',
    )
    expect(chatSource).toContain(
      'if (m.role === "assistant" && isMetalHeadroomBubbleContent(m.content)) {',
    )
  })

  it('a turn-peak admission decline gets the restart remedy, not shorten-message', () => {
    // The engine's first live refusal (turn-peak valve, ctx 101,678) shares
    // the 413/prompt_too_long envelope but has the OPPOSITE remedy: the
    // conversation is too deep for the current process, and restarting the
    // model serves the same turn (cache restores from disk — proven live).
    const body = JSON.stringify({
      error: {
        message:
          'hybrid delta: turn-peak admission declined a delta forward at a ' +
          'context of 101678 tokens — the cross-turn Metal peak walk projects ' +
          '108.7GB against a refusal threshold of 107.5GB. Growing this ' +
          'conversation further in-process would abort the engine; start a ' +
          'new conversation, restart the engine to continue this one, or ' +
          'reduce the prefix-cache memory settings.',
        type: 'prompt_too_long',
        code: 'prefill_admission_declined',
      },
    })
    const bubble = promptTooLongChatErrorContent(`API error: 413 - ${body}`)
    expect(bubble).toContain('turn-peak admission declined')
    expect(bubble).toContain('restart this model from the Server tab')
    expect(bubble).toContain('restores from the disk cache')
    expect(bubble).not.toContain('shorten this message')
    // Still an error bubble the replay filter recognises — the annotation
    // must never be replayed to the model as assistant history.
    expect(isPromptTooLongBubbleContent(bubble)).toBe(true)
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

describe('valve decline persistent bubble (ledger row 152)', () => {
  it('classifies the live valve message WITHOUT any prompt_too_long wording', () => {
    const live =
      'dots3-note-prev-JANG: prefill admission rejected chunk [0:2048) — ' +
      'active Metal working set 96.51GB plus projected transient 11.12GB ' +
      'exceeds the device working-set limit 107.50GB. A context of this ' +
      'length cannot be served on this hardware; reduce the prompt or context size.'
    const bubble = promptTooLongChatErrorContent(live)
    expect(bubble).not.toBeNull()
    expect(bubble).toContain('restart this model')
  })

  it('classifies via the machine code suffix appended by the stream detail', () => {
    const bubble = promptTooLongChatErrorContent(
      'some future wording we did not predict [prefill_admission_declined]',
    )
    expect(bubble).not.toBeNull()
    expect(bubble).toContain('restart this model')
  })

  it('still returns null for unrelated errors', () => {
    expect(promptTooLongChatErrorContent('ECONNRESET while reading')).toBeNull()
  })
})
