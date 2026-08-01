import { describe, expect, it, vi } from 'vitest'
import {
  detectedConfigFromRemoteCapabilities,
  fetchRemoteModelCapabilities,
  generationDefaultsFromRemoteCapabilities,
} from '../src/shared/remoteModelCapabilities'

const DSV4_CAPABILITIES = {
  id: 'dsv4-final94',
  family: 'deepseek_v4',
  supports_tools: true,
  tool_parser: 'dsml',
  supports_thinking: true,
  supports_instruct_mode: true,
  reasoning_parser: 'deepseek_r1',
  think_in_template: true,
  reasoning_efforts: ['low', 'high', 'max', 'invalid'],
  default_reasoning_effort: 'low',
  modalities: ['text'],
  media: {
    runtime_modalities: ['text'],
    declared_modalities: ['text'],
  },
  cache: {
    type: 'disabled',
    paged: false,
    native: {
      family: 'deepseek_v4',
      schema: 'deepseek_v4_v10_delta',
      pool_quant: { requested: true, enabled: true, observed: true },
    },
  },
  quantization: {
    codec: 'affine_quantized_matmul',
    profile: 'JANG_DSV4_NR8',
  },
  sampling_defaults: { temperature: 1, top_p: 0.95, top_k: 0 },
  effective_defaults: {
    temperature: 1,
    top_p: 0.95,
    top_k: 0,
    min_p: 0,
    max_output_tokens: 16384,
  },
  max_prompt_tokens: 1048576,
}

describe('remote model capability hydration', () => {
  it('maps the live DSV4 parser, reasoning, cache, and modality contract', () => {
    const detected = detectedConfigFromRemoteCapabilities(DSV4_CAPABILITIES)
    expect(detected).toMatchObject({
      family: 'deepseek-v4',
      toolParser: 'dsml',
      reasoningParser: 'deepseek_r1',
      supportsThinking: true,
      supportsInstructMode: true,
      supportedReasoningEfforts: ['low', 'high', 'max'],
      defaultReasoningEffort: 'low',
      thinkInTemplate: true,
      dsv4PoolQuantDefault: true,
      cacheType: 'disabled',
      usePagedCache: false,
      enableAutoToolChoice: true,
      isMultimodal: false,
      architectureHints: {
        cacheSchema: 'deepseek_v4_v10_delta',
        nativeCacheFamily: 'deepseek_v4',
      },
      quantizationLabel: 'JANG_DSV4_NR8',
    })
    expect(detected).not.toHaveProperty('maxContextLength')
    expect(detected).not.toHaveProperty('nativeMtp')
  })

  it('maps exact live sampling values without fabricating absent penalties', () => {
    expect(generationDefaultsFromRemoteCapabilities(DSV4_CAPABILITIES)).toEqual({
      temperature: 1,
      topP: 0.95,
      topK: 0,
    })
  })

  it('does not promote runtime-effective fallbacks into bundle sampling defaults', () => {
    expect(generationDefaultsFromRemoteCapabilities({
      sampling_defaults: { temperature: 1 },
      effective_defaults: {
        temperature: 0.4,
        top_p: 0.8,
        top_k: 40,
        min_p: 0.1,
        repetition_penalty: 1.2,
        max_output_tokens: 4096,
      },
    })).toEqual({ temperature: 1 })
  })

  it('maps only explicit live native MTP fields', () => {
    expect(detectedConfigFromRemoteCapabilities({
      family: 'qwen3',
      cache: { native: { cache_type: 'hybrid_kv_ssm' } },
      mtp: {
        runtime_available: true,
        effective_depth: 2,
        effective_depth_source: 'runtime_config',
        runtime_scope: 'text+vl',
        requires_deterministic_sampling: false,
      },
    })).toMatchObject({
      nativeMtp: {
        supported: true,
        depth: 2,
        depthSource: 'runtime_config',
        runtimeScope: 'text+vl',
        nativeCacheType: 'hybrid_kv_ssm',
        requiresDeterministicSampling: false,
      },
    })
  })

  it('keeps missing capability fields absent', () => {
    expect(detectedConfigFromRemoteCapabilities({ family: 'custom_family' })).toEqual({
      family: 'custom_family',
      description: 'Live runtime capabilities: custom_family',
    })
    expect(generationDefaultsFromRemoteCapabilities({ family: 'custom_family' })).toBeNull()
  })

  it('fetches model-specific capabilities with main-process credentials', async () => {
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(DSV4_CAPABILITIES), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })) as unknown as typeof fetch

    await expect(fetchRemoteModelCapabilities({
      remoteUrl: 'http://engine.test/',
      remoteApiKey: 'secret',
      remoteModel: 'dsv4 final/94',
      remoteOrganization: 'org-test',
    }, fetchImpl)).resolves.toMatchObject({ id: 'dsv4-final94' })

    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(fetchImpl).toHaveBeenCalledWith(
      'http://engine.test/v1/models/dsv4%20final%2F94/capabilities',
      expect.objectContaining({
        headers: {
          Accept: 'application/json',
          Authorization: 'Bearer secret',
          'OpenAI-Organization': 'org-test',
        },
      }),
    )
  })

  it('falls back to the single-model capability route', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 404 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(DSV4_CAPABILITIES), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })) as unknown as typeof fetch

    await expect(fetchRemoteModelCapabilities({
      remoteUrl: 'http://engine.test',
      remoteModel: 'dsv4-final94',
    }, fetchImpl)).resolves.toMatchObject({ family: 'deepseek_v4' })
    expect(fetchImpl).toHaveBeenNthCalledWith(
      2,
      'http://engine.test/v1/capabilities',
      expect.any(Object),
    )
  })

  it('rejects unqualified fallback capabilities for a different model', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 404 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...DSV4_CAPABILITIES,
        id: 'another-model',
        loaded_model: 'another-loaded-model',
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })) as unknown as typeof fetch

    await expect(fetchRemoteModelCapabilities({
      remoteUrl: 'http://engine.test',
      remoteModel: 'dsv4-final94',
    }, fetchImpl)).resolves.toBeNull()
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })

  it('accepts an unqualified fallback when loaded_model exactly matches', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(new Response('', { status: 404 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...DSV4_CAPABILITIES,
        id: 'served-alias',
        loaded_model: 'dsv4-final94',
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })) as unknown as typeof fetch

    await expect(fetchRemoteModelCapabilities({
      remoteUrl: 'http://engine.test',
      remoteModel: 'dsv4-final94',
    }, fetchImpl)).resolves.toMatchObject({ loaded_model: 'dsv4-final94' })
  })

  it('reserves part of one total timeout budget for the compatibility route', async () => {
    const fetchImpl = vi.fn((_: string | URL | Request, init?: RequestInit) => {
      if (fetchImpl.mock.calls.length === 1) {
        return new Promise<Response>((_, reject) => {
          const signal = init?.signal
          if (!signal) return reject(new Error('missing timeout signal'))
          signal.addEventListener('abort', () => reject(signal.reason), { once: true })
        })
      }
      return Promise.resolve(new Response(JSON.stringify(DSV4_CAPABILITIES), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }))
    }) as unknown as typeof fetch

    await expect(fetchRemoteModelCapabilities({
      remoteUrl: 'http://engine.test',
      remoteModel: 'dsv4-final94',
    }, fetchImpl, 200)).resolves.toMatchObject({ id: 'dsv4-final94' })
    expect(fetchImpl).toHaveBeenCalledTimes(2)
  })
})
