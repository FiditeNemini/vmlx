import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import {
  loadChatSettingsCompatibility,
  loadChatSettingsHydration,
  readPersistedSessionGenerationDefaults,
  resolveChatSettingsHydration,
} from '../src/shared/chatSettingsHydration'

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

describe('Chat Settings hydration', () => {
  it('primes only unambiguous persisted bundle defaults before the live read', () => {
    expect(readPersistedSessionGenerationDefaults({
      defaultSamplingDefaultsDeclared: true,
      defaultDoSample: true,
      defaultTemperature: 100,
      defaultTopP: 95,
      defaultTopK: 0,
      defaultMinP: 0,
      defaultRepetitionPenalty: 105,
      defaultMaxNewTokens: 8192,
    })).toEqual({
      temperature: 1,
      topP: 0.95,
      repeatPenalty: 1.05,
      maxTokens: 8192,
    })

    expect(readPersistedSessionGenerationDefaults({
      defaultSamplingDefaultsDeclared: true,
      defaultDoSample: false,
      defaultTemperature: 90,
      defaultTopP: 80,
      defaultTopK: 40,
    })).toMatchObject({
      temperature: 0,
      topP: 1,
      topK: 0,
    })
  })

  it('never invents absent persisted fields from the aggregate declaration bit', () => {
    expect(readPersistedSessionGenerationDefaults({
      defaultSamplingDefaultsDeclared: true,
      defaultDoSample: true,
      defaultTemperature: 70,
      // Historical session rows stored every absent field as zero.
      defaultTopP: 0,
      defaultTopK: 0,
      defaultMinP: 0,
      defaultRepetitionPenalty: 0,
    })).toEqual({
      temperature: 0.7,
    })
  })

  it('does not merge stale persisted fields into a successful live bundle read', () => {
    const result = resolveChatSettingsHydration({
      defaultSamplingDefaultsDeclared: true,
      defaultTemperature: 60,
      defaultTopP: 90,
    }, {
      overrides: { status: 'fulfilled', value: {} },
      generationDefaults: {
        status: 'fulfilled',
        value: { temperature: 0.7 },
      },
      detectedConfig: { status: 'fulfilled', value: null },
    })

    expect(result.defaultsState).toBe('ready')
    expect(result.modelDefaults).toEqual({ temperature: 0.7 })
  })

  it('renders an explicit live do_sample false policy without inventing other fields', () => {
    const result = resolveChatSettingsHydration({}, {
      overrides: { status: 'fulfilled', value: {} },
      generationDefaults: {
        status: 'fulfilled',
        value: {
          doSample: false,
          repeatPenalty: 1.1,
        },
      },
      detectedConfig: { status: 'fulfilled', value: null },
    })

    expect(result.defaultsState).toBe('ready')
    expect(result.modelDefaults).toEqual({
      temperature: 0,
      topP: 1,
      topK: 0,
      repeatPenalty: 1.1,
    })
  })

  it('starts every source concurrently and reduces out-of-order completions once', async () => {
    const started: string[] = []
    const overrides = deferred<{ temperature: number; topK: number; minP: number }>()
    const generationDefaults = deferred<{
      temperature: number
      topP: number
      topK: number
      minP: number
      repeatPenalty: number
    }>()
    const detectedConfig = deferred<{ family: string }>()

    let completed = false
    const pending = loadChatSettingsHydration({}, {
      overrides: () => {
        started.push('overrides')
        return overrides.promise
      },
      generationDefaults: () => {
        started.push('generationDefaults')
        return generationDefaults.promise
      },
      detectedConfig: () => {
        started.push('detectedConfig')
        return detectedConfig.promise
      },
    }).then((result) => {
      completed = true
      return result
    })

    expect(started).toEqual([
      'overrides',
      'generationDefaults',
      'detectedConfig',
    ])

    generationDefaults.resolve({
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
    })
    detectedConfig.resolve({ family: 'qwen3' })
    await Promise.resolve()
    expect(completed).toBe(false)

    overrides.resolve({ temperature: 0, topK: 0, minP: 0 })
    const result = await pending
    expect(result.defaultsState).toBe('ready')
    expect(result.overrides).toEqual({ temperature: 0, topK: 0, minP: 0 })
    expect(result.modelDefaults).toMatchObject({
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
    })
    expect(result.detectedConfig?.family).toBe('qwen3')
    expect(result.partialFailure).toBe(false)
  })

  it('loads chat compatibility metadata independently from settings readiness', async () => {
    const chat = deferred<{ modelPath: string }>()
    const messages = deferred<unknown[]>()
    const pending = loadChatSettingsCompatibility({
      chat: () => chat.promise,
      messages: () => messages.promise,
    })

    messages.resolve([{}, {}, {}])
    chat.resolve({ modelPath: '/models/qwen' })
    await expect(pending).resolves.toEqual({
      partialFailure: false,
      savedChatModelPath: '/models/qwen',
      messageCount: 3,
    })
  })

  it('keeps fulfilled sources and a persisted session fallback when another source rejects', async () => {
    const result = await loadChatSettingsHydration({
      defaultSamplingDefaultsDeclared: true,
      defaultDoSample: true,
      defaultTemperature: 60,
      defaultTopP: 92,
      defaultTopK: 32,
      defaultMinP: 4,
      defaultRepetitionPenalty: 108,
    }, {
      overrides: async () => ({ temperature: 0 }),
      generationDefaults: async () => {
        throw new Error('bundle unavailable')
      },
      detectedConfig: async () => ({ family: 'laguna' }),
    })

    expect(result.defaultsState).toBe('session-fallback')
    expect(result.partialFailure).toBe(true)
    expect(result.overrides).toEqual({ temperature: 0 })
    expect(result.modelDefaults).toMatchObject({
      temperature: 0.6,
      topP: 0.92,
      topK: 32,
      minP: 0.04,
      repeatPenalty: 1.08,
    })
    expect(result.detectedConfig?.family).toBe('laguna')
  })

  it('distinguishes truthful engine fallback from unavailable hydration', () => {
    const fulfilledWithoutBundleDefaults = resolveChatSettingsHydration({}, {
      overrides: { status: 'fulfilled', value: {} },
      generationDefaults: { status: 'fulfilled', value: null },
      detectedConfig: { status: 'fulfilled', value: null },
    })
    expect(fulfilledWithoutBundleDefaults.defaultsState).toBe('engine-fallback')
    expect(fulfilledWithoutBundleDefaults.overridesLoaded).toBe(true)

    const unavailable = resolveChatSettingsHydration({}, {
      overrides: { status: 'rejected', reason: new Error('db unavailable') },
      generationDefaults: { status: 'rejected', reason: new Error('bundle unavailable') },
      detectedConfig: { status: 'fulfilled', value: null },
    })
    expect(unavailable.defaultsState).toBe('unavailable')
    expect(unavailable.overridesLoaded).toBe(false)
    expect(unavailable.partialFailure).toBe(true)
    expect(unavailable.modelDefaults).toEqual({})
  })

  it('bounds a hung source without replacing it with a fake value', async () => {
    const never = new Promise<never>(() => {})
    const result = await loadChatSettingsHydration({}, {
      overrides: async () => ({ temperature: 0 }),
      generationDefaults: async () => ({
        temperature: 0.7,
        topP: 0.9,
      }),
      detectedConfig: () => never,
    }, 10)

    expect(result.overridesLoaded).toBe(true)
    expect(result.defaultsState).toBe('ready')
    expect(result.modelDefaults).toMatchObject({
      temperature: 0.7,
      topP: 0.9,
    })
    expect(result.detectedConfig).toBeUndefined()
    expect(result.partialFailure).toBe(true)
  })

  it('gates first paint and only renders sampler fields with a real source', () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, '../src/renderer/src/components/chat/ChatSettings.tsx'),
      'utf8',
    )
    expect(source).toContain("const inferenceReady = hydrationCurrent && overridesLoaded")
    expect(source).toContain('displayedTemperature != null')
    expect(source).toContain('displayedTopP != null')
    expect(source).toContain('displayedTopK != null')
    expect(source).toContain('displayedMinP != null')
    expect(source).toContain('displayedRepeatPenalty != null')
    expect(source).not.toContain('displayedModelDefaults.temperature ?? 0')
    expect(source).not.toContain('displayedModelDefaults.topP ?? 1')
    expect(source).not.toContain('displayedModelDefaults.repeatPenalty ?? 1.0')
    expect(source).toContain("chat.settings.defaultsEngineFallback")
  })
})
