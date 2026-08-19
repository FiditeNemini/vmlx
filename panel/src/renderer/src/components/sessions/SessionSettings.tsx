import { useState, useEffect, useRef } from 'react'
import { resolveSlowFamilyTimeoutSeconds } from '../../../../shared/slowFamilyTimeouts'
import { ArrowLeft, ChevronRight } from 'lucide-react'
import {
  SessionConfigForm,
  SessionConfig,
  DEFAULT_CONFIG,
  DSV4_MAX_CACHE_BLOCKS,
  DSV4_PAGED_CACHE_BLOCK_SIZE,
  commitActiveSettingsInput,
} from './SessionConfigForm'
import { useTranslation } from '../../i18n'
import { resolveCacheLaunchPolicy } from '../../../../shared/cacheControlPolicy'
import {
  DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV,
  isLagunaMixedSwaTurboQuantEffective,
  shouldDisableLagunaJitDefault,
} from '../../../../shared/lagunaCachePolicy'
import { buildMcpPolicyArgs } from '../../../../shared/mcpPolicy'
import { resolveEffectiveReasoningParser } from '../../../../shared/reasoningParserAliases'
import { resolveEffectiveToolParser } from '../../../../shared/toolParserAliases'
import { buildToolLaunchArgs } from '../../../../shared/toolLaunchArgs'
import { normalizeDetectedFamilyName, isZayaCcaFamily } from '../../../../shared/detectedFamilyNames'
import {
  applyBundleDsv4PoolQuantToSessionConfig,
  applyBundleGenerationDefaultsToSessionConfig,
} from '../../../../shared/sessionGenerationDefaults'
import {
  cacheTypeRequiresPaged,
  cacheSubtypeRequiresPaged,
  cacheTypeSupportsBlockDiskOnly,
  cacheSubtypeSupportsBlockDiskOnly,
} from '../../../../shared/cacheTypeCapabilities'
import { computeEffectiveJit } from '../../../../shared/jitPolicy'
import { storedKvQuantMustBeExact } from '../../../../shared/storedKvQuantPolicy'
import {
  filterAdditionalArgs,
  finiteNonNegativeNumber,
  finitePositiveInteger,
  finitePositiveNumber,
} from '../../../../shared/launchArgValues'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  config: string
  createdAt: number
  updatedAt: number
}

interface SessionSettingsProps {
  sessionId: string
  onBack: () => void
}







// The CLI preview must show what launch ACTUALLY passes. It knew only
// deepseek-v4, so it displayed --timeout 300 for six of the seven slow families
// whose launch passes 900 — a UI-truthfulness bug. Shares one table with
// sessions.ts, chat.ts and the gateway now.
function effectiveSessionTimeoutSeconds(config: Partial<SessionConfig>, family?: string): number {
  const configured = config.timeout
  if (configured != null && configured <= 0) return 86400
  return resolveSlowFamilyTimeoutSeconds(
    configured,
    normalizeDetectedFamilyName(family),
  )
}





const IMAGE_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--host',
  '--port',
  '--uds',
  '--api-key',
  '--rate-limit',
  '--timeout',
  '--inference-endpoints',
  '--wake-timeout',
  '--log-level',
  '--allowed-origins',
  '--image-mode',
  '--image-quantize',
  '--served-model-name',
  '--mflux-class',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
])

const TEXT_ADDITIONAL_ARG_BLOCKLIST = new Set([
  ...IMAGE_ADDITIONAL_ARG_BLOCKLIST,
  '--chat-template',
  '--chat-template-kwargs',
  '--continuous-batching',
  '--no-continuous-batching',
  '--enable-prefix-cache',
  '--disable-prefix-cache',
  '--use-paged-cache',
  '--no-paged-cache',
  '--paged-cache-block-size',
  '--max-cache-blocks',
  '--kv-cache-quantization',
  '--kv-cache-group-size',
  '--max-num-seqs',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--completion-batch-size',
  '--max-tokens',
  '--max-prompt-tokens',
  '--stream-interval',
  '--ssm-state-cache-size',
  '--ssm-state-cache-mb',
  '--enable-jit',
  '--no-jit',
  '--no-memory-aware-cache',
  '--prefix-cache-size',
  '--prefix-cache-max-bytes',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--enable-disk-cache',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--enable-block-disk-cache',
  '--disable-block-disk-cache',
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--smelt',
  '--smelt-experts',
  '--flash-moe',
  '--flash-moe-slot-bank',
  '--flash-moe-prefetch',
  '--flash-moe-io-split',
  '--distributed',
  '--distributed-mode',
  '--worker-nodes',
  '--cluster-secret',
  '--speculative-model',
  '--num-draft-tokens',
  '--native-mtp-depth',
  '--native-mtp-sampling-policy',
  '--disable-native-mtp',
  '--enable-pld',
  '--pld-summary-interval',
  '--is-mllm',
  '--enable-auto-tool-choice',
  '--tool-call-parser',
  '--reasoning-parser',
  '--embedding-model',
  '--default-temperature',
  '--default-top-p',
  '--default-top-k',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-enable-thinking',
])

const DSV4_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--continuous-batching',
  '--no-continuous-batching',
  '--host',
  '--port',
  '--uds',
  '--api-key',
  '--rate-limit',
  '--timeout',
  '--inference-endpoints',
  '--wake-timeout',
  '--log-level',
  '--allowed-origins',
  '--served-model-name',
  '--dsv4-enable-prefix-cache',
  '--enable-prefix-cache',
  '--disable-prefix-cache',
  '--use-paged-cache',
  '--no-paged-cache',
  '--paged-cache-block-size',
  '--max-cache-blocks',
  '--kv-cache-quantization',
  '--kv-cache-group-size',
  '--max-num-seqs',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--completion-batch-size',
  '--max-tokens',
  '--max-prompt-tokens',
  '--stream-interval',
  '--prefill-keep-alloc',
  '--no-state-machine-stops',
  '--ssm-state-cache-size',
  '--ssm-state-cache-mb',
  '--enable-jit',
  '--no-jit',
  '--no-memory-aware-cache',
  '--prefix-cache-size',
  '--prefix-cache-max-bytes',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--enable-disk-cache',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--enable-block-disk-cache',
  '--disable-block-disk-cache',
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--image-mode',
  '--image-quantize',
  '--mflux-class',
  '--smelt',
  '--smelt-experts',
  '--flash-moe',
  '--flash-moe-slot-bank',
  '--flash-moe-prefetch',
  '--flash-moe-io-split',
  '--distributed',
  '--distributed-mode',
  '--worker-nodes',
  '--cluster-secret',
  '--speculative-model',
  '--num-draft-tokens',
  '--native-mtp-depth',
  '--native-mtp-sampling-policy',
  '--disable-native-mtp',
  '--enable-pld',
  '--pld-summary-interval',
  '--is-mllm',
  '--model-family',
  '--text-only',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--omni-backend',
  '--enable-auto-tool-choice',
  '--tool-call-parser',
  '--reasoning-parser',
  '--embedding-model',
  '--default-temperature',
  '--default-top-p',
  '--default-top-k',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-enable-thinking',
  '--chat-template',
  '--chat-template-kwargs',
])


/**
 * Build a preview of the CLI command from config.
 * This MUST mirror the logic in sessions.ts buildArgs() exactly.
 * When editing buildArgs(), update this function too (and vice versa).
 */
function buildCommandPreview(
  modelPath: string,
  config: SessionConfig,
  detected?: { toolParser?: string; reasoningParser?: string; supportsThinking?: boolean; isMultimodal?: boolean; forceTextOnly?: boolean; isTurboQuant?: boolean; usePagedCache?: boolean; enableAutoToolChoice?: boolean; cacheType?: string; cacheSubtype?: string; family?: string; dsv4PoolQuantDefault?: boolean; architectureHints?: Record<string, string | number | boolean>; nativeMtp?: { supported?: boolean; depth?: number; depthSource?: string; blockedReason?: string } } | null
): string {
  const parts = ['vmlx-engine serve', modelPath]
  const requestedDistributed = !!(config as any).distributedEnabled
  const requestedFlashMoe = !!(config as any).flashMoe
  const detectedFamily = normalizeDetectedFamilyName(detected?.family)
  const dsv4Active = detectedFamily === 'deepseek-v4'
  const m3Active = detectedFamily === 'minimax_m3'
  const openPanguExactTypedCache = detectedFamily === 'openpangu_v2'
  const effectiveSmelt = !!(config as any).smelt && !dsv4Active
  // User explicitly toggled multimodal OFF (Force Off) — must beat detected VL.
  // Mirror buildArgs (sessions.ts): m3Active stands in for m3VlRoute since the
  // registry sets m3VlRoute for every minimax_m3 bundle while keeping isMultimodal
  // true, so M3 must emit NEITHER --is-mllm NOR --text-only.
  const userForceTextOnly = config.isMultimodal === false
  // Omni media is injected into the Nemotron-H text decoder by the engine;
  // attachments stay enabled without advertising the generic mlx_vlm route.
  const omniBackendActive = detectedFamily === 'nemotron-h' &&
    detected?.isMultimodal === true &&
    !userForceTextOnly &&
    !detected?.forceTextOnly
  const isVLM = dsv4Active || effectiveSmelt || detected?.forceTextOnly || userForceTextOnly || m3Active || omniBackendActive ? false
    : detected?.isMultimodal ? true
      : config.isMultimodal === true ? true
        : false
  const zayaCcaActive = isZayaCcaFamily(detectedFamily)
  const turboQuantActive = !!detected?.isTurboQuant
  const hybridCacheActive = cacheTypeRequiresPaged(detected?.cacheType)
  const architectureBlockDiskOnlySupported =
    (cacheTypeSupportsBlockDiskOnly(detected?.cacheType) ||
      cacheSubtypeSupportsBlockDiskOnly(detected?.cacheSubtype) ||
      m3Active ||
      dsv4Active) &&
    !zayaCcaActive &&
    !openPanguExactTypedCache
  const subtypePagedCacheActive = cacheSubtypeRequiresPaged(detected?.cacheSubtype)
  const effectiveDistributed = requestedDistributed && !dsv4Active
  const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active
  if (dsv4Active && typeof detected?.dsv4PoolQuantDefault === 'boolean') {
    parts[0] = `DSV4_POOL_QUANT=${detected.dsv4PoolQuantDefault ? '1' : '0'} ${parts[0]}`
  }
  const lagunaMixedSwaTurboQuantActive = isLagunaMixedSwaTurboQuantEffective({
    detected,
    kvCacheQuantization: config.kvCacheQuantization,
    explicitKvCacheQuantizationApplied:
      config.continuousBatching !== false &&
      config.enablePrefixCache !== false &&
      !!config.kvCacheQuantization &&
      config.kvCacheQuantization !== 'auto',
  })
  const effectiveEnableJit = computeEffectiveJit({
      enableJitRequested: !!config.enableJit,
      isMultimodal: isVLM,
      flashMoeActive: effectiveFlashMoe,
      distributedActive: effectiveDistributed,
      dsv4Active,
      m3Active,
      zayaCcaActive,
      turboQuantActive,
      lagunaMixedSwaTurboQuantActive,
      hybridCacheActive,
    })
  if (shouldDisableLagunaJitDefault({
    detected,
    kvCacheQuantization: config.kvCacheQuantization,
    explicitKvCacheQuantizationApplied:
      config.continuousBatching !== false &&
      config.enablePrefixCache !== false &&
      !!config.kvCacheQuantization &&
      config.kvCacheQuantization !== 'auto',
    enableJitRequested: !!config.enableJit,
  })) {
    parts[0] = `${DISABLE_JANG_AFFINE_JIT_DEFAULT_ENV}=1 vmlx-engine serve`
  }

  // Server settings
  parts.push('--host', config.host)
  parts.push('--port', config.port.toString())
  parts.push('--timeout', effectiveSessionTimeoutSeconds(config, detectedFamily).toString())

  if (config.apiKey) parts.push('# VLLM_API_KEY=*** (env var)')
  const rateLimit = finitePositiveInteger(config.rateLimit)
  if (rateLimit != null) parts.push('--rate-limit', rateLimit.toString())

  // Concurrent processing
  const effectiveMaxNumSeqs = dsv4Active ? 1 : finitePositiveInteger(config.maxNumSeqs)
  if (effectiveMaxNumSeqs && effectiveMaxNumSeqs > 0) parts.push('--max-num-seqs', effectiveMaxNumSeqs.toString())
  const prefillBatchSize = finitePositiveInteger(config.prefillBatchSize)
  if (!dsv4Active && prefillBatchSize != null) parts.push('--prefill-batch-size', prefillBatchSize.toString())
  const prefillStepSize = finitePositiveInteger(config.prefillStepSize)
  if (prefillStepSize != null) parts.push('--prefill-step-size', prefillStepSize.toString())
  const completionBatchSize = finitePositiveInteger(config.completionBatchSize)
  if (!dsv4Active && completionBatchSize != null) parts.push('--completion-batch-size', completionBatchSize.toString())

  if (isVLM) parts.push('--is-mllm')
  else if (!dsv4Active && !effectiveSmelt && !m3Active && !omniBackendActive && detected?.isMultimodal && (userForceTextOnly || detected?.forceTextOnly)) {
    // Model autodetects as VL but must run TEXT-ONLY (user Force-Off, or a family
    // whose VL runtime path isn't wired). Mirrors buildArgs: --text-only forces
    // is_mllm_model->False so the engine doesn't re-autodetect VL from config.json.
    parts.push('--text-only')
  }
  const dflash2Speculative = /dflash2/i.test(config.speculativeModel || '')
  const cacheStackActive = dsv4Active
    ? true
    : dflash2Speculative
      ? false
      : config.continuousBatching !== false
  if (cacheStackActive) {
    parts.push('--continuous-batching')
  } else {
    parts.push('--no-continuous-batching')
  }

    // Parser resolution: User explicit choice -> Detected config -> Fallback
  // (mirrors buildArgs: user choice wins over detection)
  // "" = user chose "None" → engine needs the literal "none" flag to opt out
  // (absence auto-configures from the registry). Mirrors buildArgs in sessions.ts.
  const effectiveToolParser = resolveEffectiveToolParser({
    configuredParser: config.toolCallParser,
    detectedParser: detected?.toolParser,
  })
  const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
  const effectiveReasoningParser = resolveEffectiveReasoningParser({
    configuredParser: config.reasoningParser,
    detectedParser: detected?.reasoningParser,
    supportsThinking: detected?.supportsThinking,
  })

  // Prefix cache (mirrors buildArgs): explicit user opt-out stays off even
  // when tools are configured. Tool sessions benefit from cache but do not
  // silently own the cache toggle.
  const zayaTypedCacheRequiresPaged = zayaCcaActive
  const architectureRequiresPagedCache =
    zayaTypedCacheRequiresPaged ||
    ((cacheTypeRequiresPaged(detected?.cacheType) || subtypePagedCacheActive) && detected?.usePagedCache === true)
  const cacheLaunchPolicy = resolveCacheLaunchPolicy({
    continuousBatching: cacheStackActive,
    enablePrefixCache: config.enablePrefixCache !== false,
    usePagedCache: dsv4Active
      ? config.usePagedCache === true
      : config.usePagedCache ?? detected?.usePagedCache ?? false,
    enableDiskCache: !!config.enableDiskCache,
    enableBlockDiskCache: !!config.enableBlockDiskCache,
    architectureRequiresPagedCache,
    architectureSupportsBlockDiskOnly: architectureBlockDiskOnlySupported,
  })
  const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff
  const usePagedCache = cacheLaunchPolicy.effectiveUsePagedCache
  const blockDiskOnly = cacheLaunchPolicy.enableBlockDiskCache && !usePagedCache

  if (prefixCacheOff) {
    parts.push('--disable-prefix-cache')
  } else {
    if (!dsv4Active && config.noMemoryAwareCache) {
      parts.push('--no-memory-aware-cache')
      const prefixCacheSize = finitePositiveInteger(config.prefixCacheSize)
      if (prefixCacheSize != null) parts.push('--prefix-cache-size', prefixCacheSize.toString())
      const prefixCacheMaxBytes = finitePositiveInteger(config.prefixCacheMaxBytes)
      if (prefixCacheMaxBytes != null) parts.push('--prefix-cache-max-bytes', prefixCacheMaxBytes.toString())
    } else {
      // H1 parity: --cache-memory-mb/--cache-memory-percent set the paged L1 RAM
      // byte ceiling and DO reach the engine under paged (sessions.ts emits them
      // unconditionally). The preview must show them in both modes or it lies
      // about the launch. Only --cache-ttl-minutes is genuinely inert under
      // paged. SSD-only mode omits both controls because no RAM payload tier is
      // retained.
      const cacheMemoryMb = finitePositiveInteger(config.cacheMemoryMb)
      if (!blockDiskOnly && cacheMemoryMb != null) parts.push('--cache-memory-mb', cacheMemoryMb.toString())
      const cacheMemoryPercent = finitePositiveNumber(config.cacheMemoryPercent)
      if (!blockDiskOnly && cacheMemoryPercent != null) parts.push('--cache-memory-percent', (cacheMemoryPercent / 100).toString())
      const cacheTtlMinutes = finitePositiveNumber(config.cacheTtlMinutes)
      if (cacheTtlMinutes != null && !usePagedCache && !blockDiskOnly) parts.push('--cache-ttl-minutes', cacheTtlMinutes.toString())
    }
  }

  // Paged cache — requires prefix cache ON (works for both LLM and VLM)
  if (!prefixCacheOff && usePagedCache) {
    parts.push('--use-paged-cache')
  } else if (!prefixCacheOff && !usePagedCache) {
    // Explicit user Off remains real even when block SSD L2 stays enabled.
    parts.push('--no-paged-cache')
  }
  if (!prefixCacheOff && (usePagedCache || cacheLaunchPolicy.enableBlockDiskCache)) {
    const effectivePagedCacheBlockSize = dsv4Active
      ? DSV4_PAGED_CACHE_BLOCK_SIZE
      : config.pagedCacheBlockSize
    const pagedCacheBlockSize = finitePositiveInteger(effectivePagedCacheBlockSize)
    if (pagedCacheBlockSize != null) parts.push('--paged-cache-block-size', pagedCacheBlockSize.toString())
    const maxCacheBlocks = finitePositiveInteger(config.maxCacheBlocks)
    if (maxCacheBlocks != null) parts.push('--max-cache-blocks', maxCacheBlocks.toString())
  }

  // KV cache quantization — requires prefix cache ON (works for both LLM and VLM)
  // Hybrid/Mamba models allowed — Python scheduler only quantizes KVCache layers
  // Mixed sliding/full attention bundles must not receive a lossy stored
  // codec: it changes their answers on a cache HIT. The main process rewrites
  // a saved q8/q4 to 'auto' for these bundles; mirror that here so the command
  // preview shows what will actually be launched.
  const storedKvMustBeExact = storedKvQuantMustBeExact({
    cacheType: detected?.cacheType,
    cacheSubtype: detected?.cacheSubtype,
    architectureHints: detected?.architectureHints,
  })
  const lossyStoredCodecSuppressed =
    storedKvMustBeExact &&
    !!config.kvCacheQuantization &&
    config.kvCacheQuantization !== 'auto' &&
    config.kvCacheQuantization !== 'none'
  if (!prefixCacheOff && !dsv4Active && !m3Active && !openPanguExactTypedCache && !lossyStoredCodecSuppressed && config.kvCacheQuantization && config.kvCacheQuantization !== 'auto') {
    parts.push('--kv-cache-quantization', config.kvCacheQuantization)
    const kvCacheGroupSize = finitePositiveInteger(config.kvCacheGroupSize)
    if (config.kvCacheQuantization !== 'none' && kvCacheGroupSize != null && kvCacheGroupSize !== 64) {
      parts.push('--kv-cache-group-size', kvCacheGroupSize.toString())
    }
  }

  // Disk cache (L2 persistent cache) — mirrors sessions.ts buildArgs().
  if (cacheLaunchPolicy.enableLegacyDiskCache) {
    parts.push('--enable-disk-cache')
    if (config.diskCacheDir) parts.push('--disk-cache-dir', config.diskCacheDir)
    const diskCacheMaxGb = finiteNonNegativeNumber(config.diskCacheMaxGb)
    if (diskCacheMaxGb != null) parts.push('--disk-cache-max-gb', diskCacheMaxGb.toString())
  }

  // Block-level disk cache (L2 for paged cache blocks) — mirrors sessions.ts buildArgs().
  if (cacheLaunchPolicy.enableBlockDiskCache) {
    parts.push('--enable-block-disk-cache')
    if (config.blockDiskCacheDir) parts.push('--block-disk-cache-dir', config.blockDiskCacheDir)
    const blockDiskCacheMaxGb = finiteNonNegativeNumber(config.blockDiskCacheMaxGb)
    if (blockDiskCacheMaxGb != null) parts.push('--block-disk-cache-max-gb', blockDiskCacheMaxGb.toString())
  } else if (cacheLaunchPolicy.effectiveUsePagedCache) {
    // The engine defaults paged-compatible L2 on. Emit the explicit negative
    // flag so a user-disabled UI toggle remains a real opt-out.
    parts.push('--disable-block-disk-cache')
  }

  // Performance
  const streamInterval = finitePositiveInteger(config.streamInterval)
  if (streamInterval != null) parts.push('--stream-interval', streamInterval.toString())
  const maxTokens = finitePositiveInteger(config.maxTokens)
  if (maxTokens != null) {
    parts.push('--max-tokens', maxTokens.toString())
  }
  const maxContextLength = finitePositiveInteger(config.maxContextLength)
  if (maxContextLength != null) {
    parts.push('--max-prompt-tokens', maxContextLength.toString())
  }
  // Tool integration — mirrors buildArgs in sessions.ts
  parts.push(...buildToolLaunchArgs({
    toolParser: effectiveToolParser,
    enableAutoToolChoice: effectiveAutoTool,
  }))
  if (effectiveReasoningParser) parts.push('--reasoning-parser', effectiveReasoningParser)
  if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)
  parts.push(...buildMcpPolicyArgs(config))

  // Smelt mode (partial expert loading)
  if (effectiveSmelt) {
    parts.push('--smelt')
    const pct = finitePositiveInteger((config as any).smeltExperts) ?? 50
    if (pct !== 50) {
      parts.push('--smelt-experts', pct.toString())
    }
  }

  // Flash MoE (SSD expert streaming)
  if (effectiveFlashMoe) {
    parts.push('--flash-moe')
    const slotBank = finitePositiveInteger((config as any).flashMoeSlotBank)
    if (slotBank != null) {
      parts.push('--flash-moe-slot-bank', slotBank.toString())
    }
    const prefetch = (config as any).flashMoePrefetch
    if (prefetch && prefetch !== 'none') {
      parts.push('--flash-moe-prefetch', prefetch)
    }
    const ioSplit = finitePositiveInteger((config as any).flashMoeIoSplit)
    if (ioSplit != null) {
      parts.push('--flash-moe-io-split', ioSplit.toString())
    }
  }

  // Distributed compute
  if (effectiveDistributed) {
    parts.push('--distributed')
    const mode = (config as any).distributedMode || 'pipeline'
    if (mode !== 'pipeline') {
      parts.push('--distributed-mode', mode)
    }
  }

  // Served model name
  if (config.servedModelName) parts.push('--served-model-name', config.servedModelName)

  // Custom chat template
  if ((config as any).chatTemplate) parts.push('--chat-template', '"..."')

  // Speculative decoding
  const loopedNanbeige = detected?.family === 'nanbeige' || detected?.architectureHints?.cacheSchema === 'looped_kv_v1'
  const compatibleExternalSpeculative = !!config.speculativeModel && (
    dflash2Speculative
      ? !dsv4Active && isVLM
      : !dsv4Active && !isVLM && !cacheStackActive && !loopedNanbeige
  )
  if (compatibleExternalSpeculative) {
    parts.push('--speculative-model', config.speculativeModel)
    const numDraftTokens = finitePositiveInteger(config.numDraftTokens)
    if (numDraftTokens != null && numDraftTokens !== 3) {
      parts.push('--num-draft-tokens', numDraftTokens.toString())
    }
  }

  // Native in-model MTP mirrors sessions.ts. Auto pins omitted request
  // sampling to greedy defaults so the preserved MTP head actually runs.
  const nativeMtp = detected?.nativeMtp
  if (!dsv4Active && nativeMtp?.supported) {
    const mode = (config as any).nativeMtpMode || 'auto'
    if (mode === 'off') {
      parts.push('--disable-native-mtp')
    } else {
      const configuredDepth = (config as any).nativeMtpDepthOverride === true
        ? (config as any).nativeMtpDepth
        : nativeMtp.depth
      const depth = Math.max(1, Math.min(3, finitePositiveInteger(configuredDepth) || finitePositiveInteger(nativeMtp.depth) || 3))
      parts.push('--native-mtp-depth', depth.toString())
      parts.push('--native-mtp-sampling-policy', 'deterministic-defaults')
    }
  }

  // Generation defaults are resolved inside vmlx_engine.server from
  // jang_config/generation_config. The panel preview must not synthesize
  // --default-* flags that would override the engine's bundle lookup.

  // Embedding model
  if (config.embeddingModel) parts.push('--embedding-model', config.embeddingModel)

  // Thinking defaults are resolved by the engine and explicit chat/API requests.

  if (effectiveEnableJit) parts.push('--enable-jit')

  if (omniBackendActive && (config as any).omniBackend && (config as any).omniBackend !== 'stage1') {
    parts.push('--omni-backend', (config as any).omniBackend)
  }

  if (config.logLevel && config.logLevel !== 'INFO') {
    parts.push('--log-level', config.logLevel)
  }
  if (config.corsOrigins && config.corsOrigins !== '*') {
    parts.push('--allowed-origins', config.corsOrigins)
  }

  if (config.additionalArgs && config.additionalArgs.trim()) {
    const filtered = filterAdditionalArgs(
      config.additionalArgs,
      dsv4Active ? DSV4_ADDITIONAL_ARG_BLOCKLIST : TEXT_ADDITIONAL_ARG_BLOCKLIST,
    )
    if (filtered.length) parts.push(filtered.join(' '))
  }

  return parts.join(' \\\n  ')
}

export function SessionSettings({ sessionId, onBack }: SessionSettingsProps) {
  const { t } = useTranslation()
  const [session, setSession] = useState<Session | null>(null)
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [detectedConfig, setDetectedConfig] = useState<{ toolParser?: string; reasoningParser?: string; supportsThinking?: boolean; cacheType?: string; cacheSubtype?: string; isMultimodal?: boolean; forceTextOnly?: boolean; isTurboQuant?: boolean; usePagedCache?: boolean; enableAutoToolChoice?: boolean; family?: string; dsv4PoolQuantDefault?: boolean; architectureHints?: Record<string, string | number | boolean>; maxContextLength?: number; nativeMtp?: { supported?: boolean; depth?: number; depthSource?: string; blockedReason?: string } } | null>(null)
  const sessionIdRef = useRef(sessionId)
  const resetRequestRef = useRef(0)
  sessionIdRef.current = sessionId

  useEffect(() => {
    let active = true
    resetRequestRef.current += 1
    setDetectedConfig(null)
    const load = async () => {
      const s = await window.api.sessions.get(sessionId)
      if (s && active) {
        // Parse stored config JSON, merge with defaults
        try {
          const stored = JSON.parse(s.config)
          const base = { ...DEFAULT_CONFIG, ...stored }
          // Do not expose DEFAULT_CONFIG while the bundle lookup is pending,
          // and merge only bundle-owned default* metadata into any edit made
          // after the persisted config becomes visible.
          setConfig(base)
          setDirty(false)
          setSession(s)
          const generationDefaults = await window.api.models.getGenerationDefaults(s.modelPath).catch(() => null)
          if (active) {
            setConfig(current => applyBundleGenerationDefaultsToSessionConfig(current, generationDefaults))
          }
        } catch {
          if (active) {
            setConfig(DEFAULT_CONFIG)
            setMessage({ type: 'error', text: t('sessions.settings.configCorrupt') })
            setDirty(true)
          }
        }
        // Load auto-detected config for preview resolution
        try {
          const det = await window.api.models.detectConfig(s.modelPath)
          if (active) {
            setConfig(current => applyBundleDsv4PoolQuantToSessionConfig(current, det))
            setDetectedConfig(det && det.family !== 'unknown' ? det : null)
          }
        } catch (_) {
          if (active) setDetectedConfig(null)
        }
      }
    }
    load()
    return () => { active = false }
  }, [sessionId])

  // Listen for session status changes
  useEffect(() => {
    const unsubStopped = window.api.sessions.onStopped((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'stopped', pid: undefined } : prev)
        setRestarting(false)
      }
    })
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'running' } : prev)
        setRestarting(false)
        setMessage({ type: 'success', text: t('sessions.settings.restartedToast') })
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'error' } : prev)
        setRestarting(false)
        setMessage({ type: 'error', text: t('sessions.settings.restartFailed', { error: data.error }) })
      }
    })
    return () => {
      unsubStopped()
      unsubReady()
      unsubError()
    }
  }, [sessionId])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setDirty(true)
    setMessage(null)
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const result = await window.api.sessions.update(sessionId, config)
      if (result.success) {
        setDirty(false)
        setMessage({
          type: 'success',
          text: result.restartRequired
            ? t('sessions.settings.savedRestartRequired', { keys: result.changedKeys?.join(', ') ?? '' })
            : t('sessions.settings.savedToast')
        })
        // Refresh session data
        const s = await window.api.sessions.get(sessionId)
        if (s) setSession(s)
      } else {
        setMessage({ type: 'error', text: result.error || t('sessions.settings.saveFailed') })
      }
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  const handleSaveAndRestart = async () => {
    setSaving(true)
    setMessage(null)
    try {
      // Save first
      const saveResult = await window.api.sessions.update(sessionId, config)
      if (!saveResult.success) {
        setMessage({ type: 'error', text: saveResult.error || t('sessions.settings.saveFailed') })
        setSaving(false)
        return
      }
      setDirty(false)

      // Stop and wait for the process to actually exit
      setRestarting(true)
      setMessage({ type: 'success', text: t('sessions.settings.stopping') })
      const stopResult = await window.api.sessions.stop(sessionId)
      if (!stopResult.success) {
        setMessage({ type: 'error', text: t('sessions.settings.stopFailed', { error: stopResult.error ?? '' }) })
        setRestarting(false)
        setSaving(false)
        return
      }

      // sessions.stop resolves only after stopSession has completed, so the
      // new process can start immediately without an arbitrary renderer delay.
      setMessage({ type: 'success', text: t('sessions.settings.starting') })
      const startResult = await window.api.sessions.start(sessionId)
      if (!startResult.success) {
        setMessage({ type: 'error', text: t('sessions.settings.startFailed', { error: startResult.error ?? '' }) })
        setRestarting(false)
      }
      // Success/failure will be handled by event listeners above

      // Refresh session data
      const s = await window.api.sessions.get(sessionId)
      if (s) setSession(s)
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
      setRestarting(false)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    const resetSessionId = sessionId
    const resetRequest = ++resetRequestRef.current
    const resetStillCurrent = () => (
      sessionIdRef.current === resetSessionId
      && resetRequestRef.current === resetRequest
    )
    const base = { ...DEFAULT_CONFIG, host: config.host, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (session?.modelPath) {
      try {
        const detected = await window.api.models.detectConfig(session.modelPath)
        if (!resetStillCurrent()) return
        if (detected && detected.family !== 'unknown') {
          // Reset restores model-derived Auto, not a sticky explicit On/Off.
          base.enableAutoToolChoice = undefined
          if (detected.family === 'deepseek-v4') {
            base.dsv4PrefixCache = true
            base.dsv4PoolQuant = typeof detected.dsv4PoolQuantDefault === 'boolean'
              ? detected.dsv4PoolQuantDefault
              : undefined
            base.enablePrefixCache = true
            base.usePagedCache = true
            base.enableDiskCache = false
            base.enableBlockDiskCache = true
            base.kvCacheQuantization = 'auto'
            base.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
            base.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS
          } else if (detected.family === 'openpangu_v2') {
            base.enablePrefixCache = true
            base.usePagedCache = false
            base.enableDiskCache = true
            base.enableBlockDiskCache = false
            base.noMemoryAwareCache = false
            base.kvCacheQuantization = 'none'
          } else {
            base.usePagedCache = detected.usePagedCache === true
            base.enableDiskCache = false
            // Generic exact/partial-prefix L2 is independent of paged RAM.
            // The launch policy still suppresses this for model families with
            // explicit typed/native cache exceptions.
            base.enableBlockDiskCache = true
          }
          // VLM models: set isMultimodal flag unless this model has a
          // runtime forceTextOnly policy. Set both sides explicitly: leaving
          // this undefined lets updateSessionConfig merge an old Force On
          // value back into the reset configuration even though the preview
          // and launch detector are text-only.
          base.isMultimodal = detected.forceTextOnly === true
            ? false
            : detected.isMultimodal === true
        }
      } catch (_) {
        if (!resetStillCurrent()) return
      }
      // Bundle generation defaults are independent of family/cache detection.
      // Keep hydrating them even if detectConfig cannot classify the model.
      const generationDefaults = await window.api.models.getGenerationDefaults(session.modelPath).catch(() => null)
      if (!resetStillCurrent()) return
      Object.assign(base, applyBundleGenerationDefaultsToSessionConfig(base, generationDefaults))
    }
    if (!resetStillCurrent()) return
    setConfig(base)
    setDirty(true)
    setMessage(null)
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">{t('sessions.settings.loading')}</p>
      </div>
    )
  }

  const shortName = session.modelName || session.modelPath.split('/').pop() || session.modelPath
  const isRunning = session.status === 'running' || session.status === 'loading'

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button onClick={onBack} className="text-muted-foreground hover:text-foreground flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" /> {t('common.back')}
          </button>
          <h1 className="text-2xl font-bold">{t('sessions.settings.title')}</h1>
        </div>

        {/* Session Info */}
        <div className="mb-4 p-3 bg-card border border-border rounded">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-muted-foreground">{t('sessions.settings.modelLabel')}</span>
              <p className="font-medium text-sm truncate" title={session.modelPath}>{shortName}</p>
              <p className="text-xs text-muted-foreground truncate">{session.modelPath}</p>
            </div>
            <div className="text-right text-sm">
              <span className={`inline-flex items-center gap-1.5 ${isRunning ? 'text-primary' : 'text-muted-foreground'}`}>
                <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-primary' : 'bg-muted-foreground'}`} />
                {restarting ? t('status.restarting') : session.status}
              </span>
              <p className="text-xs text-muted-foreground">{config.host}:{config.port}</p>
            </div>
          </div>
        </div>

        {/* Running warning */}
        {isRunning && !restarting && (
          <div className="mb-4 p-3 bg-warning/10 border border-warning/30 rounded-lg text-sm text-warning">
            {t('sessions.settings.runningWarning')}
          </div>
        )}

        {/* Status message */}
        {message && (
          <div className={`mb-4 p-3 rounded-lg text-sm ${message.type === 'success'
            ? 'bg-primary/10 border border-primary/30 text-primary'
            : 'bg-destructive/10 border border-destructive/30 text-destructive'
            }`}>
            {message.text}
          </div>
        )}

        {/* Config Form */}
        <SessionConfigForm config={config} onChange={handleChange} onReset={handleReset} detectedCacheType={detectedConfig?.cacheType} detectedUsePagedCache={detectedConfig?.usePagedCache} detectedCacheSubtype={detectedConfig?.cacheSubtype} detectedFamily={detectedConfig?.family} detectedArchitectureHints={detectedConfig?.architectureHints} detectedToolParser={detectedConfig?.toolParser} detectedReasoningParser={detectedConfig?.reasoningParser} detectedEnableAutoToolChoice={detectedConfig?.enableAutoToolChoice} detectedIsTurboQuant={detectedConfig?.isTurboQuant} detectedIsMultimodal={detectedConfig?.isMultimodal} detectedForceTextOnly={detectedConfig?.forceTextOnly} detectedMaxContext={detectedConfig?.maxContextLength} detectedNativeMtp={(detectedConfig as any)?.nativeMtp} modelType={(() => { try { return JSON.parse(session.config || '{}').modelType } catch { return undefined } })()} sessionId={sessionId} modelIdentity={`${session.modelName || ''} ${session.modelPath}`} />

        {/* Command Preview */}
        <div className="mt-4">
          <button
            onClick={() => setShowPreview(!showPreview)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            <ChevronRight className={`h-4 w-4 transition-transform ${showPreview ? 'rotate-90' : ''}`} /> {t('sessions.settings.cliCommandPreview')}
          </button>
          {showPreview && (
            <pre className="mt-2 p-3 bg-background/80 text-primary text-xs font-mono rounded-lg overflow-x-auto whitespace-pre-wrap">
              {buildCommandPreview(session.modelPath, config, detectedConfig)}
            </pre>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3 mt-6 pb-6">
          <button onClick={onBack} className="px-4 py-2 border border-border rounded hover:bg-accent">
            {t('common.back')}
          </button>
          <button
            onPointerDown={commitActiveSettingsInput}
            onClick={handleSave}
            disabled={!dirty || saving || restarting}
            className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-40"
          >
            {saving && !restarting
              ? t('common.saving')
              : t(isRunning ? 'sessions.settings.saveForNextRestart' : 'sessions.settings.saveSettings')}
          </button>
          {isRunning && (
            <button
              onPointerDown={commitActiveSettingsInput}
              onClick={handleSaveAndRestart}
              disabled={saving || restarting}
              className="px-6 py-2 bg-success text-success-foreground rounded hover:bg-success/90 font-medium disabled:opacity-40"
            >
              {restarting ? t('sessions.settings.restarting') : t('sessions.settings.saveAndRestart')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
