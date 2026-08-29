export type BenchmarkProfileId = 'peak' | 'representative'

export type BenchmarkFamilyId =
  | 'qwen38-flash-next'
  | 'qwen38-27b'
  | 'dsv4-flash'
  | 'glm53-flash'
  | 'generic'

export type BenchmarkScenarioKind = 'decode' | 'prefill' | 'mixed'

export interface BenchmarkScenario {
  id: string
  label: string
  kind: BenchmarkScenarioKind
  maxTokens: number
  temperature: number
  repetitions: number
  disableThinking: boolean
  timeoutMs: number
  prompt?: string
  systemPrompt?: string
  targetPromptTokens?: number
}

export interface BenchmarkProfile {
  id: BenchmarkProfileId
  familyId: BenchmarkFamilyId
  familyLabel: string
  label: string
  disclosure: string
  scenarios: BenchmarkScenario[]
}

const PREFILL_FILLER =
  'The harbor master logged the arrival of the cargo vessel at dawn, noting the ' +
  'manifest listed turbines, ceramic insulators, and forty crates of instrument ' +
  'wire. Customs inspection began at the east pier under a light rain. '

const PEAK_CODE_PROMPT =
  'Complete this Python function. Return only code.\n\n' +
  'def merge_intervals(intervals):\n' +
  '    """Merge overlapping [start, end] intervals and return the merged list, sorted."""\n'

const FAMILY_LABELS: Record<BenchmarkFamilyId, string> = {
  'qwen38-flash-next': 'Qwen3.8 Flash-Next',
  'qwen38-27b': 'Qwen3.8 27B',
  'dsv4-flash': 'DeepSeek V4 Flash',
  'glm53-flash': 'GLM-5.3 Flash',
  generic: 'Detected model',
}

export function detectBenchmarkFamily(identity: string): BenchmarkFamilyId {
  const normalized = identity.toLowerCase().replace(/[^a-z0-9]+/g, '-')
  if (
    /qwen3-?8.*flash-?next/.test(normalized) ||
    normalized.includes('qwen4-exp')
  ) {
    return 'qwen38-flash-next'
  }
  if (/qwen3-?8.*27b/.test(normalized)) return 'qwen38-27b'
  if (
    /deepseek.*v4.*flash/.test(normalized) ||
    /dsv4.*flash/.test(normalized)
  ) {
    return 'dsv4-flash'
  }
  if (/glm-?5-?3.*flash/.test(normalized) || normalized.includes('glm5-next')) {
    return 'glm53-flash'
  }
  return 'generic'
}

function peakPrefillScenarios(
  familyId: BenchmarkFamilyId,
): BenchmarkScenario[] {
  // Flash-Next and 27B use the best retained arms from the 2026-08-29 speed
  // audit. DSV4/GLM have no source-matched tuning receipt yet, so their peak
  // profile runs a bounded shape sweep and lets the result card name the
  // highest observed rate instead of pretending one hard-coded shape won.
  const targets =
    familyId === 'qwen38-flash-next'
      ? [12_000]
      : familyId === 'qwen38-27b'
        ? [6_000]
        : familyId === 'glm53-flash'
          ? [2_000, 6_000]
          : [2_000, 6_000, 12_000]
  const repetitions =
    familyId === 'qwen38-flash-next' || familyId === 'qwen38-27b' ? 3 : 1

  return targets.map((target) => ({
    id: `peak-prefill-${target}`,
    label: `Peak clean prefill (~${target / 1000}K)`,
    kind: 'prefill',
    targetPromptTokens: target,
    maxTokens: 8,
    temperature: 0,
    repetitions,
    disableThinking: true,
    timeoutMs: 30 * 60 * 1000,
  }))
}

function peakScenarios(familyId: BenchmarkFamilyId): BenchmarkScenario[] {
  return [
    ...peakPrefillScenarios(familyId),
    {
      id: 'peak-code-burst',
      label: 'Peak code burst (128)',
      kind: 'decode',
      prompt: PEAK_CODE_PROMPT,
      maxTokens: 128,
      temperature: 0,
      repetitions: 3,
      disableThinking: true,
      timeoutMs: 10 * 60 * 1000,
    },
  ]
}

const REPRESENTATIVE_SCENARIOS: BenchmarkScenario[] = [
  {
    id: 'representative-short',
    label: 'Short generation',
    kind: 'mixed',
    prompt: 'Write a haiku about silicon.',
    maxTokens: 64,
    temperature: 0.7,
    repetitions: 1,
    disableThinking: false,
    timeoutMs: 5 * 60 * 1000,
  },
  {
    id: 'representative-medium',
    label: 'Medium generation',
    kind: 'mixed',
    prompt:
      'Explain how a transformer neural network processes a sentence, step by step.',
    maxTokens: 256,
    temperature: 0.7,
    repetitions: 1,
    disableThinking: false,
    timeoutMs: 10 * 60 * 1000,
  },
  {
    id: 'representative-long',
    label: 'Long generation',
    kind: 'mixed',
    prompt:
      'Write a detailed technical blog post about the advantages and challenges of running large language models on Apple Silicon. Cover memory bandwidth, unified memory architecture, and the role of quantization.',
    maxTokens: 512,
    temperature: 0.7,
    repetitions: 1,
    disableThinking: false,
    timeoutMs: 15 * 60 * 1000,
  },
  {
    id: 'representative-prefill',
    label: 'Long prompt (prefill)',
    kind: 'prefill',
    targetPromptTokens: 2_000,
    maxTokens: 8,
    temperature: 0,
    repetitions: 1,
    disableThinking: false,
    timeoutMs: 15 * 60 * 1000,
  },
]

export function getBenchmarkProfile(
  profileId: BenchmarkProfileId,
  identity: string,
): BenchmarkProfile {
  const familyId = detectBenchmarkFamily(identity)
  const familyLabel = FAMILY_LABELS[familyId]
  if (profileId === 'representative') {
    return {
      id: profileId,
      familyId,
      familyLabel,
      label: 'Representative',
      disclosure:
        'Mixed prompts with normal sampling. Use this for a broader workload view, not a peak headline.',
      scenarios: REPRESENTATIVE_SCENARIOS,
    }
  }
  return {
    id: profileId,
    familyId,
    familyLabel,
    label: 'Peak / best-case',
    disclosure:
      'Best-case microbenchmark: cache-busted prefill plus predictable 128-token code bursts, thinking off. Peak cards show the best observed trial, not sustained agentic throughput.',
    scenarios: peakScenarios(familyId),
  }
}

export function buildBenchmarkMessages(
  scenario: BenchmarkScenario,
  nonce: string,
): Array<{ role: string; content: string }> {
  if (scenario.targetPromptTokens) {
    // This deliberately matches the retained speed-audit token-shape builder.
    // The exact server-side prompt token count is recorded in each result.
    const wordsNeeded = Math.floor(scenario.targetPromptTokens / 1.35)
    const fillerWords = PREFILL_FILLER.trim().split(/\s+/)
    const words: string[] = []
    while (words.length < wordsNeeded) words.push(...fillerWords)
    const body = words.slice(0, wordsNeeded).join(' ')
    const messages: Array<{ role: string; content: string }> = []
    if (scenario.systemPrompt) {
      messages.push({ role: 'system', content: scenario.systemPrompt })
    }
    messages.push({
      role: 'user',
      content:
        `[benchmark:${nonce}] Read the report below, then answer with exactly one word: OK.\n\n` +
        `${body}\n\nAnswer:`,
    })
    return messages
  }
  return [
    {
      role: 'user',
      content: `${scenario.prompt || ''}\n# benchmark:${nonce}`,
    },
  ]
}
