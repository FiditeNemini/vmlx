import { closeSync, ftruncateSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve } from 'path'
import { afterEach, describe, expect, it } from 'vitest'
import { inspectQwen4ResidentWeights } from '../src/main/qwenFileBackedResidency'
import { estimateModelFileBytes, estimateModelLaunchMemory, MODEL_LAUNCH_FIXED_OVERHEAD_BYTES } from '../src/main/modelLaunchMemory'

const temporary: string[] = []
afterEach(() => temporary.splice(0).forEach(dir => rmSync(dir, { recursive: true, force: true })))

function fixture(): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-qwen-header-estimate-'))
  temporary.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen4_exp',
    text_config: { hidden_size: 64, vocab_size: 8, split_ngram_parts: 1 } }))
  return dir
}

type Spec = { dtype: 'F16' | 'U32'; shape: number[] }
const table = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0'
const specs: Record<string, Spec> = {
  [`${table}.weight`]: { dtype: 'U32', shape: [256, 8] },
  [`${table}.scales`]: { dtype: 'F16', shape: [256, 1] },
  [`${table}.biases`]: { dtype: 'F16', shape: [256, 1] },
  'model.language_model.layers.1.ple.layer_multipliers': { dtype: 'U32', shape: [8] },
  'model.language_model.layers.1.ple.gate.weight': { dtype: 'F16', shape: [64, 64] },
  'lm_head.weight': { dtype: 'U32', shape: [8, 16] },
  'lm_head.scales': { dtype: 'F16', shape: [8, 1] },
  'lm_head.biases': { dtype: 'F16', shape: [8, 1] },
}

function writeShard(dir: string, name: string, entries: Record<string, Spec> = specs): void {
  let offset = 0
  const header: Record<string, unknown> = {}
  for (const [key, spec] of Object.entries(entries)) {
    const bytes = spec.shape.reduce((n, dim) => n * dim, 1) * (spec.dtype === 'U32' ? 4 : 2)
    header[key] = { ...spec, data_offsets: [offset, offset + bytes] }
    offset += bytes
  }
  const json = Buffer.from(JSON.stringify(header))
  const prefix = Buffer.alloc(8)
  prefix.writeBigUInt64LE(BigInt(json.length))
  const file = join(dir, name)
  mkdirSync(resolve(file, '..'), { recursive: true })
  writeFileSync(file, Buffer.concat([prefix, json]))
  // Sparse payload: the estimator must inspect metadata without allocating it.
  const fd = openSync(file, 'r+')
  try { ftruncateSync(fd, 8 + json.length + offset) } finally { closeSync(fd) }
}

function writeIndex(dir: string, names: Record<string, string> = Object.fromEntries(Object.keys(specs).map(key => [key, 'model.safetensors']))): void {
  writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({ weight_map: names }))
}

function bundle(): string {
  const dir = fixture()
  writeShard(dir, 'model.safetensors')
  writeIndex(dir)
  return dir
}

describe('Qwen indexed file-backed residency', () => {
  it('counts resident tensors, hash buffers and projections; excludes only the actual PLE table', () => {
    const dir = bundle()
    // An unindexed nested q4 sidecar or arbitrary file never gets counted twice.
    writeShard(dir, 'mtp_draft/proposal.safetensors', { 'lm_head.weight': { dtype: 'U32', shape: [8, 8] } })
    writeFileSync(join(dir, 'notes.txt'), 'unrelated bundle data')
    const estimate = estimateModelLaunchMemory(dir, estimateModelFileBytes(dir), 128 * 1024 ** 3)
    expect(estimate.source).toBe('indexed-tensor-headers')
    expect(estimate.fileBackedPleBytes).toBe(9216)
    expect(estimate.proposalHeadReserveBytes).toBe(288)
    expect(estimate.expectedResidentBytes).toBe(32 + 8192 + 544 + 288)
    expect(estimate.launchResidentBytes).toBe(estimate.expectedResidentBytes + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES)
    expect(estimate.launchAdmissionBytes).toBe(estimate.launchResidentBytes)
    expect(estimate.reason).toBe('conservative-q4-reserve')
  })

  it('omits the proposal reserve only when the loader override is known disabled', () => {
    const dir = bundle()
    const estimate = estimateModelLaunchMemory(dir, estimateModelFileBytes(dir), 0, { proposalHeadEnabled: false })
    expect(estimate.expectedResidentBytes).toBe(32 + 8192 + 544)
    expect(estimate.proposalHeadReserveBytes).toBe(0)
    expect(estimate.reason).toBe('disabled')
  })

  it('supports the other loader table namespace, and keeps table-like tensors in other layers resident', () => {
    const dir = fixture()
    const entries = Object.fromEntries(Object.entries(specs).map(([key, spec]) => [key.replace('model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_', 'language_model.model.layers.1.ple.ngram_embedding.shards.'), spec]))
    entries['language_model.model.layers.2.ple.ngram_embedding.shards.0.weight'] = { dtype: 'U32', shape: [1, 4] }
    writeShard(dir, 'model.safetensors', entries)
    writeIndex(dir, Object.fromEntries(Object.keys(entries).map(key => [key, 'model.safetensors'])))
    const result = inspectQwen4ResidentWeights(dir)
    expect(result.source).toBe('indexed-tensor-headers')
    if (result.source === 'indexed-tensor-headers') {
      expect(result.fileBackedPleBytes).toBe(9216)
      expect(result.residentTensorBytes).toBe(32 + 8192 + 544 + 16)
    }
  })

  it('does not reuse stale header accounting after a shard/index replacement', () => {
    const dir = bundle()
    const before = inspectQwen4ResidentWeights(dir)
    const changed = { ...specs, 'new_projection.weight': { dtype: 'F16' as const, shape: [64, 64] } }
    writeShard(dir, 'replacement.safetensors', changed)
    writeIndex(dir, Object.fromEntries(Object.keys(changed).map(key => [key, 'replacement.safetensors'])))
    const after = inspectQwen4ResidentWeights(dir)
    if (before.source !== 'indexed-tensor-headers' || after.source !== 'indexed-tensor-headers') throw new Error('expected indexed headers')
    expect(after.residentTensorBytes - before.residentTensorBytes).toBe(8192)
  })

  it.each(['missing-index', 'unreadable-header', 'incomplete-table', 'wrong-index', 'outside-bundle', 'oversize-header'])(
    'falls back with provenance, never a fixed fraction, for %s', failure => {
      const dir = bundle()
      if (failure === 'missing-index') rmSync(join(dir, 'model.safetensors.index.json'))
      if (failure === 'unreadable-header') writeFileSync(join(dir, 'model.safetensors'), 'bad')
      if (failure === 'incomplete-table') {
        const entries = { ...specs }; delete entries[`${table}.biases`]
        writeShard(dir, 'model.safetensors', entries)
        writeIndex(dir, Object.fromEntries(Object.keys(entries).map(key => [key, 'model.safetensors'])))
      }
      if (failure === 'wrong-index') writeIndex(dir, { absent: 'model.safetensors' })
      if (failure === 'outside-bundle') writeIndex(dir, { absent: '../outside.safetensors' })
      if (failure === 'oversize-header') {
        const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(9n * 1024n ** 2n)
        writeFileSync(join(dir, 'model.safetensors'), bytes)
      }
      const estimate = estimateModelLaunchMemory(dir, 100e9, 0)
      expect(estimate.source).toBe('full-file-fallback')
      expect(estimate.reason).toBeTruthy()
      expect(estimate.launchResidentBytes).toBe(100e9 + MODEL_LAUNCH_FIXED_OVERHEAD_BYTES)
      expect(estimate.expectedResidentBytes).toBe(100e9)
      expect(estimate.fileBackedPleBytes).toBe(0)
    },
  )

  it('uses the corrected estimate for both launch and wake progress consumers', () => {
    const source = readFileSync(resolve(__dirname, '../src/main/sessions.ts'), 'utf8')
    expect(source).toContain('expectedResidentBytes: wakeEstimate.expectedResidentBytes')
    expect(source).toContain('expectedResidentBytes: launchEstimate.expectedResidentBytes')
    expect(source).not.toMatch(/modelFileBytes\s*\*\s*(wakeProfile|residentProfile)\.ratio/)
  })
})
