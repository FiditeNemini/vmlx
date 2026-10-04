import { closeSync, fstatSync, openSync, readSync, realpathSync, statSync } from 'fs'
import { isAbsolute, join, relative, resolve } from 'path'

const MAX_HEADER_BYTES = 8 * 1024 ** 2
const MAX_METADATA_BYTES = 32 * 1024 ** 2
const MAX_SHARDS = 4096
const TYPE_BYTES: Record<string, number> = {
  BOOL: 1, U8: 1, I8: 1, U16: 2, I16: 2, F16: 2, BF16: 2,
  U32: 4, I32: 4, F32: 4, U64: 8, I64: 8, F64: 8,
  F8_E4M3: 1, F8_E5M2: 1,
}

// The file-backed table contract owned by qwen4_exp/loader.py:_PLE_TABLE_RE.
// Hash buffers, PLE projections and other layers remain resident.
const PLE_TABLE = /(?:^|\.)layers\.1\.ple\.(?:ple_embedding\.)?ngram_embedding\.(shard_|shards\.)(\d+)\.(weight|scales|biases)$/

interface TensorHeader {
  dtype: string
  shape: number[]
  data_offsets: [number, number]
}

export interface QwenResidentWeights {
  source: 'indexed-tensor-headers'
  residentTensorBytes: number
  fileBackedPleBytes: number
  proposalHeadReserveBytes: number
  proposalHeadSource: 'disabled' | 'conservative-q4-reserve'
}

export type QwenResidentInspection = QwenResidentWeights | {
  source: 'unavailable'
  reason: string
}

/** False is accepted only when the caller knows the loader's proposal override. */
export interface QwenResidencyOptions { proposalHeadEnabled?: boolean }

function checkedAdd(left: number, right: number): number {
  const value = left + right
  if (!Number.isSafeInteger(value) || value < 0) throw new Error('unsafe byte total')
  return value
}

function readExact(fd: number, bytes: number, position: number): Buffer {
  const buffer = Buffer.alloc(bytes)
  let offset = 0
  while (offset < bytes) {
    const count = readSync(fd, buffer, offset, bytes - offset, position + offset)
    if (!count) throw new Error('truncated metadata')
    offset += count
  }
  return buffer
}

function readObject(path: string, maxBytes: number): Record<string, any> {
  const fd = openSync(path, 'r')
  try {
    const bytes = fstatSync(fd).size
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > maxBytes) throw new Error('metadata size limit')
    const value = JSON.parse(readExact(fd, bytes, 0).toString('utf8'))
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('metadata is not an object')
    return value
  } finally { closeSync(fd) }
}

function containedFile(root: string, name: string): string {
  if (isAbsolute(name)) throw new Error('absolute shard path')
  const path = realpathSync(resolve(root, name))
  const rel = relative(root, path)
  if (!rel || rel === '..' || rel.startsWith('../') || isAbsolute(rel)) throw new Error('shard outside bundle')
  return path
}

function readHeader(path: string, budget: { bytes: number }): Record<string, TensorHeader> {
  const fd = openSync(path, 'r')
  try {
    const before = fstatSync(fd, { bigint: true })
    const length = readExact(fd, 8, 0).readBigUInt64LE()
    if (length <= 0n || length > BigInt(MAX_HEADER_BYTES) || 8n + length > before.size) throw new Error('invalid header length')
    budget.bytes = checkedAdd(budget.bytes, Number(length) + 8)
    if (budget.bytes > MAX_METADATA_BYTES) throw new Error('metadata budget exceeded')
    const value = JSON.parse(readExact(fd, Number(length), 8).toString('utf8'))
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('header is not an object')
    const entries: Record<string, TensorHeader> = Object.create(null)
    const ranges: [number, number][] = []
    for (const [name, tensor] of Object.entries(value)) {
      if (name === '__metadata__') continue
      const info = tensor as TensorHeader
      if (!info || !TYPE_BYTES[info.dtype] || !Array.isArray(info.shape) ||
          !info.shape.every(dim => Number.isSafeInteger(dim) && dim >= 0) ||
          !Array.isArray(info.data_offsets) || info.data_offsets.length !== 2) throw new Error('invalid tensor header')
      const [start, end] = info.data_offsets
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start ||
          BigInt(end) + length + 8n > before.size) throw new Error('invalid tensor range')
      const elements = info.shape.reduce((size, dim) => size * dim, 1)
      if (!Number.isSafeInteger(elements) || elements * TYPE_BYTES[info.dtype] !== end - start) throw new Error('tensor byte geometry mismatch')
      entries[name] = info
      if (end > start) ranges.push([start, end])
    }
    ranges.sort((a, b) => a[0] - b[0])
    if (ranges.some((range, i) => i > 0 && range[0] < ranges[i - 1][1])) throw new Error('overlapping tensor ranges')
    const after = fstatSync(fd, { bigint: true })
    if (before.size !== after.size || before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) throw new Error('shard changed during inspection')
    return entries
  } finally { closeSync(fd) }
}

function proposalReserve(tensors: Map<string, TensorHeader>, config: Record<string, any>, options: QwenResidencyOptions): number {
  if (options.proposalHeadEnabled === false || config.tie_word_embeddings === true || config.text_config?.tie_word_embeddings === true) return 0
  const heads = [...tensors.keys()].filter(key => key === 'lm_head.weight' || key.endsWith('.lm_head.weight'))
  if (heads.length !== 1) throw new Error('proposal source head is unknown')
  const stem = heads[0].slice(0, -'.weight'.length)
  const weight = tensors.get(heads[0])!
  const scales = tensors.get(`${stem}.scales`)
  const biases = tensors.get(`${stem}.biases`)
  // The override/stamp belongs to the engine process. Reserve the q4 allocation
  // even if launch mode disables MTP: this loader prepares its head independently.
  // This is a reserve, never an assertion that the sidecar was opened.
  if (!scales || !biases || weight.dtype !== 'U32' || weight.shape.length !== 2 ||
      scales.shape.length !== 2 || biases.shape.length !== 2 ||
      scales.shape.some((dim, i) => dim !== biases.shape[i]) ||
      scales.shape[0] !== weight.shape[0]) return 0 // No affine source: loader cannot construct this head.
  const hidden = Number(config.text_config?.hidden_size ?? config.hidden_size)
  if (!Number.isSafeInteger(hidden) || hidden <= 0 || hidden % scales.shape[1] || hidden % 8) throw new Error('proposal geometry is unknown')
  return checkedAdd(weight.shape[0] * hidden / 2,
    (scales.data_offsets[1] - scales.data_offsets[0]) + (biases.data_offsets[1] - biases.data_offsets[0]))
}

/** Bounded index/header inspection only; never reads tensor payloads or loads MLX. */
export function inspectQwen4ResidentWeights(modelPath: string, options: QwenResidencyOptions = {}): QwenResidentInspection {
  try {
    const root = realpathSync(modelPath)
    const config = readObject(join(root, 'config.json'), MAX_HEADER_BYTES)
    const indexPath = join(root, 'model.safetensors.index.json')
    const indexBefore = statSync(indexPath, { bigint: true })
    const index = readObject(indexPath, MAX_HEADER_BYTES)
    const map = index.weight_map
    if (!map || typeof map !== 'object' || Array.isArray(map) || !Object.keys(map).length ||
        Object.values(map).some(name => typeof name !== 'string' || !name.endsWith('.safetensors'))) throw new Error('missing or invalid weight index')
    const names = [...new Set(Object.values(map) as string[])]
    if (names.length > MAX_SHARDS) throw new Error('shard count limit')
    const tensors = new Map<string, TensorHeader>()
    const tensorFiles = new Map<string, string>()
    const budget = { bytes: indexBefore.size <= BigInt(MAX_METADATA_BYTES) ? Number(indexBefore.size) : MAX_METADATA_BYTES }
    for (const name of names) {
      for (const [key, info] of Object.entries(readHeader(containedFile(root, name), budget))) {
        if (tensors.has(key)) throw new Error('duplicate indexed tensor')
        tensors.set(key, info)
        tensorFiles.set(key, name)
      }
    }
    for (const [key, name] of Object.entries(map)) {
      if (tensorFiles.get(key) !== name) throw new Error('weight index/header mismatch')
    }
    const nParts = config.text_config?.split_ngram_parts ?? 128
    if (!Number.isSafeInteger(nParts) || nParts <= 0 || nParts > MAX_SHARDS) throw new Error('invalid PLE shard count')
    const formats = new Set<string>()
    let residentTensorBytes = 0
    let fileBackedPleBytes = 0
    for (const [key, info] of tensors) {
      const bytes = info.data_offsets[1] - info.data_offsets[0]
      const match = key.match(PLE_TABLE)
      if (match) {
        fileBackedPleBytes = checkedAdd(fileBackedPleBytes, bytes)
        formats.add(key.slice(0, -`${match[1]}${match[2]}.${match[3]}`.length) + match[1])
      } else residentTensorBytes = checkedAdd(residentTensorBytes, bytes)
    }
    const complete = [...formats].filter(format => Array.from({ length: nParts }, (_, i) => i)
      .every(i => ['weight', 'scales', 'biases'].every(suffix => tensors.has(`${format}${i}.${suffix}`))))
    if (complete.length !== 1 || fileBackedPleBytes <= 0) throw new Error('incomplete or ambiguous file-backed PLE table')
    const indexAfter = statSync(indexPath, { bigint: true })
    if (indexBefore.size !== indexAfter.size || indexBefore.mtimeNs !== indexAfter.mtimeNs || indexBefore.ctimeNs !== indexAfter.ctimeNs) throw new Error('index changed during inspection')
    const proposalHeadReserveBytes = proposalReserve(tensors, config, options)
    return { source: 'indexed-tensor-headers', residentTensorBytes, fileBackedPleBytes,
      proposalHeadReserveBytes, proposalHeadSource: proposalHeadReserveBytes > 0 ? 'conservative-q4-reserve' : 'disabled' }
  } catch (error) {
    return { source: 'unavailable', reason: error instanceof Error ? error.message : 'metadata inspection failed' }
  }
}
