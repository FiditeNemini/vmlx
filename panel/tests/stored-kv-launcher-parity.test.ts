import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  allowedStoredKvQuantOptions,
  storedKvQuantMustBeExact,
} from "../src/shared/storedKvQuantPolicy"

/**
 * Production Electron sessions preserve architecture-native cache state.
 *
 * A model may own full KV, sparse-index state, recurrent companions, rotating
 * metadata, or a native compressed representation. The UI must not add q4/q8
 * to any of them, and a stale saved value must not reach either argv builder.
 */
const MAIN = "src/main/sessions.ts"
const RENDERER = "src/renderer/src/components/sessions/SessionSettings.tsx"
const FORM = "src/renderer/src/components/sessions/SessionConfigForm.tsx"

const read = (rel: string) => readFileSync(resolve(__dirname, "..", rel), "utf8")

describe("production stored cache stays architecture-native", () => {
  it("normalizes every persisted value to canonical auto", () => {
    const main = read(MAIN)
    const start = main.indexOf("function normalizeCacheStackMutualExclusion")
    const block = main.slice(start, start + 1500)
    expect(block).toContain("config.kvCacheQuantization !== 'auto'")
    expect(block).toContain("config.kvCacheQuantization = 'auto'")
  })

  it("the main launch builder never emits a generic stored codec", () => {
    const main = read(MAIN)
    expect(main).not.toContain("args.push('--kv-cache-quantization'")
    expect(main).not.toContain('args.push("--kv-cache-quantization"')
  })

  it("the renderer command preview never emits a generic stored codec", () => {
    const renderer = read(RENDERER)
    expect(renderer).not.toContain("parts.push('--kv-cache-quantization'")
    expect(renderer).not.toContain('parts.push("--kv-cache-quantization"')
  })

  it("the visible setting is a disabled single native option", () => {
    const form = read(FORM)
    const start = form.indexOf("{/* Stored prefix representation. Auto adds no codec. */}")
    const block = form.slice(start, start + 800)
    expect(block).toContain('<select value={effectiveStoredCacheQuantization} className="cfg-input" disabled>')
    expect(block).toContain('<option value="auto">')
    expect(block).not.toContain('<option value="none">')
    expect(block).not.toContain('value="q8"')
    expect(block).not.toContain('value="q4"')
  })

  it("the shared policy rejects added codecs for every topology", () => {
    for (const input of [
      { cacheType: "kv" },
      { cacheType: "hybrid" },
      { cacheType: "rotating_kv" },
      { cacheSubtype: "mixed_swa_kv" },
      { cacheSubtype: "step3p7_full_sliding_kv" },
      { architectureHints: { cacheSchema: "hybrid_ssm_v1" } },
    ]) {
      expect(storedKvQuantMustBeExact(input)).toBe(true)
      expect(allowedStoredKvQuantOptions(input)).toEqual(["auto"])
    }
  })

  it("all reset surfaces use the same canonical value", () => {
    for (const rel of [
      RENDERER,
      "src/renderer/src/components/sessions/ServerSettingsDrawer.tsx",
      "src/renderer/src/components/sessions/CreateSession.tsx",
    ]) {
      const source = read(rel)
      expect(source).not.toContain("base.kvCacheQuantization = 'none'")
      expect(source).toContain("base.kvCacheQuantization = 'auto'")
    }
  })
})
