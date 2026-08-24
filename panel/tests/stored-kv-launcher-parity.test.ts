import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { storedKvQuantMustBeExact } from "../src/shared/storedKvQuantPolicy"

/**
 * The display layer alone is not enough.
 *
 * Withholding q8/q4 from the SELECTOR only helps new sessions. A session that
 * saved q8 before that guard existed still carried it, and both arg builders
 * pushed `--kv-cache-quantization q8`, which the engine honours as an explicit
 * override. The UI then showed "only exact storage is offered" while q8 was on
 * the wire — strictly worse than showing q8 honestly.
 *
 * These pin the rule at the two places that actually decide what launches.
 */
const MAIN = "src/main/sessions.ts"
const RENDERER = "src/renderer/src/components/sessions/SessionSettings.tsx"
const FORM = "src/renderer/src/components/sessions/SessionConfigForm.tsx"

const read = (rel: string) => readFileSync(resolve(__dirname, "..", rel), "utf8")

describe("a saved lossy stored codec cannot reach a mixed-SWA launch", () => {
  it("the main process consults the shared policy", () => {
    const main = read(MAIN)
    expect(main).toContain("shared/storedKvQuantPolicy")
    expect(main).toContain("storedKvQuantMustBeExact(")
  })

  it("the main process rewrites a saved q8/q4 rather than only hiding it", () => {
    const main = read(MAIN)
    expect(main).toContain("config.kvCacheQuantization = 'auto'")
    // 'auto' and NOT 'none': both now preserve native state, but Auto is the
    // canonical app value and omits a redundant CLI flag.
    const idx = main.indexOf("storedKvQuantMustBeExact(")
    const window = main.slice(idx, idx + 1400)
    expect(window).toContain("'auto'")
    expect(window).not.toContain("config.kvCacheQuantization = 'none'")
  })

  it("the renderer preview suppresses the same flag", () => {
    const r = read(RENDERER)
    expect(r).toContain("shared/storedKvQuantPolicy")
    expect(r).toContain("lossyStoredCodecSuppressed")
    expect(r).toContain("!lossyStoredCodecSuppressed")
  })

  it("the selector is driven by the policy's option list, not a re-inlined gate", () => {
    const form = read(FORM)
    expect(form).toContain("allowedStoredKvQuantOptions(")
    expect(form).toContain("storedKvQuantOptions.includes('q8')")
    expect(form).toContain("storedKvQuantOptions.includes('q4')")
  })

  it("all three consumers agree with the policy for a mixed-SWA bundle", () => {
    expect(storedKvQuantMustBeExact({ cacheSubtype: "mixed_swa_kv" })).toBe(true)
    expect(
      storedKvQuantMustBeExact({ cacheSubtype: "step3p7_full_sliding_kv" }),
    ).toBe(true)
    expect(storedKvQuantMustBeExact({ cacheType: "rotating_kv" })).toBe(true)
    // and leaves everyone else alone
    expect(storedKvQuantMustBeExact({ cacheType: "kv" })).toBe(false)
  })
})
