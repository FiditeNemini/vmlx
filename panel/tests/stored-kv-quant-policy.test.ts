import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  allowedStoredKvQuantOptions,
  isMixedSwaBundle,
  storedKvQuantMustBeExact,
} from "../src/shared/storedKvQuantPolicy"

const FORM = "src/renderer/src/components/sessions/SessionConfigForm.tsx"
const LOCALES = ["en", "ko", "ja", "zh", "es"]

describe("mixed-SWA detection", () => {
  it("matches on cache subtype", () => {
    expect(isMixedSwaBundle({ cacheSubtype: "mixed_swa_kv" })).toBe(true)
    expect(isMixedSwaBundle({ cacheSubtype: "step3p7_full_sliding_kv" })).toBe(
      true,
    )
  })

  it("matches on the rotating_kv cache type", () => {
    expect(isMixedSwaBundle({ cacheType: "rotating_kv" })).toBe(true)
  })

  it("matches on architecture hints when subtype is absent", () => {
    expect(
      isMixedSwaBundle({ architectureHints: { cacheSchema: "mixed_swa_kv_v1" } }),
    ).toBe(true)
    expect(
      isMixedSwaBundle({
        architectureHints: { attentionArch: "full_and_sliding_kv" },
      }),
    ).toBe(true)
  })

  it("does not match uniform-attention bundles", () => {
    expect(isMixedSwaBundle({})).toBe(false)
    expect(isMixedSwaBundle({ cacheType: "kv", cacheSubtype: "kv" })).toBe(false)
    expect(isMixedSwaBundle({ architectureHints: null })).toBe(false)
    expect(
      isMixedSwaBundle({ architectureHints: { attentionArch: "full_kv" } }),
    ).toBe(false)
  })
})

describe("stored KV quantization policy", () => {
  // The policy went global after the mixed-SWA proof (Laguna-S, temp 0:
  // stored q8 cold bb040715 -> hit 633c133d DIVERGED). Stored prefix state
  // is exact for EVERY family now — one native value, one launch path, no
  // family-specific exception — and disk bytes confirmed full-precision
  // records with zero scale/bias tensors across families.
  it("requires exact storage for every family", () => {
    expect(storedKvQuantMustBeExact({ cacheSubtype: "mixed_swa_kv" })).toBe(true)
    expect(storedKvQuantMustBeExact({ cacheSubtype: "kv" })).toBe(true)
  })

  it("offers no lossy option to any bundle", () => {
    for (const cacheSubtype of ["mixed_swa_kv", "kv", "hybrid_ssm_v1"]) {
      const opts = allowedStoredKvQuantOptions({ cacheSubtype })
      expect(opts).toEqual(["auto"])
      expect(opts).not.toContain("q8")
      expect(opts).not.toContain("q4")
    }
  })
})

describe("the form delegates instead of re-inlining", () => {
  const form = readFileSync(resolve(__dirname, "..", FORM), "utf8")

  it("imports the shared policy module for mixed-SWA detection", () => {
    // Once the stored-codec policy went globally exact there was no
    // per-family branch left to call — the form only needs the mixed-SWA
    // detector. If storedKvQuantMustBeExact/allowedStoredKvQuantOptions
    // reappear in the form, that is fine; a re-inlined boolean is not.
    expect(form).toContain("shared/storedKvQuantPolicy")
    expect(form).toContain("isMixedSwaBundle(")
  })

  it("no longer hand-copies the three-condition mixed-SWA chain", () => {
    // There were TWO identical copies (mixedSwaCacheActive and
    // mixedSwaBlockDiskOnlySupported). If this fails, one came back.
    const inlined =
      /detectedCacheType === 'rotating_kv' \|\|\s*\n\s*detectedCacheSubtype === 'mixed_swa_kv'/
    expect(inlined.test(form)).toBe(false)
  })

  it("never re-offers a lossy stored-prefix codec", () => {
    // The stored prefix record is exact for every family. A q8/q4 stored
    // codec selector coming back to the form would reintroduce the
    // answer-changing bug the policy exists to prevent.
    expect(/storedKvQuant[a-zA-Z]*\s*[:=][^\n]*['"]q8['"]/.test(form)).toBe(false)
    expect(/storedKvQuant[a-zA-Z]*\s*[:=][^\n]*['"]q4['"]/.test(form)).toBe(false)
  })
})

describe("i18n coverage", () => {
  it("defines the warning in every locale", () => {
    // A missing key renders as a raw key path in the UI.
    for (const loc of LOCALES) {
      const json = JSON.parse(
        readFileSync(
          resolve(__dirname, "..", `src/renderer/src/i18n/locales/${loc}.json`),
          "utf8",
        ),
      )
      const text = json?.sessions?.config?.storedKvExactRequired
      expect(typeof text, `${loc} missing storedKvExactRequired`).toBe("string")
      expect(text.length, `${loc} empty storedKvExactRequired`).toBeGreaterThan(10)
    }
  })
})
