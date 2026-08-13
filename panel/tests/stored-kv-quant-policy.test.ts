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
  it("requires exact storage for mixed-SWA and allows lossy elsewhere", () => {
    expect(storedKvQuantMustBeExact({ cacheSubtype: "mixed_swa_kv" })).toBe(true)
    expect(storedKvQuantMustBeExact({ cacheSubtype: "kv" })).toBe(false)
  })

  it("offers no lossy option to a mixed-SWA bundle", () => {
    // q8 changed the answer on a cache HIT (Laguna-S, temp 0:
    // cold bb040715 -> hit 633c133d), so it must not be selectable.
    const opts = allowedStoredKvQuantOptions({ cacheSubtype: "mixed_swa_kv" })
    expect(opts).toEqual(["auto", "none"])
    expect(opts).not.toContain("q8")
    expect(opts).not.toContain("q4")
  })

  it("leaves other families their full choice", () => {
    expect(allowedStoredKvQuantOptions({ cacheSubtype: "kv" })).toEqual([
      "auto",
      "none",
      "q8",
      "q4",
    ])
  })
})

describe("the form delegates instead of re-inlining", () => {
  const form = readFileSync(resolve(__dirname, "..", FORM), "utf8")

  it("imports and calls the shared policy", () => {
    expect(form).toContain("shared/storedKvQuantPolicy")
    expect(form).toContain("isMixedSwaBundle(")
    expect(form).toContain("storedKvQuantMustBeExact(")
  })

  it("no longer hand-copies the three-condition mixed-SWA chain", () => {
    // There were TWO identical copies (mixedSwaCacheActive and
    // mixedSwaBlockDiskOnlySupported). If this fails, one came back.
    const inlined =
      /detectedCacheType === 'rotating_kv' \|\|\s*\n\s*detectedCacheSubtype === 'mixed_swa_kv'/
    expect(inlined.test(form)).toBe(false)
  })

  it("gates the lossy stored-codec options via the policy's option list", () => {
    // Driven by allowedStoredKvQuantOptions rather than a re-inlined boolean,
    // so the module cannot go stale against the form it governs.
    expect(form).toContain("allowedStoredKvQuantOptions(")
    expect(form).toContain("storedKvQuantOptions.includes('q8')")
    expect(form).toContain("storedKvQuantOptions.includes('q4')")
  })

  it("warns the user why the choice is unavailable", () => {
    expect(form).toContain("sessions.config.storedKvExactRequired")
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
