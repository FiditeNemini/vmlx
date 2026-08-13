import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import {
  computeEffectiveJit,
  jitSuppressionReason,
} from "../src/shared/jitPolicy"

const ENABLED = {
  enableJitRequested: true,
  isMultimodal: false,
  flashMoeActive: false,
  distributedActive: false,
  dsv4Active: false,
  m3Active: false,
  zayaCcaActive: false,
  turboQuantActive: false,
  lagunaMixedSwaTurboQuantActive: false,
  hybridCacheActive: false,
}

/** Every consumer that must not re-inline the rule. */
const CONSUMERS = [
  "src/main/sessions.ts",
  "src/renderer/src/components/sessions/SessionSettings.tsx",
  "src/renderer/src/components/sessions/SessionConfigForm.tsx",
]

describe("JIT suppression policy", () => {
  it("enables JIT only when requested and no runtime owns its own kernels", () => {
    expect(computeEffectiveJit(ENABLED)).toBe(true)
    expect(jitSuppressionReason(ENABLED)).toBeNull()
  })

  it("reports a reason for every suppressing condition", () => {
    const cases: Array<[keyof typeof ENABLED, string]> = [
      ["isMultimodal", "multimodal"],
      ["flashMoeActive", "flash_moe"],
      ["distributedActive", "distributed"],
      ["dsv4Active", "dsv4"],
      ["m3Active", "m3"],
      ["zayaCcaActive", "zaya_cca"],
      ["turboQuantActive", "turboquant"],
      ["lagunaMixedSwaTurboQuantActive", "laguna_mixed_swa_turboquant"],
      ["hybridCacheActive", "hybrid_cache"],
    ]
    for (const [field, reason] of cases) {
      const input = { ...ENABLED, [field]: true }
      expect(jitSuppressionReason(input), field).toBe(reason)
      expect(computeEffectiveJit(input), field).toBe(false)
    }
  })

  it("treats an unset toggle as disabled regardless of runtime", () => {
    expect(
      jitSuppressionReason({ ...ENABLED, enableJitRequested: false }),
    ).toBe("disabled")
  })

  it("keeps all consumers on the shared policy instead of re-inlining it", () => {
    // The rule previously lived as three hand-copied boolean expressions that
    // spelled the multimodal condition differently (isVLM vs multimodalActive),
    // so the checkbox and the launched flag could disagree. If this fails,
    // someone re-inlined it -- route the call site through computeEffectiveJit.
    for (const rel of CONSUMERS) {
      const src = readFileSync(resolve(__dirname, "..", rel), "utf8")
      expect(src, `${rel} must import the shared policy`).toContain(
        "shared/jitPolicy",
      )
      expect(src, `${rel} must call computeEffectiveJit`).toContain(
        "computeEffectiveJit(",
      )
      expect(
        /!!config\.enableJit\s*&&\s*!/.test(src),
        `${rel} re-inlined the JIT suppression chain`,
      ).toBe(false)
    }
  })
})
