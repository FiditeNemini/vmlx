import { describe, expect, it } from "vitest";

import { formatCacheSelection, mergeCacheDetails } from "../src/shared/cacheMetrics";

describe('cache selection labels', () => {
  it.each(['native-glm+disk', 'paged', 'block-disk', 'miss', 'bypass'])('renders the observed %s selection', (value) => {
    expect(formatCacheSelection(value)).toBe(value)
  })
  it('preserves structured adaptive selection and rejected candidate', () => {
    expect(formatCacheSelection({ selected: 'ssd', rejected: 'prefill' })).toBe('ssd ← prefill')
    expect(formatCacheSelection({ selected: 'prefill', reason: 'cost' })).toBe('prefill')
  })
  it.each([null, undefined, '', ' ', 0, false, {}, { selected: 2 }])('does not invent a label for %j', (value) => {
    expect(formatCacheSelection(value)).toBeNull()
  })
})

describe("mergeCacheDetails", () => {
  it("retains a disk tier when a later tool iteration reports only resident cache", () => {
    expect(mergeCacheDetails("paged+dsv4+disk", "paged+dsv4")).toBe(
      "paged+dsv4+disk",
    );
  });

  it("adds newly observed tiers once in observation order", () => {
    expect(mergeCacheDetails("paged+ssm", "paged+ssm+disk+tq-native")).toBe(
      "paged+ssm+disk+tq-native",
    );
  });

  it("ignores empty and duplicate components", () => {
    expect(mergeCacheDetails("", "paged++disk+disk")).toBe("paged+disk");
  });
});
