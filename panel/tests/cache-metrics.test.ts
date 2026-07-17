import { describe, expect, it } from "vitest";

import { mergeCacheDetails } from "../src/shared/cacheMetrics";

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
