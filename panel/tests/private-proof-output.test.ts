import {
  chmodSync,
  mkdtempSync,
  mkdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";

// @ts-expect-error The production helper is deliberately plain ESM for Node scripts.
import { resolvePrivateProofDirectory } from "../scripts/private-proof-output.mjs";

const productRepo = path.resolve(__dirname, "..", "..");

describe("private proof output", () => {
  it("requires an explicit private root and creates a unique family/run path", () => {
    const root = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    try {
      expect(() =>
        resolvePrivateProofDirectory({
          repoDir: productRepo,
          overrideEnv: "VMLX_TEST_PROOF_DIR",
          proofName: "live matrix",
          family: "Gemma 4/MoE",
          env: {},
          pid: 123,
        }),
      ).toThrow(/VMLX_PRIVATE_EVIDENCE_ROOT is required/);

      const result = resolvePrivateProofDirectory({
        repoDir: productRepo,
        overrideEnv: "VMLX_TEST_PROOF_DIR",
        proofName: "live matrix",
        family: "Gemma 4/MoE",
        env: {
          VMLX_PRIVATE_EVIDENCE_ROOT: root,
          VMLX_PROOF_RUN_ID: "paired run 01",
        },
        pid: 123,
      });
      expect(result).toBe(
        path.join(
          root,
          "live-proofs",
          "Gemma-4-MoE",
          "live-matrix-paired-run-01",
        ),
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects public-worktree and other-Git destinations", () => {
    const root = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    const otherRepo = path.join(root, "other-repo");
    mkdirSync(otherRepo);
    const initialized = spawnSync("/usr/bin/git", ["init", "-q", otherRepo]);
    expect(initialized.status).toBe(0);
    try {
      expect(() =>
        resolvePrivateProofDirectory({
          repoDir: productRepo,
          overrideEnv: "VMLX_TEST_PROOF_DIR",
          proofName: "proof",
          family: "qwen",
          env: {
            VMLX_TEST_PROOF_DIR: path.join(
              productRepo,
              "build",
              "private-proof",
            ),
          },
        }),
      ).toThrow(/outside the public Git worktree/);

      expect(() =>
        resolvePrivateProofDirectory({
          repoDir: productRepo,
          overrideEnv: "VMLX_TEST_PROOF_DIR",
          proofName: "proof",
          family: "qwen",
          env: {
            VMLX_TEST_PROOF_DIR: path.join(otherRepo, "proof"),
          },
        }),
      ).toThrow(/outside every Git worktree/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects stale explicit output directories", () => {
    const root = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    const existing = path.join(root, "existing");
    mkdirSync(existing);
    try {
      expect(() =>
        resolvePrivateProofDirectory({
          repoDir: productRepo,
          overrideEnv: "VMLX_TEST_PROOF_DIR",
          proofName: "proof",
          family: "lfm",
          env: { VMLX_TEST_PROOF_DIR: existing },
        }),
      ).toThrow(/must name a new proof directory/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("does not trust an ambient PATH git shim for worktree classification", () => {
    const root = mkdtempSync(path.join(tmpdir(), "vmlx-private-proof-"));
    const otherRepo = path.join(root, "other-repo");
    const fakeBin = path.join(root, "fake-bin");
    mkdirSync(otherRepo);
    mkdirSync(fakeBin);
    expect(spawnSync("/usr/bin/git", ["init", "-q", otherRepo]).status).toBe(0);
    const fakeGit = path.join(fakeBin, "git");
    writeFileSync(fakeGit, "#!/bin/sh\nexit 128\n", "utf8");
    chmodSync(fakeGit, 0o755);
    try {
      expect(() =>
        resolvePrivateProofDirectory({
          repoDir: productRepo,
          overrideEnv: "VMLX_TEST_PROOF_DIR",
          proofName: "proof",
          family: "qwen",
          env: {
            PATH: fakeBin,
            VMLX_TEST_PROOF_DIR: path.join(otherRepo, "proof"),
          },
        }),
      ).toThrow(/outside every Git worktree/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
