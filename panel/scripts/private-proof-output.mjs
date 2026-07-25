import { existsSync, realpathSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

function nearestExistingDirectory(candidate) {
  let current = path.resolve(candidate);
  while (!existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) return current;
    current = parent;
  }
  return current;
}

function canonicalExistingAncestor(candidate) {
  const existing = nearestExistingDirectory(candidate);
  return realpathSync(existing);
}

function isWithin(candidate, parent) {
  const relative = path.relative(parent, candidate);
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) && relative !== "..")
  );
}

function isInAnyGitContext(candidate) {
  const existing = canonicalExistingAncestor(candidate);
  const gitEnv = { ...process.env, LC_ALL: "C" };
  delete gitEnv.GIT_DIR;
  delete gitEnv.GIT_WORK_TREE;
  delete gitEnv.GIT_CEILING_DIRECTORIES;
  const result = spawnSync(
    "/usr/bin/git",
    [
      "-C",
      existing,
      "rev-parse",
      "--is-inside-work-tree",
      "--is-inside-git-dir",
    ],
    {
      encoding: "utf8",
      env: gitEnv,
    },
  );
  if (result.status !== 0) {
    return !String(result.stderr || "")
      .toLowerCase()
      .includes("not a git repository");
  }
  return String(result.stdout || "")
    .split(/\r?\n/)
    .some((line) => line.trim().toLowerCase() === "true");
}

function safeSegment(value, fallback) {
  const sanitized = String(value || "")
    .replace(/[^A-Za-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 96);
  return sanitized || fallback;
}

export function resolvePrivateProofDirectory({
  repoDir,
  overrideEnv,
  proofName,
  family,
  env = process.env,
  now = new Date(),
  pid = process.pid,
}) {
  const repository = realpathSync(path.resolve(repoDir));
  const override = String(env[overrideEnv] || "").trim();
  let candidate;
  if (override) {
    candidate = path.resolve(override);
    if (existsSync(candidate)) {
      throw new Error(`${overrideEnv} must name a new proof directory`);
    }
  } else {
    const privateRoot = String(env.VMLX_PRIVATE_EVIDENCE_ROOT || "").trim();
    if (!privateRoot) {
      throw new Error(
        `${overrideEnv} or VMLX_PRIVATE_EVIDENCE_ROOT is required; proof artifacts must not default into the public checkout`,
      );
    }
    const runId = safeSegment(
      env.VMLX_PROOF_RUN_ID ||
        `${now.toISOString().replace(/[:.]/g, "-")}-pid${pid}`,
      `pid${pid}`,
    );
    candidate = path.resolve(
      privateRoot,
      "live-proofs",
      safeSegment(family, "unknown-family"),
      `${safeSegment(proofName, "proof")}-${runId}`,
    );
    if (existsSync(candidate)) {
      throw new Error("private proof run directory already exists");
    }
  }

  const existing = canonicalExistingAncestor(candidate);
  if (isWithin(existing, repository) || isWithin(candidate, repository)) {
    throw new Error(
      "proof output must resolve outside the public Git worktree",
    );
  }
  if (isInAnyGitContext(candidate)) {
    throw new Error("proof output must resolve outside every Git worktree");
  }
  return candidate;
}
