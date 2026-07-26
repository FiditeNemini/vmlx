"use strict";

const {
  closeSync,
  constants,
  existsSync,
  fchmodSync,
  fsyncSync,
  fstatSync,
  linkSync,
  lstatSync,
  openSync,
  readFileSync,
  readSync,
  realpathSync,
  statSync,
  unlinkSync,
  writeSync,
} = require("node:fs");
const { createHash, randomUUID } = require("node:crypto");
const { dirname, isAbsolute, join, resolve } = require("node:path");
const { spawnSync } = require("node:child_process");

const FIXED_PATH = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin";
const PLAN_SCHEMA = 1;

function sha256Bytes(value) {
  return createHash("sha256").update(value).digest("hex");
}

function sha256Fd(fd) {
  const digest = createHash("sha256");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  let position = 0;
  for (;;) {
    const count = readSync(fd, buffer, 0, buffer.length, position);
    if (count === 0) break;
    digest.update(buffer.subarray(0, count));
    position += count;
  }
  return digest.digest("hex");
}

function openIdentity(path, { allowMultipleLinks = false } = {}) {
  const flags = constants.O_RDONLY | (constants.O_NOFOLLOW || 0);
  const fd = openSync(path, flags);
  try {
    const before = fstatSync(fd, { bigint: true });
    if (
      !before.isFile() ||
      (!allowMultipleLinks && before.nlink !== 1n) ||
      (before.mode & 0o111n) === 0n
    ) {
      throw new Error(`unsafe pinned release executable: ${path}`);
    }
    const sha256 = sha256Fd(fd);
    const after = fstatSync(fd, { bigint: true });
    if (
      before.dev !== after.dev ||
      before.ino !== after.ino ||
      before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs
    ) {
      throw new Error(`pinned release executable changed while read: ${path}`);
    }
    return {
      fd,
      record: {
        path,
        device: before.dev.toString(),
        inode: before.ino.toString(),
        size: before.size.toString(),
        mode: Number(before.mode & 0o777n),
        sha256,
      },
    };
  } catch (error) {
    closeSync(fd);
    throw error;
  }
}

function regularFileIdentity(path, { allowMultipleLinks = false } = {}) {
  const flags = constants.O_RDONLY | (constants.O_NOFOLLOW || 0);
  const fd = openSync(path, flags);
  try {
    const before = fstatSync(fd, { bigint: true });
    if (!before.isFile() || (!allowMultipleLinks && before.nlink !== 1n)) {
      throw new Error(`unsafe pinned release file: ${path}`);
    }
    const sha256 = sha256Fd(fd);
    const after = fstatSync(fd, { bigint: true });
    if (
      before.dev !== after.dev ||
      before.ino !== after.ino ||
      before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs
    ) {
      throw new Error(`pinned release file changed while read: ${path}`);
    }
    return {
      path,
      device: before.dev.toString(),
      inode: before.ino.toString(),
      size: before.size.toString(),
      mode: Number(before.mode & 0o777n),
      sha256,
    };
  } finally {
    closeSync(fd);
  }
}

function sameIdentity(actual, expected, label) {
  for (const field of ["device", "inode", "size", "sha256"]) {
    if (String(actual[field]) !== String(expected[field])) {
      throw new Error(`${label} ${field} changed`);
    }
  }
}

function bindReleasePython(aliasPath, options = {}) {
  if (!isAbsolute(aliasPath)) {
    throw new Error("release Python alias must be absolute");
  }
  const alias = resolve(aliasPath);
  const sourcePath = realpathSync(alias);
  const source = openIdentity(sourcePath);
  const binDir = dirname(alias);
  const sourceBinDir = dirname(sourcePath);
  const pyvenvPath = resolve(binDir, "..", "pyvenv.cfg");
  const pyvenv = regularFileIdentity(pyvenvPath);
  const verifier = regularFileIdentity(__filename);
  const nonce = randomUUID().replaceAll("-", "");
  // Keep the executable hardlink beside the physical interpreter. Standalone
  // CPython builds resolve libpython through an executable-relative @rpath;
  // moving the hardlink into a venv bin directory makes dyld search the venv
  // for a library that lives beside the physical interpreter instead.
  const actionPath = join(sourceBinDir, `.vmlx-r18-python-${nonce}`);
  const planPath = join(binDir, `.vmlx-r18-python-${nonce}.json`);
  let actionCreated = false;
  let planCreated = false;
  try {
    const createHardlink = options.link || linkSync;
    createHardlink(sourcePath, actionPath);
    actionCreated = true;
    const action = openIdentity(actionPath, { allowMultipleLinks: true });
    let actionRecord;
    try {
      sameIdentity(action.record, source.record, "release Python hardlink");
      actionRecord = action.record;
    } finally {
      closeSync(action.fd);
    }
    const payload = {
      schema_version: PLAN_SCHEMA,
      fixed_path: FIXED_PATH,
      alias,
      source: source.record,
      action: { ...actionRecord, path: actionPath },
      pyvenv,
      verifier,
    };
    const canonical = Buffer.from(
      `${canonicalJson(payload)}\n`,
      "utf8",
    );
    const planFd = openSync(
      planPath,
      constants.O_WRONLY |
        constants.O_CREAT |
        constants.O_EXCL |
        (constants.O_NOFOLLOW || 0),
      0o600,
    );
    planCreated = true;
    try {
      writeSync(planFd, canonical);
      fsyncSync(planFd);
      fchmodSync(planFd, 0o400);
      fsyncSync(planFd);
    } finally {
      closeSync(planFd);
    }
    return {
      planPath,
      planSha256: sha256Bytes(canonical),
      actionPath,
      sourcePath,
      sourceSha256: source.record.sha256,
      pyvenvSha256: pyvenv.sha256,
    };
  } catch (error) {
    if (planCreated) {
      try {
        unlinkSync(planPath);
      } catch {}
    }
    if (actionCreated) {
      try {
        unlinkSync(actionPath);
      } catch {}
    }
    throw error;
  } finally {
    closeSync(source.fd);
  }
}

function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function readBinding(planPath, expectedSha256) {
  if (!isAbsolute(planPath) || !/^[0-9a-f]{64}$/.test(expectedSha256 || "")) {
    throw new Error("release Python binding path/digest is invalid");
  }
  const flags = constants.O_RDONLY | (constants.O_NOFOLLOW || 0);
  const fd = openSync(planPath, flags);
  let bytes;
  let metadata;
  try {
    metadata = fstatSync(fd, { bigint: true });
    if (
      !metadata.isFile() ||
      metadata.nlink !== 1n ||
      Number(metadata.mode & 0o777n) !== 0o400
    ) {
      throw new Error("release Python binding plan is not immutable");
    }
    bytes = readFileSync(fd);
  } finally {
    closeSync(fd);
  }
  if (sha256Bytes(bytes) !== expectedSha256) {
    throw new Error("release Python binding plan digest changed");
  }
  const plan = JSON.parse(bytes.toString("utf8"));
  if (
    plan.schema_version !== PLAN_SCHEMA ||
    plan.fixed_path !== FIXED_PATH ||
    process.env.PATH !== FIXED_PATH ||
    realpathSync(plan.alias) !== plan.source.path
  ) {
    throw new Error("release Python binding plan or alias changed");
  }
  const verifier = regularFileIdentity(__filename);
  sameIdentity(verifier, plan.verifier, "release Python verifier");
  const pyvenv = regularFileIdentity(plan.pyvenv.path);
  sameIdentity(pyvenv, plan.pyvenv, "release Python pyvenv");
  const source = openIdentity(plan.source.path, { allowMultipleLinks: true });
  const action = openIdentity(plan.action.path, { allowMultipleLinks: true });
  sameIdentity(source.record, plan.source, "release Python source");
  sameIdentity(action.record, plan.action, "release Python action");
  sameIdentity(action.record, source.record, "release Python hardlink");
  return { plan, source, action, planRecord: regularFileIdentity(planPath) };
}

function bindingFromEnvironment() {
  const planPath = process.env.VMLX_R18_RELEASE_PYTHON_PLAN || "";
  const expectedSha256 =
    process.env.VMLX_R18_RELEASE_PYTHON_PLAN_SHA256 || "";
  return { planPath, expectedSha256 };
}

function scriptBinding(args) {
  const argv = [...args];
  const inlineIndex = argv.indexOf("-c");
  if (inlineIndex >= 0 && inlineIndex + 1 < argv.length) {
    return {
      argv,
      input: undefined,
      record: {
        kind: "inline",
        sha256: sha256Bytes(Buffer.from(argv[inlineIndex + 1], "utf8")),
      },
      cleanup() {},
      revalidate() {},
    };
  }
  const moduleIndex = argv.indexOf("-m");
  if (moduleIndex >= 0 && moduleIndex + 1 < argv.length) {
    return {
      argv,
      input: undefined,
      record: {
        kind: "module",
        value: argv[moduleIndex + 1],
        argv_sha256: sha256Bytes(Buffer.from(canonicalJson(argv), "utf8")),
      },
      cleanup() {},
      revalidate() {},
    };
  }
  const stdinIndex = argv.indexOf("-");
  if (stdinIndex >= 0) {
    const input = readFileSync(0);
    return {
      argv,
      input,
      record: { kind: "stdin", sha256: sha256Bytes(input) },
      cleanup() {},
      revalidate() {},
    };
  }
  const scriptIndex = argv.findIndex(
    (value) => !value.startsWith("-") && existsSync(resolve(value)),
  );
  if (scriptIndex < 0) {
    return {
      argv,
      input: undefined,
      record: {
        kind: "argv",
        sha256: sha256Bytes(Buffer.from(canonicalJson(argv), "utf8")),
      },
      cleanup() {},
      revalidate() {},
    };
  }
  const originalPath = realpathSync(resolve(argv[scriptIndex]));
  const original = regularFileIdentity(originalPath);
  const actionPath = join(
    dirname(originalPath),
    `.${originalPath.split("/").at(-1)}.vmlx-r18-${randomUUID().replaceAll("-", "")}`,
  );
  linkSync(originalPath, actionPath);
  const action = regularFileIdentity(actionPath, { allowMultipleLinks: true });
  sameIdentity(action, original, "release Python script hardlink");
  argv[scriptIndex] = actionPath;
  return {
    argv,
    input: undefined,
    record: { kind: "script", original, action },
    revalidate() {
      sameIdentity(
        regularFileIdentity(originalPath, { allowMultipleLinks: true }),
        original,
        "release Python script source",
      );
      sameIdentity(
        regularFileIdentity(actionPath, { allowMultipleLinks: true }),
        action,
        "release Python script action",
      );
    },
    cleanup() {
      try {
        unlinkSync(actionPath);
      } catch {}
    },
  };
}

function sanitizedPythonEnvironment(launcher) {
  const env = { ...process.env, PATH: FIXED_PATH };
  for (const key of Object.keys(env)) {
    if (key.startsWith("PYTHON")) delete env[key];
  }
  // Executing the source-adjacent hardlink preserves the interpreter rpath;
  // this pinned launcher preserves the authoritative venv prefix/import path.
  env.__PYVENV_LAUNCHER__ = launcher;
  return env;
}

function assertBindingUnchanged(before, current, phase) {
  sameIdentity(
    current.source.record,
    before.source.record,
    `release Python source ${phase}`,
  );
  sameIdentity(
    current.action.record,
    before.action.record,
    `release Python action ${phase}`,
  );
  if (
    current.planRecord.sha256 !== before.planRecord.sha256 ||
    current.planRecord.inode !== before.planRecord.inode
  ) {
    throw new Error(`release Python binding plan changed ${phase}`);
  }
}

function closeBinding(binding) {
  closeSync(binding.source.fd);
  closeSync(binding.action.fd);
}

function runPinnedReleasePythonAction(args, options = {}) {
  const binding = bindingFromEnvironment();
  const before = readBinding(binding.planPath, binding.expectedSha256);
  let script;
  let proc;
  try {
    script = scriptBinding(args);
    if (options.beforeSpawn) {
      options.beforeSpawn({
        plan: before.plan,
        script: script.record,
      });
    }
    // Re-open and revalidate both executable and script identities after the
    // final synchronous pre-spawn boundary. This prevents a caller-controlled
    // mutation in that boundary from being executed and keeps the post-spawn
    // check as a second, independent mutation detector.
    script.revalidate();
    const immediatelyBeforeSpawn = readBinding(
      binding.planPath,
      binding.expectedSha256,
    );
    try {
      assertBindingUnchanged(
        before,
        immediatelyBeforeSpawn,
        "immediately before spawn",
      );
    } finally {
      closeBinding(immediatelyBeforeSpawn);
    }
    proc = spawnSync(before.plan.action.path, script.argv, {
      cwd: options.cwd ? resolve(options.cwd) : process.cwd(),
      env: sanitizedPythonEnvironment(before.plan.alias),
      encoding: options.capture ? "utf8" : undefined,
      input: script.input,
      stdio: options.capture
        ? undefined
        : [script.input ? "pipe" : "inherit", "inherit", "inherit"],
    });
    try {
      // The child has completed; all held and named identities are checked below.
    } finally {
      script.revalidate();
      let after;
      try {
        after = readBinding(binding.planPath, binding.expectedSha256);
        assertBindingUnchanged(before, after, "after spawn");
      } finally {
        if (after) closeBinding(after);
      }
    }
  } finally {
    if (script) script.cleanup();
    closeBinding(before);
  }
  if (proc.error) throw proc.error;
  if (proc.status !== 0) {
    const detail = options.capture && proc.stderr ? `: ${proc.stderr.trim()}` : "";
    throw new Error(`pinned release Python failed with exit ${proc.status}${detail}`);
  }
  return options.capture ? proc.stdout || "" : "";
}

function cleanupReleasePythonBinding() {
  const binding = bindingFromEnvironment();
  const current = readBinding(binding.planPath, binding.expectedSha256);
  try {
    unlinkSync(current.plan.action.path);
    unlinkSync(binding.planPath);
  } finally {
    closeSync(current.source.fd);
    closeSync(current.action.fd);
  }
}

function cli(argv) {
  const command = argv[0];
  if (command === "bind") {
    const aliasIndex = argv.indexOf("--python");
    if (aliasIndex < 0 || !argv[aliasIndex + 1]) {
      throw new Error("bind requires --python");
    }
    process.stdout.write(`${JSON.stringify(bindReleasePython(argv[aliasIndex + 1]))}\n`);
    return;
  }
  if (command === "run") {
    const cwdIndex = argv.indexOf("--cwd");
    const delimiter = argv.indexOf("--");
    const cwd = cwdIndex >= 0 ? argv[cwdIndex + 1] : process.cwd();
    const pythonArgs = delimiter >= 0 ? argv.slice(delimiter + 1) : [];
    runPinnedReleasePythonAction(pythonArgs, { cwd });
    return;
  }
  if (command === "cleanup") {
    cleanupReleasePythonBinding();
    return;
  }
  throw new Error(`unsupported release Python action: ${command || "missing"}`);
}

if (require.main === module) {
  try {
    cli(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

module.exports = {
  FIXED_PATH,
  bindReleasePython,
  bindingFromEnvironment,
  cleanupReleasePythonBinding,
  readBinding,
  runPinnedReleasePythonAction,
};
