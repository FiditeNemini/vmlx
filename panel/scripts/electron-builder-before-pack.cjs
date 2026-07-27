const { spawnSync } = require("node:child_process");
const { createHash } = require("node:crypto");
const {
  chmodSync,
  closeSync,
  constants,
  existsSync,
  fchmodSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readSync,
  readdirSync,
  readlinkSync,
  realpathSync,
  rmSync,
  statSync,
  writeSync,
} = require("node:fs");
const { dirname, join, relative, resolve } = require("node:path");
const {
  bindingFromEnvironment,
  readBinding: readReleasePythonBinding,
  runPinnedReleasePythonAction,
} = require("./release-python-action.cjs");

const R19_VERSION = "1.6.19";
const R19_SCOPE = "r19_production";
const R19_TEAM_ID = "55KGF2S5AY";
const R19_CODESIGN_IDENTITY =
  "Developer ID Application: ShieldStack LLC (55KGF2S5AY)";
// electron-builder's CSC_NAME is a certificate selector, not the full
// codesign Authority value. Newer electron-builder releases reject selectors
// that include the "Developer ID Application:" certificate-type prefix.
const R19_CSC_NAME = "ShieldStack LLC (55KGF2S5AY)";
const R19_FIXED_PATH = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin";
const R19_RUNTIME_CONTRACTS = {
  sequoia: {
    mlx_wheel_platform: "macosx_14_0_arm64",
    minimum_system_version: "14.5.0",
  },
  tahoe: {
    mlx_wheel_platform: "macosx_26_0_arm64",
    minimum_system_version: "26.0.0",
  },
};
const R19_PINNED_TOOL_NAMES = [
  "git",
  "node",
  "npm",
  "npx",
  "shasum",
  "awk",
  "file",
  "find",
  "asar",
  "app_builder",
  "electron_builder",
];

function run(cmd, args, cwd) {
  const proc = spawnSync(cmd, args, {
    cwd,
    stdio: "inherit",
    env: process.env,
  });
  if (proc.error) {
    throw proc.error;
  }
  if (proc.status !== 0) {
    throw new Error(`${cmd} ${args.join(" ")} failed with exit ${proc.status}`);
  }
}

function runCapture(cmd, args, cwd) {
  const proc = spawnSync(cmd, args, {
    cwd,
    encoding: "utf8",
    env: process.env,
  });
  if (proc.error) {
    throw proc.error;
  }
  if (proc.status !== 0) {
    throw new Error(
      `${cmd} ${args.join(" ")} failed with exit ${proc.status}: ${(proc.stderr || "").trim()}`,
    );
  }
  return (proc.stdout || "").trim();
}

function readFileSnapshot(filePath, requireSingleLink = true) {
  const fd = openSync(filePath, constants.O_RDONLY | constants.O_NOFOLLOW);
  const digest = createHash("sha256");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  const chunks = [];
  try {
    const before = fstatSync(fd, { bigint: true });
    if (
      !before.isFile() ||
      (requireSingleLink && before.nlink !== 1n)
    ) {
      throw new Error(`release input is not a single-link regular file: ${filePath}`);
    }
    for (;;) {
      const count = readSync(fd, buffer, 0, buffer.length, null);
      if (count === 0) break;
      const chunk = Buffer.from(buffer.subarray(0, count));
      chunks.push(chunk);
      digest.update(chunk);
    }
    const after = fstatSync(fd, { bigint: true });
    if (
      before.dev !== after.dev ||
      before.ino !== after.ino ||
      before.size !== after.size ||
      before.mtimeNs !== after.mtimeNs ||
      before.mode !== after.mode ||
      before.nlink !== after.nlink
    ) {
      throw new Error(`release input changed while it was read: ${filePath}`);
    }
    return {
      record: {
        path: resolve(filePath),
        sha256: digest.digest("hex"),
        size: Number(after.size),
        mode: Number(after.mode & 0o7777n),
      },
      bytes: Buffer.concat(chunks),
      identity: {
        device: before.dev,
        inode: before.ino,
      },
    };
  } finally {
    closeSync(fd);
  }
}

function fileIdentity(filePath, requireSingleLink = true) {
  return readFileSnapshot(filePath, requireSingleLink).record;
}

function sha256File(filePath) {
  return fileIdentity(filePath).sha256;
}

function readBoundJson(
  filePath,
  expectedSha256,
  label,
  requireSingleLink = true,
) {
  if (!/^[0-9a-f]{64}$/.test(expectedSha256 || "")) {
    throw new Error(`${label} independently supplied SHA-256 is invalid`);
  }
  const snapshot = readFileSnapshot(filePath, requireSingleLink);
  if (snapshot.record.sha256 !== expectedSha256) {
    throw new Error(`${label} SHA-256 does not match`);
  }
  let value;
  try {
    value = JSON.parse(snapshot.bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return { ...snapshot, value };
}

function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function treePayload(rootPath) {
  const root = resolve(rootPath);
  const rootStat = lstatSync(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new Error(`release payload root is not a real directory: ${root}`);
  }
  const entries = {};
  function walk(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort(
      (left, right) => left.name.localeCompare(right.name),
    )) {
      const path = join(directory, entry.name);
      const name = relative(root, path).split("\\").join("/");
      const metadata = lstatSync(path);
      const mode = metadata.mode & 0o7777;
      if (metadata.isSymbolicLink()) {
        entries[name] = { kind: "symlink", target: readlinkSync(path), mode };
      } else if (metadata.isDirectory()) {
        entries[name] = { kind: "directory", mode };
        walk(path);
      } else if (metadata.isFile()) {
        const identity = fileIdentity(path);
        entries[name] = {
          kind: "file",
          sha256: identity.sha256,
          size: identity.size,
          mode,
        };
      } else {
        throw new Error(`unsupported release payload entry: ${path}`);
      }
    }
  }
  walk(root);
  const rootMode = rootStat.mode & 0o7777;
  const encoded = canonicalJson({ root_mode: rootMode, entries });
  return {
    root_mode: rootMode,
    entry_count: Object.keys(entries).length,
    tree_sha256: createHash("sha256").update(encoded).digest("hex"),
    entries,
  };
}

function exactObjectKeys(value, expected) {
  return (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.keys(value).sort().join(",") === [...expected].sort().join(",")
  );
}

function metadataField(text, name) {
  const prefix = `${name}:`;
  const line = text
    .split(/\r?\n/)
    .find((candidate) => candidate.startsWith(prefix));
  return line ? line.slice(prefix.length).trim() : "";
}

function wheelDistributionRecord(sitePackages, distribution, expectedPlatform) {
  const prefix = distribution === "mlx" ? "mlx-" : "mlx_metal-";
  const candidates = readdirSync(sitePackages, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.name.startsWith(prefix) &&
        entry.name.endsWith(".dist-info") &&
        entry.isDirectory() &&
        !entry.isSymbolicLink(),
    )
    .map((entry) => join(sitePackages, entry.name));
  if (candidates.length !== 1) {
    throw new Error(
      `vMLX ${R19_VERSION} bundled runtime must contain one ${distribution} dist-info`,
    );
  }
  const distInfo = candidates[0];
  const metadataPath = join(distInfo, "METADATA");
  const wheelPath = join(distInfo, "WHEEL");
  const metadataIdentity = fileIdentity(metadataPath);
  const wheelIdentity = fileIdentity(wheelPath);
  const metadata = readFileSync(metadataPath, "utf8");
  const wheel = readFileSync(wheelPath, "utf8");
  const canonicalName = metadataField(metadata, "Name")
    .toLowerCase()
    .replaceAll("_", "-");
  const version = metadataField(metadata, "Version");
  const tags = [
    ...new Set(
      wheel
        .split(/\r?\n/)
        .filter((line) => line.startsWith("Tag:"))
        .map((line) => line.slice(4).trim()),
    ),
  ].sort();
  const platforms = new Set(
    tags.map((tag) => tag.slice(tag.lastIndexOf("-") + 1)),
  );
  if (
    canonicalName !== distribution ||
    !version ||
    tags.length === 0 ||
    tags.some((tag) => tag.split("-").length < 3) ||
    platforms.size !== 1 ||
    !platforms.has(expectedPlatform)
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} bundled ${distribution} wheel identity is not ${expectedPlatform}`,
    );
  }
  return {
    distribution,
    version,
    dist_info: distInfo.slice(distInfo.lastIndexOf("/") + 1),
    tags,
    metadata_sha256: metadataIdentity.sha256,
    wheel_sha256: wheelIdentity.sha256,
  };
}

function inspectBundleRuntimeContract(bundleRoot, flavor, version = R19_VERSION) {
  const expected = R19_RUNTIME_CONTRACTS[flavor];
  if (!expected) {
    throw new Error(`vMLX ${R19_VERSION} unsupported runtime flavor ${flavor}`);
  }
  const pythonLib = join(bundleRoot, "python", "lib");
  const siteCandidates = readdirSync(pythonLib, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.name.startsWith("python") &&
        entry.isDirectory() &&
        !entry.isSymbolicLink() &&
        existsSync(join(pythonLib, entry.name, "site-packages")),
    )
    .map((entry) => join(pythonLib, entry.name, "site-packages"));
  if (siteCandidates.length !== 1) {
    throw new Error(
      `vMLX ${R19_VERSION} ${flavor} runtime must contain one site-packages`,
    );
  }
  const provenancePath = join(bundleRoot, "vmlx-bundle-provenance.json");
  const provenanceIdentity = fileIdentity(provenancePath);
  const provenance = JSON.parse(readFileSync(provenancePath, "utf8"));
  if (
    !exactObjectKeys(provenance, [
      "schema_version",
      "vmlx",
      "jang",
      "mlx_wheel_platform",
    ]) ||
    provenance.schema_version !== 1 ||
    provenance?.vmlx?.version !== version ||
    provenance.mlx_wheel_platform !== expected.mlx_wheel_platform
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} ${flavor} bundle provenance is invalid`,
    );
  }
  const distributions = {
    mlx: wheelDistributionRecord(
      siteCandidates[0],
      "mlx",
      expected.mlx_wheel_platform,
    ),
    "mlx-metal": wheelDistributionRecord(
      siteCandidates[0],
      "mlx-metal",
      expected.mlx_wheel_platform,
    ),
  };
  if (distributions.mlx.version !== distributions["mlx-metal"].version) {
    throw new Error(
      `vMLX ${R19_VERSION} ${flavor} mlx wheel versions differ`,
    );
  }
  return {
    flavor,
    mlx_wheel_platform: expected.mlx_wheel_platform,
    minimum_system_version: expected.minimum_system_version,
    mlx_version: distributions.mlx.version,
    bundle_provenance_sha256: provenanceIdentity.sha256,
    bundle_provenance: provenance,
    distributions,
  };
}

function inspectAppRuntimeContract(app, flavor) {
  const expected = R19_RUNTIME_CONTRACTS[flavor];
  const infoPath = join(app, "Contents", "Info.plist");
  const infoIdentity = fileIdentity(infoPath);
  const info = readFileSync(infoPath, "utf8");
  const plistString = (key) => {
    const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = new RegExp(
      `<key>${escaped}</key>\\s*<string>([^<]+)</string>`,
    ).exec(info);
    return match ? match[1] : "";
  };
  if (
    !expected ||
    plistString("CFBundleIdentifier") !== "net.vmlx.app" ||
    plistString("CFBundleShortVersionString") !== R19_VERSION ||
    plistString("CFBundleVersion") !== R19_VERSION ||
    plistString("LSMinimumSystemVersion") !== expected.minimum_system_version
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} ${flavor} app identity/version/minimum-system contract is not exact`,
    );
  }
  return {
    ...inspectBundleRuntimeContract(
      join(app, "Contents", "Resources", "bundled-python"),
      flavor,
    ),
    info_plist_sha256: infoIdentity.sha256,
  };
}

function validatePlanBundleRuntime(plan) {
  const expected = R19_RUNTIME_CONTRACTS[plan.current_flavor];
  if (
    !expected ||
    !exactObjectKeys(plan.flavor_contract, [
      "mlx_wheel_platform",
      "minimum_system_version",
    ]) ||
    canonicalJson(plan.flavor_contract) !== canonicalJson(expected) ||
    !exactObjectKeys(plan.bundle_runtime, ["path", "sha256"]) ||
    !/^[0-9a-f]{64}$/.test(plan.bundle_runtime.sha256 || "")
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} release plan runtime contract is invalid`,
    );
  }
  const boundRuntime = readBoundJson(
    plan.bundle_runtime.path,
    plan.bundle_runtime.sha256,
    `vMLX ${R19_VERSION} sealed bundle runtime`,
  );
  const record = boundRuntime.record;
  const payload = boundRuntime.value;
  if (
    record.sha256 !== plan.bundle_runtime.sha256 ||
    payload?.schema_version !== 1 ||
    payload?.scope !== R19_SCOPE ||
    payload?.stage !== "bundle_runtime" ||
    payload?.version !== R19_VERSION ||
    payload?.flavor !== plan.current_flavor ||
    payload?.source?.commit !== plan.source_commit ||
    payload?.source?.tree !== plan.source_tree ||
    canonicalJson(payload.runtime_contract) !==
      canonicalJson({
        ...payload.runtime_contract,
        ...expected,
      })
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} sealed bundle runtime does not match the release plan`,
    );
  }
  return payload;
}

function verifyPinnedToolchain(plan) {
  if (
    plan.fixed_path !== R19_FIXED_PATH ||
    process.env.PATH !== plan.fixed_path ||
    process.env.VMLX_R19_FIXED_PATH !== plan.fixed_path ||
    !plan.tools ||
    Object.keys(plan.tools).sort().join(",") !==
      [...R19_PINNED_TOOL_NAMES].sort().join(",")
  ) {
    throw new Error(`vMLX ${R19_VERSION} release toolchain or fixed PATH changed`);
  }
  for (const name of R19_PINNED_TOOL_NAMES) {
    const tool = plan.tools[name];
    const envPrefix = `VMLX_R19_TOOL_${name.toUpperCase()}`;
    if (
      !tool ||
      process.env[`${envPrefix}_PATH`] !== tool.path ||
      process.env[`${envPrefix}_REALPATH`] !== tool.realpath ||
      process.env[`${envPrefix}_SHA256`] !== tool.sha256 ||
      resolve(tool.realpath) !== tool.realpath ||
      realpathSync(tool.path) !== tool.realpath ||
      fileIdentity(tool.realpath, false).sha256 !== tool.sha256 ||
      (statSync(tool.realpath).mode & 0o111) === 0
    ) {
      throw new Error(`vMLX ${R19_VERSION} pinned ${name} identity changed`);
    }
  }
  return true;
}

function readBoundReleasePlanForAction(panelDir, expectedPlanSha256) {
  const rootDir = resolve(panelDir, "..");
  const planPath = join(rootDir, "build", "r19-release-driver-plan.json");
  requireExactEnv("VMLX_R19_RELEASE_PLAN", planPath);
  requireExactEnv("VMLX_R19_RELEASE_PLAN_SHA256", expectedPlanSha256);
  const bound = readBoundJson(
    planPath,
    expectedPlanSha256,
    `vMLX ${R19_VERSION} release-driver plan`,
  );
  verifyPinnedToolchain(bound.value);
  return { ...bound, planPath };
}

function runPlanToolAction(
  panelDir,
  expectedPlanSha256,
  action,
  args,
  options = {},
) {
  const before = readBoundReleasePlanForAction(panelDir, expectedPlanSha256);
  const tools = before.value.tools;
  const argv = Array.isArray(args) ? args.map((value) => String(value)) : [];
  const actionCommands = {
    node: [tools.node.realpath, argv],
    npm: [tools.node.realpath, [tools.npm.realpath, ...argv]],
    npx: [tools.node.realpath, [tools.npx.realpath, ...argv]],
    asar: [tools.node.realpath, [tools.asar.realpath, ...argv]],
    "app-builder": [tools.app_builder.realpath, argv],
    "electron-builder": [
      tools.node.realpath,
      [tools.electron_builder.realpath, ...argv],
    ],
    git: [tools.git.realpath, argv],
  };
  const command = actionCommands[action];
  if (!command) {
    throw new Error(`unsupported vMLX ${R19_VERSION} pinned-tool action: ${action}`);
  }

  let proc;
  let invocationError;
  try {
    proc = spawnSync(command[0], command[1], {
      cwd: options.cwd ? resolve(options.cwd) : resolve(panelDir),
      encoding: options.capture ? "utf8" : undefined,
      stdio: options.capture ? undefined : "inherit",
      env: process.env,
    });
    invocationError = proc.error;
  } finally {
    const after = readBoundReleasePlanForAction(panelDir, expectedPlanSha256);
    if (
      before.identity.device !== after.identity.device ||
      before.identity.inode !== after.identity.inode ||
      before.record.sha256 !== after.record.sha256
    ) {
      throw new Error(
        `vMLX ${R19_VERSION} release-driver plan identity changed across pinned-tool action`,
      );
    }
  }

  if (invocationError) {
    throw invocationError;
  }
  if (!proc || proc.status !== 0) {
    throw new Error(
      `vMLX ${R19_VERSION} pinned-tool action ${action} failed with exit ${
        proc ? proc.status : "unknown"
      }: ${proc && proc.stderr ? proc.stderr.trim() : ""}`,
    );
  }
  return {
    action,
    returncode: proc.status,
    stdout: options.capture ? (proc.stdout || "").trim() : "",
    stderr: options.capture ? (proc.stderr || "").trim() : "",
    planSha256: expectedPlanSha256,
  };
}

function writeExclusiveSealedJson(outputPath, payload) {
  const encoded = Buffer.from(`${JSON.stringify(payload, null, 2)}\n`);
  const fd = openSync(
    outputPath,
    constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL,
    0o400,
  );
  try {
    let offset = 0;
    while (offset < encoded.length) {
      offset += writeSync(fd, encoded, offset, encoded.length - offset);
    }
    fsyncSync(fd);
    fchmodSync(fd, 0o400);
  } finally {
    closeSync(fd);
  }
  return {
    path: resolve(outputPath),
    sha256: createHash("sha256").update(encoded).digest("hex"),
  };
}

function assertExactResolvedPathSet(
  actualPaths,
  expectedPaths,
  errorMessage = "artifact paths do not match",
) {
  if (!Array.isArray(actualPaths) || !Array.isArray(expectedPaths)) {
    throw new Error(errorMessage);
  }
  const actual = actualPaths.map((artifact) => resolve(artifact)).sort();
  const expected = expectedPaths.map((artifact) => resolve(artifact)).sort();
  if (
    actual.length !== expected.length ||
    actual.some((artifact, index) => artifact !== expected[index])
  ) {
    throw new Error(errorMessage);
  }
  return actual;
}

function verifyExactDmgDirectory(
  distDir,
  expectedPaths,
  errorMessage = "DMG paths do not match",
) {
  const resolvedDistDir = resolve(distDir);
  const dmgEntries = readdirSync(resolvedDistDir, {
    withFileTypes: true,
  }).filter((entry) => entry.name.toLowerCase().endsWith(".dmg"));
  if (dmgEntries.some((entry) => !entry.isFile())) {
    throw new Error(errorMessage);
  }
  return assertExactResolvedPathSet(
    dmgEntries.map((entry) => join(resolvedDistDir, entry.name)),
    expectedPaths,
    errorMessage,
  );
}

function canonicalGitHubRepo(rawUrl) {
  const value = String(rawUrl || "").trim();
  const scp = /^git@github\.com:([^/:\s]+\/[^/:\s]+?)(?:\.git)?$/i.exec(value);
  if (scp) {
    return scp[1].toLowerCase();
  }

  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return "";
  }
  const isHttps =
    parsed.protocol === "https:" &&
    !parsed.username &&
    !parsed.password &&
    !parsed.port;
  const isSsh =
    parsed.protocol === "ssh:" &&
    parsed.username === "git" &&
    !parsed.password &&
    !parsed.port;
  if (
    (!isHttps && !isSsh) ||
    parsed.hostname.toLowerCase() !== "github.com" ||
    parsed.search ||
    parsed.hash
  ) {
    return "";
  }
  const repoPath = parsed.pathname
    .replace(/^\/+|\/+$/g, "")
    .replace(/\.git$/i, "");
  if (!/^[^/:\s]+\/[^/:\s]+$/.test(repoPath)) {
    return "";
  }
  return repoPath.toLowerCase();
}

function requireExactEnv(name, expected) {
  const actual = process.env[name];
  if (actual !== expected) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging requires ${name}=${expected}`,
    );
  }
}

function verifyR19ReleasePlan(
  rootDir,
  manifestSha256,
  sourceCommit,
  sourceTree,
  context,
) {
  const planPath = join(rootDir, "build", "r19-release-driver-plan.json");
  requireExactEnv("VMLX_R19_RELEASE_PLAN", planPath);
  const expectedPlanHash = process.env.VMLX_R19_RELEASE_PLAN_SHA256;
  if (!expectedPlanHash || !/^[0-9a-f]{64}$/.test(expectedPlanHash)) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging requires a hashed release-driver plan`,
    );
  }
  const boundPlan = readBoundJson(
    planPath,
    expectedPlanHash,
    `vMLX ${R19_VERSION} release-driver plan`,
  );
  const plan = boundPlan.value;
  const driverPid = Number(process.env.VMLX_R19_RELEASE_DRIVER_PID);
  const expectedArtifact = resolve(
    process.env.VMLX_R19_RELEASE_EXPECTED_ARTIFACT || "",
  );
  const expectedFlavor = process.env.VMLX_R19_RELEASE_CURRENT_FLAVOR;
  const expectedPhase = process.env.VMLX_R19_RELEASE_PHASE;
  const expectedNonce = process.env.VMLX_R19_RELEASE_DRIVER_NONCE;
  const hookDirectory = resolve(
    process.env.VMLX_R19_HOOK_ATTESTATION_DIR || "",
  );
  const expectedStagedApp =
    expectedPhase === "stage"
      ? join(expectedArtifact, "mac-arm64", "vMLX.app")
      : join(
          dirname(expectedArtifact),
          `${expectedFlavor}-app`,
          "mac-arm64",
          "vMLX.app",
        );
  const expectedHookAttestation = join(
    hookDirectory,
    `${expectedFlavor}.completion.json`,
  );
  const expectedBundleRuntime = join(
    hookDirectory,
    `${expectedFlavor}.bundle-runtime.json`,
  );
  requireExactEnv("VMLX_R19_RELEASE_REQUESTED_FLAVOR", "all");
  if (
    !Number.isSafeInteger(driverPid) ||
    driverPid <= 1 ||
    plan.schema_version !== 3 ||
    plan.scope !== R19_SCOPE ||
    plan.version !== R19_VERSION ||
    plan.source_commit !== sourceCommit ||
    plan.source_tree !== sourceTree ||
    plan.manifest_sha256 !== manifestSha256 ||
    plan.requested_flavor !== "all" ||
    plan.current_flavor !== expectedFlavor ||
    !["sequoia", "tahoe"].includes(expectedFlavor) ||
    plan.phase !== expectedPhase ||
    !["stage", "dmg"].includes(expectedPhase) ||
    resolve(plan.expected_artifact || "") !== expectedArtifact ||
    resolve(plan.staged_app || "") !== expectedStagedApp ||
    resolve(plan.hook_attestation || "") !== expectedHookAttestation ||
    resolve(plan?.bundle_runtime?.path || "") !== expectedBundleRuntime ||
    plan.driver_pid !== driverPid ||
    plan.nonce !== expectedNonce ||
    !/^[0-9a-f]{64}$/.test(plan.nonce || "")
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} release-driver plan does not match this artifact phase`,
    );
  }
  verifyPinnedToolchain(plan);
  validatePlanBundleRuntime(plan);
  try {
    process.kill(driverPid, 0);
  } catch {
    throw new Error(`vMLX ${R19_VERSION} release-driver process is not active`);
  }

  const isPackContext =
    context &&
    context.packager &&
    typeof context.packager.projectDir === "string" &&
    typeof context.outDir === "string" &&
    typeof context.appOutDir === "string" &&
    typeof context.electronPlatformName === "string" &&
    Number.isSafeInteger(context.arch) &&
    Array.isArray(context.targets);

  const isArtifactStart =
    context &&
    typeof context.targetPresentableName === "string" &&
    typeof context.file === "string" &&
    Object.prototype.hasOwnProperty.call(context, "arch") &&
    (context.arch === null || Number.isSafeInteger(context.arch));

  const isBuildResult =
    context &&
    typeof context.outDir === "string" &&
    Array.isArray(context.artifactPaths) &&
    context.platformToTargets instanceof Map &&
    context.configuration &&
    typeof context.configuration === "object";

  const matchedContextKinds = [
    isPackContext && "beforePack",
    isArtifactStart && "artifactBuildStarted",
    isBuildResult && "afterAllArtifactBuild",
  ].filter(Boolean);
  if (matchedContextKinds.length !== 1) {
    throw new Error(
      `vMLX ${R19_VERSION} release-driver received an unrecognized or ambiguous electron-builder hook context`,
    );
  }

  if (matchedContextKinds[0] === "beforePack") {
    if (
      expectedPhase !== "stage" ||
      resolve(context.outDir) !== expectedArtifact
    ) {
      throw new Error(
        `vMLX ${R19_VERSION} app staging is outside the release-driver plan`,
      );
    }
  } else if (matchedContextKinds[0] === "artifactBuildStarted") {
    if (
      expectedPhase !== "dmg" ||
      context.targetPresentableName !== "DMG" ||
      resolve(context.file) !== expectedArtifact
    ) {
      throw new Error(
        `vMLX ${R19_VERSION} artifact build is outside the release-driver plan`,
      );
    }
  } else {
    const expectedOutDir =
      expectedPhase === "stage" ? expectedArtifact : dirname(expectedArtifact);
    if (resolve(context.outDir) !== expectedOutDir) {
      throw new Error(
        `vMLX ${R19_VERSION} completed output directory is outside the release-driver plan`,
      );
    }
    assertExactResolvedPathSet(
      context.artifactPaths,
      expectedPhase === "stage"
        ? []
        : [expectedArtifact, `${expectedArtifact}.blockmap`],
      `vMLX ${R19_VERSION} completed artifacts do not exactly match the release-driver plan`,
    );
  }

  return plan;
}

function emitR19CompletionAttestation(panelDir, plan, planSha256) {
  if (plan.phase !== "dmg") {
    return null;
  }
  const outputPath = resolve(plan.hook_attestation);
  const outputDirectory = dirname(outputPath);
  const configuredDirectory = resolve(
    process.env.VMLX_R19_HOOK_ATTESTATION_DIR || "",
  );
  if (
    outputDirectory !== configuredDirectory ||
    existsSync(outputPath) ||
    !lstatSync(outputDirectory).isDirectory() ||
    lstatSync(outputDirectory).isSymbolicLink() ||
    (statSync(outputDirectory).mode & 0o077) !== 0
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} hook attestation output is reused or not private`,
    );
  }
  const app = resolve(plan.staged_app);
  const dmg = resolve(plan.expected_artifact);
  const blockmap = `${dmg}.blockmap`;
  if (
    !lstatSync(app).isDirectory() ||
    lstatSync(app).isSymbolicLink() ||
    !existsSync(join(app, "Contents", "Resources", "app.asar"))
  ) {
    throw new Error(`vMLX ${R19_VERSION} staged application is missing or wrong`);
  }
  const extracted = mkdtempSync(
    join(outputDirectory, `.${plan.current_flavor}.asar.`),
  );
  try {
    runPlanToolAction(
      panelDir,
      planSha256,
      "asar",
      ["extract", join(app, "Contents", "Resources", "app.asar"), extracted],
      { cwd: panelDir },
    );
    const bundleRuntime = validatePlanBundleRuntime(plan);
    const runtimeContract = inspectAppRuntimeContract(
      app,
      plan.current_flavor,
    );
    const runtimeWithoutInfo = { ...runtimeContract };
    delete runtimeWithoutInfo.info_plist_sha256;
    if (
      canonicalJson(runtimeWithoutInfo) !==
      canonicalJson(bundleRuntime.runtime_contract)
    ) {
      throw new Error(
        `vMLX ${R19_VERSION} staged runtime differs from sealed bundle provenance`,
      );
    }
    const payload = {
      schema_version: 1,
      scope: R19_SCOPE,
      stage: "electron_builder_completion",
      version: R19_VERSION,
      flavor: plan.current_flavor,
      source: {
        commit: plan.source_commit,
        tree: plan.source_tree,
      },
      preflight_sha256: plan.manifest_sha256,
      plan: {
        path: resolve(process.env.VMLX_R19_RELEASE_PLAN || ""),
        sha256: planSha256,
        nonce: plan.nonce,
        driver_pid: plan.driver_pid,
      },
      fixed_path: plan.fixed_path,
      tools: plan.tools,
      bundle_runtime: plan.bundle_runtime,
      runtime_contract: runtimeContract,
      staged_app: {
        path: app,
        payload: treePayload(app),
      },
      extracted_asar: {
        payload: treePayload(extracted),
      },
      artifacts: {
        dmg: fileIdentity(dmg),
        blockmap: fileIdentity(blockmap),
      },
    };
    return {
      ...writeExclusiveSealedJson(outputPath, payload),
      payload,
    };
  } finally {
    rmSync(extracted, { recursive: true, force: true });
  }
}

function verifyR19PackagingContext(panelDir, context) {
  const packagePath = join(panelDir, "package.json");
  if (!existsSync(packagePath)) {
    throw new Error(`Missing package metadata: ${packagePath}`);
  }
  const packageJson = JSON.parse(readFileSync(packagePath, "utf8"));
  if (packageJson.version !== R19_VERSION) {
    if (
      process.env.VMLX_RELEASE_SCOPE === R19_SCOPE ||
      process.env.VMLINUX_RELEASE_SCOPE === R19_SCOPE ||
      process.env.VMLX_R19_OFFICIAL_PACKAGING === "1"
    ) {
      throw new Error(
        `VMLX_RELEASE_SCOPE=${R19_SCOPE} requires package version ${R19_VERSION}, found ${packageJson.version}`,
      );
    }
    return { guarded: false, version: packageJson.version };
  }

  const rootDir = resolve(panelDir, "..");
  requireExactEnv("VMLX_RELEASE_SCOPE", R19_SCOPE);
  requireExactEnv("VMLX_R19_OFFICIAL_PACKAGING", "1");
  requireExactEnv("VMLX_R19_EXPECTED_TEAM_ID", R19_TEAM_ID);
  requireExactEnv("VMLX_R19_EXPECTED_CODESIGN_IDENTITY", R19_CODESIGN_IDENTITY);
  requireExactEnv("CSC_NAME", R19_CSC_NAME);

  const configuredTeam = packageJson?.build?.mac?.notarize?.teamId;
  if (configuredTeam !== R19_TEAM_ID) {
    throw new Error(
      `vMLX ${R19_VERSION} package notarization team must be ${R19_TEAM_ID}`,
    );
  }

  const manifestPathRaw = process.env.VMLX_R19_PREPACKAGE_MANIFEST;
  const expectedManifestHash = process.env.VMLX_R19_PREPACKAGE_MANIFEST_SHA256;
  if (!manifestPathRaw || !expectedManifestHash) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging requires the official prepackage manifest and SHA-256`,
    );
  }
  const manifestPath = resolve(manifestPathRaw);
  const boundManifest = readBoundJson(
    manifestPath,
    expectedManifestHash,
    `vMLX ${R19_VERSION} prepackage manifest`,
  );
  const actualManifestHash = boundManifest.record.sha256;
  const manifest = boundManifest.value;
  if (
    manifest.status !== "pass" ||
    manifest.scope !== R19_SCOPE ||
    manifest.version !== R19_VERSION
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} prepackage manifest is not a passing ${R19_SCOPE} manifest`,
    );
  }

  const plan = verifyR19ReleasePlan(
    rootDir,
    actualManifestHash,
    manifest?.source?.commit,
    manifest?.source?.tree,
    context,
  );
  const planSha256 = process.env.VMLX_R19_RELEASE_PLAN_SHA256;
  const runPinnedGit = (args) =>
    runPlanToolAction(panelDir, planSha256, "git", args, {
      cwd: rootDir,
      capture: true,
    }).stdout;
  const head = runPinnedGit(["-C", rootDir, "rev-parse", "HEAD"]);
  const tree = runPinnedGit(["-C", rootDir, "rev-parse", "HEAD^{tree}"]);
  const dirty = runPinnedGit([
    "-C",
    rootDir,
    "status",
    "--porcelain",
    "--untracked-files=all",
  ]);
  if (manifest?.source?.commit !== head || manifest?.source?.tree !== tree) {
    throw new Error(
      `vMLX ${R19_VERSION} prepackage manifest is not bound to the packaging source`,
    );
  }
  if (dirty) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging source has uncommitted or untracked files`,
    );
  }
  const upstream = runPinnedGit(["-C", rootDir, "rev-parse", "@{upstream}"]);
  if (upstream !== head || manifest?.source?.upstream_commit !== upstream) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging source is not the attested pushed revision`,
    );
  }
  const origin = runPinnedGit(["-C", rootDir, "remote", "get-url", "origin"]);
  const originIdentity = canonicalGitHubRepo(origin);
  if (
    originIdentity !== "jjang-ai/vmlx" ||
    manifest?.source?.remote_identity !== originIdentity
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging source is not the canonical HTTPS/SSH repository`,
    );
  }
  const remoteMainOutput = runPinnedGit([
    "-C",
    rootDir,
    "ls-remote",
    "--exit-code",
    "origin",
    "refs/heads/main",
  ]);
  const remoteMain = remoteMainOutput.split(/\s+/)[0] || "";
  if (
    remoteMain !== head ||
    manifest?.source?.remote_main_commit !== remoteMain
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} packaging source is not the live canonical main revision`,
    );
  }
  const authoritativePython = join(rootDir, ".venv", "bin", "python");
  requireExactEnv("VMLX_R19_RELEASE_PYTHON", authoritativePython);
  if (!existsSync(authoritativePython)) {
    throw new Error(
      `Missing authoritative release Python: ${authoritativePython}`,
    );
  }
  const pythonBindingEnv = bindingFromEnvironment();
  const validatedPython = readReleasePythonBinding(
    pythonBindingEnv.planPath,
    pythonBindingEnv.expectedSha256,
  );
  const pythonPlan = validatedPython.plan;
  closeSync(validatedPython.source.fd);
  closeSync(validatedPython.action.fd);
  const probeSource = [
    "import importlib.util, json, pathlib, sys, vmlx_engine",
    "root = pathlib.Path.cwd().resolve()",
    "init_path = pathlib.Path(vmlx_engine.__file__).resolve()",
    'server_spec = importlib.util.find_spec("vmlx_engine.server")',
    "server_path = pathlib.Path(server_spec.origin).resolve() if server_spec and server_spec.origin else None",
    "print(json.dumps({",
    '    "executable": sys.executable,',
    '    "prefix": sys.prefix,',
    '    "init_path": str(init_path),',
    '    "server_path": str(server_path) if server_path else "",',
    "}))",
  ].join("\n");
  const probe = JSON.parse(
    runR19ReleasePythonAction(["-I", "-c", probeSource], {
      cwd: rootDir,
      capture: true,
    }),
  );
  const expectedInit = join(rootDir, "vmlx_engine", "__init__.py");
  const expectedServer = join(rootDir, "vmlx_engine", "server.py");
  const expectedPrefix = join(rootDir, ".venv");
  if (
    resolve(probe.executable) !== resolve(authoritativePython) ||
    resolve(pythonPlan.alias) !== resolve(authoritativePython) ||
    realpathSync(authoritativePython) !== pythonPlan.source.path ||
    resolve(probe.prefix) !== expectedPrefix ||
    resolve(probe.init_path) !== expectedInit ||
    resolve(probe.server_path) !== expectedServer
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} release Python does not import the packaging source`,
    );
  }
  const initHash = sha256File(expectedInit);
  const serverHash = sha256File(expectedServer);
  if (
    process.env.VMLX_R19_RELEASE_PYTHON_EXECUTABLE_SHA256 !==
      pythonPlan.source.sha256 ||
    process.env.VMLX_R19_RELEASE_PYTHON_PYVENV_SHA256 !==
      pythonPlan.pyvenv.sha256 ||
    process.env.VMLX_R19_RELEASE_PYTHON_INIT_SHA256 !== initHash ||
    process.env.VMLX_R19_RELEASE_PYTHON_SERVER_SHA256 !== serverHash
  ) {
    throw new Error(
      `vMLX ${R19_VERSION} release Python source hashes do not match`,
    );
  }

  const isBuildResult =
    context &&
    typeof context.outDir === "string" &&
    Array.isArray(context.artifactPaths) &&
    context.platformToTargets instanceof Map &&
    context.configuration &&
    typeof context.configuration === "object";
  let stagedRuntimeContract = null;
  if (isBuildResult && plan.phase === "stage") {
    const sealedBundle = validatePlanBundleRuntime(plan);
    stagedRuntimeContract = inspectAppRuntimeContract(
      plan.staged_app,
      plan.current_flavor,
    );
    const stagedWithoutInfo = { ...stagedRuntimeContract };
    delete stagedWithoutInfo.info_plist_sha256;
    if (
      canonicalJson(stagedWithoutInfo) !==
      canonicalJson(sealedBundle.runtime_contract)
    ) {
      throw new Error(
        `vMLX ${R19_VERSION} staged runtime differs from sealed bundle provenance`,
      );
    }
  }
  const completionAttestation =
    isBuildResult && plan.phase === "dmg"
      ? emitR19CompletionAttestation(
          panelDir,
          plan,
          process.env.VMLX_R19_RELEASE_PLAN_SHA256,
        )
      : null;

  return {
    guarded: true,
    version: packageJson.version,
    manifestSha256: actualManifestHash,
    sourceCommit: head,
    sourceTree: tree,
    python: authoritativePython,
    initSha256: initHash,
    serverSha256: serverHash,
    plan,
    stagedRuntimeContract,
    completionAttestation,
  };
}

function runR19ReleasePythonAction(args, options = {}) {
  return runPinnedReleasePythonAction(args, options);
}

async function beforePack(context) {
  const isElectronBuilderPack = !!(
    context &&
    context.packager &&
    context.packager.projectDir
  );
  const panelDir = isElectronBuilderPack
    ? context.packager.projectDir
    : process.cwd();

  const packaging = isElectronBuilderPack
    ? verifyR19PackagingContext(panelDir, context)
    : { guarded: false };

  const verifyScript = join(panelDir, "scripts", "verify-bundled-python.sh");
  if (!existsSync(verifyScript)) {
    throw new Error(`Missing bundled-python verifier: ${verifyScript}`);
  }

  run("/bin/bash", [verifyScript], panelDir);

  if (process.env.VMLX_BEFORE_PACK_SKIP_VITE === "1") {
    if (isElectronBuilderPack) {
      throw new Error(
        "VMLX_BEFORE_PACK_SKIP_VITE is only allowed for direct hook smoke tests, not electron-builder packaging",
      );
    }
    console.log("VMLX_BEFORE_PACK_SKIP_VITE=1: skipped electron-vite build");
    return;
  }

  if (packaging.guarded) {
    runPlanToolAction(
      panelDir,
      process.env.VMLX_R19_RELEASE_PLAN_SHA256,
      "npx",
      ["--no-install", "electron-vite", "build"],
      { cwd: panelDir },
    );
  } else {
    run(
      "/opt/homebrew/bin/npx",
      ["--no-install", "electron-vite", "build"],
      panelDir,
    );
  }
}

module.exports = beforePack;
module.exports.assertExactResolvedPathSet = assertExactResolvedPathSet;
module.exports.canonicalGitHubRepo = canonicalGitHubRepo;
module.exports.verifyExactDmgDirectory = verifyExactDmgDirectory;
module.exports.verifyR19ReleasePlan = verifyR19ReleasePlan;
module.exports.verifyR19PackagingContext = verifyR19PackagingContext;
module.exports.verifyPinnedToolchain = verifyPinnedToolchain;
module.exports.runPlanToolAction = runPlanToolAction;
module.exports.inspectBundleRuntimeContract = inspectBundleRuntimeContract;
module.exports.inspectAppRuntimeContract = inspectAppRuntimeContract;
module.exports.emitR19CompletionAttestation = emitR19CompletionAttestation;
module.exports.runR19ReleasePythonAction = runR19ReleasePythonAction;
module.exports.treePayload = treePayload;
module.exports.R19_CODESIGN_IDENTITY = R19_CODESIGN_IDENTITY;
module.exports.R19_TEAM_ID = R19_TEAM_ID;

if (require.main === module) {
  beforePack().catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
  });
}
