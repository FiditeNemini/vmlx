const {
  verifyR18PackagingContext,
} = require("./electron-builder-before-pack.cjs");
const { resolve } = require("node:path");

function createReleaseArtifactLifecycleHook(
  verifyPackagingContext = verifyR18PackagingContext,
) {
  return async function releaseArtifactLifecycleHook(context) {
  // Electron-builder's BuildResult intentionally has no packager/projectDir.
  // Anchor this hook to its checked-in script location so invoking
  // electron-builder from a parent directory cannot bypass or misdirect the
  // release-source checks. This module is intentionally registered for both
  // artifactBuildStarted and afterAllArtifactBuild: the first blocks a
  // --prepackaged bypass before an artifact is created and the second
  // revalidates the source after all artifact work.
    const panelDir = resolve(__dirname, "..");
    verifyPackagingContext(panelDir, context);
    return [];
  };
}

module.exports = createReleaseArtifactLifecycleHook();
module.exports.createReleaseArtifactLifecycleHook =
  createReleaseArtifactLifecycleHook;
