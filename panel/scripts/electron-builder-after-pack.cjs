const { spawnSync } = require('node:child_process')
const {
  chmodSync,
  existsSync,
  lstatSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} = require('node:fs')
const { basename, dirname, join } = require('node:path')

function walk(dir, out = []) {
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, ent.name)
    if (ent.isDirectory()) {
      walk(path, out)
    } else if (ent.isFile()) {
      out.push(path)
    }
  }
  return out
}

function isMachO(path) {
  const proc = spawnSync('file', ['-b', path], {
    stdio: 'pipe',
    encoding: 'utf8',
  })
  return proc.status === 0 && (proc.stdout || '').includes('Mach-O')
}

function removeSignature(path) {
  const proc = spawnSync('codesign', ['--remove-signature', path], {
    stdio: 'pipe',
    encoding: 'utf8',
  })
  // Unsigned files are fine; this hook only normalizes wheels that already
  // carry upstream signatures before electron-builder signs the whole app.
  const output = `${proc.stdout || ''}${proc.stderr || ''}`
  if (proc.status !== 0 && !output.includes('code object is not signed at all')) {
    throw new Error(`codesign --remove-signature failed for ${path}\n${output}`)
  }
}

function signAdhoc(path) {
  const proc = spawnSync('codesign', ['--force', '--sign', '-', '--timestamp=none', path], {
    stdio: 'pipe',
    encoding: 'utf8',
  })
  const output = `${proc.stdout || ''}${proc.stderr || ''}`
  if (proc.status !== 0) {
    throw new Error(`codesign --sign - failed for ${path}\n${output}`)
  }
}

function isBundledPipDistlibWindowsLauncher(path) {
  if (!path.endsWith('.exe')) return false
  const parent = dirname(path).split(/[\\/]+/).slice(-4).join('/')
  return parent === 'site-packages/pip/_vendor/distlib'
}

function removeBundledWindowsLaunchers(files) {
  const launchers = files.filter(isBundledPipDistlibWindowsLauncher)
  for (const file of launchers) {
    rmSync(file, { force: true })
  }
  return launchers
}

function detachHardlinkedTree(root) {
  if (!existsSync(root)) return []

  const detached = []
  let sequence = 0
  for (const file of walk(root)) {
    const before = lstatSync(file, { bigint: true })
    if (!before.isFile() || before.nlink === 1n) continue

    const bytes = readFileSync(file)
    const mode = Number(before.mode & 0o7777n)
    const temporary = join(
      dirname(file),
      `.${basename(file)}.vmlx-detach-${process.pid}-${sequence++}`,
    )
    try {
      writeFileSync(temporary, bytes, { flag: 'wx', mode })
      chmodSync(temporary, mode)
      const copied = lstatSync(temporary, { bigint: true })
      if (!copied.isFile() || copied.nlink !== 1n || copied.size !== before.size) {
        throw new Error(`detached release resource is not a single-link copy: ${file}`)
      }
      if (!readFileSync(temporary).equals(bytes)) {
        throw new Error(`detached release resource changed while copying: ${file}`)
      }
      renameSync(temporary, file)
    } finally {
      rmSync(temporary, { force: true })
    }

    const after = lstatSync(file, { bigint: true })
    if (!after.isFile() || after.nlink !== 1n || after.size !== before.size) {
      throw new Error(`packaged release resource stayed hard-linked: ${file}`)
    }
    detached.push(file)
  }
  return detached
}

async function afterPack(context) {
  const appOutDir = context && context.appOutDir
  const appName = context && context.packager && context.packager.appInfo
    ? context.packager.appInfo.productFilename
    : 'vMLX'
  if (!appOutDir) {
    throw new Error('electron-builder afterPack hook missing appOutDir')
  }

  const packagedEngineSource = join(
    appOutDir,
    `${appName}.app`,
    'Contents',
    'Resources',
    'vmlx-engine-source',
  )
  const detachedEngineFiles = detachHardlinkedTree(packagedEngineSource)
  if (detachedEngineFiles.length > 0) {
    console.log(
      `[afterPack] detached ${detachedEngineFiles.length} hard-linked vMLX source files from the checkout`,
    )
  }

  const bundledPython = join(
    appOutDir,
    `${appName}.app`,
    'Contents',
    'Resources',
    'bundled-python',
    'python',
  )
  if (!existsSync(bundledPython)) {
    console.log(`[afterPack] bundled Python not found, skipping signature normalization: ${bundledPython}`)
    return
  }

  const allFiles = walk(bundledPython)
  const removedWindowsLaunchers = removeBundledWindowsLaunchers(allFiles)
  const nativeFiles = allFiles
    .filter(file => !removedWindowsLaunchers.includes(file))
    .filter(isMachO)
  for (const file of nativeFiles) {
    removeSignature(file)
    signAdhoc(file)
  }
  if (removedWindowsLaunchers.length > 0) {
    console.log(
      `[afterPack] removed ${removedWindowsLaunchers.length} bundled Python Windows launcher stubs: ` +
        removedWindowsLaunchers.map(file => basename(file)).join(', '),
    )
  }
  console.log(`[afterPack] normalized ad-hoc signatures for ${nativeFiles.length} bundled Python native files`)
}

module.exports = afterPack
module.exports.detachHardlinkedTree = detachHardlinkedTree
module.exports.removeBundledWindowsLaunchers = removeBundledWindowsLaunchers
module.exports.isBundledPipDistlibWindowsLauncher = isBundledPipDistlibWindowsLauncher

if (require.main === module) {
  afterPack({
    appOutDir: join(process.cwd(), 'release', 'mac-arm64'),
    packager: { appInfo: { productFilename: 'vMLX' } },
  }).catch((error) => {
    console.error(error && error.stack ? error.stack : error)
    process.exit(1)
  })
}
