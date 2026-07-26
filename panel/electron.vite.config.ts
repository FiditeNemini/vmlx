import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { execFileSync } from 'child_process'

function rendererSourceCommit(): string {
  const commit = execFileSync(
    'git',
    ['-C', __dirname, 'rev-parse', 'HEAD'],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }
  ).trim()
  if (!/^[0-9a-f]{40}$/.test(commit)) {
    throw new Error(`Cannot attest renderer source commit: ${commit || 'empty'}`)
  }
  return commit
}

const sourceCommit = rendererSourceCommit()

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin({ exclude: ['uuid'] })],
    build: {
      outDir: 'dist/main',
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'src/main/index.ts')
        },
        output: {
          format: 'es'
        }
      }
    }
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      outDir: 'dist/preload',
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'src/preload/index.ts')
        }
      }
    }
  },
  renderer: {
    define: {
      // Derived directly from this checkout. Release proof independently
      // requires a clean tree and compares this value to the frozen HEAD.
      __VMLINUX_BUILD_SOURCE_COMMIT__: JSON.stringify(sourceCommit)
    },
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src/renderer')
      }
    },
    plugins: [react()]
  }
})
