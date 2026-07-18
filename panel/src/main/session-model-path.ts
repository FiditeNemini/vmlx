import { existsSync, statSync } from 'fs'
import { join } from 'path'
import { sessionMatchesModelPath } from '../shared/sessionUtils'

export interface SessionModelPathRecord {
  id: string
  modelPath: string
  type?: 'local' | 'remote'
}

export interface SessionModelPathClassification {
  modelPathMissing: boolean
  usableTwinId?: string
}

export type ClassifiedSessionModelPath<T extends SessionModelPathRecord> =
  T & SessionModelPathClassification

export interface ModelBundleDirectoryValidation {
  valid: boolean
  path: string
  error?: string
}

function pathExists(path: string): boolean {
  try {
    return existsSync(path)
  } catch {
    return false
  }
}

/**
 * Add filesystem truth to persisted sessions without mutating or pruning them.
 * A twin is another local session with the same model identity and a path that
 * currently exists. The flags are deliberately ephemeral list metadata: a
 * disconnected drive can change the answer without any database mutation.
 */
export function classifySessionModelPaths<T extends SessionModelPathRecord>(
  sessions: readonly T[],
  exists: (path: string) => boolean = pathExists,
): Array<ClassifiedSessionModelPath<T>> {
  const existingLocalSessions = sessions.filter(
    (session) => session.type !== 'remote' && exists(session.modelPath),
  )

  return sessions.map((session) => {
    if (session.type === 'remote') {
      return { ...session, modelPathMissing: false }
    }

    const modelPathMissing = !exists(session.modelPath)
    if (!modelPathMissing) {
      return { ...session, modelPathMissing: false }
    }

    const usableTwin = existingLocalSessions.find(
      (candidate) =>
        candidate.id !== session.id &&
        sessionMatchesModelPath(candidate.modelPath, session.modelPath),
    )

    return {
      ...session,
      modelPathMissing: true,
      ...(usableTwin ? { usableTwinId: usableTwin.id } : {}),
    }
  })
}

/** Validate the renderer-selected directory before any session or chat write. */
export function validateModelBundleDirectory(
  candidatePath: string,
): ModelBundleDirectoryValidation {
  const normalizedPath = String(candidatePath || '').trim().replace(/\/+$/, '')
  if (!normalizedPath) {
    return { valid: false, path: '', error: 'No model directory was selected.' }
  }

  try {
    if (!existsSync(normalizedPath) || !statSync(normalizedPath).isDirectory()) {
      return {
        valid: false,
        path: normalizedPath,
        error: 'The selected model directory does not exist.',
      }
    }

    const configPath = join(normalizedPath, 'config.json')
    if (!existsSync(configPath) || !statSync(configPath).isFile()) {
      return {
        valid: false,
        path: normalizedPath,
        error: 'The selected directory is not a model bundle: config.json is missing.',
      }
    }
  } catch {
    return {
      valid: false,
      path: normalizedPath,
      error: 'The selected model directory could not be read.',
    }
  }

  return { valid: true, path: normalizedPath }
}
