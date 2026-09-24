import { ipcMain, dialog } from 'electron'
import { writeFileSync, readFileSync, statSync } from 'fs'
import { db, type Message } from '../database'
import { randomUUID } from 'crypto'
import { renderSessionMarkdown, parseSessionMarkdown } from '../session-export'

/**
 * Chat export/import IPC handlers.
 * Supports JSON, Markdown, and ShareGPT formats.
 */

export function registerExportHandlers(): void {
  // Export a single chat
  ipcMain.handle('chat:export', async (_, chatId: string, format: 'json' | 'markdown' | 'sharegpt') => {
    const chat = db.getChat(chatId)
    if (!chat) throw new Error('Chat not found')

    const messages = db.getMessages(chatId)
    let content: string
    let ext: string

    switch (format) {
      case 'markdown': {
        content = renderSessionMarkdown(chat, messages)
        ext = 'md'
        break
      }
      case 'sharegpt': {
        const conversations = messages.map(m => ({
          from: m.role === 'assistant' ? 'gpt' : m.role === 'user' ? 'human' : 'system',
          value: m.content
        }))
        content = JSON.stringify({ conversations }, null, 2)
        ext = 'json'
        break
      }
      default: {
        content = JSON.stringify({
          title: chat.title,
          modelId: chat.modelId,
          modelPath: chat.modelPath,
          createdAt: chat.createdAt,
          messages: messages.map(m => ({
            role: m.role as 'system' | 'user' | 'assistant',
            content: m.content,
            timestamp: m.timestamp,
            ...(m.reasoningContent ? { reasoning: m.reasoningContent } : {}),
            ...(m.generationRecordJson ? { generationRecord: JSON.parse(m.generationRecordJson) } : {})
          }))
        }, null, 2)
        ext = 'json'
      }
    }

    const safeName = chat.title.replace(/[^a-zA-Z0-9 _-]/g, '').slice(0, 50).trim() || 'chat'
    const result = await dialog.showSaveDialog({
      title: format === 'markdown' ? 'Export session' : 'Export Chat',
      defaultPath: `${safeName}.${ext}`,
      filters: ext === 'md'
        ? [{ name: 'Markdown', extensions: ['md'] }]
        : [{ name: 'JSON', extensions: ['json'] }]
    })

    if (result.canceled || !result.filePath) return { success: false }

    try {
      writeFileSync(result.filePath, content, 'utf-8')
    } catch (err) {
      throw new Error(`Failed to write file: ${(err as Error).message}`)
    }
    return { success: true, path: result.filePath }
  })

  // Import a chat from file
  ipcMain.handle('chat:import', async (_, modelPath?: string) => {
    const result = await dialog.showOpenDialog({
      title: 'Import Chat',
      filters: [{ name: 'Chat files', extensions: ['json', 'md'] }],
      properties: ['openFile'],
      securityScopedBookmarks: true
    })

    if (result.canceled || result.filePaths.length === 0) return { success: false }

    const filePath = result.filePaths[0]

    // Security-scoped bookmark: start access before reading, stop after.
    // showOpenDialog grants temporary access, but with securityScopedBookmarks
    // we must explicitly manage the lifecycle.
    let stopAccess: (() => void) | undefined
    if (result.bookmarks && result.bookmarks.length > 0) {
      try {
        const { app } = await import('electron')
        stopAccess = app.startAccessingSecurityScopedResource(result.bookmarks[0]) as unknown as () => void
      } catch (_) { }
    }

    // Guard against excessively large files
    const fileStats = statSync(filePath)
    if (fileStats.size > 50 * 1024 * 1024) {
      throw new Error('File too large (max 50MB)')
    }

    const raw = readFileSync(filePath, 'utf-8')
    stopAccess?.()

    let title = 'Imported Chat'
    let savedModelId: string | undefined
    let savedModelPath: string | undefined
    let savedCreatedAt: number | undefined
    let messages: Array<Partial<Message> & { role: string; content: string; reasoning?: string }> = []

    if (filePath.endsWith('.json')) {
      let parsed: any
      try {
        parsed = JSON.parse(raw)
      } catch {
        throw new Error('Invalid JSON file — could not parse')
      }

      // ShareGPT format
      if (parsed.conversations && Array.isArray(parsed.conversations)) {
        title = 'Imported (ShareGPT)'
        messages = parsed.conversations.map((c: any) => ({
          role: c.from === 'gpt' ? 'assistant' : c.from === 'human' ? 'user' : c.from || 'user',
          content: c.value || ''
        }))
      }
      // vMLX native format
      else if (parsed.messages && Array.isArray(parsed.messages)) {
        title = parsed.title || 'Imported Chat'
        if (typeof parsed.modelId === 'string') savedModelId = parsed.modelId
        if (typeof parsed.modelPath === 'string') savedModelPath = parsed.modelPath
        if (Number.isFinite(parsed.createdAt)) savedCreatedAt = parsed.createdAt
        messages = parsed.messages.map((m: any) => ({
          role: m.role || 'user',
          content: m.content || '',
          ...(Number.isFinite(m.timestamp) ? { timestamp: m.timestamp } : {}),
          ...(m.reasoning ? { reasoning: m.reasoning } : {}),
          ...(m.generationRecord ? { generationRecordJson: JSON.stringify(m.generationRecord) } : {})
        }))
      }
    } else if (filePath.endsWith('.md')) {
      const session = parseSessionMarkdown(raw)
      if (session) {
        title = session.title
        savedModelId = session.modelId
        savedModelPath = session.modelPath
        savedCreatedAt = session.createdAt
        messages = session.messages.map(m => ({ ...m, role: m.role || 'user', content: m.content || '' }))
      } else {
        // Parse legacy markdown: ## User / ## Assistant / ## System sections
        title = 'Imported (Markdown)'
        const sections = raw.split(/^## /m).slice(1)
        for (const section of sections) {
          const firstLine = section.split('\n')[0].trim().toLowerCase()
          let content = section.split('\n').slice(1).join('\n').trim()
          const role = firstLine.includes('assistant') || firstLine.includes('gpt') ? 'assistant'
            : firstLine.includes('system') ? 'system' : 'user'
          // Extract reasoning from <details><summary>Thinking</summary>...</details> blocks
          let reasoning: string | undefined
          const detailsMatch = content.match(/^<details><summary>Thinking<\/summary>\s*\n\n([\s\S]*?)\n\n<\/details>\s*\n?/)
          if (detailsMatch) {
            reasoning = detailsMatch[1].trim()
            content = content.slice(detailsMatch[0].length).trim()
          }
          if (content) messages.push({ role, content, ...(reasoning ? { reasoning } : {}) })
        }
      }
    }

    if (messages.length === 0) {
      throw new Error('No messages found in file')
    }

    // Validate message roles
    const validRoles = new Set(['system', 'user', 'assistant', 'tool'])
    for (const msg of messages) {
      if (!validRoles.has(msg.role)) {
        msg.role = 'user' // Normalize unknown roles
      }
    }

    // Create chat and add messages
    const chatId = randomUUID()
    const now = Date.now()
    db.createChat({
      id: chatId,
      title,
      modelId: modelPath ? 'default' : savedModelId || 'default',
      modelPath: modelPath || savedModelPath || '',
      folderId: undefined,
      createdAt: savedCreatedAt ?? now,
      updatedAt: now
    })

    for (const msg of messages) {
      db.addMessage({
        id: randomUUID(),
        chatId,
        role: msg.role as 'system' | 'user' | 'assistant',
        content: msg.content,
        timestamp: msg.timestamp ?? now,
        ...(msg.reasoning || msg.reasoningContent ? { reasoningContent: msg.reasoning || msg.reasoningContent } : {}),
        generationRecordJson: msg.generationRecordJson,
        reasoningSegmentsJson: msg.reasoningSegmentsJson,
        toolCallsOaiJson: msg.toolCallsOaiJson,
        toolResultsOaiJson: msg.toolResultsOaiJson,
        toolCallId: msg.toolCallId,
        toolCapabilityFingerprint: msg.toolCapabilityFingerprint,
        toolCallsJson: msg.toolCallsJson,
        warningsJson: msg.warningsJson,
        metricsJson: msg.metricsJson,
      })
    }

    return { success: true, chatId, title, messageCount: messages.length }
  })
}
