#!/usr/bin/env node
import { spawn, spawnSync } from 'node:child_process'
import { createServer } from 'node:http'
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import path from 'node:path'

const require = createRequire(import.meta.url)
const { chromium } = require('playwright-core')

const panelDir = path.resolve(new URL('..', import.meta.url).pathname)
const repoDir = path.resolve(panelDir, '..')
const proofDir = path.join(
  repoDir,
  'docs',
  'internal',
  'release-gates',
  '20260719_midstream_failure_recovery',
)
const python = '/Users/example/mlx/vllm-mlx/.venv/bin/python'
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function freePort() {
  return await new Promise((resolve, reject) => {
    const server = createServer()
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      server.close(() => resolve(address.port))
    })
    server.on('error', reject)
  })
}

async function waitFor(url, timeoutMs = 30_000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok) return await response.json()
    } catch {}
    await sleep(150)
  }
  throw new Error(`Timed out waiting for ${url}`)
}

function runCurl(label, url, body, headers = []) {
  const args = [
    '-sS', '-N', url,
    '-H', 'Content-Type: application/json',
    ...headers.flatMap((header) => ['-H', header]),
    '-d', JSON.stringify(body),
  ]
  const result = spawnSync('curl', args, { encoding: 'utf8' })
  if (result.status !== 0) {
    throw new Error(`${label} curl failed: ${result.stderr}`)
  }
  const file = path.join(proofDir, `${label}.sse`)
  writeFileSync(file, result.stdout)
  return result.stdout
}

function assertRawStreams(streams) {
  const failures = []
  const requireText = (stream, text, label) => {
    if (!stream.includes(text)) failures.push(`${label} missing ${text}`)
  }
  requireText(streams.responsesFail, '"delta": "RESP-PARTIAL-"', 'responses fail')
  requireText(streams.responsesFail, '"delta": "VISIBLE"', 'responses fail')
  requireText(streams.responsesFail, 'event: error', 'responses fail')
  requireText(streams.responsesFail, 'event: response.failed', 'responses fail')
  requireText(streams.responsesFail, '"output_tokens": 2', 'responses fail usage')
  if (streams.responsesFail.includes('response.completed')) {
    failures.push('responses fail incorrectly completed')
  }
  requireText(streams.responsesRecover, '"delta": "RESP-RECOVERY-"', 'responses recovery')
  requireText(streams.responsesRecover, '"delta": "OK"', 'responses recovery')
  requireText(streams.responsesRecover, 'event: response.completed', 'responses recovery')
  requireText(streams.chatFail, '"content": "CHAT-PARTIAL-"', 'chat fail')
  requireText(streams.chatFail, '"content": "VISIBLE"', 'chat fail')
  requireText(streams.chatFail, '"error":', 'chat fail')
  requireText(streams.chatFail, '"completion_tokens": 2', 'chat fail usage')
  requireText(streams.chatFail, 'data: [DONE]', 'chat fail')
  const chatError = streams.chatFail.indexOf('"error":')
  const chatUsage = streams.chatFail.lastIndexOf('"usage": {')
  if (!(chatError >= 0 && chatUsage > chatError)) {
    failures.push('chat fail usage did not follow error')
  }
  requireText(streams.chatRecover, '"content": "CHAT-RECOVERY-"', 'chat recovery')
  requireText(streams.chatRecover, '"content": "OK"', 'chat recovery')
  requireText(streams.chatRecover, 'data: [DONE]', 'chat recovery')
  if (failures.length) throw new Error(`Raw stream proof failed:\n- ${failures.join('\n- ')}`)
}

function terminateTree(child) {
  if (!child?.pid) return
  try { process.kill(-child.pid, 'SIGTERM') } catch {}
}

async function sendVisibleTurn(page, prompt, partial, terminal, screenshotStem) {
  const textarea = page.locator('textarea').last()
  await textarea.waitFor({ state: 'visible', timeout: 10_000 })
  if (!(await textarea.isEnabled())) throw new Error(`textarea disabled before ${prompt}`)
  await textarea.fill(prompt)
  await textarea.press('Enter')
  await page.waitForFunction(
    (text) => document.body.innerText.includes(text),
    prompt,
    { timeout: 10_000 },
  )
  await page.waitForFunction(
    (text) => document.body.innerText.includes(text),
    partial,
    { timeout: 10_000 },
  )
  const partialScreenshot = path.join(proofDir, `${screenshotStem}-partial.png`)
  await page.screenshot({ path: partialScreenshot, fullPage: true })
  await page.waitForFunction(
    (text) => document.body.innerText.includes(text),
    terminal,
    { timeout: 15_000 },
  )
  await sleep(250)
  const terminalScreenshot = path.join(proofDir, `${screenshotStem}-terminal.png`)
  await page.screenshot({ path: terminalScreenshot, fullPage: true })
  return { partialScreenshot, terminalScreenshot }
}

async function main() {
  mkdirSync(proofDir, { recursive: true })
  const requestLog = path.join(proofDir, 'request-bodies.jsonl')
  writeFileSync(requestLog, '')
  const serverPort = await freePort()
  const cdpPort = await freePort()
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-midstream-userdata-'))
  const serverLogs = []
  const appLogs = []

  const proofServer = spawn(
    python,
    [
      path.join(repoDir, 'tests/cross_matrix/live_midstream_failure_server.py'),
      '--port', String(serverPort),
      '--request-log', requestLog,
    ],
    {
      cwd: repoDir,
      env: { ...process.env, PYTHONPATH: repoDir },
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )
  proofServer.stdout.on('data', (data) => serverLogs.push(data.toString()))
  proofServer.stderr.on('data', (data) => serverLogs.push(data.toString()))

  let app
  let browser
  try {
    const health = await waitFor(`http://127.0.0.1:${serverPort}/health`)
    const base = `http://127.0.0.1:${serverPort}`
    const streams = {
      responsesFail: runCurl('raw-responses-fail', `${base}/v1/responses`, {
        model: 'midstream-live-proof', input: 'RAW-RESP-FAIL', stream: true,
        max_output_tokens: 64, enable_thinking: false,
      }, ['x-vmlx-stream-usage: incremental']),
      responsesRecover: runCurl('raw-responses-recover', `${base}/v1/responses`, {
        model: 'midstream-live-proof', input: 'RAW-RESP-RECOVER', stream: true,
        max_output_tokens: 64, enable_thinking: false,
      }),
      chatFail: runCurl('raw-chat-fail', `${base}/v1/chat/completions`, {
        model: 'midstream-live-proof',
        messages: [{ role: 'user', content: 'RAW-CHAT-FAIL' }],
        stream: true, stream_options: { include_usage: true },
        max_tokens: 64, enable_thinking: false,
      }),
      chatRecover: runCurl('raw-chat-recover', `${base}/v1/chat/completions`, {
        model: 'midstream-live-proof',
        messages: [{ role: 'user', content: 'RAW-CHAT-RECOVER' }],
        stream: true, stream_options: { include_usage: true },
        max_tokens: 64, enable_thinking: false,
      }),
    }
    assertRawStreams(streams)

    app = spawn(
      'npm',
      ['run', 'dev', '--', '--', `--user-data-dir=${userDataDir}`, `--remote-debugging-port=${cdpPort}`],
      {
        cwd: panelDir,
        env: {
          ...process.env,
          PATH: `/Users/example/mlx/vllm-mlx/.venv/bin:${process.env.PATH}`,
          VMLX_SKIP_UPDATE_CHECK: '1',
        },
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    )
    app.stdout.on('data', (data) => appLogs.push(data.toString()))
    app.stderr.on('data', (data) => appLogs.push(data.toString()))
    await waitFor(`http://127.0.0.1:${cdpPort}/json/version`, 60_000)
    browser = await chromium.connectOverCDP(`http://127.0.0.1:${cdpPort}`)
    const contexts = browser.contexts()
    const context = contexts[0]
    const pages = context.pages()
    const page = pages.find((candidate) => candidate.url().startsWith('http')) || pages[0]
    await page.setViewportSize({ width: 1440, height: 1000 })
    await page.waitForFunction(() => Boolean(window.api?.sessions && window.api?.chat), null, { timeout: 30_000 })
    await page.waitForFunction(() => document.getElementById('root')?.children.length, null, { timeout: 30_000 })
    const gotIt = page.getByRole('button', { name: /Got it/i })
    if (await gotIt.count()) await gotIt.first().click().catch(() => {})

    const remote = await page.evaluate(async (port) => {
      await window.api.chat.clearAllLocks().catch(() => null)
      const result = await window.api.sessions.createRemote({
        remoteUrl: `http://127.0.0.1:${port}`,
        remoteModel: 'midstream-live-proof',
      })
      if (!result.success) throw new Error(result.error || 'remote session create failed')
      const started = await window.api.sessions.start(result.session.id)
      if (!started.success) throw new Error(started.error || 'remote session start failed')
      window.dispatchEvent(new CustomEvent('vmlx:navigate', {
        detail: { mode: 'server', panel: 'session', sessionId: result.session.id },
      }))
      return result.session
    }, serverPort)

    await page.locator('textarea').last().waitFor({ state: 'visible', timeout: 30_000 })
    await page.waitForFunction(
      (port) => document.body.innerText.includes(`127.0.0.1:${port}`)
        && document.body.innerText.includes('Connected'),
      serverPort,
      { timeout: 30_000 },
    )
    const existingChatIds = await page.evaluate(async (modelPath) => (
      await window.api.chat.getByModel(modelPath)
    ).map((chat) => chat.id), remote.modelPath)
    await page.getByRole('button', { name: '+ Chat', exact: true }).click()
    await page.waitForTimeout(400)
    const responsesChat = await page.evaluate(async ({ modelPath, existingChatIds }) => {
      const chats = await window.api.chat.getByModel(modelPath)
      const chat = chats.find((candidate) => !existingChatIds.includes(candidate.id))
      if (!chat) throw new Error('fresh Responses chat was not created')
      await window.api.chat.setOverrides(chat.id, {
        chatId: chat.id,
        wireApi: 'responses',
        builtinToolsEnabled: false,
        enableThinking: false,
        maxTokens: 64,
      })
      return chat
    }, { modelPath: remote.modelPath, existingChatIds })

    const responseFailShots = await sendVisibleTurn(
      page,
      'UI-RESP-FAIL',
      'RESP-PARTIAL-',
      '[Generation interrupted]',
      'electron-responses-fail',
    )
    const responsesFailureMessages = await page.evaluate(
      (chatId) => window.api.chat.getMessages(chatId),
      responsesChat.id,
    )
    const responsesFailedAssistant = [...responsesFailureMessages].reverse()
      .find((message) => message.role === 'assistant')

    const responseRecoverShots = await sendVisibleTurn(
      page,
      'UI-RESP-RECOVER',
      'RESP-RECOVERY-',
      'RESP-RECOVERY-OK',
      'electron-responses-recover',
    )

    const preChatCompletionIds = await page.evaluate(async (modelPath) => (
      await window.api.chat.getByModel(modelPath)
    ).map((chat) => chat.id), remote.modelPath)
    await page.getByRole('button', { name: '+ Chat', exact: true }).click()
    await page.waitForTimeout(500)
    const chatCompletionChat = await page.evaluate(async ({ modelPath, existingChatIds }) => {
      const chats = await window.api.chat.getByModel(modelPath)
      const chat = chats.find((candidate) => !existingChatIds.includes(candidate.id))
      if (!chat) throw new Error('new Chat Completions chat was not created')
      await window.api.chat.setOverrides(chat.id, {
        chatId: chat.id,
        wireApi: 'completions',
        builtinToolsEnabled: false,
        enableThinking: false,
        maxTokens: 64,
      })
      return chat
    }, { modelPath: remote.modelPath, existingChatIds: preChatCompletionIds })

    const chatFailShots = await sendVisibleTurn(
      page,
      'UI-CHAT-FAIL',
      'CHAT-PARTIAL-',
      '[Generation interrupted]',
      'electron-chat-fail',
    )
    const chatFailureMessages = await page.evaluate(
      (chatId) => window.api.chat.getMessages(chatId),
      chatCompletionChat.id,
    )
    const chatFailedAssistant = [...chatFailureMessages].reverse()
      .find((message) => message.role === 'assistant')
    const chatRecoverShots = await sendVisibleTurn(
      page,
      'UI-CHAT-RECOVER',
      'CHAT-RECOVERY-',
      'CHAT-RECOVERY-OK',
      'electron-chat-recover',
    )

    const finalMessages = {
      responses: await page.evaluate((chatId) => window.api.chat.getMessages(chatId), responsesChat.id),
      chat: await page.evaluate((chatId) => window.api.chat.getMessages(chatId), chatCompletionChat.id),
    }
    const assertions = {
      responsesPartialContent: responsesFailedAssistant?.content,
      responsesPartialMetrics: responsesFailedAssistant?.metricsJson
        ? JSON.parse(responsesFailedAssistant.metricsJson) : null,
      chatPartialContent: chatFailedAssistant?.content,
      chatPartialMetrics: chatFailedAssistant?.metricsJson
        ? JSON.parse(chatFailedAssistant.metricsJson) : null,
      responsesRecoveryContent: [...finalMessages.responses].reverse()
        .find((message) => message.role === 'assistant')?.content,
      chatRecoveryContent: [...finalMessages.chat].reverse()
        .find((message) => message.role === 'assistant')?.content,
    }
    const failures = []
    if (assertions.responsesPartialContent !== 'RESP-PARTIAL-VISIBLE\n\n[Generation interrupted]') {
      failures.push(`Responses partial persistence mismatch: ${assertions.responsesPartialContent}`)
    }
    if (assertions.responsesPartialMetrics?.tokenCount !== 2 || assertions.responsesPartialMetrics?.promptTokens !== 5) {
      failures.push(`Responses authoritative partial usage mismatch: ${JSON.stringify(assertions.responsesPartialMetrics)}`)
    }
    if (assertions.chatPartialContent !== 'CHAT-PARTIAL-VISIBLE\n\n[Generation interrupted]') {
      failures.push(`Chat partial persistence mismatch: ${assertions.chatPartialContent}`)
    }
    if (assertions.chatPartialMetrics?.tokenCount !== 2 || assertions.chatPartialMetrics?.promptTokens !== 5) {
      failures.push(`Chat authoritative partial usage mismatch: ${JSON.stringify(assertions.chatPartialMetrics)}`)
    }
    if (assertions.responsesRecoveryContent !== 'RESP-RECOVERY-OK') {
      failures.push(`Responses immediate recovery mismatch: ${assertions.responsesRecoveryContent}`)
    }
    if (assertions.chatRecoveryContent !== 'CHAT-RECOVERY-OK') {
      failures.push(`Chat immediate recovery mismatch: ${assertions.chatRecoveryContent}`)
    }
    const requestBodies = readFileSync(requestLog, 'utf8')
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line))
    const responsesRecoveryRequest = [...requestBodies].reverse()
      .find((item) => item.endpoint === '/v1/responses' && JSON.stringify(item.body).includes('UI-RESP-RECOVER'))
    const chatRecoveryRequest = [...requestBodies].reverse()
      .find((item) => item.endpoint === '/v1/chat/completions' && JSON.stringify(item.body).includes('UI-CHAT-RECOVER'))
    for (const [label, request] of [
      ['Responses', responsesRecoveryRequest],
      ['Chat', chatRecoveryRequest],
    ]) {
      const serialized = JSON.stringify(request?.body || {})
      if (!request) failures.push(`${label} recovery request was not captured`)
      if (serialized.includes('[Generation interrupted]')) {
        failures.push(`${label} recovery replay leaked the UI interruption marker`)
      }
    }
    if (!JSON.stringify(responsesRecoveryRequest?.body || {}).includes('RESP-PARTIAL-VISIBLE')) {
      failures.push('Responses recovery did not replay the safe partial prefix')
    }
    if (!JSON.stringify(chatRecoveryRequest?.body || {}).includes('CHAT-PARTIAL-VISIBLE')) {
      failures.push('Chat recovery did not replay the safe partial prefix')
    }
    if (failures.length) throw new Error(`Electron proof failed:\n- ${failures.join('\n- ')}`)

    const result = {
      verdict: 'PASS',
      sourceHead: spawnSync('git', ['rev-parse', 'HEAD'], { cwd: repoDir, encoding: 'utf8' }).stdout.trim(),
      health,
      serverPort,
      cdpPort,
      cdpNote: 'Used isolated local CDP port because 9335 is occupied by an existing SSH tunnel.',
      remoteSessionId: remote.id,
      responsesChatId: responsesChat.id,
      chatCompletionChatId: chatCompletionChat.id,
      assertions,
      historyReplay: {
        interruptionMarkerStripped: true,
        safePartialPrefixReplayed: true,
      },
      screenshots: {
        responseFailShots,
        responseRecoverShots,
        chatFailShots,
        chatRecoverShots,
      },
      rawSse: Object.keys(streams),
    }
    writeFileSync(path.join(proofDir, 'live-proof.json'), `${JSON.stringify(result, null, 2)}\n`)
    writeFileSync(path.join(proofDir, 'electron-app.log'), appLogs.join(''))
    writeFileSync(path.join(proofDir, 'proof-server.log'), serverLogs.join(''))
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
  } finally {
    if (browser) await browser.close().catch(() => {})
    terminateTree(app)
    terminateTree(proofServer)
    await sleep(500)
    writeFileSync(path.join(proofDir, 'electron-app.log'), appLogs.join(''))
    writeFileSync(path.join(proofDir, 'proof-server.log'), serverLogs.join(''))
  }
}

main().catch((error) => {
  console.error(error.stack || error)
  process.exitCode = 1
})
