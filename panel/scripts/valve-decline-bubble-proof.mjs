#!/usr/bin/env node
import { createServer } from 'node:http'
import net from 'node:net'
import crypto from 'node:crypto'
import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

const panelDir = path.resolve(new URL('..', import.meta.url).pathname)
const repoDir = path.resolve(panelDir, '..')
const proofBasename = process.env.VMLX_LIVE_PROOF_BASENAME
  || `${new Date().toISOString().slice(0, 10)}-valve-decline-bubble`

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function isSocketDisconnectError(error) {
  const code = String(error?.code || '')
  const message = String(error?.message || error || '')
  const cause = error?.cause
  const nestedErrors = Array.isArray(error?.errors) ? error.errors : []
  return (
    code === 'EPIPE'
    || code === 'ECONNRESET'
    || code === 'ERR_STREAM_DESTROYED'
    || code === 'ERR_STREAM_WRITE_AFTER_END'
    || /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message)
    || (cause ? isSocketDisconnectError(cause) : false)
    || nestedErrors.some((nested) => isSocketDisconnectError(nested))
  )
}

function attachChildProcessStreamErrorGuard(stream, logs) {
  stream?.on('error', (error) => {
    if (isSocketDisconnectError(error)) return
    logs.push(`child stdio stream error: ${error?.message || String(error)}`)
  })
}

function safeHttpWrite(res, chunk) {
  if (res.destroyed || res.writableEnded) return false
  try {
    return res.write(chunk)
  } catch (error) {
    if (isSocketDisconnectError(error)) return false
    throw error
  }
}

function safeHttpEnd(res, chunk) {
  if (res.destroyed || res.writableEnded) return false
  try {
    res.end(chunk)
    return true
  } catch (error) {
    if (isSocketDisconnectError(error)) return false
    throw error
  }
}

async function freePort() {
  return await new Promise((resolve, reject) => {
    const server = createServer()
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port
      server.close(() => resolve(port))
    })
    server.on('error', reject)
  })
}

async function requestJson(url, timeoutMs = 1000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: controller.signal })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}

function collectRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []
    req.on('data', (chunk) => chunks.push(chunk))
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8')
      if (!raw) return resolve({})
      try {
        resolve(JSON.parse(raw))
      } catch (error) {
        reject(error)
      }
    })
    req.on('error', reject)
  })
}

function writeSseEvent(res, event, data) {
  if (event && !safeHttpWrite(res, `event: ${event}\n`)) return false
  return safeHttpWrite(res, `data: ${JSON.stringify(data)}\n\n`)
}


// Mock engine that DECLINES the chat span exactly like the live prefill
// admission valve (ledger rows 150/152): the chat stream emits an SSE error
// event carrying the valve prose + machine code, which is the path that
// rendered NOTHING before the fix (only a vanishing toast).
const VALVE_MESSAGE = 'dots3-note-prev-JANG: prefill admission rejected chunk [0:2048) — '
  + 'active Metal working set 96.51GB plus projected transient 11.12GB exceeds the '
  + 'device working-set limit 107.50GB. A context of this length cannot be served on '
  + 'this hardware; reduce the prompt or context size.'

async function startMockServer() {
  const requests = []
  const port = await freePort()
  const server = createServer(async (req, res) => {
    try {
      if (req.method === 'GET' && req.url === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' })
        safeHttpEnd(res, JSON.stringify({ object: 'list', data: [{ id: 'vmlx-valve-mock', object: 'model' }] }))
        return
      }
      if (req.method === 'POST' && (req.url === '/v1/chat/completions' || req.url === '/v1/responses')) {
        const body = await collectRequestBody(req)
        requests.push({ url: req.url, body })
        res.writeHead(200, {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        })
        await sleep(50)
        if (req.url === '/v1/responses') {
          writeSseEvent(res, 'response.error', {
            sequence_number: 1,
            error: { message: VALVE_MESSAGE, type: 'prompt_too_long', code: 'prefill_admission_declined' },
          })
        } else {
          safeHttpWrite(res, 'data: ' + JSON.stringify({
            error: { message: VALVE_MESSAGE, type: 'prompt_too_long', code: 'prefill_admission_declined' },
          }) + '\n\n')
        }
        safeHttpWrite(res, 'data: [DONE]\n\n')
        safeHttpEnd(res)
        return
      }
      res.writeHead(404, { 'content-type': 'application/json' })
      safeHttpEnd(res, JSON.stringify({ error: `Unhandled ${req.method} ${req.url}` }))
    } catch (error) {
      if (isSocketDisconnectError(error)) { safeHttpEnd(res); return }
      res.writeHead(500, { 'content-type': 'application/json' })
      safeHttpEnd(res, JSON.stringify({ error: error.message }))
    }
  })
  await new Promise((resolve, reject) => {
    server.listen(port, '127.0.0.1', resolve)
    server.on('error', reject)
  })
  return { port, requests, close: () => new Promise((resolve) => server.close(resolve)) }
}

class CdpSocket {
  constructor(socket) {
    this.socket = socket
    this.buffer = Buffer.alloc(0)
    this.nextId = 1
    this.pending = new Map()
    this.closed = false
    socket.on('data', (chunk) => this.onData(chunk))
    socket.on('error', (error) => {
      if (isSocketDisconnectError(error)) this.closed = true
      this.rejectPending(error)
    })
    socket.on('close', () => {
      this.closed = true
      this.rejectPending(new Error('CDP socket closed before response'))
    })
    socket.on('end', () => {
      this.closed = true
      this.rejectPending(new Error('CDP socket ended before response'))
    })
  }

  static async connect(wsUrl) {
    const url = new URL(wsUrl)
    const key = crypto.randomBytes(16).toString('base64')
    const socket = net.connect(Number(url.port || 80), url.hostname)
    await new Promise((resolve, reject) => {
      socket.once('connect', resolve)
      socket.once('error', reject)
    })
    try {
      socket.write([
        `GET ${url.pathname}${url.search} HTTP/1.1`,
        `Host: ${url.host}`,
        'Upgrade: websocket',
        'Connection: Upgrade',
        `Sec-WebSocket-Key: ${key}`,
        'Sec-WebSocket-Version: 13',
        '\r\n',
      ].join('\r\n'))
    } catch (error) {
      try { socket.destroy() } catch {}
      throw error
    }
    let handshake = Buffer.alloc(0)
    const connected = await new Promise((resolve, reject) => {
      const onData = (chunk) => {
        handshake = Buffer.concat([handshake, chunk])
        const idx = handshake.indexOf('\r\n\r\n')
        if (idx < 0) return
        socket.off('data', onData)
        const header = handshake.slice(0, idx).toString('utf8')
        if (!header.includes(' 101 ')) {
          reject(new Error(`WebSocket upgrade failed: ${header.split('\r\n')[0]}`))
          return
        }
        const rest = handshake.slice(idx + 4)
        const cdp = new CdpSocket(socket)
        if (rest.length) cdp.onData(rest)
        resolve(cdp)
      }
      socket.on('data', onData)
      socket.once('error', reject)
    })
    return connected
  }

  send(method, params = {}, timeoutMs = 30_000) {
    const id = this.nextId++
    const payload = JSON.stringify({ id, method, params })
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          reject(new Error(`CDP timeout: ${method}`))
        }
      }, timeoutMs)
      this.pending.set(id, { resolve, reject, timer })
      try {
        this.writeClientFrame(payload)
      } catch (error) {
        clearTimeout(timer)
        this.pending.delete(id)
        reject(error)
      }
    })
  }

  close() {
    this.closed = true
    try { this.socket.end() } catch {}
    try { this.socket.destroy() } catch {}
  }

  rejectPending(error) {
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer)
      reject(error)
    }
    this.pending.clear()
  }

  writeClientFrame(payload) {
    if (this.socket.destroyed || this.closed) {
      const error = new Error('CDP socket closed before write')
      error.code = 'ERR_STREAM_DESTROYED'
      this.rejectPending(error)
      return false
    }
    try {
      this.socket.write(encodeClientFrame(payload))
      return true
    } catch (error) {
      if (isSocketDisconnectError(error)) {
        this.closed = true
        this.rejectPending(error)
        return false
      }
      throw error
    }
  }

  onData(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk])
    while (this.buffer.length >= 2) {
      const b0 = this.buffer[0]
      const opcode = b0 & 0x0f
      let len = this.buffer[1] & 0x7f
      let offset = 2
      if (len === 126) {
        if (this.buffer.length < 4) return
        len = this.buffer.readUInt16BE(2)
        offset = 4
      } else if (len === 127) {
        if (this.buffer.length < 10) return
        const high = this.buffer.readUInt32BE(2)
        const low = this.buffer.readUInt32BE(6)
        if (high !== 0) throw new Error('CDP frame too large')
        len = low
        offset = 10
      }
      if (this.buffer.length < offset + len) return
      const payload = this.buffer.slice(offset, offset + len)
      this.buffer = this.buffer.slice(offset + len)
      if (opcode === 1) {
        const msg = JSON.parse(payload.toString('utf8'))
        if (msg.id && this.pending.has(msg.id)) {
          const { resolve, reject, timer } = this.pending.get(msg.id)
          this.pending.delete(msg.id)
          clearTimeout(timer)
          if (msg.error) reject(new Error(JSON.stringify(msg.error)))
          else resolve(msg.result)
        }
      } else if (opcode === 8) {
        this.close()
      }
    }
  }
}

function encodeClientFrame(text) {
  const payload = Buffer.from(text, 'utf8')
  const len = payload.length
  const headerLen = len < 126 ? 2 : len < 65536 ? 4 : 10
  const header = Buffer.alloc(headerLen + 4)
  header[0] = 0x81
  if (len < 126) {
    header[1] = 0x80 | len
  } else if (len < 65536) {
    header[1] = 0x80 | 126
    header.writeUInt16BE(len, 2)
  } else {
    header[1] = 0x80 | 127
    header.writeUInt32BE(0, 2)
    header.writeUInt32BE(len, 6)
  }
  const maskOffset = headerLen
  const mask = crypto.randomBytes(4)
  mask.copy(header, maskOffset)
  const out = Buffer.alloc(header.length + payload.length)
  header.copy(out, 0)
  for (let i = 0; i < payload.length; i++) {
    out[header.length + i] = payload[i] ^ mask[i % 4]
  }
  return out
}

async function waitForTarget(debugPort, appLogs) {
  const started = Date.now()
  while (Date.now() - started < 60_000) {
    if (appLogs.some((line) => line.includes('Failed to get lock') || line.includes('second-instance'))) {
      throw new Error('Electron app did not acquire single-instance lock')
    }
    try {
      const targets = await requestJson(`http://127.0.0.1:${debugPort}/json/list`, 1000)
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)
      if (page) return page
    } catch {}
    await sleep(250)
  }
  throw new Error(`Timed out waiting for DevTools target on ${debugPort}`)
}

async function evaluate(cdp, expression) {
  const result = await cdp.send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
    timeout: 30_000,
  })
  if (result.exceptionDetails) {
    throw new Error(JSON.stringify(result.exceptionDetails, null, 2))
  }
  return result.result?.value
}

async function capturePng(cdp, filePath) {
  const shot = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: true,
  })
  writeFileSync(filePath, Buffer.from(shot.data, 'base64'))
  return filePath
}


async function main() {
  const userDataDir = mkdtempSync(path.join(tmpdir(), 'vmlx-valve-userdata-'))
  const mock = await startMockServer()
  const debugPort = await freePort()
  const appLogs = []
  const app = spawn('npm', [
    'run', 'dev', '--', '--',
    `--user-data-dir=${userDataDir}`,
    `--remote-debugging-port=${debugPort}`,
  ], {
    cwd: panelDir,
    env: { ...process.env, VMLX_SKIP_UPDATE_CHECK: '1' },
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  app.stdout.on('data', (d) => appLogs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  app.stderr.on('data', (d) => appLogs.push(...d.toString().split(/\r?\n/).filter(Boolean)))
  attachChildProcessStreamErrorGuard(app.stdout, appLogs)
  attachChildProcessStreamErrorGuard(app.stderr, appLogs)

  let cdp
  try {
    const target = await waitForTarget(debugPort, appLogs)
    cdp = await CdpSocket.connect(target.webSocketDebuggerUrl)
    await cdp.send('Runtime.enable')
    await cdp.send('Page.enable')
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false })
    await evaluate(cdp, `
      new Promise((resolve, reject) => {
        const started = Date.now();
        const check = () => {
          if (window.api?.chat && window.api?.sessions) resolve(true);
          else if (Date.now() - started > 30000) reject(new Error('window.api not ready'));
          else setTimeout(check, 100);
        };
        check();
      })
    `)
    const rendererResult = await evaluate(cdp, `
      (async () => {
        const mockPort = ${JSON.stringify(mock.port)};
        await new Promise((resolve, reject) => {
          const started = Date.now();
          const check = () => {
            if (document.getElementById('root')?.children.length) resolve(true);
            else if (Date.now() - started > 30000) reject(new Error('React root not mounted'));
            else setTimeout(check, 100);
          };
          check();
        });
        await window.api.engine.checkInstallation().catch(() => null);
        await new Promise((resolve) => setTimeout(resolve, 1500));
        await window.api.chat.clearAllLocks().catch(() => null);
        const remote = await window.api.sessions.createRemote({
          remoteUrl: 'http://127.0.0.1:' + mockPort,
          remoteModel: 'vmlx-valve-mock',
        });
        if (!remote.success) throw new Error(remote.error || 'remote session create failed');
        await window.api.sessions.start(remote.session.id);
        return { sessionId: remote.session.id, modelPath: remote.session.modelPath };
      })()
    `)
    const visual = await evaluate(cdp, `
      (async () => {
        const wait = (predicate, timeoutMs = 20000) => new Promise((resolve, reject) => {
          const started = Date.now();
          const tick = () => {
            try { const v = predicate(); if (v) return resolve(v); } catch (_) {}
            if (Date.now() - started > timeoutMs) return reject(new Error('timeout :: ' + document.body.innerText.slice(0, 900)));
            setTimeout(tick, 100);
          };
          tick();
        });
        const dismiss = [...document.querySelectorAll('button')].find((b) => b.innerText.includes('Got it'));
        if (dismiss) { dismiss.click(); await new Promise((r) => setTimeout(r, 150)); }
        window.dispatchEvent(new CustomEvent('vmlx:navigate', { detail: { mode: 'chat' } }));
        await wait(() => document.body.innerText.includes('New Chat'));
        const newChat = [...document.querySelectorAll('button')].find((b) => b.innerText.trim() === 'New Chat')
          || [...document.querySelectorAll('button')].find((b) => b.innerText.includes('New Chat'));
        if (!newChat) throw new Error('New Chat button not found');
        newChat.click();
        const composer = await wait(() => document.querySelector('textarea[placeholder*="essage" i], textarea'));
        const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        setter.call(composer, 'trigger the valve');
        composer.dispatchEvent(new Event('input', { bubbles: true }));
        await new Promise((r) => setTimeout(r, 200));
        composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
        await new Promise((r) => setTimeout(r, 300));
        if (document.querySelector('textarea') && document.querySelector('textarea').value === 'trigger the valve') {
          const send = [...document.querySelectorAll('button')].find((b) =>
            (b.getAttribute('aria-label') || '').match(/send/i) || b.type === 'submit');
          if (send) send.click();
        }
        await wait(() => document.body.innerText.includes('Message not sent'), 30000);
        const text = document.body.innerText;
        const chats = await window.api.chat.getAll().catch(() => []);
        const uiChat = chats[0];
        const messages = uiChat ? await window.api.chat.getMessages(uiChat.id).catch(() => []) : [];
        const assistant = [...messages].reverse().find((m) => m.role === 'assistant');
        return {
          bubbleVisible: text.includes('Message not sent'),
          restartGuidanceVisible: text.includes('restart this model'),
          valveProseVisible: text.includes('prefill admission rejected chunk'),
          bubblePersisted: !!assistant && typeof assistant.content === 'string'
            && assistant.content.startsWith('Message not sent — ')
            && assistant.content.includes('restart this model'),
          assistantContentHead: (assistant?.content || '').slice(0, 300),
          chatTitle: uiChat?.title || null,
          messageCount: messages.length,
        };
      })()
    `)
    const proofDir = path.join(repoDir, 'build', 'private-evidence')
    mkdirSync(proofDir, { recursive: true })
    const screenshot = await capturePng(cdp, path.join(proofDir, `${proofBasename}-chat.png`))
    const result = {
      generatedAt: new Date().toISOString(),
      script: 'panel/scripts/valve-decline-bubble-proof.mjs',
      repoDir,
      panelDir,
      mockRequests: mock.requests.map((r) => r.url),
      ...rendererResult,
      visual,
      screenshot,
    }
    const failures = []
    if (!visual.bubblePersisted) failures.push('bubble not persisted in chat history: ' + visual.assistantContentHead)
    if (!visual.bubbleVisible) failures.push('bubble not visible in the rendered chat')
    if (!visual.restartGuidanceVisible) failures.push('restart guidance not visible')
    if (!visual.valveProseVisible) failures.push('valve prose not visible')
    result.ok = failures.length === 0
    result.failures = failures
    writeFileSync(path.join(proofDir, `${proofBasename}-proof.json`), JSON.stringify(result, null, 2))
    if (failures.length) throw new Error('Valve bubble proof failed:\n- ' + failures.join('\n- '))
    console.log(JSON.stringify({ ok: true, screenshot, proof: `${proofBasename}-proof.json`, visual }, null, 2))
  } finally {
    try { cdp?.close() } catch {}
    try { process.kill(-app.pid, 'SIGTERM') } catch {}
    await sleep(1500)
    try { process.kill(-app.pid, 'SIGKILL') } catch {}
    await mock.close().catch(() => {})
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
