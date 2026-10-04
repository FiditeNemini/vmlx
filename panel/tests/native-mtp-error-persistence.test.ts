import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { nativeMtpChatErrorWarning } from '../src/shared/chatErrorDisplay'

const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
const tree = ts.createSourceFile('chat.ts', source, ts.ScriptTarget.Latest, true)
function find(node: ts.Node, predicate: (n: ts.Node) => boolean): ts.Node | undefined {
  if (predicate(node)) return node
  let result: ts.Node | undefined
  node.forEachChild(n => { result ||= find(n, predicate) })
  return result
}
const catchNode = find(tree, n => ts.isCatchClause(n) && n.getText(tree).includes('const nativeMtpErrorWarning')) as ts.CatchClause
const statements = [...catchNode.block.statements]
const declaration = (name: string) => statements.find(n => ts.isVariableStatement(n) && n.declarationList.declarations.some(d => d.name.getText(tree) === name))!
const save = statements.find(n => ts.isIfStatement(n) && n.expression.getText(tree) === 'hadVisibleActivity') as ts.IfStatement
const saveBody = (save.thenStatement as ts.Block).statements
// Execute the actual owning content/diagnostic/reasoning statements, without metrics/DB/network.
const owned = saveBody.filter(n => {
  const s = n.getText(tree)
  return s.startsWith('assistantMessage.content =') || s.startsWith('assistantMessage.tokens =') ||
    s.startsWith('if (abortWarnings.length') || s.startsWith('if (abortReasoningContent)') ||
    s.startsWith('if (abortReasoningSegments.length')
})
const complete = find(save, n => ts.isCallExpression(n) && n.expression.getText(tree) === 'win.webContents.send' && n.arguments[0]?.getText(tree) === '"chat:complete"') as ts.CallExpression
function run(code: string, env: Record<string, unknown>): any {
  const js = ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function(...Object.keys(env), js)(...Object.values(env))
}
function persist(partialContent: string, reasoning: string, aborted = false) {
  const assistantMessage: any = { id: 'answer' }
  const env: any = { assistantMessage, partialContent, abortReasoningContent: reasoning, abortReasoningSegments: reasoning ? [reasoning] : [],
    abortTotalTokens: reasoning ? 69 : 0, projectedMetalHeadroomErrorContent: null, promptTooLongErrorContent: null, runtimeFailureErrorContent: null,
    collectedToolStatuses: [], abortController: { signal: { aborted } }, errMsg: 'Server error: Stream generation failed: RuntimeError: NativeMTPError: pending-verify cache rejected rollback',
    responseWarnings: ['Existing notice'], nativeMtpChatErrorWarning,
    chatId: 'chat', proofRequestId: undefined, wireRequestIds: [], activeRequests: new Map(), abortFinishReason: null, abortMetrics: {},
    // The abort completion payload carries the finalized display timeline
    // (undefined when the recorder could not admit the interrupted stream).
    generationRecord: { version: 1, status: 'interrupted', passes: [], displayTimeline: undefined },
  }
  return run(`${declaration('nativeMtpErrorWarning').getText(tree)}\n${declaration('abortWarnings').getText(tree)}\n${declaration('hadVisibleActivity').getText(tree)}\nif (hadVisibleActivity) {${owned.map(n => n.getText(tree)).join('\n')}}\nreturn { assistantMessage, hadVisibleActivity: Boolean(hadVisibleActivity), complete: ${complete.arguments[1].getText(tree)} };`, env)
}

describe('native MTP persistent diagnostic', () => {
  it('recognizes actual wrapped errors without classifying unrelated failures', () => {
    expect(nativeMtpChatErrorWarning('Server error: NativeMTPError: rollback failed')).toBe('Generation failed: NativeMTPError: rollback failed')
    expect(nativeMtpChatErrorWarning('API error: 500 - {"error":{"message":"NativeMTPError: rollback failed"}}')).toContain('NativeMTPError: rollback failed')
    const captured = 'Server error: Stream generation failed: RuntimeError: NativeMTPError: native MTP pending-verify rollback failed'
    const diagnostic = 'Generation failed: NativeMTPError: native MTP pending-verify rollback failed'
    expect(nativeMtpChatErrorWarning(captured)).toBe(diagnostic)
    expect(nativeMtpChatErrorWarning(`API error: 500 - ${JSON.stringify({ error: { message: captured } })}`)).toBe(diagnostic)
    expect(nativeMtpChatErrorWarning(JSON.stringify({ detail: captured }))).toBe(diagnostic)
    expect(nativeMtpChatErrorWarning('Stream generation failed: RuntimeError: unrelated NativeMTPError: text')).toBeNull()
    for (const value of ['fetch failed', 'AbortError', 'The answer mentions NativeMTPError:', 'NativeMTPError:', null]) expect(nativeMtpChatErrorWarning(value)).toBeNull()
  })
  it.each([['Partial answer', 'Real reasoning'], ['', 'Real reasoning'], ['', '']])('persists error alongside content %j and reasoning %j', (content, reasoning) => {
    const { assistantMessage: m, complete, hadVisibleActivity } = persist(content, reasoning)
    expect(hadVisibleActivity).toBe(true)
    expect(m.content).toBe(content ? `${content}\n\n[Generation interrupted]` : '[Generation interrupted]')
    expect(m.reasoningContent).toBe(reasoning || undefined)
    expect(m.reasoningSegmentsJson).toBe(reasoning ? JSON.stringify([reasoning]) : undefined)
    expect(JSON.parse(m.warningsJson)).toEqual(['Existing notice', 'Generation failed: NativeMTPError: pending-verify cache rejected rollback'])
    expect(complete.warnings).toEqual(JSON.parse(m.warningsJson))
    expect(complete.content).toBe(m.content)
    expect(complete.reasoningContent).toBe(reasoning || undefined)
    // Replay reads authored content and strips its existing UI marker, not diagnostic metadata.
    const replayIf = find(tree, n => ts.isIfStatement(n) && n.expression.getText(tree) === 'm.role === "assistant" && typeof msgContent === "string"') as ts.IfStatement
    const strip = (replayIf.thenStatement as ts.Block).statements[0].getText(tree)
    const replay = run(`let msgContent = m.content; ${strip}; return msgContent;`, { m })
    expect(replay).toBe(content)
    expect(replay).not.toContain('NativeMTPError')
  })
  it('does not relabel user cancellation as backend failure', () => {
    const { assistantMessage: m, complete } = persist('Partial answer', 'Real reasoning', true)
    expect(JSON.parse(m.warningsJson)).toEqual(['Existing notice'])
    expect(complete.warnings).toEqual(['Existing notice'])
  })
})
