import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  requestsDirectAnswerAfterSingleTool,
  requestsNoToolCalls,
  shouldAutoContinueAfterToolUse,
  shouldFinishZayaAppleScriptToolRound,
} from '../src/shared/toolAutoContinue'

describe('tool auto-continue policy', () => {
  it('continues when a model stops after tools with no visible response', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: '',
        iterationTokenCount: 0,
        finishReason: 'stop',
        thresholdTokens: 100,
      }),
    ).toBe(true)
  })

  it('continues short content only when the model hit the length limit', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: 'partial sentence',
        iterationTokenCount: 4,
        finishReason: 'length',
        thresholdTokens: 100,
      }),
    ).toBe(true)
  })

  it('does not duplicate a short normal final answer after tool results', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: 'Done after tools.',
        iterationTokenCount: 4,
        finishReason: 'stop',
        thresholdTokens: 100,
      }),
    ).toBe(false)
  })

  it('finishes the specialized ZAYA AppleScript bundle after its native action result', () => {
    expect(
      shouldFinishZayaAppleScriptToolRound(true, ['run_applescript']),
    ).toBe(true)
    expect(
      shouldFinishZayaAppleScriptToolRound(false, ['run_applescript']),
    ).toBe(false)
    expect(
      shouldFinishZayaAppleScriptToolRound(true, ['run_applescript', 'read_file']),
    ).toBe(false)
    expect(shouldFinishZayaAppleScriptToolRound(true, [])).toBe(false)
  })

  it('recognizes only explicit one-tool exact-final contracts', () => {
    expect(
      requestsDirectAnswerAfterSingleTool(
        'Call file_info exactly once. After the tool result, reply exactly DONE and nothing else.',
      ),
    ).toBe(true)
    expect(
      requestsDirectAnswerAfterSingleTool(
        'Continue this same chat. Call the built-in file_info tool exactly once with path pyproject.toml. After the real tool result, reply exactly B1-NONE-MT2-DONE and nothing else.',
      ),
    ).toBe(true)
    expect(
      requestsDirectAnswerAfterSingleTool(
        'Call the built-in file_info tool exactly once with path panel/package.json. After its result, reply exactly B1-ELECTRON-TOOL-TEMPLATE1-DONE and nothing else.',
      ),
    ).toBe(true)
    expect(
      requestsDirectAnswerAfterSingleTool(
        'Use tools as needed, then reply exactly DONE.',
      ),
    ).toBe(false)
    expect(
      requestsDirectAnswerAfterSingleTool(
        'Call file_info exactly once, then summarize the result.',
      ),
    ).toBe(false)
    expect(
      requestsDirectAnswerAfterSingleTool(
        'Call file_info exactly once. After checking prerequisites. The tool result may be long; reply exactly DONE.',
      ),
    ).toBe(false)
  })

  it('maps an explicit current-turn no-tool directive to the API contract', () => {
    expect(
      requestsNoToolCalls(
        '[FOLLOW] Do not call any tool. Use only the previous result.',
      ),
    ).toBe(true)
    expect(requestsNoToolCalls('Please never use tools. Answer directly.')).toBe(true)
    expect(
      requestsNoToolCalls('Do not call any tool unless the file is missing.'),
    ).toBe(false)
    expect(
      requestsNoToolCalls('Explain why someone might say "do not call any tool".'),
    ).toBe(false)
  })

  it('sends tool_choice none and suppresses the generic tool prompt when requested', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('requestsNoToolCalls(latestUserText)')
    expect(source.match(/obj\.tool_choice = "none"/g) || []).toHaveLength(2)
    expect(source).toContain('!userForbidsToolCalls')
  })

  it('checks the terminal AppleScript round before sending a follow-up', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const policy = source.indexOf('shouldFinishZayaAppleScriptToolRound(')
    const terminalBreak = source.indexOf('if (finishAfterNativeToolResult)', policy)
    const followUp = source.indexOf('if (!(await sendFollowUp())) break;', policy)

    expect(policy).toBeGreaterThan(-1)
    expect(terminalBreak).toBeGreaterThan(policy)
    expect(followUp).toBeGreaterThan(terminalBreak)
  })

  it('increments the auto-continue counter once per follow-up attempt', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('shouldAutoContinueAfterToolUse({'),
      source.indexOf('const hasContent = fullContent.trim().length > 0'),
    )

    expect(branch.match(/autoContinueCount\+\+/g) || []).toHaveLength(1)
  })

  it('uses one answer-only recovery instead of repeating reasoning-only tool follow-ups', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const MAX_AUTO_CONTINUES = 1')
    expect(source).toContain('let finalAnswerRecovery = false')
    expect(source).toContain('finalAnswerRecovery = true')
    expect(source).toContain('delete obj.tools')
    expect(source).toContain('obj.enable_thinking = false')
    expect(source).toContain(
      'The tool completed, but the model produced no visible answer after one direct-answer recovery.',
    )
  })

  it('removes tools for an explicit single-tool exact-final follow-up', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const planned = source.indexOf('plannedDirectAnswerPass =')
    const followUp = source.indexOf('if (!(await sendFollowUp())) break;', planned)

    expect(source).toContain('requestsDirectAnswerAfterSingleTool(latestUserText)')
    expect(source).toContain('if (!(finalAnswerRecovery || plannedDirectAnswerPass)) return')
    expect(planned).toBeGreaterThan(-1)
    expect(followUp).toBeGreaterThan(planned)
  })

  it('resets token timing at follow-up stream boundaries and counts long in-stream gaps', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const followUp = source.slice(
      source.indexOf('const sendFollowUp = async'),
      source.indexOf('// ─── Helper: execute tool calls', source.indexOf('const sendFollowUp = async')),
    )

    expect(followUp).toContain('lastTokenTime = null')
    expect(source).toContain('if (gap > 0) generationMs += gap')
    expect(source).not.toContain('if (gap < 5000) generationMs += gap')
  })

  it('does not replace the measured live stream rate with a buffered usage burst', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const finalTps =')
    expect(source).toContain('liveTps > 0')
    expect(source).toContain('? liveTps')
  })

  it('drops only superseded empty-response warnings after a successful recovery', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const start = source.indexOf('if (\n          finalAnswerRecovery &&')
    const end = source.indexOf('if (\n          toolIteration > 0 &&', start)
    const branch = source.slice(start, end)

    expect(start).toBeGreaterThan(-1)
    expect(branch).toContain('allGeneratedContent.trim() || fullContent.trim()')
    expect(branch).toContain('dropSupersededRecoveryWarnings(responseWarnings)')
  })

  it('resets text-chat tool streaming state before chained follow-up requests', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('receivedToolCalls = [];'),
      source.indexOf('if (!(await sendFollowUp())) break;', source.indexOf('receivedToolCalls = [];')),
    )

    for (const required of [
      'receivedToolCalls = []',
      'fullContent = ""',
      'rawAccumulated = ""',
      'lastFinishReason = undefined',
      'clientToolCallBuffering = false',
      'clientSideThinkParsing = false',
      'serverSendsUsage = false',
      'currentEventType = ""',
      'seenResponsesApiEvents.clear()',
    ]) {
      expect(branch).toContain(required)
    }
  })

  it('clears Responses tool-call buffers before follow-up and on stalled buffering', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const toolLoopStart = source.indexOf('await executeToolCalls()')
    const followUpStart = source.indexOf('if (!(await sendFollowUp())) break;', toolLoopStart)
    const followUpBranch = source.slice(toolLoopStart, followUpStart)

    expect(toolLoopStart).toBeGreaterThan(-1)
    expect(followUpStart).toBeGreaterThan(toolLoopStart)
    expect(followUpBranch).toContain('receivedToolCalls = []')
    expect(followUpBranch).toContain('clientToolCallBuffering = false')
    expect(followUpBranch.indexOf('receivedToolCalls = []')).toBeLessThan(
      followUpBranch.indexOf('clientToolCallBuffering = false'),
    )

    const stallStart = source.indexOf('Tool call generation stalled')
    const stallBranch = source.slice(stallStart, source.indexOf('await rdr.cancel()', stallStart) + 200)

    expect(stallStart).toBeGreaterThan(-1)
    expect(stallBranch).toContain('clientToolCallBuffering = false')
    expect(stallBranch).toContain('await rdr.cancel()')
    expect(stallBranch).not.toContain('executeToolCalls')
  })

  it('responses stream parser accepts data-only event types from parsed payloads', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const responsesEventType =')
    expect(source).toContain(
      'typeof parsed.type === "string" ? parsed.type : currentEventType',
    )

    const functionCallBranch = source.slice(
      source.indexOf('// Handle function_call items (tool calls) from Responses API'),
      source.indexOf('// Real-time usage from response.usage events'),
    )

    expect(functionCallBranch).toContain(
      'responsesEventType === "response.output_item.done"',
    )
  })

  it('loopback remote sessions use node streaming fetch for SSE', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('function isLoopbackUrl')
    expect(source).toContain('const useNodeStreamingFetch =')
    expect(source).toContain('!isRemote || isLoopbackUrl(apiUrl)')
    expect(source).toContain('!isRemote || isLoopbackUrl(url)')
  })

  it('suppresses generic agentic instructions for native ZAYA and LFM2 prompts', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('function shouldSuppressGenericAgenticPromptForNativeTools')
    expect(source).toContain('detectedFamily === "zaya"')
    expect(source).toContain('detectedFamily === "zaya1-vl"')
    expect(source).toContain('detectedFamily === "zaya1_vl"')
    expect(source).toContain('detectedFamily === "lfm2"')
    expect(source).toContain('modelNameOrPath')
    expect(source).toContain('modelName.includes("zaya")')
    expect(source).toContain('modelName.includes("lfm2")')
    expect(source).toContain('const suppressGenericAgenticToolPromptForNativeTools =')

    const promptBranch = source.slice(
      source.indexOf('const suppressGenericAgenticToolPromptForNativeTools ='),
      source.indexOf('// No default system prompt injected'),
    )
    expect(promptBranch).toContain('!suppressGenericAgenticToolPromptForNativeTools')
    expect(promptBranch).toContain('chat.modelPath || chat.modelId')
    expect(promptBranch).toContain('AGENTIC_SYSTEM_PROMPT + directMediaAttachmentRule')
    expect(promptBranch).toContain('directMediaAttachmentRule.trim()')
  })

  it('panel max tool iterations caps tool loops', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10;'),
      source.indexOf('if (toolIteration > 0 || collectedToolStatuses.length > 0)'),
    )

    expect(branch).toContain('const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10')
    expect(branch).toContain('while (toolIteration < MAX_TOOL_ITERATIONS)')
    expect(branch).toContain('toolIteration++')
  })
})
