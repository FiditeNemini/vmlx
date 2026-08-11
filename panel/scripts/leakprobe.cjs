#!/usr/bin/env node
/* Run an arbitrary DOM probe against the live dev-build app over CDP.
 *
 * uidrv.cjs takes probe JS as a shell argument, which mangles anything
 * containing quotes, backslashes or `&&`. This reads the probe from a FILE
 * instead, so a real assertion can be written without shell escaping.
 *
 * Usage:  node scripts/leakprobe.cjs /path/to/probe.js
 *         VMLX_CDP=http://127.0.0.1:9333 (default)
 *
 * The probe file must evaluate to a single expression — typically an IIFE
 * returning a JSON-serialisable object. Its result is printed as JSON.
 *
 * Example probe (marker-leak check):
 *   (() => {
 *     const M = ['<|start|>', '<|message|>', '<|eom|>', '<|eot|>'];
 *     const body = document.body.innerText;
 *     return { leaked: M.filter(m => body.includes(m)) };
 *   })()
 */
const { chromium } = require('playwright-core')
const fs = require('fs')

const CDP = process.env.VMLX_CDP || 'http://127.0.0.1:9333'

async function main() {
  const probePath = process.argv[2]
  if (!probePath) {
    console.error('usage: leakprobe.cjs <probe-file.js>')
    process.exit(2)
  }
  const browser = await chromium.connectOverCDP(CDP)
  let page
  for (const ctx of browser.contexts()) {
    for (const p of ctx.pages()) {
      const url = p.url()
      if (url.includes('5173') || url.includes('index.html') || url.startsWith('file:')) {
        page = p
      }
    }
  }
  if (!page) {
    console.error('NO_PAGE')
    process.exit(2)
  }
  const js = fs.readFileSync(probePath, 'utf8')
  console.log(JSON.stringify(await page.evaluate(js), null, 1))
  // Do not close the browser: over CDP that can terminate the live app.
  await new Promise((resolve) => process.stdout.write('', resolve))
  process.exit(0)
}

main().catch((e) => {
  console.error('PROBE_ERR', e.message)
  process.exit(1)
})
