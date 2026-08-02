#!/usr/bin/env node
/* R21 DSV4 campaign UI driver (CDP). Acts on the real visible Chat UI.
 * Usage: node r21drv.cjs <cmd> [args...]
 *   state                  - dump dialogs/buttons/inputs/text
 *   shot <path>            - screenshot
 *   click <text>           - click first visible element matching text
 *   clicknth <text> <n>    - click nth match
 *   type <selector> <text> - fill input
 *   press <key>            - keyboard press
 *   send <text>            - type into the chat composer and submit
 *   chat                   - dump the rendered chat transcript
 */
const { chromium } = require('playwright-core')
const CDP = process.env.VMLX_CDP || 'http://127.0.0.1:19759'

async function getPage(browser) {
  for (const ctx of browser.contexts()) {
    for (const p of ctx.pages()) {
      const u = p.url()
      if (u.includes('5173') || u.includes('index.html') || u.startsWith('file:')) return p
    }
  }
  return browser.contexts()[0].pages()[0]
}

async function main() {
  const [cmd, a1, a2] = process.argv.slice(2)
  const browser = await chromium.connectOverCDP(CDP)
  const page = await getPage(browser)
  if (!page) { console.error('NO_PAGE'); process.exit(2) }
  await page.waitForLoadState('domcontentloaded').catch(() => {})

  if (cmd === 'state') {
    const info = await page.evaluate(() => {
      const vis = (el) => {
        const r = el.getBoundingClientRect()
        const s = getComputedStyle(el)
        return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'
      }
      const t = (el) => (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().replace(/\s+/g, ' ').slice(0, 90)
      const leaf = [...document.querySelectorAll('div,span,p,li,td,label,h1,h2,h3,code')]
        .filter(e => e.children.length === 0 && vis(e) && (e.innerText || '').trim())
        .map(e => e.innerText.trim().replace(/\s+/g, ' ').slice(0, 110))
      return {
        buttons: [...document.querySelectorAll('button,[role=button]')].filter(vis).map(t).filter(Boolean).slice(0, 60),
        inputs: [...document.querySelectorAll('input,textarea')].filter(vis).map(e => ({
          tag: e.tagName.toLowerCase(), type: e.type || '', ph: e.placeholder || '', val: (e.value || '').slice(0, 60),
        })),
        text: [...new Set(leaf)].slice(0, 70),
      }
    })
    console.log(JSON.stringify(info, null, 1))
  }

  if (cmd === 'shot') {
    await page.screenshot({ path: a1 || '/tmp/r21.png', fullPage: false })
    console.log('SHOT', a1 || '/tmp/r21.png')
  }

  if (cmd === 'click') {
    await page.getByText(a1, { exact: false }).first().click({ timeout: 8000 })
    console.log('CLICKED', a1)
  }

  if (cmd === 'clicknth') {
    await page.getByText(a1, { exact: false }).nth(Number(a2 || 0)).click({ timeout: 8000 })
    console.log('CLICKED', a1, a2)
  }

  if (cmd === 'type') {
    await page.fill(a1, a2, { timeout: 8000 })
    console.log('TYPED', a1)
  }

  if (cmd === 'xy') {
    await page.mouse.click(Number(a1), Number(a2))
    console.log('CLICKED_XY', a1, a2)
  }

  if (cmd === 'vp') {
    const v = await page.evaluate(() => ({ w: innerWidth, h: innerHeight, dpr: devicePixelRatio }))
    console.log(JSON.stringify(v))
  }

  if (cmd === 'scroll') {
    await page.mouse.move(Number(a1 || 1200), 500)
    await page.mouse.wheel(0, Number(a2 || 600))
    console.log('SCROLLED', a2)
  }

  if (cmd === 'toggles') {
    const out = await page.evaluate(() => {
      const vis = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 }
      return [...document.querySelectorAll('input[type=checkbox],[role=switch],button[aria-pressed]')]
        .filter(vis).map(el => ({
          kind: el.tagName.toLowerCase() + (el.type ? ':' + el.type : ''),
          checked: el.checked ?? el.getAttribute('aria-checked') ?? el.getAttribute('aria-pressed'),
          label: (el.closest('label')?.innerText || el.parentElement?.innerText || '').trim().replace(/\s+/g, ' ').slice(0, 70),
        }))
    })
    console.log(JSON.stringify(out, null, 1))
  }

  if (cmd === 'press') {
    await page.keyboard.press(a1)
    console.log('PRESSED', a1)
  }

  if (cmd === 'send') {
    // The chat composer only -- never the settings panel's system-prompt textarea.
    const box = page.locator('[placeholder="Message..."]').first()
    await box.click({ timeout: 8000 })
    await box.fill(a1, { timeout: 8000 })
    await page.keyboard.press('Enter')
    console.log('SENT', a1.slice(0, 60))
  }

  if (cmd === 'chat') {
    const msgs = await page.evaluate(() => {
      const vis = (el) => {
        const r = el.getBoundingClientRect()
        return r.width > 0 && r.height > 0
      }
      const nodes = [...document.querySelectorAll('[class*=message], [class*=Message], [data-role]')]
        .filter(vis)
        .map(e => e.innerText.trim().replace(/\n{3,}/g, '\n\n').slice(0, 1500))
        .filter(Boolean)
      return [...new Set(nodes)]
    })
    console.log(JSON.stringify(msgs, null, 1))
  }

  await browser.close().catch(() => {})
}

main().catch(e => { console.error('ERR', e.message); process.exit(1) })
