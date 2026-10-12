import { readFileSync } from 'node:fs'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ToastItem } from '../src/renderer/src/components/Toast'

// The toast stack sits in the bottom-right corner over the Chat Settings Save button and the composer's Send
// button; an error toast stays 10 s. In-app row 2026-10-11 (Gemma-4, audio-400 turn): the next Save click landed on
// the toast, the settings change was never saved and the following turn silently ran without tools. Toasts are
// click-through except their own buttons.
describe('toast stack does not swallow clicks meant for controls beneath it', () => {
  it('the stack is pointer-events-none', () => {
    const source = readFileSync('src/renderer/src/components/Toast.tsx', 'utf8')
    expect(source).toMatch(/className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md pointer-events-none"/)
  })

  it('only the dismiss button takes clicks back', () => {
    const html = renderToStaticMarkup(React.createElement(ToastItem, {
      toast: { id: 1, type: 'error', title: 'Message failed', detail: '400 unsupported media modality audio' },
      onDismiss: () => {},
    }))
    expect(html).toContain('data-vmlx-control="toast-dismiss"')
    expect(html).toMatch(/<button[^>]*class="[^"]*pointer-events-auto[^"]*"/)
    // the card itself must not opt back in
    expect(html.split('<button')[0]).not.toContain('pointer-events-auto')
  })
})
