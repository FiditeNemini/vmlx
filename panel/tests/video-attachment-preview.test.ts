import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { VideoAttachmentPreview } from '../src/renderer/src/components/chat/VideoAttachmentPreview'
const state = vi.hoisted(() => ({ failedSrc: null as string | null }))
vi.mock('react', async () => ({ ...(await vi.importActual<typeof import('react')>('react')),
  useState: () => [state.failedSrc, (value: string | null) => { state.failedSrc = value }],
}))
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
const html = (src: string) => renderToStaticMarkup(React.createElement(VideoAttachmentPreview, {src}))
describe('video attachment preview errors', () => {
  it('preserves supported controls and original source', () => {
    state.failedSrc = null
    expect(html('clip.mp4')).toContain('controls=""')
    expect(html('clip.mp4')).toContain('src="clip.mp4"')
    expect(html('clip.mp4')).not.toContain('role="status"')
  })
  it('reports failure while retaining the attachment source', () => {
    state.failedSrc = 'clip.mp4'
    expect(html('clip.mp4')).toContain('role="status"')
    expect(html('clip.mp4')).toContain('videoPreviewUnavailableDetail')
    expect(html('clip.mp4')).toContain('src="clip.mp4"')
  })
  it('does not carry failure onto a replacement source', () => {
    state.failedSrc = 'old.mp4'
    expect(html('new.mp4')).not.toContain('role="status"')
  })
  it('wires browser error and recovery', () => {
    const tree = VideoAttachmentPreview({ src: 'clip.mp4' })
    const video = tree.props.children[0]
    video.props.onError()
    expect(state.failedSrc).toBe('clip.mp4')
    video.props.onLoadedMetadata()
    expect(state.failedSrc).toBeNull()
  })
})
