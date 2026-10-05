import { useState } from 'react'
import { useTranslation } from '../../i18n'

/** Browser codec support is independent of the model's media decoder. */
export function VideoAttachmentPreview({ src }: { src: string }) {
  const { t } = useTranslation()
  const [failedSrc, setFailedSrc] = useState<string | null>(null)
  const failed = failedSrc === src
  return (
    <div className="max-w-[360px]">
      <video
        src={src}
        controls
        preload="metadata"
        onError={() => setFailedSrc(src)}
        onLoadedMetadata={() => setFailedSrc(null)}
        className="max-w-[360px] max-h-[240px] rounded-md border border-white/10 bg-black"
        hidden={failed}
      />
      {failed && (
        <p role="status" className="p-2 text-xs text-inherit border border-current/30 rounded-md"
          title={t('chat.bubble.videoPreviewUnavailableDetail')}>
          {t('chat.bubble.videoPreviewUnavailable')}
          <span className="block mt-1">{t('chat.bubble.videoPreviewUnavailableDetail')}</span>
        </p>
      )}
    </div>
  )
}
