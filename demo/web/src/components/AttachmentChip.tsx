import { ImagePlaceholder } from "./ImagePlaceholder";

export type AttachmentChipProps = {
  attachmentId: string;
  mediaType?: string;
  audience?: string;
  expanded?: boolean;
};

export function AttachmentChip({ attachmentId, mediaType, audience, expanded = false }: AttachmentChipProps) {
  if (expanded) {
    return (
      <div className="att-card">
        <ImagePlaceholder attachmentId={attachmentId} mediaType={mediaType} audience={audience} size="lg" />
        <div className="att-card-meta">
          <div className="att-card-id">[att:{attachmentId}]</div>
          <div className="att-card-caption">{mediaType ?? "image attachment"}</div>
          <div className="att-card-stats">
            <span>{audience ?? "unscoped"}</span>
            <span>·</span>
            <span>backend bytes when available</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="att-chip" title={mediaType ?? attachmentId}>
      <ImagePlaceholder attachmentId={attachmentId} mediaType={mediaType} audience={audience} size="sm" />
      <div className="att-chip-meta">
        <div className="att-chip-id">[att:{attachmentId}]</div>
        <div className="att-chip-hash">{mediaType ?? "image"}</div>
      </div>
    </div>
  );
}
