import { useEffect, useMemo, useState, type CSSProperties } from "react";

import { attachmentBytesUrl } from "../api/client";

const IMG_SIZES = {
  xs: [28, 28],
  sm: [44, 44],
  md: [88, 64],
  lg: [160, 100],
  xl: [260, 160]
} as const;

export type ImagePlaceholderProps = {
  attachmentId?: string;
  mediaType?: string;
  audience?: string;
  size?: keyof typeof IMG_SIZES;
  quarantined?: boolean;
  onClick?: () => void;
  style?: CSSProperties;
};

function seedFrom(value: string): number {
  let acc = 0;
  for (let index = 0; index < value.length; index += 1) {
    acc = (acc + value.charCodeAt(index) * (index + 17)) % 997;
  }
  return acc / 997;
}

export function ImagePlaceholder({
  attachmentId,
  mediaType,
  audience,
  size = "md",
  quarantined = false,
  onClick,
  style
}: ImagePlaceholderProps) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [width, height] = IMG_SIZES[size] ?? IMG_SIZES.md;
  const seed = useMemo(() => seedFrom(attachmentId ?? mediaType ?? "image"), [attachmentId, mediaType]);

  useEffect(() => {
    if (attachmentId === undefined) {
      return undefined;
    }

    let cancelled = false;
    let localUrl: string | null = null;

    void fetch(attachmentBytesUrl(attachmentId, audience))
      .then((response) => (response.ok ? response.blob() : null))
      .then((blob) => {
        if (blob === null || cancelled) {
          return;
        }
        localUrl = URL.createObjectURL(blob);
        setObjectUrl(localUrl);
      })
      .catch(() => {
        setObjectUrl(null);
      });

    return () => {
      cancelled = true;
      if (localUrl !== null) {
        URL.revokeObjectURL(localUrl);
      }
    };
  }, [attachmentId, audience]);

  if (objectUrl !== null) {
    return (
      <div
        className={`img-ph ${quarantined ? "quarantined" : ""}`}
        onClick={onClick}
        style={{ width, height, cursor: onClick === undefined ? "default" : "pointer", ...style }}
        title={attachmentId}
      >
        <img src={objectUrl} alt={attachmentId ?? "attachment"} width={width} height={height} style={{ objectFit: "cover" }} />
        <div className="img-ph-label">{attachmentId ?? "image"}</div>
        {quarantined ? <div className="img-ph-quar">quarantined</div> : null}
      </div>
    );
  }

  const hue = Math.round(seed * 360);
  const c1 = `oklch(0.36 0.07 ${hue})`;
  const c2 = `oklch(0.28 0.05 ${hue})`;
  const c3 = `oklch(0.45 0.09 ${hue})`;
  const short = attachmentId ?? "image";
  const dots = Array.from({ length: 6 }, (_, index) => {
    const f = (seed * (index + 1) * 17.31) % 1;
    return {
      x: Math.floor(f * width),
      y: Math.floor(((f * 11.3) % 1) * height),
      r: 1 + ((f * 7) % 3)
    };
  });

  return (
    <div
      className={`img-ph ${quarantined ? "quarantined" : ""}`}
      onClick={onClick}
      style={{ width, height, cursor: onClick === undefined ? "default" : "pointer", ...style }}
      title={short}
    >
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} preserveAspectRatio="none">
        <defs>
          <pattern
            id={`stripe-${short.replace(/[^a-zA-Z0-9_-]/g, "_")}`}
            width="6"
            height="6"
            patternTransform={`rotate(${Math.floor(seed * 90)})`}
            patternUnits="userSpaceOnUse"
          >
            <rect width="6" height="6" fill={c1} />
            <line x1="0" y1="0" x2="0" y2="6" stroke={c2} strokeWidth="1" />
          </pattern>
        </defs>
        <rect width={width} height={height} fill={`url(#stripe-${short.replace(/[^a-zA-Z0-9_-]/g, "_")})`} />
        {dots.map((dot, index) => (
          <circle key={index} cx={dot.x} cy={dot.y} r={dot.r} fill={c3} opacity="0.6" />
        ))}
      </svg>
      <div className="img-ph-label">{short}</div>
      {quarantined ? <div className="img-ph-quar">quarantined</div> : null}
    </div>
  );
}
