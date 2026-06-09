import { useId } from "react";

import type { MoodHistoryEntry, MoodSnapshot, SocialMemoryItem } from "../../api/types";

function scopedSvgId(prefix: string, reactId: string): string {
  return `${prefix}-${reactId.replace(/:/g, "")}`;
}

function clampRatio(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function scale(
  value: number,
  min: number,
  max: number,
  outputMin: number,
  outputMax: number,
): number {
  if (max <= min) {
    return (outputMin + outputMax) / 2;
  }
  const ratio = clampRatio((value - min) / (max - min));
  return outputMin + ratio * (outputMax - outputMin);
}

export function ValenceArousalPlane({
  current,
  history,
}: {
  current: MoodSnapshot;
  history: readonly MoodHistoryEntry[];
}) {
  const reactId = useId();
  const gradientId = scopedSvgId("matlas-plane-current", reactId);
  const clipId = scopedSvgId("matlas-plane-clip", reactId);
  const width = 360;
  const height = 210;
  const pad = 28;
  const plotWidth = width - pad * 2;
  const plotHeight = height - pad * 2;
  const arousalValues = [current.arousal, ...history.map((point) => point.arousal)];
  const arousalMin = Math.min(0, ...arousalValues);
  const arousalMax = Math.max(1, ...arousalValues);
  const points = [...history].sort((left, right) => left.ts - right.ts);

  const x = (valence: number) => scale(valence, -1, 1, pad, width - pad);
  const y = (arousal: number) => scale(arousal, arousalMin, arousalMax, height - pad, pad);

  return (
    <svg
      className="matlas-plane"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="valence arousal plane"
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={pad} y={pad} width={plotWidth} height={plotHeight} rx="4" />
        </clipPath>
        <radialGradient id={gradientId}>
          <stop offset="0%" stopColor="var(--acc)" stopOpacity="1" />
          <stop offset="100%" stopColor="var(--acc)" stopOpacity="0.25" />
        </radialGradient>
      </defs>
      <rect className="matlas-plane-frame" x={pad} y={pad} width={plotWidth} height={plotHeight} />
      <line className="matlas-plane-axis" x1={x(0)} y1={pad} x2={x(0)} y2={height - pad} />
      <line className="matlas-plane-axis" x1={pad} y1={y(0)} x2={width - pad} y2={y(0)} />
      <g clipPath={`url(#${clipId})`}>
        {points.map((point, index) => {
          const opacity = points.length <= 1 ? 0.78 : 0.28 + (index / (points.length - 1)) * 0.5;
          return (
            <circle
              key={point.id}
              className="matlas-plane-history-point"
              data-testid="matlas-plane-history-point"
              cx={x(point.valence)}
              cy={y(point.arousal)}
              r="4"
              opacity={opacity}
            />
          );
        })}
        {points.length > 1 ? (
          <polyline
            className="matlas-plane-trail"
            points={points.map((point) => `${x(point.valence)},${y(point.arousal)}`).join(" ")}
          />
        ) : null}
      </g>
      <circle
        className="matlas-plane-current"
        data-testid="matlas-plane-current"
        cx={x(current.valence)}
        cy={y(current.arousal)}
        r="8"
        fill={`url(#${gradientId})`}
      />
      <text className="matlas-plane-label" x={pad} y={height - 8}>
        valence
      </text>
      <text className="matlas-plane-label" x={width - 84} y={18}>
        arousal
      </text>
    </svg>
  );
}

export function SocialTrustScatter({
  items,
  selectedId,
  onSelect,
}: {
  items: readonly SocialMemoryItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const reactId = useId();
  const glowId = scopedSvgId("matlas-social-glow", reactId);
  const width = 360;
  const height = 220;
  const pad = 30;
  const trustValues = items.map((item) => item.trust);
  const attachmentValues = items.map((item) => item.attachment);
  const trustMin = Math.min(0, ...trustValues);
  const trustMax = Math.max(1, ...trustValues);
  const attachmentMin = Math.min(0, ...attachmentValues);
  const attachmentMax = Math.max(1, ...attachmentValues);
  const maxInteractions = Math.max(1, ...items.map((item) => item.interaction_count));

  const x = (trust: number) => scale(trust, trustMin, trustMax, pad, width - pad);
  const y = (attachment: number) =>
    scale(attachment, attachmentMin, attachmentMax, height - pad, pad);

  return (
    <svg
      className="matlas-social-scatter"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="trust attachment scatter"
    >
      <defs>
        <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect
        className="matlas-scatter-frame"
        x={pad}
        y={pad}
        width={width - pad * 2}
        height={height - pad * 2}
      />
      <line
        className="matlas-scatter-axis"
        x1={pad}
        y1={height - pad}
        x2={width - pad}
        y2={height - pad}
      />
      <line className="matlas-scatter-axis" x1={pad} y1={pad} x2={pad} y2={height - pad} />
      {items.map((item) => {
        const label = item.name ?? item.entity_id;
        const radius = 4 + Math.sqrt(item.interaction_count / maxInteractions) * 8;
        const selected = selectedId === item.entity_id;
        return (
          <g
            key={item.entity_id}
            className={`matlas-scatter-dot ${selected ? "selected" : ""}`}
            role="button"
            tabIndex={0}
            aria-label={`select ${label}`}
            onClick={() => onSelect(item.entity_id)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(item.entity_id);
              }
            }}
          >
            <circle
              cx={x(item.trust)}
              cy={y(item.attachment)}
              r={radius}
              filter={selected ? `url(#${glowId})` : undefined}
              data-testid="matlas-social-dot"
            />
            <text x={x(item.trust) + radius + 4} y={y(item.attachment) + 3}>
              {label}
            </text>
          </g>
        );
      })}
      <text className="matlas-scatter-label" x={pad} y={height - 8}>
        trust
      </text>
      <text className="matlas-scatter-label" x={width - 112} y={18}>
        attachment
      </text>
    </svg>
  );
}
