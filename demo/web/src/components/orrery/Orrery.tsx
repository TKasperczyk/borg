import { useId, type KeyboardEvent } from "react";

import { isDreamProcessName, type RouteId, type RouteNavigationOptions } from "../../routes";
import { shortId } from "../../screens/screen-utils";
import type { OrreryViewModel } from "./useOrreryData";

export type OrreryInspectTarget = {
  type: "turn";
  id: string;
  hint?: unknown;
};

export type OrreryProps = {
  size: "full" | "compact";
  data: OrreryViewModel;
  onNavigate: (view: RouteId, options?: RouteNavigationOptions) => void;
  onInspect: (target: OrreryInspectTarget) => void;
};

type Point = {
  x: number;
  y: number;
};

const VIEW_SIZE = 560;
const CENTER = VIEW_SIZE / 2;
const MEMORY_RING_START = 72;
const MEMORY_RING_STEP = 18;
const MEMORY_LABEL_ANGLE_START = -126;
const DREAM_ORBIT_RADIUS = 232;
const FAULT_LIMIT = 5;
const LABEL_PAD = 14;
const LABEL_GAP = 7;

type Box = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

function classNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

function polarPoint(radius: number, angleDegrees: number): Point {
  const angle = ((angleDegrees - 90) * Math.PI) / 180;
  return {
    x: CENTER + radius * Math.cos(angle),
    y: CENTER + radius * Math.sin(angle),
  };
}

function arcPath(radius: number, startAngle: number, endAngle: number): string {
  const start = polarPoint(radius, endAngle);
  const end = polarPoint(radius, startAngle);
  const largeArc = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${radius} ${radius} 0 ${largeArc} 0 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
}

function onKeyboardActivate(event: KeyboardEvent<SVGGElement>, action: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function countLabel(count: number, lowerBound: boolean): string {
  const formatted = count.toLocaleString();
  return lowerBound ? `${formatted}+` : formatted;
}

function memoryStrokeWidth(count: number, maxCount: number): number {
  if (count <= 0 || maxCount <= 0) {
    return 1.2;
  }
  return 1.2 + (Math.log1p(count) / Math.log1p(maxCount)) * 4.2;
}

function governanceSweep(count: number, criticalCount = 0): number {
  if (count <= 0) {
    return 0;
  }
  const countScale = Math.log1p(Math.min(count, 24)) / Math.log1p(24);
  const criticalBoost = criticalCount > 0 ? Math.min(18, criticalCount * 3) : 0;
  return Math.min(128, 28 + countScale * 82 + criticalBoost);
}

function governanceStrokeWidth(count: number): number {
  if (count <= 0) {
    return 0;
  }
  const countScale = Math.log1p(Math.min(count, 24)) / Math.log1p(24);
  return 2.5 + countScale * 4.5;
}

function estimateTextWidth(text: string, fontSize: number): number {
  return text.length * fontSize * 0.62;
}

function boxOverlaps(left: Box, right: Box): boolean {
  return (
    left.left < right.right &&
    left.right > right.left &&
    left.top < right.bottom &&
    left.bottom > right.top
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

type LabelSide = "left" | "right" | "top" | "bottom";

export type OrreryLaidOutLabel = {
  id: string;
  kind: "memory" | "satellite" | "governance" | "fault";
  text: string;
  subtext?: string;
  anchor: Point;
  side: LabelSide;
  x: number;
  y: number;
  textAnchor: "start" | "middle" | "end";
  box: Box;
  leader: string;
  badge?: Box;
};

type LabelCandidate = Omit<OrreryLaidOutLabel, "box" | "leader" | "textAnchor" | "x" | "y"> & {
  preferredX: number;
  preferredY: number;
  width: number;
  height: number;
};

type PlacedLabelCandidate = LabelCandidate &
  Pick<OrreryLaidOutLabel, "box" | "textAnchor" | "x" | "y">;

function labelBoxFor(
  candidate: LabelCandidate,
  crossAxis: number,
): Pick<OrreryLaidOutLabel, "box" | "textAnchor" | "x" | "y"> {
  if (candidate.side === "left") {
    const right = clamp(
      Math.min(candidate.preferredX, candidate.anchor.x - 18),
      LABEL_PAD + candidate.width,
      CENTER - 34,
    );
    return {
      x: right,
      y: crossAxis,
      textAnchor: "end",
      box: {
        left: right - candidate.width,
        right,
        top: crossAxis - candidate.height / 2,
        bottom: crossAxis + candidate.height / 2,
      },
    };
  }

  if (candidate.side === "right") {
    const left = clamp(
      Math.max(candidate.preferredX, candidate.anchor.x + 18),
      CENTER + 34,
      VIEW_SIZE - LABEL_PAD - candidate.width,
    );
    return {
      x: left,
      y: crossAxis,
      textAnchor: "start",
      box: {
        left,
        right: left + candidate.width,
        top: crossAxis - candidate.height / 2,
        bottom: crossAxis + candidate.height / 2,
      },
    };
  }

  const centerX = clamp(
    crossAxis,
    LABEL_PAD + candidate.width / 2,
    VIEW_SIZE - LABEL_PAD - candidate.width / 2,
  );
  const centerY =
    candidate.side === "top"
      ? clamp(candidate.preferredY, LABEL_PAD + candidate.height / 2, CENTER - 42)
      : clamp(candidate.preferredY, CENTER + 42, VIEW_SIZE - LABEL_PAD - candidate.height / 2);
  return {
    x: centerX,
    y: centerY,
    textAnchor: "middle",
    box: {
      left: centerX - candidate.width / 2,
      right: centerX + candidate.width / 2,
      top: centerY - candidate.height / 2,
      bottom: centerY + candidate.height / 2,
    },
  };
}

function leaderPath(anchor: Point, box: Box, side: LabelSide): string {
  if (side === "left") {
    const y = clamp(anchor.y, box.top + 4, box.bottom - 4);
    const elbowX = box.right + 8;
    return `M ${anchor.x.toFixed(1)} ${anchor.y.toFixed(1)} L ${elbowX.toFixed(1)} ${y.toFixed(1)} L ${box.right.toFixed(1)} ${y.toFixed(1)}`;
  }
  if (side === "right") {
    const y = clamp(anchor.y, box.top + 4, box.bottom - 4);
    const elbowX = box.left - 8;
    return `M ${anchor.x.toFixed(1)} ${anchor.y.toFixed(1)} L ${elbowX.toFixed(1)} ${y.toFixed(1)} L ${box.left.toFixed(1)} ${y.toFixed(1)}`;
  }
  if (side === "top") {
    const x = clamp(anchor.x, box.left + 4, box.right - 4);
    const elbowY = box.bottom + 8;
    return `M ${anchor.x.toFixed(1)} ${anchor.y.toFixed(1)} L ${x.toFixed(1)} ${elbowY.toFixed(1)} L ${x.toFixed(1)} ${box.bottom.toFixed(1)}`;
  }
  const x = clamp(anchor.x, box.left + 4, box.right - 4);
  const elbowY = box.top - 8;
  return `M ${anchor.x.toFixed(1)} ${anchor.y.toFixed(1)} L ${x.toFixed(1)} ${elbowY.toFixed(1)} L ${x.toFixed(1)} ${box.top.toFixed(1)}`;
}

function placeLane(
  candidates: readonly LabelCandidate[],
  side: LabelSide,
  avoidBoxes: readonly Box[],
): OrreryLaidOutLabel[] {
  const vertical = side === "left" || side === "right";
  const sorted = [...candidates].sort((left, right) =>
    vertical ? left.preferredY - right.preferredY : left.preferredX - right.preferredX,
  );
  const placed: PlacedLabelCandidate[] = [];

  if (!vertical) {
    for (const candidate of sorted) {
      const crossMin = LABEL_PAD + candidate.width / 2;
      const crossMax = VIEW_SIZE - LABEL_PAD - candidate.width / 2;
      let crossAxis = clamp(candidate.preferredX, crossMin, crossMax);

      for (let attempts = 0; attempts < 32; attempts += 1) {
        const proposed = labelBoxFor(candidate, crossAxis);
        const overlap = placed
          .map((item) => item.box)
          .concat(avoidBoxes)
          .find((box) => boxOverlaps(proposed.box, box));
        if (overlap === undefined) {
          placed.push({ ...candidate, ...proposed });
          break;
        }

        const gap = (candidate.width + (overlap.right - overlap.left)) / 2 + LABEL_GAP;
        const overlapCenter = (overlap.left + overlap.right) / 2;
        const direction = candidate.preferredX >= overlapCenter ? 1 : -1;
        crossAxis = clamp(overlapCenter + direction * gap, crossMin, crossMax);

        if (attempts === 31) {
          placed.push({ ...candidate, ...proposed });
        }
      }
    }

    return finalizePlacedLabels(placed);
  }

  let cursor = LABEL_PAD;

  for (const candidate of sorted) {
    const minCenter = LABEL_PAD + candidate.height / 2;
    const maxCenter = VIEW_SIZE - LABEL_PAD - candidate.height / 2;
    const crossMin = LABEL_PAD + candidate.width / 2;
    const crossMax = VIEW_SIZE - LABEL_PAD - candidate.width / 2;
    const half = vertical ? candidate.height / 2 : candidate.width / 2;
    const min = vertical ? minCenter : crossMin;
    const max = vertical ? maxCenter : crossMax;
    const preferred = vertical ? candidate.preferredY : candidate.preferredX;
    let crossAxis = Math.max(clamp(preferred, min, max), cursor + half);
    let proposed = labelBoxFor(candidate, crossAxis);

    for (let attempts = 0; attempts < 24; attempts += 1) {
      const overlap = avoidBoxes.find((box) => boxOverlaps(proposed.box, box));
      if (overlap === undefined) {
        break;
      }
      crossAxis = vertical
        ? overlap.bottom + LABEL_GAP + candidate.height / 2
        : overlap.right + LABEL_GAP + candidate.width / 2;
      if (crossAxis > max) {
        crossAxis = max;
        proposed = labelBoxFor(candidate, crossAxis);
        break;
      }
      proposed = labelBoxFor(candidate, crossAxis);
    }

    placed.push({ ...candidate, ...proposed });
    cursor = vertical ? proposed.box.bottom + LABEL_GAP : proposed.box.right + LABEL_GAP;
  }

  const last = placed.at(-1);
  if (last !== undefined) {
    const overflow = vertical
      ? last.box.bottom - (VIEW_SIZE - LABEL_PAD)
      : last.box.right - (VIEW_SIZE - LABEL_PAD);
    const first = placed[0]!;
    const headroom = vertical ? first.box.top - LABEL_PAD : first.box.left - LABEL_PAD;
    const shift = Math.max(0, Math.min(overflow, headroom));
    if (shift > 0) {
      for (const item of placed) {
        if (vertical) {
          item.y -= shift;
          item.box.top -= shift;
          item.box.bottom -= shift;
        } else {
          item.x -= shift;
          item.box.left -= shift;
          item.box.right -= shift;
        }
      }
    }
  }

  return finalizePlacedLabels(placed);
}

function finalizePlacedLabels(placed: readonly PlacedLabelCandidate[]): OrreryLaidOutLabel[] {
  return placed.map((item) => ({
    id: item.id,
    kind: item.kind,
    text: item.text,
    subtext: item.subtext,
    anchor: item.anchor,
    side: item.side,
    x: item.x,
    y: item.y,
    textAnchor: item.textAnchor,
    box: item.box,
    leader: leaderPath(item.anchor, item.box, item.side),
    badge:
      item.subtext === undefined
        ? undefined
        : {
            left:
              item.textAnchor === "end"
                ? item.x - estimateTextWidth(item.subtext, 9) - 8
                : item.textAnchor === "middle"
                  ? item.x - estimateTextWidth(item.subtext, 9) / 2 - 4
                  : item.x - 4,
            right:
              item.textAnchor === "end"
                ? item.x + 4
                : item.textAnchor === "middle"
                  ? item.x + estimateTextWidth(item.subtext, 9) / 2 + 4
                  : item.x + estimateTextWidth(item.subtext, 9) + 8,
            top: item.y + 2,
            bottom: item.y + 15,
          },
  }));
}

function labelSideForPoint(point: Point): LabelSide {
  const dx = point.x - CENTER;
  const dy = point.y - CENTER;
  if (Math.abs(dx) >= Math.abs(dy) * 0.72) {
    return dx < 0 ? "left" : "right";
  }
  return dy < 0 ? "top" : "bottom";
}

export function layoutOrreryLabels(data: OrreryViewModel): OrreryLaidOutLabel[] {
  const candidates: LabelCandidate[] = [];
  const avoidBoxes: Box[] = [
    { left: CENTER - 78, right: CENTER + 78, top: CENTER - 72, bottom: CENTER + 92 },
  ];

  data.memoryBands.forEach((band, index) => {
    const radius = MEMORY_RING_START + index * MEMORY_RING_STEP;
    const angle = MEMORY_LABEL_ANGLE_START + index * 31;
    const anchor = polarPoint(radius, angle);
    const preferred = polarPoint(radius + 42, angle);
    const side = labelSideForPoint(preferred);
    const subtext = `${countLabel(band.count, band.countIsLowerBound)} records`;
    candidates.push({
      id: `memory:${band.id}`,
      kind: "memory",
      text: band.name,
      subtext,
      anchor,
      side,
      preferredX: preferred.x,
      preferredY: preferred.y,
      width: Math.max(estimateTextWidth(band.name, 10), estimateTextWidth(subtext, 9)) + 10,
      height: 25,
    });
  });

  data.dream.satellites.forEach((satellite, index) => {
    const angle = -154 + (index * 360) / Math.max(1, data.dream.satellites.length);
    const point = polarPoint(DREAM_ORBIT_RADIUS, angle);
    const preferred = polarPoint(DREAM_ORBIT_RADIUS + 36, angle);
    avoidBoxes.push({
      left: point.x - 17,
      right: point.x + 17,
      top: point.y - 17,
      bottom: point.y + 17,
    });
    candidates.push({
      id: `satellite:${satellite.name}`,
      kind: "satellite",
      text: satellite.label,
      subtext: satellite.lastStatus ?? (satellite.enabled ? "enabled" : "off"),
      anchor: point,
      side: labelSideForPoint(preferred),
      preferredX: preferred.x,
      preferredY: preferred.y,
      width:
        Math.max(
          estimateTextWidth(satellite.label, 8.8),
          estimateTextWidth(satellite.lastStatus ?? (satellite.enabled ? "enabled" : "off"), 8.5),
        ) + 10,
      height: 24,
    });
  });

  if (data.governance.commitments.total > 0) {
    candidates.push({
      id: "governance:commitments",
      kind: "governance",
      text: "commitments",
      subtext: `${data.governance.commitments.critical} critical / ${data.governance.commitments.advisory} advisory`,
      anchor: polarPoint(246, 232),
      side: "left",
      preferredX: 92,
      preferredY: 318,
      width: 156,
      height: 25,
    });
  } else {
    candidates.push({
      id: "governance:commitments",
      kind: "governance",
      text: "commitments",
      subtext: "none active",
      anchor: polarPoint(246, 232),
      side: "left",
      preferredX: 92,
      preferredY: 318,
      width: 112,
      height: 25,
    });
  }

  candidates.push({
    id: "governance:directives",
    kind: "governance",
    text: "directives",
    subtext:
      data.governance.directives.active > 0
        ? `${data.governance.directives.active} active / ${data.governance.directives.total} total`
        : data.governance.directives.total > 0
          ? `0 active / ${data.governance.directives.total} total`
          : "none active",
    anchor: polarPoint(228, 238),
    side: "left",
    preferredX: 88,
    preferredY: 352,
    width: 138,
    height: 25,
  });

  const faultAnchor = polarPoint(242, 55);
  avoidBoxes.push({
    left: faultAnchor.x - 45,
    right: faultAnchor.x + 20,
    top: faultAnchor.y - 35,
    bottom: faultAnchor.y + 20,
  });
  candidates.push({
    id: "fault:reviews",
    kind: "fault",
    text: "reviews",
    subtext: `${data.reviews.openCount} open`,
    anchor: faultAnchor,
    side: "right",
    preferredX: 430,
    preferredY: 58,
    width: 76,
    height: 24,
  });

  const placed: OrreryLaidOutLabel[] = [];
  for (const side of ["left", "right", "top", "bottom"] as const) {
    const lane = placeLane(
      candidates.filter((candidate) => candidate.side === side),
      side,
      avoidBoxes.concat(placed.map((label) => label.box)),
    );
    placed.push(...lane);
  }
  return placed;
}

function wsTone(wsState: OrreryViewModel["runtime"]["wsState"]): "live" | "warn" | "bad" {
  if (wsState === "live") {
    return "live";
  }
  if (wsState === "reconnecting") {
    return "warn";
  }
  return "bad";
}

function turnTone(data: OrreryViewModel["stream"]): "active" | "idle" | "warn" | "bad" {
  if (data.running) {
    return "active";
  }
  if (data.terminalOutcome === "error" || data.terminalOutcome === "aborted") {
    return "bad";
  }
  if (data.terminalOutcome !== null && data.terminalOutcome !== "reflected") {
    return "warn";
  }
  return "idle";
}

function GovernanceLabel({
  label,
  muted = false,
}: {
  label: OrreryLaidOutLabel | undefined;
  muted?: boolean;
}) {
  if (label === undefined) {
    return null;
  }

  return (
    <g className={classNames("orr-laidout-label", muted && "orr-governance-label-muted")}>
      <path className="orr-label-leader" d={label.leader} />
      {label.badge === undefined ? null : (
        <rect
          className="orr-label-badge"
          x={label.badge.left}
          y={label.badge.top}
          width={label.badge.right - label.badge.left}
          height={label.badge.bottom - label.badge.top}
          rx="3"
        />
      )}
      <text
        className={label.kind === "fault" ? "orr-fault-label" : "orr-governance-label"}
        x={label.x}
        y={label.y - 3}
        textAnchor={label.textAnchor}
      >
        {label.text}
      </text>
      <text
        className={classNames(
          label.kind === "fault" ? "orr-fault-label" : "orr-governance-label",
          "orr-label-subtext",
        )}
        x={label.x}
        y={label.y + 10}
        textAnchor={label.textAnchor}
      >
        {label.subtext}
      </text>
    </g>
  );
}

export function Orrery({ size, data, onNavigate, onInspect }: OrreryProps) {
  const rawId = useId().replaceAll(":", "");
  const coreGradientId = `orr-core-${rawId}`;
  const spineGradientId = `orr-spine-${rawId}`;
  const arrowMarkerId = `orr-arrow-${rawId}`;
  const maxBandCount = Math.max(1, ...data.memoryBands.map((band) => band.count));
  const healthTone = wsTone(data.runtime.wsState);
  const streamTone = turnTone(data.stream);
  const labelMap = new Map(layoutOrreryLabels(data).map((label) => [label.id, label]));
  const activeTurnLabel =
    data.stream.activeTurnId === null ? "no active turn" : shortId(data.stream.activeTurnId);

  const inspectTurn = () => {
    if (data.stream.activeTurnId === null) {
      onNavigate("cognition");
      return;
    }
    onInspect({
      type: "turn",
      id: data.stream.activeTurnId,
      hint: {
        phase: data.stream.lastPhase,
        state: data.stream.state,
      },
    });
  };

  return (
    <div
      className={classNames("orr-shell", size === "full" ? "orr-full" : "orr-compact")}
      data-testid="orrery"
      data-size={size}
    >
      <svg
        className="orr-svg"
        viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
        role="img"
        aria-label="Cognitive Orrery"
      >
        <defs>
          <radialGradient id={coreGradientId}>
            <stop offset="0%" stopColor="var(--text)" stopOpacity="0.34" />
            <stop offset="62%" stopColor="var(--info)" stopOpacity="0.18" />
            <stop offset="100%" stopColor="var(--bg-0)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id={spineGradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--info-dim)" stopOpacity="0.08" />
            <stop offset="48%" stopColor="var(--info)" stopOpacity="0.55" />
            <stop offset="100%" stopColor="var(--purple)" stopOpacity="0.42" />
          </linearGradient>
          <marker
            id={arrowMarkerId}
            markerWidth="8"
            markerHeight="8"
            refX="7"
            refY="4"
            orient="auto"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" className="orr-spine-arrow" />
          </marker>
        </defs>

        <rect className="orr-field" x="0" y="0" width={VIEW_SIZE} height={VIEW_SIZE} />
        <circle className="orr-outer-grid" cx={CENTER} cy={CENTER} r="252" />
        <circle className="orr-outer-grid orr-outer-grid-soft" cx={CENTER} cy={CENTER} r="210" />

        <g className="orr-memory-system" aria-label="memory bands">
          {data.memoryBands.map((band, index) => {
            const radius = MEMORY_RING_START + index * MEMORY_RING_STEP;
            const label = labelMap.get(`memory:${band.id}`);
            const activate = () => onNavigate("memory", { memoryBand: band.id });

            return (
              <g
                key={band.id}
                className="orr-memory-band"
                role="button"
                tabIndex={0}
                aria-label={`${band.name} memory, ${countLabel(
                  band.count,
                  band.countIsLowerBound,
                )}`}
                data-testid={`orr-memory-ring-${band.id}`}
                onClick={activate}
                onKeyDown={(event) => onKeyboardActivate(event, activate)}
              >
                <circle
                  className="orr-memory-ring"
                  cx={CENTER}
                  cy={CENTER}
                  r={radius}
                  strokeWidth={memoryStrokeWidth(band.count, maxBandCount)}
                />
                {label === undefined ? null : (
                  <>
                    <path className="orr-label-leader" d={label.leader} />
                    {label.badge === undefined ? null : (
                      <rect
                        className="orr-label-badge"
                        x={label.badge.left}
                        y={label.badge.top}
                        width={label.badge.right - label.badge.left}
                        height={label.badge.bottom - label.badge.top}
                        rx="3"
                      />
                    )}
                    <text
                      className="orr-memory-label"
                      x={label.x}
                      y={label.y - 3}
                      textAnchor={label.textAnchor}
                    >
                      {label.text}
                    </text>
                    <text
                      className="orr-memory-count"
                      x={label.x}
                      y={label.y + 11}
                      textAnchor={label.textAnchor}
                    >
                      {label.subtext}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </g>

        <g
          className={classNames(
            "orr-core",
            `orr-core-${healthTone}`,
            data.stream.running && "orr-core-active",
          )}
          data-testid="orr-core"
          data-ws-state={data.runtime.wsState}
          data-active-turn={data.stream.running ? "true" : "false"}
        >
          <circle
            className="orr-core-glow"
            cx={CENTER}
            cy={CENTER}
            r="62"
            fill={`url(#${coreGradientId})`}
          />
          <circle className="orr-core-body" cx={CENTER} cy={CENTER} r="31" />
          <circle className="orr-core-dot-glow" cx={CENTER} cy={CENTER} r="8.5" />
          <circle className="orr-core-dot" cx={CENTER} cy={CENTER} r="5" />
          <text className="orr-core-label" x={CENTER} y={CENTER - 43} textAnchor="middle">
            runtime
          </text>
          <text className="orr-core-health" x={CENTER} y={CENTER + 50} textAnchor="middle">
            {data.runtime.wsState}
          </text>
        </g>

        <g className="orr-dream-system" aria-label="dream processes">
          <circle className="orr-dream-orbit" cx={CENTER} cy={CENTER} r={DREAM_ORBIT_RADIUS} />
          {data.dream.satellites.map((satellite, index) => {
            const angle = -154 + (index * 360) / Math.max(1, data.dream.satellites.length);
            const point = polarPoint(DREAM_ORBIT_RADIUS, angle);
            const label = labelMap.get(`satellite:${satellite.name}`);
            const statusClass =
              satellite.lastStatus === "error"
                ? "orr-dream-satellite-bad"
                : satellite.running
                  ? "orr-dream-satellite-running"
                  : satellite.enabled
                    ? "orr-dream-satellite-enabled"
                    : "orr-dream-satellite-muted";
            const activate = () => {
              if (isDreamProcessName(satellite.name)) {
                onNavigate("dream", { dreamProcess: satellite.name });
                return;
              }
              onNavigate("dream");
            };

            return (
              <g
                key={satellite.name}
                className={classNames("orr-dream-satellite", statusClass)}
                role="button"
                tabIndex={0}
                aria-label={`${satellite.label} dream process${
                  satellite.phase === null ? "" : ` ${satellite.phase}`
                }`}
                data-testid={`orr-dream-satellite-${satellite.name}`}
                data-running={satellite.running ? "true" : "false"}
                onClick={activate}
                onKeyDown={(event) => onKeyboardActivate(event, activate)}
              >
                <line
                  className="orr-dream-tether"
                  x1={CENTER}
                  y1={CENTER}
                  x2={point.x}
                  y2={point.y}
                />
                <circle className="orr-dream-orbit-glow" cx={point.x} cy={point.y} r="12" />
                <circle className="orr-dream-orbit-mark" cx={point.x} cy={point.y} r="12" />
                <circle className="orr-dream-dot" cx={point.x} cy={point.y} r="4.5" />
                {label === undefined ? null : (
                  <>
                    <path className="orr-label-leader" d={label.leader} />
                    {label.badge === undefined ? null : (
                      <rect
                        className="orr-label-badge orr-label-badge-dream"
                        x={label.badge.left}
                        y={label.badge.top}
                        width={label.badge.right - label.badge.left}
                        height={label.badge.bottom - label.badge.top}
                        rx="3"
                      />
                    )}
                    <text
                      className="orr-dream-label"
                      x={label.x}
                      y={label.y - 3}
                      textAnchor={label.textAnchor}
                    >
                      {label.text}
                    </text>
                    <text
                      className="orr-dream-status"
                      x={label.x}
                      y={label.y + 10}
                      textAnchor={label.textAnchor}
                    >
                      {label.subtext}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </g>

        <g className="orr-governance-system" aria-label="governance constraints">
          {data.governance.commitments.total > 0 ? (
            <g
              className={classNames(
                "orr-governance-arc",
                data.governance.commitments.critical > 0
                  ? "orr-governance-critical"
                  : "orr-governance-advisory",
              )}
              role="button"
              tabIndex={0}
              aria-label={`${data.governance.commitments.total} active commitments`}
              data-testid="orr-governance-commitments"
              onClick={() => onNavigate("governance", { governanceTab: "commitments" })}
              onKeyDown={(event) =>
                onKeyboardActivate(event, () =>
                  onNavigate("governance", { governanceTab: "commitments" }),
                )
              }
            >
              <path
                d={arcPath(
                  246,
                  210,
                  210 +
                    governanceSweep(
                      data.governance.commitments.total,
                      data.governance.commitments.critical,
                    ),
                )}
                style={{ strokeWidth: governanceStrokeWidth(data.governance.commitments.total) }}
              />
              <GovernanceLabel label={labelMap.get("governance:commitments")} />
            </g>
          ) : (
            <GovernanceLabel muted label={labelMap.get("governance:commitments")} />
          )}
          {data.governance.directives.active > 0 ? (
            <g
              className="orr-governance-arc orr-governance-directives"
              role="button"
              tabIndex={0}
              aria-label={`${data.governance.directives.active} active directives`}
              data-testid="orr-governance-directives"
              onClick={() => onNavigate("governance", { governanceTab: "shared_state" })}
              onKeyDown={(event) =>
                onKeyboardActivate(event, () =>
                  onNavigate("governance", { governanceTab: "shared_state" }),
                )
              }
            >
              <path
                d={arcPath(228, 216, 216 + governanceSweep(data.governance.directives.active))}
                style={{ strokeWidth: governanceStrokeWidth(data.governance.directives.active) }}
              />
              <GovernanceLabel label={labelMap.get("governance:directives")} />
            </g>
          ) : (
            <GovernanceLabel muted label={labelMap.get("governance:directives")} />
          )}
        </g>

        <g
          className={classNames("orr-fault-system", `orr-fault-${data.reviews.severity}`)}
          role={data.reviews.openCount > 0 ? "button" : undefined}
          tabIndex={data.reviews.openCount > 0 ? 0 : undefined}
          aria-label={`${data.reviews.openCount} open reviews`}
          data-testid="orr-fault-system"
          onClick={data.reviews.openCount > 0 ? () => onNavigate("review") : undefined}
          onKeyDown={
            data.reviews.openCount > 0
              ? (event) => onKeyboardActivate(event, () => onNavigate("review"))
              : undefined
          }
        >
          {Array.from({ length: Math.min(FAULT_LIMIT, data.reviews.openCount) }).map((_, index) => {
            const point = polarPoint(242, 44 + index * 7);
            return (
              <circle
                key={index}
                className="orr-fault-node"
                cx={point.x}
                cy={point.y}
                r={8 - index * 0.6}
                data-testid={index === 0 ? "orr-fault-node" : `orr-fault-node-${index}`}
              />
            );
          })}
          <GovernanceLabel label={labelMap.get("fault:reviews")} />
        </g>

        <g
          className={classNames("orr-stream-system", `orr-stream-${streamTone}`)}
          role="button"
          tabIndex={0}
          aria-label={
            data.stream.activeTurnId === null
              ? "open cognition"
              : `inspect turn ${data.stream.activeTurnId}`
          }
          data-testid="orr-stream-spine"
          onClick={inspectTurn}
          onKeyDown={(event) => onKeyboardActivate(event, inspectTurn)}
        >
          <path
            className="orr-stream-spine"
            d="M 106 438 C 188 390 218 360 280 360 C 342 360 372 390 454 438"
            stroke={`url(#${spineGradientId})`}
            markerEnd={`url(#${arrowMarkerId})`}
          />
          <circle className="orr-turn-glow" cx="280" cy="360" r="13" />
          <circle
            className="orr-turn-pulse"
            cx="280"
            cy="360"
            r={data.stream.activeTurnId === null ? 6 : 9}
            data-testid="orr-active-turn-pulse"
            data-active={data.stream.activeTurnId === null ? "false" : "true"}
            data-running={data.stream.running ? "true" : "false"}
          />
          <text className="orr-stream-phase" x="280" y="386" textAnchor="middle">
            {data.stream.lastPhase}
          </text>
          <text className="orr-stream-turn" x="280" y="402" textAnchor="middle">
            {activeTurnLabel}
          </text>
        </g>

        {data.loading ? (
          <text
            className="orr-state-note"
            x="280"
            y="28"
            textAnchor="middle"
            data-testid="orr-loading"
          >
            loading substrate
          </text>
        ) : null}
        {data.error === null ? null : (
          <text
            className="orr-state-note orr-state-note-bad"
            x="280"
            y="48"
            textAnchor="middle"
            data-testid="orr-error"
          >
            {data.error}
          </text>
        )}
      </svg>
    </div>
  );
}
