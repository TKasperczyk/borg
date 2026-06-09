import { useId, type KeyboardEvent } from "react";

import type { RouteId, RouteNavigationOptions } from "../../routes";
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
  return `${lowerBound ? ">" : ""}${count.toLocaleString()}`;
}

function memoryStrokeWidth(count: number, maxCount: number): number {
  if (count <= 0 || maxCount <= 0) {
    return 1.2;
  }
  return 1.2 + (Math.log1p(count) / Math.log1p(maxCount)) * 4.2;
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

export function Orrery({ size, data, onNavigate, onInspect }: OrreryProps) {
  const rawId = useId().replaceAll(":", "");
  const coreGradientId = `orr-core-${rawId}`;
  const spineGradientId = `orr-spine-${rawId}`;
  const arrowMarkerId = `orr-arrow-${rawId}`;
  const maxBandCount = Math.max(1, ...data.memoryBands.map((band) => band.count));
  const healthTone = wsTone(data.runtime.wsState);
  const streamTone = turnTone(data.stream);
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
            const angle = MEMORY_LABEL_ANGLE_START + index * 31;
            const label = polarPoint(radius + 12, angle);
            const anchor = label.x < CENTER - 8 ? "end" : label.x > CENTER + 8 ? "start" : "middle";
            const activate = () => onNavigate("memory");

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
                <text className="orr-memory-label" x={label.x} y={label.y} textAnchor={anchor}>
                  {band.name}
                </text>
                <text className="orr-memory-count" x={label.x} y={label.y + 12} textAnchor={anchor}>
                  {countLabel(band.count, band.countIsLowerBound)}
                </text>
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
            const label = polarPoint(DREAM_ORBIT_RADIUS + 18, angle);
            const statusClass =
              satellite.lastStatus === "error"
                ? "orr-dream-satellite-bad"
                : satellite.running
                  ? "orr-dream-satellite-running"
                  : satellite.enabled
                    ? "orr-dream-satellite-enabled"
                    : "orr-dream-satellite-muted";
            const activate = () => onNavigate("dream");

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
                <circle className="orr-dream-orbit-mark" cx={point.x} cy={point.y} r="12" />
                <circle className="orr-dream-dot" cx={point.x} cy={point.y} r="4.5" />
                <text className="orr-dream-label" x={label.x} y={label.y} textAnchor="middle">
                  {satellite.label}
                </text>
              </g>
            );
          })}
        </g>

        <g className="orr-governance-system" aria-label="governance constraints">
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
            <path d={arcPath(246, 210, 310)} />
            <text className="orr-governance-label" x="70" y="315">
              cmt {data.governance.commitments.critical}/{data.governance.commitments.advisory}
            </text>
          </g>
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
            <path d={arcPath(228, 216, 304)} />
            <text className="orr-governance-label" x="78" y="336">
              dir {data.governance.directives.active}/{data.governance.directives.total}
            </text>
          </g>
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
          <text className="orr-fault-label" x="430" y="92" textAnchor="middle">
            review {data.reviews.openCount}
          </text>
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
