import type { WsState } from "../../api/types";

export type MiniOrreryProps = {
  wsState: WsState;
  dreamRunning: boolean;
  openReviews: number;
};

function classNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

function healthTone(wsState: WsState): "live" | "warn" | "bad" {
  if (wsState === "live") {
    return "live";
  }
  if (wsState === "reconnecting") {
    return "warn";
  }
  return "bad";
}

export function MiniOrrery({ wsState, dreamRunning, openReviews }: MiniOrreryProps) {
  const tone = healthTone(wsState);

  return (
    <div
      className={classNames(
        "orr-mini",
        `orr-mini-${tone}`,
        dreamRunning && "orr-mini-dream-running",
        openReviews > 0 && "orr-mini-has-reviews",
      )}
      role="img"
      aria-label={`orrery ${wsState}, dream ${
        dreamRunning ? "running" : "idle"
      }, reviews ${openReviews}`}
      data-testid="mini-orrery"
      data-ws-state={wsState}
      data-dream-running={dreamRunning ? "true" : "false"}
      data-open-reviews={openReviews}
    >
      <svg className="orr-mini-svg" viewBox="0 0 42 24" aria-hidden="true">
        <circle className="orr-mini-ring" cx="18" cy="12" r="9" />
        <circle className="orr-mini-core" cx="18" cy="12" r="3.5" />
        <circle className="orr-mini-dream" cx="31" cy="8" r="2.8" />
        <circle className="orr-mini-review" cx="31" cy="17" r="2.8" />
      </svg>
    </div>
  );
}
