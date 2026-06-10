export type MiniOrreryProps = {
  dreamRunning: boolean;
};

function classNames(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(" ");
}

export function MiniOrrery({ dreamRunning }: MiniOrreryProps) {
  return (
    <div
      className={classNames("orr-mini", dreamRunning && "orr-mini-dream-running")}
      role="img"
      aria-label={`dream ${dreamRunning ? "running" : "idle"}`}
      data-testid="mini-orrery"
      data-dream-running={dreamRunning ? "true" : "false"}
    >
      <svg className="orr-mini-svg" viewBox="0 0 42 24" aria-hidden="true">
        <circle className="orr-mini-ring" cx="18" cy="12" r="9" />
        <circle className="orr-mini-core" cx="18" cy="12" r="3.5" />
        <circle className="orr-mini-dream" cx="31" cy="8" r="2.8" />
      </svg>
    </div>
  );
}
