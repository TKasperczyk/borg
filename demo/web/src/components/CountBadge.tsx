import type { SeverityRank } from "./SeverityChip";

export type CountBadgeProps = {
  count: number;
  severity?: SeverityRank;
  label?: string;
};

export function CountBadge({ count, severity = 1, label }: CountBadgeProps) {
  return (
    <span
      className={`count-badge sev-${severity}`}
      aria-label={label === undefined ? undefined : `${label}: ${count}`}
    >
      {count.toLocaleString()}
    </span>
  );
}
