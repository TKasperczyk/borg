import { Tag } from "./Tag";

export type SeverityRank = 1 | 2 | 3 | 4;

export type SeverityChipProps = {
  rank: SeverityRank;
  children?: string;
};

export function SeverityChip({ rank, children = `severity ${rank}` }: SeverityChipProps) {
  return <Tag kind={`sev-${rank}`}>{children}</Tag>;
}
