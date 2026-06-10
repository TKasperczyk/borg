import { resolveObjectType, type ObjectType } from "./inspector-id";
import type { InspectorTab } from "./inspector-registry";
import { IdChip } from "./IdChip";

export type IdRefProps = {
  id: string;
  type?: ObjectType | null;
  label?: string;
  active?: boolean;
  ariaLabel?: string;
  title?: string;
  presetTab?: InspectorTab;
  hint?: unknown;
};

export function IdRef({
  id,
  type,
  label,
  active = false,
  ariaLabel,
  title,
  presetTab,
  hint,
}: IdRefProps) {
  const resolvedType = type ?? resolveObjectType(id);

  return (
    <IdChip
      id={id}
      type={resolvedType}
      label={label}
      active={active}
      ariaLabel={ariaLabel}
      title={title}
      presetTab={presetTab}
      hint={hint}
      copy={false}
      className={`superseded-by-chip ${active ? "primary" : "ghost"}`}
    />
  );
}
