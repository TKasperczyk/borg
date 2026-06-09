import { IdRef } from "./Inspector/IdRef";
import type { ObjectType } from "./Inspector/inspector-id";

export type SupersededByChipProps = {
  id: string;
  label: string;
  onOpen: (id: string) => void;
  active?: boolean;
  ariaLabel?: string;
  title?: string;
  inspectType?: ObjectType;
  inspectHint?: unknown;
  inspectAriaLabel?: string;
};

export function SupersededByChip({
  id,
  label,
  onOpen,
  active = false,
  ariaLabel,
  title,
  inspectType,
  inspectHint,
  inspectAriaLabel,
}: SupersededByChipProps) {
  return (
    <>
      <button
        type="button"
        className={`btn sm superseded-by-chip ${active ? "primary" : "ghost"}`}
        title={title ?? `Jump to ${id}`}
        aria-label={ariaLabel ?? `jump to ${id}`}
        aria-current={active ? "true" : undefined}
        onClick={(event) => {
          event.stopPropagation();
          onOpen(id);
        }}
      >
        {label}
      </button>
      {inspectType === undefined ? null : (
        <IdRef
          id={id}
          type={inspectType}
          label="inspect"
          title={`Inspect ${id}`}
          ariaLabel={inspectAriaLabel ?? `inspect ${id}`}
          hint={inspectHint}
        />
      )}
    </>
  );
}
