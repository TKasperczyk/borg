export type SupersededByChipProps = {
  id: string;
  label: string;
  onOpen: (id: string) => void;
  active?: boolean;
  ariaLabel?: string;
  title?: string;
};

export function SupersededByChip({
  id,
  label,
  onOpen,
  active = false,
  ariaLabel,
  title,
}: SupersededByChipProps) {
  return (
    <button
      type="button"
      className={`btn sm ${active ? "primary" : "ghost"}`}
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
  );
}
