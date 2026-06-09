import { shortId } from "../../screens/screen-utils";
import { useInspector } from "./inspector-context";
import { resolveObjectType, type ObjectType } from "./inspector-id";
import type { InspectorTab } from "./inspector-registry";

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
  const inspector = useInspector();
  const resolvedType = type ?? resolveObjectType(id);
  const disabled = resolvedType === null;

  return (
    <button
      type="button"
      className={`btn sm superseded-by-chip ${active ? "primary" : "ghost"}`}
      title={title ?? (disabled ? `Unknown object type for ${id}` : `Jump to ${id}`)}
      aria-label={ariaLabel ?? (disabled ? `unknown object ${id}` : `jump to ${id}`)}
      aria-current={active ? "true" : undefined}
      disabled={disabled}
      onClick={(event) => {
        event.stopPropagation();
        if (resolvedType !== null) {
          inspector.openObject({ type: resolvedType, id, presetTab, hint });
        }
      }}
    >
      {label ?? shortId(id)}
    </button>
  );
}
