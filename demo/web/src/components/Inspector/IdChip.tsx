import { useState } from "react";

import { copyText } from "../../lib/clipboard";
import { shortId } from "../../screens/screen-utils";
import { useInspector } from "./inspector-context";
import { resolveObjectType, type ObjectType } from "./inspector-id";
import type { InspectorTab } from "./inspector-registry";

export type IdChipProps = {
  id: string;
  type?: ObjectType | null;
  label?: string;
  active?: boolean;
  ariaLabel?: string;
  title?: string;
  presetTab?: InspectorTab;
  hint?: unknown;
  copy?: boolean;
  className?: string;
};

export function IdChip({
  id,
  type,
  label,
  active = false,
  ariaLabel,
  title,
  presetTab,
  hint,
  copy = true,
  className = "",
}: IdChipProps) {
  const inspector = useInspector();
  const [copied, setCopied] = useState(false);
  const resolvedType = type ?? resolveObjectType(id);
  const display = label ?? shortId(id);

  async function copyId(): Promise<void> {
    await copyText(id);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <span className={`id-chip ${active ? "active" : ""} ${className}`.trim()}>
      {resolvedType !== null ? (
        <button
          type="button"
          className="id-chip-main"
          title={title ?? `Jump to ${id}`}
          aria-label={ariaLabel ?? `jump to ${id}`}
          aria-current={active ? "true" : undefined}
          onClick={(event) => {
            event.stopPropagation();
            inspector.openObject({ type: resolvedType, id, presetTab, hint });
          }}
        >
          {display}
        </button>
      ) : (
        <span
          className="id-chip-main id-chip-static"
          title={title ?? `Unknown object type for ${id}`}
          aria-label={ariaLabel ?? `unknown object ${id}`}
        >
          {display}
        </span>
      )}
      {copy ? (
        <button
          type="button"
          className="id-chip-copy"
          title={copied ? "Copied full id" : `Copy ${id}`}
          aria-label={copied ? `copied ${id}` : `copy ${id}`}
          onClick={(event) => {
            event.stopPropagation();
            void copyId();
          }}
        >
          {copied ? "copied" : "copy"}
        </button>
      ) : null}
    </span>
  );
}
