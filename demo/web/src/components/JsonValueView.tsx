import { useState } from "react";

const COLLECTION_PREVIEW_LIMIT = 50;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function formatValue(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? String(value);
}

function collectionSummary(value: unknown): string | null {
  if (Array.isArray(value)) {
    if (value.length <= COLLECTION_PREVIEW_LIMIT) {
      return null;
    }

    return value.every((entry) => typeof entry === "number")
      ? `${value.length} numbers`
      : `${value.length} entries`;
  }

  if (isRecord(value)) {
    const entries = Object.entries(value);
    if (entries.length <= COLLECTION_PREVIEW_LIMIT) {
      return null;
    }

    return entries.every(([, entry]) => typeof entry === "number")
      ? `${entries.length} numbers`
      : `${entries.length} entries`;
  }

  return null;
}

export function JsonValueView({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const [expanded, setExpanded] = useState(false);
  const summary = collectionSummary(value);

  if (summary !== null && !expanded) {
    return (
      <span className="json-collection-summary">
        <span>{summary}</span>
        <button type="button" className="btn sm ghost" onClick={() => setExpanded(true)}>
          expand
        </button>
      </span>
    );
  }

  if (value === null || typeof value !== "object") {
    return <span className="json-scalar">{String(value)}</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="json-scalar">[]</span>;
    }

    return <pre className="json-pre">{formatValue(value)}</pre>;
  }

  if (!isRecord(value)) {
    return <span className="json-scalar">{String(value)}</span>;
  }

  const entries = Object.entries(value);
  if (entries.length === 0) {
    return <span className="json-scalar">{"{}"}</span>;
  }

  if (depth > 0) {
    return <pre className="json-pre">{formatValue(value)}</pre>;
  }

  return (
    <div className="props json-value-props">
      {entries.map(([key, entry]) => (
        <div key={key} className="row">
          <span className="k">{key}</span>
          <span className="v">
            <JsonValueView value={entry} depth={depth + 1} />
          </span>
        </div>
      ))}
    </div>
  );
}
