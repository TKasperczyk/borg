function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function formatValue(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? String(value);
}

export function JsonValueView({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || typeof value !== "object") {
    return <span className="why-scalar">{String(value)}</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="why-scalar">[]</span>;
    }

    return <pre className="why-pre">{formatValue(value)}</pre>;
  }

  if (!isRecord(value)) {
    return <span className="why-scalar">{String(value)}</span>;
  }

  const entries = Object.entries(value);
  if (entries.length === 0) {
    return <span className="why-scalar">{"{}"}</span>;
  }

  if (depth > 0) {
    return <pre className="why-pre">{formatValue(value)}</pre>;
  }

  return (
    <div className="why-kv">
      {entries.map(([key, entry]) => (
        <div key={key} className="why-row">
          <span className="why-key">{key}</span>
          <JsonValueView value={entry} depth={depth + 1} />
        </div>
      ))}
    </div>
  );
}
