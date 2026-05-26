import { useEffect, useState } from "react";

import { getWhy } from "../api/client";
import type { WhyResponse } from "../api/types";
import { Modal } from "./Modal";

type WhyDrawerProps = {
  open: boolean;
  id: string | null;
  onClose: () => void;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function formatValue(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? String(value);
}

function JsonValueView({ value, depth = 0 }: { value: unknown; depth?: number }) {
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

export function WhyDrawer({ open, id, onClose }: WhyDrawerProps) {
  const [data, setData] = useState<WhyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!open || id === null) {
      setData(null);
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);

    void getWhy(id)
      .then((result) => {
        if (!cancelled) {
          setData(result);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught : new Error(String(caught)));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [id, open]);

  const entries = data === null ? [] : Object.entries(data);

  return (
    <Modal open={open} title={id === null ? "why" : `why ${id}`} onClose={onClose}>
      <div className="why-drawer">
        {loading ? <div className="notice">loading provenance</div> : null}
        {error === null ? null : <div className="notice bad">{error.message}</div>}
        {!loading && error === null && data !== null ? (
          entries.length === 0 ? (
            <div className="notice">no provenance fields</div>
          ) : (
            entries.map(([key, value]) => (
              <details key={key} className="why-section" open>
                <summary>{key}</summary>
                <JsonValueView value={value} />
              </details>
            ))
          )
        ) : null}
      </div>
    </Modal>
  );
}
