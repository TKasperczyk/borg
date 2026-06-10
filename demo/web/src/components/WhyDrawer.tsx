import { useEffect, useState } from "react";

import { getWhy } from "../api/client";
import type { WhyResponse } from "../api/types";
import { ErrorState } from "./ErrorState";
import { Empty } from "./Empty";
import { JsonValueView } from "./JsonValueView";
import { Loading } from "./Loading";
import { Modal } from "./Modal";

type WhyDrawerProps = {
  open: boolean;
  id: string | null;
  onClose: () => void;
};

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
        {loading ? <Loading>loading provenance</Loading> : null}
        {error === null ? null : <ErrorState>{error.message}</ErrorState>}
        {!loading && error === null && data !== null ? (
          entries.length === 0 ? (
            <Empty>no provenance fields</Empty>
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
