import { useEffect, useState } from "react";

import { ApiError, getLedger } from "../../api/client";
import type { EvidenceLedger, EvidenceLedgerEntry, EvidenceLedgerSection } from "../../api/types";
import { AttachmentChip } from "../../components/AttachmentChip";
import { Empty } from "../../components/Empty";
import { Tag, type TagKind } from "../../components/Tag";

export type LedgerViewProps = {
  turnId: string | null;
  cachedLedger?: EvidenceLedger;
  active: boolean;
  audience: string;
};

type LedgerGroup = {
  id: string;
  label: string;
  entries: EvidenceLedgerEntry[];
};

type LedgerState = {
  turnId: string;
  ledger: EvidenceLedger;
};

function groupForSection(section: EvidenceLedgerSection): LedgerGroup {
  return {
    id: section.id,
    label: section.label,
    entries: section.entries,
  };
}

function trustKind(entry: EvidenceLedgerEntry): TagKind {
  if (entry.taint === "quarantined" || entry.taint === "contested") {
    return "bad";
  }
  if (entry.state === "tentative" || entry.state === "pending") {
    return "warn";
  }
  if (entry.state === "locked" || entry.state === "live") {
    return "acc";
  }
  return "";
}

export function LedgerView({ turnId, cachedLedger, active, audience }: LedgerViewProps) {
  const [ledgerState, setLedgerState] = useState<LedgerState | null>(() =>
    turnId === null || cachedLedger === undefined ? null : { turnId, ledger: cachedLedger },
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);

    if (turnId === null) {
      setLedgerState(null);
      return;
    }

    if (cachedLedger !== undefined) {
      setLedgerState({ turnId, ledger: cachedLedger });
      return;
    }

    setLedgerState((current) => (current?.turnId === turnId ? current : null));
  }, [cachedLedger, turnId]);

  useEffect(() => {
    if (!active || turnId === null || cachedLedger !== undefined) {
      return;
    }

    let cancelled = false;
    setError(null);
    void getLedger(turnId)
      .then((response) => {
        if (!cancelled) {
          setLedgerState({ turnId, ledger: response.ledger });
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setLedgerState(null);
          setError(
            caught instanceof ApiError && caught.status === 404
              ? "ledger not retained (pre-restart) - durable replay lands in a later persistence sprint"
              : caught instanceof Error
                ? caught.message
                : String(caught),
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [active, cachedLedger, turnId]);

  if (turnId === null) {
    return <Empty>send a turn to build an evidence ledger</Empty>;
  }

  const ledger = ledgerState?.turnId === turnId ? ledgerState.ledger : null;

  if (ledger === null) {
    return <Empty>{error ?? "ledger not loaded yet"}</Empty>;
  }

  const groups = ledger.sections.map(groupForSection);

  return (
    <div>
      <div
        style={{
          padding: "10px 14px",
          borderBottom: "1px solid var(--line)",
          color: "var(--text-mute)",
          fontSize: "10.5px",
        }}
      >
        prompt-visible substrate · {ledger.estimatedTokens} estimated tokens
      </div>
      {groups.map((group) => (
        <div key={group.id} className="lgr-section">
          <div className="lgr-section-head">
            <span>▸</span>
            <span>{group.label}</span>
            <span className="count">[{group.entries.length}]</span>
          </div>
          {group.entries.length === 0 ? (
            <div className="lgr-empty">— none —</div>
          ) : null}
          {group.entries.map((entry) => (
            <div key={entry.id} className="lgr-item">
              <div className="head">
                <span className="id">
                  [{group.id}:{entry.id}]
                </span>
                {entry.state === undefined ? null : (
                  <Tag kind={trustKind(entry)}>{entry.state}</Tag>
                )}
                {entry.taint === undefined || entry.taint === "none" ? null : (
                  <Tag kind={trustKind(entry)}>{entry.taint}</Tag>
                )}
                <span className="trust">trust {entry.trust_rank}</span>
              </div>
              <div className="text">{entry.text ?? entry.value ?? "(no text)"}</div>
              {entry.citations === undefined || entry.citations.length === 0 ? null : (
                <div className="cite">citations · {entry.citations.join(", ")}</div>
              )}
            </div>
          ))}
        </div>
      ))}

      {ledger.imageAttachments === undefined || ledger.imageAttachments.length === 0 ? null : (
        <div className="lgr-section">
          <div className="lgr-section-head">
            <span className="info">▸</span>
            <span>attached images</span>
            <span className="count">[{ledger.imageAttachments.length}]</span>
          </div>
          <div style={{ padding: "0 14px" }}>
            {ledger.imageAttachments.map((image) => (
              <AttachmentChip
                key={image.attachment_id}
                attachmentId={image.attachment_id}
                audience={audience}
                expanded
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
