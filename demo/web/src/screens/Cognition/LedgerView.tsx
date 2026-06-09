import { Fragment, useEffect, useState } from "react";

import { ApiError, getLedger } from "../../api/client";
import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSourceType,
} from "../../api/types";
import { AttachmentChip } from "../../components/AttachmentChip";
import { Empty } from "../../components/Empty";
import { IdRef } from "../../components/Inspector/IdRef";
import { resolveObjectType, type ObjectType } from "../../components/Inspector/inspector-id";
import { Tag, type TagKind } from "../../components/Tag";

export type LedgerViewProps = {
  turnId: string | null;
  cachedLedger?: EvidenceLedger;
  active: boolean;
  audience: string;
  entryFilter?: (entry: EvidenceLedgerEntry) => boolean;
  emptyMessage?: string;
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

const LEDGER_SOURCE_OBJECT_TYPES: Partial<Record<EvidenceLedgerSourceType, ObjectType>> = {
  current_session_stream: "stream_entry",
  prior_session_stream: "stream_entry",
  episode: "episode",
  semantic_node: "semantic_node",
  semantic_edge: "semantic_edge",
  action_record: "action_record",
  relational_slot: "relational_slot",
  commitment: "commitment",
  shared_state: "shared_state_entry",
  image_attachment: "attachment",
  assistant_stream: "stream_entry",
};

const LEDGER_SOURCE_OBJECT_METADATA_KEYS: Partial<Record<ObjectType, readonly string[]>> = {
  stream_entry: [
    "stream_id",
    "stream_entry_id",
    "source_stream_id",
    "parent_entry_id",
    "stream_ids",
    "source_stream_ids",
    "source_stream_entry_ids",
    "provenance_stream_entry_ids",
    "last_updated_stream_entry_ids",
  ],
  episode: ["episode_id"],
  semantic_node: ["node_id"],
  semantic_edge: ["edge_id"],
  action_record: ["action_id", "current_action_id", "record_ids"],
  relational_slot: ["relational_slot_id", "slot_id"],
  commitment: ["commitment_id"],
  shared_state_entry: ["shared_state_entry_id", "artifact_entry_id"],
  attachment: ["attachment_id"],
};

function ledgerSourceObjectType(sourceType: EvidenceLedgerSourceType): ObjectType | null {
  return LEDGER_SOURCE_OBJECT_TYPES[sourceType] ?? null;
}

function isExpectedObjectId(id: string, type: ObjectType): boolean {
  return resolveObjectType(id) === type;
}

function addExpectedObjectId(candidates: Set<string>, id: string, type: ObjectType): void {
  if (isExpectedObjectId(id, type)) {
    candidates.add(id);
  }
}

function ledgerHandleObjectId(handle: string, type: ObjectType): string | null {
  const separatorIndex = handle.lastIndexOf(":");
  if (separatorIndex < 0 || separatorIndex === handle.length - 1) {
    return null;
  }

  const rawId = handle.slice(separatorIndex + 1);
  return isExpectedObjectId(rawId, type) ? rawId : null;
}

function metadataStringValues(metadata: Record<string, unknown>, key: string): string[] {
  const value = metadata[key];
  if (typeof value === "string") {
    return [value];
  }
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }
  return [];
}

function ledgerSourceObjectId(entry: EvidenceLedgerEntry, type: ObjectType): string | null {
  const candidates = new Set<string>();
  const handleId = ledgerHandleObjectId(entry.id, type);

  if (handleId !== null) {
    candidates.add(handleId);
  }

  if (entry.state_metadata !== undefined) {
    for (const key of LEDGER_SOURCE_OBJECT_METADATA_KEYS[type] ?? []) {
      for (const value of metadataStringValues(entry.state_metadata, key)) {
        addExpectedObjectId(candidates, value, type);
      }
    }
  }

  for (const citation of entry.citations ?? []) {
    addExpectedObjectId(candidates, citation, type);
  }

  if (candidates.size !== 1) {
    return null;
  }

  for (const candidate of candidates) {
    return candidate;
  }

  return null;
}

export function LedgerView({
  turnId,
  cachedLedger,
  active,
  audience,
  entryFilter,
  emptyMessage = "ledger not loaded yet",
}: LedgerViewProps) {
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
    return <Empty>{error ?? emptyMessage}</Empty>;
  }

  const groups = ledger.sections.map(groupForSection).map((group) => ({
    ...group,
    entries: entryFilter === undefined ? group.entries : group.entries.filter(entryFilter),
  }));
  const entryCount = groups.reduce((sum, group) => sum + group.entries.length, 0);

  if (entryFilter !== undefined && entryCount === 0) {
    return <Empty>{emptyMessage}</Empty>;
  }

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
          {group.entries.map((entry) => {
            const sourceObjectType = ledgerSourceObjectType(entry.source_type);
            const sourceObjectId =
              sourceObjectType === null ? null : ledgerSourceObjectId(entry, sourceObjectType);
            return (
              <div key={entry.id} className="lgr-item">
                <div className="head">
                  <span className="id">
                    {sourceObjectType === null || sourceObjectId === null ? (
                      <>[{group.id}:{entry.id}]</>
                    ) : (
                      <IdRef
                        id={sourceObjectId}
                        type={sourceObjectType}
                        label={`[${group.id}:${entry.id}]`}
                      />
                    )}
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
                  <div className="cite">
                    citations ·{" "}
                    {entry.citations.map((citation, index) => (
                      <Fragment key={`${citation}:${index}`}>
                        {index === 0 ? null : ", "}
                        {resolveObjectType(citation) === null ? (
                          citation
                        ) : (
                          <IdRef id={citation} />
                        )}
                      </Fragment>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
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
