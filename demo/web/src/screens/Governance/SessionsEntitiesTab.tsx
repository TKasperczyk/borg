import { useEffect, useMemo, useState, type FormEvent } from "react";

import type { CreatorDirectiveItem, EntityRecord, SessionRecord } from "../../api/types";
import { IdRef } from "../../components/Inspector/IdRef";
import { ParticipationPolicyControl } from "../../components/ParticipationPolicyControl";
import { Tag } from "../../components/Tag";
import { sourceLabel } from "../../components/SessionFleet";
import { isInteractiveDescendantEvent } from "../../lib/keyboard";
import { dateLabel, shortId } from "../screen-utils";

type SessionsEntitiesTabProps = {
  sessions: readonly SessionRecord[];
  directives: readonly CreatorDirectiveItem[];
  activeSessionId: string;
  creator: EntityRecord | null;
  operatorChatError: string | null;
  onSelectSession: (sessionId: string) => void;
  onOpenOperatorChat: () => Promise<void> | void;
  onSetCreatorByName: (name: string) => Promise<void> | void;
  onSessionPolicyChanged: () => Promise<void>;
};

type KnownEntity = {
  id: string;
  labels: Set<string>;
  sources: Set<string>;
  creator: boolean;
};

function addKnownEntity(
  entities: Map<string, KnownEntity>,
  input: { id: string | null; label: string | null; source: string; creator?: boolean },
): void {
  if (input.id === null) {
    return;
  }

  const current =
    entities.get(input.id) ??
    ({
      id: input.id,
      labels: new Set<string>(),
      sources: new Set<string>(),
      creator: false,
    } satisfies KnownEntity);
  if (input.label !== null && input.label.length > 0) {
    current.labels.add(input.label);
  }
  current.sources.add(input.source);
  current.creator = current.creator || input.creator === true;
  entities.set(input.id, current);
}

function knownEntities(input: {
  creator: EntityRecord | null;
  sessions: readonly SessionRecord[];
  directives: readonly CreatorDirectiveItem[];
}): KnownEntity[] {
  const entities = new Map<string, KnownEntity>();

  if (input.creator !== null) {
    addKnownEntity(entities, {
      id: input.creator.id,
      label: input.creator.canonical_name,
      source: "creator",
      creator: true,
    });
  }

  for (const session of input.sessions) {
    addKnownEntity(entities, {
      id: session.audience_entity_id,
      label: session.audience_label,
      source: `session ${shortId(session.session_id)}`,
    });
  }

  for (const directive of input.directives) {
    addKnownEntity(entities, {
      id: directive.subject_entity_id,
      label: directive.subject_entity_name,
      source: `directive ${shortId(directive.id)}`,
    });
  }

  return [...entities.values()].sort((left, right) => {
    if (left.creator !== right.creator) {
      return left.creator ? -1 : 1;
    }
    return left.id.localeCompare(right.id);
  });
}

export function SessionsEntitiesTab({
  sessions,
  directives,
  activeSessionId,
  creator,
  operatorChatError,
  onSelectSession,
  onOpenOperatorChat,
  onSetCreatorByName,
  onSessionPolicyChanged,
}: SessionsEntitiesTabProps) {
  const [creatorName, setCreatorName] = useState(creator?.canonical_name ?? "");
  const [creatorBusy, setCreatorBusy] = useState(false);
  const entities = useMemo(
    () => knownEntities({ creator, sessions, directives }),
    [creator, directives, sessions],
  );

  useEffect(() => {
    setCreatorName(creator?.canonical_name ?? "");
  }, [creator?.canonical_name]);

  async function submitCreator(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const name = creatorName.trim();
    if (name.length === 0) {
      return;
    }

    setCreatorBusy(true);
    try {
      await onSetCreatorByName(name);
    } finally {
      setCreatorBusy(false);
    }
  }

  return (
    <div className="governance-panel sessions-entities">
      <div className="page-head">
        <span className="desc">session participation policy and known entity handles</span>
        <span className="spacer"></span>
        <button type="button" className="btn sm primary" onClick={onOpenOperatorChat}>
          operator chat
        </button>
        <form className="governance-creator-form" onSubmit={(event) => void submitCreator(event)}>
          <input
            value={creatorName}
            onChange={(event) => setCreatorName(event.target.value)}
            placeholder={creator?.canonical_name ?? "creator name"}
            aria-label="creator name"
          />
          <button
            className="btn sm"
            type="submit"
            disabled={creatorBusy || creatorName.trim().length === 0}
          >
            mark creator
          </button>
        </form>
      </div>
      {operatorChatError === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          {operatorChatError}
        </div>
      )}
      <div className="page-body sessions-entities-body">
        <section className="scope-section" aria-label="sessions table">
          <div className="scope-section-head">
            <h2>sessions</h2>
            <span className="dim">{sessions.length.toLocaleString()} rows from getSessions</span>
          </div>
          <div style={{ overflow: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ minWidth: 180 }}>label</th>
                  <th style={{ width: 100 }}>source</th>
                  <th style={{ width: 150 }}>audience</th>
                  <th style={{ width: 110 }}>role</th>
                  <th style={{ width: 110 }}>privacy</th>
                  <th style={{ width: 90 }}>status</th>
                  <th style={{ minWidth: 220 }}>policy</th>
                  <th style={{ width: 120 }}>last turn</th>
                  <th style={{ width: 90, textAlign: "right" }}>messages</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((session) => (
                  <tr
                    key={session.session_id}
                    className={session.session_id === activeSessionId ? "selected" : ""}
                    onClick={(event) => {
                      if (!isInteractiveDescendantEvent(event.currentTarget, event.target)) {
                        onSelectSession(session.session_id);
                      }
                    }}
                    style={{ cursor: "pointer" }}
                  >
                    <td>
                      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        <button
                          type="button"
                          className="row-select-button"
                          aria-pressed={session.session_id === activeSessionId}
                          aria-label={`select session ${session.session_id}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            onSelectSession(session.session_id);
                          }}
                        >
                          {session.label}
                        </button>
                        <IdRef
                          id={session.session_id}
                          type="session"
                          label={shortId(session.session_id)}
                          hint={session}
                        />
                        <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
                          {dateLabel(session.last_activity_at)}
                        </span>
                      </div>
                    </td>
                    <td>{sourceLabel(session)}</td>
                    <td>
                      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        <span>{session.audience_label}</span>
                        {session.audience_entity_id === null ? null : (
                          <IdRef
                            id={session.audience_entity_id}
                            type="entity"
                            label={shortId(session.audience_entity_id)}
                          />
                        )}
                      </div>
                    </td>
                    <td>{session.audience_role}</td>
                    <td>{session.privacy_level}</td>
                    <td>
                      <Tag kind={session.status === "active" ? "acc" : ""}>{session.status}</Tag>
                    </td>
                    <td onClick={(event) => event.stopPropagation()}>
                      <ParticipationPolicyControl
                        sessionId={session.session_id}
                        policy={session.participation_policy}
                        locked={session.audience_role === "operator"}
                        onChanged={onSessionPolicyChanged}
                      />
                    </td>
                    <td>
                      {session.last_turn_id === null ? (
                        <span className="mute">—</span>
                      ) : (
                        <IdRef
                          id={session.last_turn_id}
                          type="turn"
                          label={shortId(session.last_turn_id)}
                        />
                      )}
                    </td>
                    <td className="tab-num" style={{ textAlign: "right" }}>
                      {session.message_count.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="scope-section" aria-label="known entities">
          <div className="scope-section-head">
            <h2>entities known from sessions/directives</h2>
            <span className="capability-note">entity creation not available from this console</span>
          </div>
          {entities.length === 0 ? (
            <div className="notice">no entity ids surfaced by sessions or directives</div>
          ) : (
            <div className="known-entities">
              {entities.map((entity) => (
                <article key={entity.id} className="known-entity">
                  <div className="known-entity-head">
                    <IdRef id={entity.id} type="entity" label={entity.id} />
                    {entity.creator ? <Tag kind="acc">creator</Tag> : null}
                  </div>
                  <div className="props">
                    <div className="row">
                      <span className="k">labels</span>
                      <span className="v">
                        {[...entity.labels].sort().join(", ") || <span className="mute">—</span>}
                      </span>
                    </div>
                    <div className="row">
                      <span className="k">sources</span>
                      <span className="v">{[...entity.sources].sort().join(", ")}</span>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
