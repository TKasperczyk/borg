import type { ActionRecord, ActionState } from "../../memory/actions/index.js";
import type {
  CommitmentKind,
  CommitmentRecord,
  CommitmentType,
} from "../../memory/commitments/index.js";
import { normalizeDirectiveFamily } from "../../memory/commitments/index.js";
import {
  defaultCommitmentCriticalDomain,
  defaultCommitmentEnforcementClass,
} from "../../memory/commitments/index.js";
import { createActionId, createCommitmentId, createStreamEntryId } from "../../util/ids.js";

const DEFAULT_TEST_TIMESTAMP_MS = 1_000;

function definedOverrides<T extends object>(overrides: Partial<T>): Partial<T> {
  return Object.fromEntries(
    Object.entries(overrides).filter(([, value]) => value !== undefined),
  ) as Partial<T>;
}

function timestampForActionState(
  state: ActionState,
  timestamp: number,
): Pick<
  ActionRecord,
  | "considering_at"
  | "committed_at"
  | "scheduled_at"
  | "completed_at"
  | "not_done_at"
  | "expired_at"
  | "archived_at"
  | "unknown_at"
> {
  return {
    considering_at: state === "considering" ? timestamp : null,
    committed_at: state === "committed_to_do" ? timestamp : null,
    scheduled_at: state === "scheduled" ? timestamp : null,
    completed_at: state === "completed" ? timestamp : null,
    not_done_at: state === "not_done" ? timestamp : null,
    expired_at: state === "expired" ? timestamp : null,
    archived_at: state === "archived" ? timestamp : null,
    unknown_at: state === "unknown" ? timestamp : null,
  };
}

export function makeActionRecord(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const state = overrides.state ?? "scheduled";
  const createdAt = overrides.created_at ?? DEFAULT_TEST_TIMESTAMP_MS;
  const updatedAt = overrides.updated_at ?? createdAt;
  const stateTimestamps = timestampForActionState(state, updatedAt);

  const defaults: ActionRecord = {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "File the Barcelona callback note",
    actor: overrides.actor ?? "borg",
    audience_entity_id: null,
    goal_id: null,
    open_question_id: null,
    state,
    confidence: overrides.confidence ?? 0.86,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: createdAt,
    updated_at: updatedAt,
    considering_at: stateTimestamps.considering_at,
    committed_at: stateTimestamps.committed_at,
    scheduled_at: stateTimestamps.scheduled_at,
    completed_at: stateTimestamps.completed_at,
    not_done_at: stateTimestamps.not_done_at,
    expired_at: stateTimestamps.expired_at,
    archived_at: stateTimestamps.archived_at,
    unknown_at: stateTimestamps.unknown_at,
    canonicalized_by_artifact_entry_id: null,
    session_scope: null,
    session_anchor_id: null,
    last_referenced_at_ms: updatedAt,
    last_referenced_turn_counter: null,
  };

  return {
    ...defaults,
    ...definedOverrides(overrides),
    state,
    created_at: createdAt,
    updated_at: updatedAt,
  };
}

export function makeCompletedActionRecord(overrides: Partial<ActionRecord> = {}): ActionRecord {
  return makeActionRecord({ ...overrides, state: "completed" });
}

export function makeCommitmentRecord(overrides: Partial<CommitmentRecord> = {}): CommitmentRecord {
  const createdAt = overrides.created_at ?? DEFAULT_TEST_TIMESTAMP_MS;
  const type: CommitmentType = overrides.type ?? "promise";
  const kind: CommitmentKind = overrides.kind ?? "assistant_commitment";
  const enforcementClass = overrides.enforcement_class ?? defaultCommitmentEnforcementClass(kind);

  const defaults: CommitmentRecord = {
    id: overrides.id ?? createCommitmentId(),
    record_version: overrides.record_version ?? 1,
    type,
    kind,
    enforcement_class: enforcementClass,
    critical_domain:
      enforcementClass === "critical"
        ? (overrides.critical_domain ?? defaultCommitmentCriticalDomain(kind, enforcementClass))
        : null,
    directive_family: normalizeDirectiveFamily("test commitment fixture"),
    closure_pressure_relevance: overrides.closure_pressure_relevance ?? "neutral",
    directive: overrides.directive ?? "Keep the test commitment active.",
    priority: overrides.priority ?? 5,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    committed_by_entity_id: null,
    provenance: overrides.provenance ?? { kind: "manual" },
    source_stream_entry_ids: overrides.source_stream_entry_ids ?? [createStreamEntryId()],
    created_at: createdAt,
    updated_at: overrides.updated_at ?? createdAt,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    canonicalized_by_artifact_entry_id: null,
    last_reinforced_at: overrides.last_reinforced_at ?? createdAt,
  };

  return {
    ...defaults,
    ...definedOverrides(overrides),
    type,
    kind,
    enforcement_class: enforcementClass,
    critical_domain:
      enforcementClass === "critical"
        ? (overrides.critical_domain ?? defaultCommitmentCriticalDomain(kind, enforcementClass))
        : null,
    created_at: createdAt,
  };
}
