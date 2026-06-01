import type {
  SharedStateAddOperation,
  SharedStateArtifact,
  SharedStateCanonicalizes,
  SharedStateEntry,
  SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";
import { createEntityId, createSharedStateEntryId, createStreamEntryId } from "../../util/ids.js";

function definedOverrides<T extends object>(overrides: Partial<T>): Partial<T> {
  return Object.fromEntries(
    Object.entries(overrides).filter(([, value]) => value !== undefined),
  ) as Partial<T>;
}

export function makeSharedStateCanonicalizes(
  overrides: Partial<SharedStateCanonicalizes> = {},
): SharedStateCanonicalizes {
  return {
    goal_ids: overrides.goal_ids ?? [],
    commitment_ids: overrides.commitment_ids ?? [],
    action_ids: overrides.action_ids ?? [],
    open_question_ids: overrides.open_question_ids ?? [],
  };
}

export function makeSharedStateEntry(overrides: Partial<SharedStateEntry> = {}): SharedStateEntry {
  const sourceIds = overrides.provenance_stream_entry_ids ??
    overrides.last_updated_stream_entry_ids ?? [createStreamEntryId()];
  const createdAt = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createSharedStateEntryId(),
    audience_entity_id: overrides.audience_entity_id ?? createEntityId(),
    state_key: overrides.state_key ?? "decision.fixture",
    kind: overrides.kind ?? "locked",
    text: overrides.text ?? "Release freeze is locked for the workstream",
    owner_entity_id: overrides.owner_entity_id ?? null,
    provenance_stream_entry_ids: sourceIds,
    last_updated_stream_entry_ids: overrides.last_updated_stream_entry_ids ?? sourceIds,
    created_at: createdAt,
    last_updated_at: overrides.last_updated_at ?? createdAt,
    last_updated_turn_global: overrides.last_updated_turn_global ?? null,
    superseded_by_id: overrides.superseded_by_id ?? null,
    rank: overrides.rank ?? 0,
    canonicalizes: makeSharedStateCanonicalizes(overrides.canonicalizes),
  };
}

export function makeLockedSharedStateEntry(
  overrides: Partial<SharedStateEntry> = {},
): SharedStateEntry {
  return makeSharedStateEntry({ ...overrides, kind: "locked" });
}

export function makeLiveSharedStateEntry(
  overrides: Partial<SharedStateEntry> = {},
): SharedStateEntry {
  return makeSharedStateEntry({ ...overrides, kind: "live" });
}

export function makeTentativeSharedStateEntry(
  overrides: Partial<SharedStateEntry> = {},
): SharedStateEntry {
  return makeSharedStateEntry({ ...overrides, kind: "tentative" });
}

export function makeInvalidatedSharedStateEntry(
  overrides: Partial<SharedStateEntry> = {},
): SharedStateEntry {
  return makeSharedStateEntry({ ...overrides, kind: "invalidated" });
}

export function makeSharedStateArtifact(
  entries: readonly SharedStateEntry[] = [makeLockedSharedStateEntry()],
  overrides: Partial<SharedStateArtifact> = {},
): SharedStateArtifact {
  const createdAt = overrides.created_at ?? 1_000;

  const defaults: SharedStateArtifact = {
    audience_entity_id:
      overrides.audience_entity_id ?? entries[0]?.audience_entity_id ?? createEntityId(),
    record_version: overrides.record_version ?? 1,
    created_at: createdAt,
    updated_at: overrides.updated_at ?? createdAt,
    last_compiled_at: createdAt,
    last_compiled_stream_entry_id: createStreamEntryId(),
    entries: overrides.entries ?? [...entries],
  };

  return {
    ...defaults,
    ...definedOverrides(overrides),
    created_at: createdAt,
  };
}

export function makeSharedStateAddOperation(
  entry: SharedStateEntry = makeLockedSharedStateEntry(),
  overrides: Partial<SharedStateAddOperation> = {},
): SharedStateAddOperation {
  return {
    type: "add",
    id: entry.id,
    kind: entry.kind as SharedStateEntryKind,
    state_key: entry.state_key ?? "decision.fixture",
    text: entry.text,
    owner_entity_id: entry.owner_entity_id,
    provenance_stream_entry_ids: entry.provenance_stream_entry_ids,
    last_updated_stream_entry_ids: entry.last_updated_stream_entry_ids,
    created_at: entry.created_at,
    last_updated_at: entry.last_updated_at,
    rank: entry.rank,
    canonicalizes: entry.canonicalizes,
    ...overrides,
  };
}
