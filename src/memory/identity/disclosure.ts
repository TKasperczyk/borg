import { legacyCommitmentSchema, type CommitmentRecord } from "../commitments/index.js";
import {
  isMemoryDisclosureLabelVisibleToAnyAudience,
  memoryDisclosureLabelFromMetadata,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import { isEpisodeAccessVisible, type Episode, type EpisodeAccessLike } from "../episodic/index.js";
import { entityIdHelpers, episodeIdHelpers, type EntityId } from "../../util/ids.js";
import type { IdentityEvent } from "./types.js";

export type CommitmentAccess = Pick<
  CommitmentRecord,
  "id" | "made_to_entity" | "restricted_audience" | "source_stream_entry_ids"
>;

export type IdentityEventValueDisclosureSources = {
  audienceEntityIds: EntityId[];
  commitmentAccesses: CommitmentAccess[];
  disclosureLabels: MemoryDisclosureLabel[];
  empty: boolean;
  episodeAccesses: EpisodeAccessLike[];
  malformed: boolean;
  recognized: boolean;
  sourceEpisodeIds: Episode["id"][];
};

export type IdentityEventDisclosureSources = IdentityEventValueDisclosureSources & {
  newValue: IdentityEventValueDisclosureSources;
  oldValue: IdentityEventValueDisclosureSources;
  provenance: IdentityEventValueDisclosureSources;
};

const SOURCE_EPISODE_ID_KEYS = [
  "episode_ids",
  "evidence_episode_ids",
  "key_episode_ids",
  "related_episode_ids",
  "resolution_evidence_episode_ids",
  "source_episode_ids",
] as const;

const DISCLOSURE_LABEL_KEYS = [
  "disclosureLabel",
  "disclosure_label",
  "resolutionDisclosureLabel",
  "resolution_disclosure_label",
] as const;

const AUDIENCE_ENTITY_ID_KEYS = ["audience_entity_id", "owner_entity_id"] as const;

const MAX_IDENTITY_EVENT_SOURCE_DEPTH = 4;

function emptyIdentityEventValueDisclosureSources(): IdentityEventValueDisclosureSources {
  return {
    audienceEntityIds: [],
    commitmentAccesses: [],
    disclosureLabels: [],
    empty: false,
    episodeAccesses: [],
    malformed: false,
    recognized: false,
    sourceEpisodeIds: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function uniqueEpisodeIds(ids: Episode["id"][]): Episode["id"][] {
  return Array.from(new Set(ids));
}

function uniqueEntityIds(ids: EntityId[]): EntityId[] {
  return Array.from(new Set(ids)).sort((left, right) => left.localeCompare(right));
}

function hasOwnKey(value: Record<string, unknown>, key: string): value is Record<string, unknown> {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function parseEpisodeIdArray(value: unknown): Episode["id"][] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const episodeIds: Episode["id"][] = [];
  for (const item of value) {
    if (typeof item !== "string" || !episodeIdHelpers.is(item)) {
      return null;
    }
    episodeIds.push(item);
  }
  return episodeIds;
}

function parseEntityIdArray(value: unknown): EntityId[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const entityIds: EntityId[] = [];
  for (const item of value) {
    if (typeof item !== "string" || !entityIdHelpers.is(item)) {
      return null;
    }
    entityIds.push(item);
  }
  return entityIds;
}

function parseEntityId(value: unknown): EntityId | null | undefined {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string" && entityIdHelpers.is(value)) {
    return value;
  }
  return undefined;
}

function parseDisclosureLabel(value: unknown): MemoryDisclosureLabel | null {
  return memoryDisclosureLabelFromMetadata(value);
}

function mergeIdentityEventValueDisclosureSources(
  target: IdentityEventValueDisclosureSources,
  source: IdentityEventValueDisclosureSources,
): void {
  target.audienceEntityIds = uniqueEntityIds([
    ...target.audienceEntityIds,
    ...source.audienceEntityIds,
  ]);
  target.commitmentAccesses.push(...source.commitmentAccesses);
  target.disclosureLabels.push(...source.disclosureLabels);
  target.empty = target.empty || source.empty;
  target.episodeAccesses.push(...source.episodeAccesses);
  target.malformed = target.malformed || source.malformed;
  target.recognized = target.recognized || source.recognized;
  target.sourceEpisodeIds = uniqueEpisodeIds([
    ...target.sourceEpisodeIds,
    ...source.sourceEpisodeIds,
  ]);
}

function identityEventEpisodeAccess(
  value: unknown,
  options: { allowNestedEpisode: boolean },
): EpisodeAccessLike | null | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  if (
    !hasOwnKey(value, "audience_entity_id") &&
    !hasOwnKey(value, "origin_audience_entity_ids") &&
    !hasOwnKey(value, "shared") &&
    !hasOwnKey(value, "episode")
  ) {
    return undefined;
  }

  if (options.allowNestedEpisode && hasOwnKey(value, "episode") && isRecord(value.episode)) {
    return identityEventEpisodeAccess(value.episode, {
      allowNestedEpisode: false,
    });
  }

  const audienceEntityId = parseEntityId(value.audience_entity_id);
  if (audienceEntityId === undefined) {
    return null;
  }

  const originAudienceEntityIds = hasOwnKey(value, "origin_audience_entity_ids")
    ? parseEntityIdArray(value.origin_audience_entity_ids)
    : [];
  if (originAudienceEntityIds === null) {
    return null;
  }

  const shared =
    value.shared === undefined
      ? false
      : typeof value.shared === "boolean"
        ? value.shared
        : undefined;
  if (shared === undefined) {
    return null;
  }

  if (audienceEntityId === null && originAudienceEntityIds.length === 0 && shared === false) {
    return null;
  }

  return {
    audience_entity_id: audienceEntityId,
    origin_audience_entity_ids: originAudienceEntityIds,
    shared,
  };
}

function collectIdentityEventValueDisclosureSources(
  value: unknown,
  recordType: IdentityEvent["record_type"],
  result: IdentityEventValueDisclosureSources,
  options: { allowNestedEpisode: boolean; depth: number },
): void {
  if (value === null || value === undefined) {
    if (options.depth === 0) {
      result.empty = true;
    }
    return;
  }

  if (options.depth > MAX_IDENTITY_EVENT_SOURCE_DEPTH) {
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectIdentityEventValueDisclosureSources(item, recordType, result, {
        allowNestedEpisode: false,
        depth: options.depth + 1,
      });
    }
    return;
  }

  if (!isRecord(value)) {
    return;
  }

  if (hasOwnKey(value, "restricted_audience")) {
    result.recognized = true;
    const parsedCommitment = legacyCommitmentSchema.safeParse(value);
    if (!parsedCommitment.success) {
      result.malformed = true;
    } else {
      result.commitmentAccesses.push({
        id: parsedCommitment.data.id,
        made_to_entity: parsedCommitment.data.made_to_entity,
        restricted_audience: parsedCommitment.data.restricted_audience,
        source_stream_entry_ids: parsedCommitment.data.source_stream_entry_ids,
      });
    }
  }

  const episodeAccess = identityEventEpisodeAccess(value, {
    allowNestedEpisode: options.allowNestedEpisode || recordType === "episode",
  });
  if (episodeAccess !== undefined) {
    result.recognized = true;
    if (episodeAccess === null) {
      result.malformed = true;
    } else {
      result.episodeAccesses.push(episodeAccess);
    }
  }

  for (const key of SOURCE_EPISODE_ID_KEYS) {
    if (!hasOwnKey(value, key)) {
      continue;
    }
    result.recognized = true;
    const episodeIds = parseEpisodeIdArray(value[key]);
    if (episodeIds === null) {
      result.malformed = true;
      continue;
    }
    result.sourceEpisodeIds = uniqueEpisodeIds([...result.sourceEpisodeIds, ...episodeIds]);
  }

  for (const key of DISCLOSURE_LABEL_KEYS) {
    if (!hasOwnKey(value, key)) {
      continue;
    }
    result.recognized = true;
    const disclosureLabel = parseDisclosureLabel(value[key]);
    if (disclosureLabel === null) {
      result.malformed = true;
      continue;
    }
    result.disclosureLabels.push(disclosureLabel);
  }

  if (recordType === "goal" && hasOwnKey(value, "block_history")) {
    // Historical audit snapshots remain verbatim. An unlabeled block rationale
    // cannot inherit the enclosing goal's owner when its change is rendered.
    result.recognized = true;
    if (!Array.isArray(value.block_history)) {
      result.malformed = true;
    } else if (
      value.block_history.some((block) => !isRecord(block) || !hasOwnKey(block, "disclosure_label"))
    ) {
      result.disclosureLabels.push(unknownMemoryDisclosureLabel());
    }
  }

  for (const key of AUDIENCE_ENTITY_ID_KEYS) {
    if (!hasOwnKey(value, key)) {
      continue;
    }
    result.recognized = true;
    const entityId = parseEntityId(value[key]);
    if (entityId === undefined) {
      result.malformed = true;
      continue;
    }
    if (entityId !== null) {
      result.audienceEntityIds = uniqueEntityIds([...result.audienceEntityIds, entityId]);
    }
  }

  for (const nestedValue of Object.values(value)) {
    collectIdentityEventValueDisclosureSources(nestedValue, recordType, result, {
      allowNestedEpisode: false,
      depth: options.depth + 1,
    });
  }
}

export function parseIdentityEventValueDisclosureSources(
  value: unknown,
  recordType: IdentityEvent["record_type"],
): IdentityEventValueDisclosureSources {
  const result = emptyIdentityEventValueDisclosureSources();
  collectIdentityEventValueDisclosureSources(value, recordType, result, {
    allowNestedEpisode: recordType === "episode",
    depth: 0,
  });
  return result;
}

export function parseIdentityEventDisclosureSources(
  event: IdentityEvent,
): IdentityEventDisclosureSources {
  const oldValue = parseIdentityEventValueDisclosureSources(event.old_value, event.record_type);
  const newValue = parseIdentityEventValueDisclosureSources(event.new_value, event.record_type);
  const provenance = parseIdentityEventValueDisclosureSources(event.provenance, event.record_type);
  const combined = emptyIdentityEventValueDisclosureSources();
  for (const source of [oldValue, newValue, provenance]) {
    mergeIdentityEventValueDisclosureSources(combined, source);
  }
  return {
    ...combined,
    newValue,
    oldValue,
    provenance,
  };
}

function visibleCommitmentAudience(
  commitment: CommitmentAccess,
  audienceEntityId: EntityId | null,
): boolean {
  return (
    commitment.restricted_audience === null || commitment.restricted_audience === audienceEntityId
  );
}

function isDisclosureLabelVisible(
  label: MemoryDisclosureLabel,
  audienceEntityId: EntityId | null,
): boolean {
  return isMemoryDisclosureLabelVisibleToAnyAudience(
    label,
    audienceEntityId === null ? [] : [audienceEntityId],
  );
}

function isIdentityEventValueVisible(
  sources: IdentityEventValueDisclosureSources,
  audienceEntityId: EntityId | null,
  options: { neutralWhenUnrecognized?: boolean } = {},
): boolean {
  if (sources.empty) {
    return true;
  }
  if (sources.malformed) {
    return false;
  }
  if (sources.sourceEpisodeIds.length > 0) {
    return false;
  }
  if (
    sources.disclosureLabels.some((label) => !isDisclosureLabelVisible(label, audienceEntityId))
  ) {
    return false;
  }
  if (sources.episodeAccesses.some((access) => !isEpisodeAccessVisible(access, audienceEntityId))) {
    return false;
  }
  if (
    sources.commitmentAccesses.some(
      (commitment) => !visibleCommitmentAudience(commitment, audienceEntityId),
    )
  ) {
    return false;
  }
  if (
    sources.audienceEntityIds.length > 0 &&
    (audienceEntityId === null || !sources.audienceEntityIds.includes(audienceEntityId))
  ) {
    return false;
  }
  return sources.recognized || options.neutralWhenUnrecognized === true;
}

export function isIdentityEventVisible(
  event: IdentityEvent,
  audienceEntityId: EntityId | null,
): boolean {
  const sources = parseIdentityEventDisclosureSources(event);
  return (
    isIdentityEventValueVisible(sources.oldValue, audienceEntityId) &&
    isIdentityEventValueVisible(sources.newValue, audienceEntityId) &&
    isIdentityEventValueVisible(sources.provenance, audienceEntityId, {
      neutralWhenUnrecognized: true,
    })
  );
}
