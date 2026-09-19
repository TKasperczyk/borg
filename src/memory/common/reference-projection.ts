import { StorageError } from "../../util/errors.js";
import { parseEntityId } from "../../util/ids.js";
import {
  parseEpisodeParticipantEntityIdTerm,
  episodeParticipantEntityIdTerm,
} from "../episodic/participant-terms.js";

/** Codecs for machine handles, never substring substitution in authored text. */
export function projectReferenceHandle(value: string, resolve: (id: string) => string): string {
  const participant = parseEpisodeParticipantEntityIdTerm(value);
  if (participant !== null)
    return episodeParticipantEntityIdTerm(parseEntityId(resolve(participant)));
  for (const prefix of ["semantic_node:", "semantic_edge:"]) {
    if (value.startsWith(prefix)) return prefix + resolve(value.slice(prefix.length));
  }
  // Some repository scope keys encode a machine object as JSON.
  if (value.startsWith("{") || value.startsWith("[")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(value);
    } catch {
      return value;
    }
    const next = projectMachineReferences(parsed, resolve);
    return JSON.stringify(parsed) === JSON.stringify(next) ? value : JSON.stringify(next);
  }
  return resolve(value);
}

// Field contracts from the stream, working-memory, disclosure, identity, review,
// semantic and repository schemas. Unlisted scalar fields are authored/opaque.
// A field's value being an ID (or valid JSON) does not make it a reference.
const HANDLE_FIELDS = new Set([
  "id",
  "entity_ids",
  "entityIds",
  "entity_id",
  "entityId",
  "audience_entity_id",
  "audienceEntityId",
  "sender_entity_id",
  "senderEntityId",
  "reply_target_entity_id",
  "replyTargetEntityId",
  "source_entity_id",
  "sourceEntityId",
  "speaker_entity_id",
  "speakerEntityId",
  "actor_entity_id",
  "actorEntityId",
  "actor",
  "self_entity_id",
  "selfEntityId",
  "owner_entity_id",
  "ownerEntityId",
  "counterparty_entity_id",
  "subject_entity_id",
  "object_entity_id",
  "identity_entity_id",
  "created_by_entity_id",
  "committed_by_entity_id",
  "filer_entity_id",
  "merged_into",
  "made_to_entity",
  "about_entity",
  "restricted_audience",
  "origin_audience_entity_ids",
  "originAudienceEntityIds",
  "private_to_entity_ids",
  "privateToEntityIds",
  "public_to_entity_ids",
  "publicToEntityIds",
  "participant_entity_ids",
  "participantEntityIds",
  "allowed_entity_ids",
  "excluded_entity_ids",
  "activation_entity_ids",
  "activation_excluded_entity_ids",
  "visibleAudienceEntityIds",
  "audienceEntityIds",
  "rankingAudienceEntityId",
  "participants",
  "hot_entities",
  "entityTerms",
  "audienceTerms",
  "node_id",
  "nodeId",
  "nodeIds",
  "edgeIds",
  "node_ids",
  "edge_id",
  "edgeId",
  "edge_ids",
  "repair_target_id",
  "witness",
  "target_id",
  "source_edge_id",
  "source_node_id",
  "from_node_id",
  "to_node_id",
  "target_node_id",
  "insight_node_id",
  "winner_node_id",
  "loser_edge_id",
  "invalidated_by_edge_id",
  "related_semantic_node_ids",
  "created_node_ids",
  "created_edge_ids",
  "superseded_by",
  "corrected_by",
  "belief_id",
  "evidence_id",
  "dependency_id",
  "record_id",
  "scope_key",
  "scopeKey",
]);
const JSON_FIELDS = new Set([
  "metadata",
  "metadata_json",
  "detection_key",
  "structural_scope_key",
  "structural_scope_keys",
  "structural_detection_key",
  "state_json",
  "refs",
  "applying_state",
  "applying_state_json",
  "disclosure_label",
  "resolution_disclosure_label",
  "source_disclosure_label",
  "old_value_json",
  "new_value_json",
  "block_history_json",
  "headway_bases_json",
  "pending_attempt_snapshot",
  "procedural_context",
  "procedural_context_json",
  "observation_metadata",
  "targets",
  "reversal",
  "provenance",
  "evidence",
]);
// Stream content is arbitrary authored JSON, not a machine metadata envelope.
const OPAQUE_FIELDS = new Set(["content", "embedding"]);

/** Shared read-time / offline projection. Arrays retain order and tuple cardinality. */
export function projectMachineReferences<T>(
  value: T,
  resolve: (id: string) => string,
  options: { table?: string } = {},
): T {
  const visit = (item: unknown, field = ""): unknown => {
    if (ArrayBuffer.isView(item)) return item;
    const handle =
      HANDLE_FIELDS.has(field) ||
      (options.table === "episode_participants" && (field === "term" || field === "value"));
    if (typeof item === "string") {
      if (handle || JSON_FIELDS.has(field)) {
        // Only schema-declared machine fields may contain serialized JSON.
        if (item.startsWith("{") || item.startsWith("[")) {
          let parsed: unknown;
          try {
            parsed = JSON.parse(item);
          } catch {
            return item;
          }
          const next = visit(parsed, field);
          return JSON.stringify(parsed) === JSON.stringify(next) ? item : JSON.stringify(next);
        }
        if (handle) return projectReferenceHandle(item, resolve);
      }
      return item;
    }
    if (Array.isArray(item)) return item.map((entry) => visit(entry, field));
    if (item === null || typeof item !== "object") return item;
    if (Object.getPrototypeOf(item) !== Object.prototype && Object.getPrototypeOf(item) !== null)
      return item;
    const result: Record<string, unknown> = {};
    for (const [key, nested] of Object.entries(item)) {
      const nextKey = projectReferenceHandle(key, resolve);
      const opaque =
        OPAQUE_FIELDS.has(key) &&
        !(key === "content" && "kind" in item && item.kind === "internal_event");
      const next = opaque ? nested : visit(nested, key);
      if (
        Object.hasOwn(result, nextKey) &&
        JSON.stringify(result[nextKey]) !== JSON.stringify(next)
      ) {
        // Recall-state's documented map values are suppression expiry turns.
        if (
          field === "suppressedHandles" &&
          typeof next === "number" &&
          typeof result[nextKey] === "number"
        ) {
          result[nextKey] = Math.max(result[nextKey], next);
          continue;
        }
        throw new StorageError(`Conflicting machine reference map keys: ${key}`, {
          code: "REFERENCE_PROJECTION_COLLISION",
        });
      }
      Object.defineProperty(result, nextKey, {
        value: next,
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
    return result;
  };
  return visit(value) as T;
}
