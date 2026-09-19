/** Offline projection validation uses the same decoders as repository reads. */
import { z } from "zod";
import { mapPeriodRow } from "../src/memory/self/autobiographical.js";
import { mapEntityRow, mapCommitmentRow } from "../src/memory/commitments/repository.js";
import { entityIdSchema } from "../src/memory/commitments/types.js";
import { mapActionRow } from "../src/memory/actions/repository.js";
import { mapActivityRow } from "../src/memory/activity/repository.js";
import { mapDaySummaryRow } from "../src/memory/activity/lived-experience-day-summary.js";
import { mapRow as mapIdentityRow } from "../src/memory/identity/repository.js";
import { mapObservedEventRow } from "../src/memory/observed-events/repository.js";
import { mapProfileRow, mapEventRow } from "../src/memory/social/repository.js";
import { mapEntryRow, mapArtifactRow } from "../src/memory/shared-state/repository.js";
import {
  proceduralEvidenceFromRow,
  skillFromRow,
  skillContextStatsFromRow,
} from "../src/memory/procedural/repository.js";
import { mapGrowthMarkerRow } from "../src/memory/self/growth-markers.js";
import { mapGoalRow, mapTraitRow, mapValueRow } from "../src/memory/self/shared/sql-mapping.js";
import {
  mapOpenQuestionRow,
  mapOpenQuestionRuminationRow,
} from "../src/memory/self/open-questions.js";
import { nodeFromRow, edgeFromRow } from "../src/memory/semantic/repository.js";
import { semanticNodeIdSchema, semanticEdgeIdSchema } from "../src/memory/semantic/types.js";
import { mapRelationalSlotRow } from "../src/memory/relational-slots/repository.js";
import { mapCreatorDirectiveRow } from "../src/memory/creator-directives/repository.js";
import { mapTrainOfThoughtJournalEntryRow } from "../src/memory/train-of-thought/repository.js";
import { fromEpisodeRow } from "../src/memory/episodic/repository.js";
import { episodeIdSchema } from "../src/memory/episodic/types.js";
import { episodeSchema } from "../src/memory/episodic/types.js";
import { mapAuditRow } from "../src/offline/audit-log.js";
import { sessionRecordSchema } from "../src/sessions/types.js";
import { recallStateSchema } from "../src/retrieval/recall-state.js";
import { attachmentIdSchema, imageMediaTypeSchema } from "../src/attachments/types.js";
import { imagePerceptionIdHelpers } from "../src/util/ids.js";
import { streamEntryIdSchema } from "../src/stream/types.js";
import { operatorAttentionRecordSchema } from "../src/memory/operator-attention/types.js";
import {
  mapReviewRow,
  ReviewQueueHandlerRegistry,
  stripReviewResolverRefs,
} from "../src/memory/review-queue/review-queue.js";
import { registerBuiltinReviewQueueHandlers } from "../src/memory/review-queue/review-handlers/defaults.js";
import { fail, type Row, type SqlChange, type LanceChange } from "./repair-entity-graph-storage.js";

const handlers = new ReviewQueueHandlerRegistry();
registerBuiltinReviewQueueHandlers(handlers);
const decoders: Record<string, (row: Row) => unknown> = {
  entities: mapEntityRow,
  commitments: mapCommitmentRow,
  action_records: mapActionRow,
  activity_events: mapActivityRow,
  lived_experience_day_summaries: mapDaySummaryRow,
  identity_events: mapIdentityRow,
  observed_events: mapObservedEventRow,
  social_profiles: mapProfileRow,
  social_events: mapEventRow,
  shared_state_entries: mapEntryRow,
  shared_state_artifacts: (row) => mapArtifactRow(row, []),
  procedural_evidence: proceduralEvidenceFromRow,
  skills: skillFromRow,
  skill_context_stats: skillContextStatsFromRow,
  growth_markers: mapGrowthMarkerRow,
  goals: mapGoalRow,
  autobiographical_periods: mapPeriodRow,
  traits: mapTraitRow,
  values: mapValueRow,
  open_questions: mapOpenQuestionRow,
  open_question_ruminations: mapOpenQuestionRuminationRow,
  semantic_nodes: (row) => nodeFromRow({ ...row, embedding: new Float32Array([0]) }),
  semantic_edges: edgeFromRow,
  relational_slots: mapRelationalSlotRow,
  creator_directives: mapCreatorDirectiveRow,
  train_of_thought_journal_entries: mapTrainOfThoughtJournalEntryRow,
  maintenance_audit: mapAuditRow,
  sessions: (row) => sessionRecordSchema.parse(row),
  operator_attention_records: (row) => operatorAttentionRecordSchema.parse(row),
  recall_state: (row) => {
    const state = recallStateSchema.parse(JSON.parse(String(row.state_json)));
    if (state.scopeKey !== row.scope_key) fail("Recall state scope key mismatch");
    return state;
  },
  review_queue: (row) => {
    const item = mapReviewRow(row);
    const handler = handlers.get(item.kind);
    if (handler !== null && item.resolved_at === null)
      handler.refsSchema.parse(stripReviewResolverRefs(item.refs));
    // Resolved rejections may intentionally retain malformed payloads as evidence.
    // Validate every stored row; only actionable refs must satisfy a handler input schema.
    return item;
  },
  episode_index: (row) => {
    episodeIdSchema.parse(row.episode_id);
    episodeSchema.shape.audience_entity_id.parse(row.audience_entity_id);
    episodeSchema.shape.origin_audience_entity_ids.parse(
      JSON.parse(String(row.origin_audience_entity_ids)),
    );
    episodeSchema.shape.shared.parse(Boolean(row.shared));
  },
  // These link/outbox tables have SQL schemas, not record codecs. Reuse the
  // repositories' branded handle schemas in addition to the projected SQL constraints.
  stream_attachments: (row) => {
    attachmentIdSchema.parse(row.attachment_id);
    imageMediaTypeSchema.parse(row.media_type);
    streamEntryIdSchema.parse(row.parent_entry_id);
    streamEntryIdSchema.nullable().parse(row.stream_entry_id);
    entityIdSchema.nullable().parse(row.audience_entity_id);
  },
  image_perception_artifacts: (row) => {
    imagePerceptionIdHelpers.parse(z.string().parse(row.artifact_id));
    imagePerceptionIdHelpers.parse(z.string().parse(row.payload_id));
    attachmentIdSchema.parse(row.attachment_id);
    streamEntryIdSchema.parse(row.parent_entry_id);
    streamEntryIdSchema.nullable().parse(row.stream_entry_id);
    entityIdSchema.nullable().parse(row.audience_entity_id);
  },
  episode_participants: (row) => {
    episodeIdSchema.parse(row.episode_id);
    z.string().min(1).parse(row.term);
    z.string().min(1).parse(row.value);
  },
  entity_external_ids: (row) => {
    entityIdSchema.parse(row.entity_id);
    z.string().min(1).parse(row.source);
    z.string().min(1).parse(row.external_id);
  },
  semantic_belief_dependencies: (row) => {
    (row.target_type === "semantic_node" ? semanticNodeIdSchema : semanticEdgeIdSchema).parse(
      row.target_id,
    );
    semanticEdgeIdSchema.parse(row.source_edge_id);
    z.enum(["supports", "derived_from"]).parse(row.dependency_kind);
  },
  semantic_edge_invalidation_events: (row) => {
    semanticEdgeIdSchema.parse(row.edge_id);
    z.number().finite().parse(row.valid_to);
  },
  semantic_node_vector_sync_outbox: (row) => {
    semanticNodeIdSchema.parse(row.node_id);
    z.number().int().positive().parse(row.generation);
  },
};

export function validateSqlChanges(changes: SqlChange[]): void {
  for (const change of changes) {
    if (change.after === null) continue;
    const decode =
      decoders[change.table] ??
      fail(
        `No repository validator for changed table ${change.table}; extend the offline adapter before applying`,
      );
    try {
      decode(change.after);
    } catch (error) {
      throw new Error(`Invalid projected ${change.table}/${change.rowid}`, { cause: error });
    }
  }
}

export function validateLanceChanges(
  changes: LanceChange[],
  sqlRows: Map<string, Map<string, Row>>,
): void {
  for (const change of changes) {
    const row = change.after;
    if (change.table === "episodes") fromEpisodeRow({ ...row, embedding: new Float32Array([0]) });
    else if (change.table === "semantic_nodes")
      nodeFromRow({
        ...sqlRows.get(change.table)?.get(change.id),
        ...row,
        embedding: Float32Array.from((row.embedding as number[] | undefined) ?? [0]),
      });
    else {
      const stored =
        sqlRows.get(change.table)?.get(change.id) ??
        fail(`Missing authoritative row for Lance ${change.table}/${change.id}`);
      const decode = decoders[change.table] ?? fail(`No Lance validator for ${change.table}`);
      decode({ ...stored, ...row });
    }
  }
}
