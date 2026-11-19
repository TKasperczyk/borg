import type { IdentityEventRepository, AnnotationTarget } from "../memory/identity/index.js";
import { createConsolidationFamilyId, createStreamEntryId } from "../util/ids.js";
import { createEpisodeFixture, type OfflineTestHarness } from "./test-support.js";

export function recordTestAnnotation(
  repository: IdentityEventRepository,
  recordId: string,
  text = "Synthetic audited correction: the observation was provisional.",
  options: { ts?: number; recordType?: AnnotationTarget["record_type"] } = {},
) {
  return repository.record({
    record_type: options.recordType ?? "episode",
    record_id: recordId,
    action: "synthetic_operator_audit",
    provenance: { kind: "manual" },
    ts: options.ts ?? 1_000,
    new_value: {
      stream_entry_id: createStreamEntryId(),
      annotation: {
        event: "episode_annotation",
        target_ids: [recordId],
        correction: text,
        authority: "Synthetic operator",
      },
    },
  });
}

export async function createRetiredEvidenceFixture(harness: OfflineTestHarness) {
  const lineage = await harness.episodicRepository.createEpisode(
    createEpisodeFixture({
      title: "Synthetic visible raw",
      narrative: "An independent raw observation remains available.",
    }),
  );
  const source = await harness.episodicRepository.createEpisode({
    ...createEpisodeFixture({
      title: "Synthetic retired consolidation",
      narrative: "Original retired narrative remains byte for byte.",
      lineage: { derived_from: [lineage.id], supersedes: [] },
    }),
    episode_kind: "consolidation_version",
    consolidation_family_id: createConsolidationFamilyId(),
    consolidation_coverage_hash: "synthetic-coverage",
  });
  // Deliberately exercise inconsistent flags: either durable archived bit is sufficient.
  harness.db.prepare("UPDATE episode_index SET archived = 1 WHERE episode_id = ?").run(source.id);
  recordTestAnnotation(harness.identityEventRepository, source.id);
  recordTestAnnotation(
    harness.identityEventRepository,
    lineage.id,
    "原始记录的独立更正。 تصحيح مستقل.",
  );
  return { source, lineage };
}

/** Compare complete independently rendered prompts; never derive a baseline by deleting fields. */
export function evidencePromptSizes(before: string, after: string) {
  return { before: before.length, after: after.length, delta: after.length - before.length };
}

export function insertMalformedTestAnnotations(
  harness: Pick<OfflineTestHarness, "db">,
  recordId: string,
) {
  for (const payload of [
    "{",
    JSON.stringify({
      stream_entry_id: "synthetic-partial",
      annotation: { event: "episode_annotation", target_ids: [recordId] },
    }),
  ]) {
    harness.db
      .prepare(
        `INSERT INTO identity_events (record_type, record_id, action, new_value_json, provenance_kind, ts) VALUES ('episode', ?, 'synthetic_audit', ?, 'manual', 1)`,
      )
      .run(recordId, payload);
  }
}

/** Valid correction surrounded by malformed rows, including an invalid attempted retraction. */
export function recordTestAnnotationWithMalformedHistory(
  harness: Pick<OfflineTestHarness, "db" | "identityEventRepository">,
  recordId: string,
) {
  const text = "Valid correction survives. 更正。 تصحيح. 🧭";
  const valid = recordTestAnnotation(harness.identityEventRepository, recordId, text);
  insertMalformedTestAnnotations(harness, recordId);
  harness.db
    .prepare(
      `INSERT INTO identity_events (record_type, record_id, action, new_value_json, provenance_kind, ts) VALUES ('episode', ?, 'synthetic_audit', ?, 'manual', 2)`,
    )
    .run(
      recordId,
      JSON.stringify({
        stream_entry_id: null,
        annotation: {
          event: "episode_annotation_retracted",
          target_ids: [recordId],
          reverses_stream_entry_id: (valid.new_value as { stream_entry_id: string })
            .stream_entry_id,
        },
      }),
    );
  harness.db.prepare("UPDATE identity_events SET ts = ? WHERE id = ?").run(1e100, valid.id);
  return text;
}
