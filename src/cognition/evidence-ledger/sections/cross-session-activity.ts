import type { BuilderSectionContext } from "../builder-context.js";
import { CROSS_SESSION_ACTIVITY_TRUST_RANK, addEntry } from "../section-buckets.js";

export function addCrossSessionSelfActivitySection(context: BuilderSectionContext): void {
  const rows = context.input.crossSessionSelfActivity ?? [];

  for (const [index, row] of rows.entries()) {
    addEntry(context.buckets, "cross_session_self_activity", {
      id: `cross_session_self_activity:${index + 1}`,
      source_type: "system_metadata",
      session_scope: "prior_session",
      actor: "system",
      trust_rank: CROSS_SESSION_ACTIVITY_TRUST_RANK,
      text: row.text,
      value: row.kind,
      state: "active",
      state_metadata: {
        event_kind: row.kind,
        occurred_at: row.occurredAt,
        relative_age: row.relativeAge,
      },
      taint: "none",
    });
  }
}
