import { type EpisodeId } from "../../../util/ids.js";
import { parseStoredProvenance } from "../../common/provenance.js";

import {
  goalSchema,
  traitSchema,
  valueSchema,
  type GoalRecord,
  type TraitRecord,
  type ValueRecord,
} from "../types.js";

function parseStoredIdArray(value: unknown): EpisodeId[] {
  if (typeof value !== "string") {
    return [];
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed)
      ? parsed.filter((item): item is EpisodeId => typeof item === "string" && item.length > 0)
      : [];
  } catch {
    return [];
  }
}

export function mapGoalRow(row: Record<string, unknown>): GoalRecord {
  return goalSchema.parse({
    id: row.id,
    description: row.description,
    priority: Number(row.priority),
    parent_goal_id:
      row.parent_goal_id === null || row.parent_goal_id === undefined
        ? null
        : String(row.parent_goal_id),
    status: row.status,
    progress_notes:
      row.progress_notes === null || row.progress_notes === undefined
        ? null
        : String(row.progress_notes),
    last_progress_ts:
      row.last_progress_ts === null || row.last_progress_ts === undefined
        ? null
        : Number(row.last_progress_ts),
    created_at: Number(row.created_at),
    target_at: row.target_at === null || row.target_at === undefined ? null : Number(row.target_at),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
  });
}

export function mapValueRow(row: Record<string, unknown>): ValueRecord {
  return valueSchema.parse({
    id: row.id,
    label: row.label,
    description: row.description,
    priority: Number(row.priority),
    created_at: Number(row.created_at),
    last_affirmed:
      row.last_affirmed === null || row.last_affirmed === undefined
        ? null
        : Number(row.last_affirmed),
    state: row.state,
    established_at:
      row.established_at === null || row.established_at === undefined
        ? null
        : Number(row.established_at),
    confidence: Number(row.confidence),
    last_tested_at:
      row.last_tested_at === null || row.last_tested_at === undefined
        ? null
        : Number(row.last_tested_at),
    last_contradicted_at:
      row.last_contradicted_at === null || row.last_contradicted_at === undefined
        ? null
        : Number(row.last_contradicted_at),
    support_count: Number(row.support_count),
    contradiction_count: Number(row.contradiction_count),
    evidence_episode_ids: parseStoredIdArray(row.evidence_episode_ids),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
  });
}

export function mapTraitRow(row: Record<string, unknown>): TraitRecord {
  return traitSchema.parse({
    id: row.id,
    label: row.label,
    strength: Number(row.strength),
    last_reinforced: Number(row.last_reinforced),
    last_decayed:
      row.last_decayed === null || row.last_decayed === undefined ? null : Number(row.last_decayed),
    state: row.state,
    established_at:
      row.established_at === null || row.established_at === undefined
        ? null
        : Number(row.established_at),
    confidence: Number(row.confidence),
    last_tested_at:
      row.last_tested_at === null || row.last_tested_at === undefined
        ? null
        : Number(row.last_tested_at),
    last_contradicted_at:
      row.last_contradicted_at === null || row.last_contradicted_at === undefined
        ? null
        : Number(row.last_contradicted_at),
    support_count: Number(row.support_count),
    contradiction_count: Number(row.contradiction_count),
    evidence_episode_ids: parseStoredIdArray(row.evidence_episode_ids),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
  });
}
