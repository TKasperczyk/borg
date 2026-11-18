import {
  ACTION_ACTOR_RESERVED_RECORDS,
  ACTIVE_ACTION_STATES,
  type ActionRecord,
  type ActionRepository,
} from "../../memory/actions/index.js";
import {
  actionMemoryDisclosureLabel,
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { EntityId } from "../../util/ids.js";

export const ACTION_CANDIDATE_LIMIT = 80;
export const ACTION_SIMILARITY_CANDIDATE_LIMIT = 40;
export const ACTION_CANDIDATE_MAX_CHARS = 48_000;
export const ACTION_GOAL_CANDIDATE_MAX_CHARS = 16_000;

export function actionCandidateForModel(action: ActionRecord): Record<string, unknown> {
  return {
    id: action.id,
    description: action.description,
    actor: action.actor,
    state: action.state,
    audience_entity_id: action.audience_entity_id,
    created_at: action.created_at,
    updated_at: action.updated_at,
    last_referenced_at_ms: action.last_referenced_at_ms ?? null,
    session_scope: action.session_scope,
    session_anchor_id: action.session_anchor_id,
    last_referenced_turn_counter: action.last_referenced_turn_counter,
    last_referenced_turn_global: action.last_referenced_turn_global ?? null,
    ...memoryDisclosurePayloadFields(actionMemoryDisclosureLabel(action)),
  };
}

// Budget whole rows, never rewrite the model's wording to fit. The reserve covers
// array punctuation and population fields. Omitted rows cannot be selected later.
function boundedRows<T extends { id: string }>(
  records: readonly T[],
  serialize: (record: T) => Record<string, unknown>,
  maxChars: number,
) {
  const shown: T[] = [];
  const rows: Record<string, unknown>[] = [];
  let chars = 256;
  for (const record of new Map(records.map((row) => [row.id, row])).values()) {
    const row = serialize(record);
    const size = JSON.stringify(row).length + 1;
    if (shown.length >= ACTION_CANDIDATE_LIMIT || chars + size > maxChars) continue;
    shown.push(record);
    rows.push(row);
    chars += size;
  }
  return { shown, rows };
}

export function presentActionCandidates(records: readonly ActionRecord[], total: number) {
  const presentation = boundedRows(records, actionCandidateForModel, ACTION_CANDIDATE_MAX_CHARS);
  return { ...presentation, omitted: Math.max(0, total - presentation.shown.length) };
}

export function presentActionGoals(goals: readonly GoalRecord[]) {
  const presentation = boundedRows(
    goals,
    (goal) => ({
      id: goal.id,
      description: goal.description,
      status: goal.status,
      ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal)),
    }),
    ACTION_GOAL_CANDIDATE_MAX_CHARS,
  );
  return { ...presentation, omitted: Math.max(0, goals.length - presentation.shown.length) };
}

export async function retrieveActionCandidates(input: {
  repository: ActionRepository;
  text: string;
  audienceEntityId: EntityId | null;
  speakerEntityId?: EntityId | null;
  participantEntityIds?: readonly EntityId[];
  onSearchFailure: (error: unknown) => void;
}): Promise<{ records: ActionRecord[]; total: number }> {
  const recent = input.repository.list({
    recallAllAudiences: true,
    prioritizeActive: true,
    rankByActivity: true,
    reserveActors: [
      "borg",
      ...(input.speakerEntityId == null ? [] : [input.speakerEntityId]),
      ...(input.participantEntityIds ?? []),
    ],
    rankAudienceEntityIds: input.audienceEntityId === null ? [] : [input.audienceEntityId],
    rankActorEntityIds: input.speakerEntityId == null ? [] : [input.speakerEntityId],
    limit: ACTION_CANDIDATE_LIMIT,
  });
  let similar: ActionRecord[] = [];
  try {
    similar = await input.repository.findByDescription(
      input.text,
      ACTION_SIMILARITY_CANDIDATE_LIMIT,
    );
  } catch (error) {
    input.onSearchFailure(error);
  }
  // Similarity is a recall rank only. Interleave with recency so neither source
  // can fill the whole presentation before the other is considered.
  const records = new Map<string, ActionRecord>();
  // Keep the reserved prefix ahead of both count and character truncation.
  const actors = new Set(["borg", input.speakerEntityId, ...(input.participantEntityIds ?? [])]);
  const actorCounts = new Map<string, number>();
  for (const record of recent) {
    if (
      ACTIVE_ACTION_STATES.some((state) => state === record.state) &&
      actors.has(record.actor) &&
      (actorCounts.get(record.actor) ?? 0) < ACTION_ACTOR_RESERVED_RECORDS
    ) {
      records.set(record.id, record);
      actorCounts.set(record.actor, (actorCounts.get(record.actor) ?? 0) + 1);
    }
  }
  for (let index = 0; index < Math.max(recent.length, similar.length); index += 1) {
    for (const record of [similar[index], recent[index]]) {
      if (record !== undefined) records.set(record.id, record);
    }
  }
  return { records: [...records.values()], total: input.repository.count() };
}
