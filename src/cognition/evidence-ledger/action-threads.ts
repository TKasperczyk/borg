import {
  ACTION_STATE_METADATA,
  ACTION_STATES,
  type ActionDescriptionSimilarityPair,
  type ActionRecord,
  type ActionState,
  type ActionStateTimestampField,
} from "../../memory/actions/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type { MemoryDisclosureLabel } from "../../retrieval/index.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { actionMemoryDisclosureLabel } from "../disclosure-labels.js";
import type { ActiveParticipant } from "../participants.js";
import type { ActionLedgerRepository } from "./builder-types.js";
import { isActionVisibleToSession } from "./audience-visibility.js";
import { actionScope, combineScopes, type ScopeResolver } from "./scope-resolver.js";
import type { EvidenceLedgerActionSalienceClass, EvidenceLedgerSessionScope } from "./types.js";

export const DEFAULT_ACTION_THREAD_RENDER_LIMIT = 12;
export const DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD = 0.85;
export const DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT = 256;
export const PARTICIPANT_RECENT_ACTION_TURN_WINDOW = 3;
export const PARTICIPANT_DORMANT_ACTION_TURN_WINDOW = 15;
export const STALE_PARTICIPANT_ACTION_RENDER_LIMIT = 5;

const OLDER_ACTION_THREAD_SAMPLE_LIMIT = 4;
const OLDER_ACTION_THREAD_SAMPLE_MAX_CHARS = 80;

export type ActionThread = {
  id: string;
  records: ActionRecord[];
  origin: ActionRecord;
  current: ActionRecord;
  scope: EvidenceLedgerSessionScope;
};

export type ActionThreadWithSalience = ActionThread & {
  salienceClass: EvidenceLedgerActionSalienceClass;
};

export type ActionCandidateForCognition = {
  record: ActionRecord;
  disclosureLabel: MemoryDisclosureLabel;
};

export const PROMPT_SALIENT_ACTION_SALIENCE_CLASSES = [
  "borg_current_turn_action",
  "borg_memory_tracking_action",
  "participant_pending_recent",
  "group_pending",
] as const satisfies readonly EvidenceLedgerActionSalienceClass[];

const PROMPT_SALIENT_ACTION_SALIENCE_CLASS_SET = new Set<EvidenceLedgerActionSalienceClass>(
  PROMPT_SALIENT_ACTION_SALIENCE_CLASSES,
);

export type ActionPromptSalienceSummary = {
  promptSalientActionsTotal: number;
  borgOwnedSalientActiveActions: number;
  participantOwnedSalientActiveActions: number;
  staleActionsOmittedFromPrompt: number;
};

export function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  return value === undefined ? fallback : Math.max(1, Math.floor(value));
}

export function normalizeUnitInterval(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(0, Math.min(1, value));
}

function uniqueEntityIds(entityIds: readonly (EntityId | null | undefined)[]): EntityId[] {
  return [...new Set(entityIds.filter((entityId): entityId is EntityId => entityId != null))];
}

function actionCognitionRank(input: {
  action: ActionRecord;
  audienceEntityId: EntityId | null;
  participantEntityIds: ReadonlySet<EntityId>;
}): number {
  if (
    input.audienceEntityId !== null &&
    input.action.audience_entity_id === input.audienceEntityId
  ) {
    return 0;
  }

  if (
    input.action.actor !== "borg" &&
    input.action.actor !== "user" &&
    input.participantEntityIds.has(input.action.actor)
  ) {
    return 1;
  }

  if (
    input.action.audience_entity_id !== null &&
    input.participantEntityIds.has(input.action.audience_entity_id)
  ) {
    return 2;
  }

  if (input.action.audience_entity_id === null) {
    return 3;
  }

  return 4;
}

export function listActionCandidatesForCognition(input: {
  actionRepository: ActionLedgerRepository;
  audienceEntityId: EntityId | null;
  activeParticipants?: readonly ActiveParticipant[];
  rankParticipantEntityIds?: readonly EntityId[];
  states?: readonly ActionState[];
  state?: ActionState;
  actor?: ActionRecord["actor"];
  limit: number;
}): ActionCandidateForCognition[] {
  const participantEntityIds = uniqueEntityIds([
    ...(input.activeParticipants ?? []).map((participant) => participant.entityId),
    ...(input.rankParticipantEntityIds ?? []),
  ]);
  const participantEntityIdSet = new Set(participantEntityIds);
  const rankAudienceEntityIds = uniqueEntityIds([input.audienceEntityId, ...participantEntityIds]);
  const records = input.actionRepository.list({
    ...(input.state === undefined ? {} : { state: input.state }),
    ...(input.states === undefined ? {} : { states: input.states }),
    ...(input.actor === undefined ? {} : { actor: input.actor }),
    recallAllAudiences: true,
    rankAudienceEntityIds,
    rankActorEntityIds: participantEntityIds,
    limit: input.limit,
  });

  return records
    .sort(
      (left, right) =>
        actionCognitionRank({
          action: left,
          audienceEntityId: input.audienceEntityId,
          participantEntityIds: participantEntityIdSet,
        }) -
          actionCognitionRank({
            action: right,
            audienceEntityId: input.audienceEntityId,
            participantEntityIds: participantEntityIdSet,
          }) ||
        right.updated_at - left.updated_at ||
        left.id.localeCompare(right.id),
    )
    .slice(0, input.limit)
    .map((record) => ({
      record,
      disclosureLabel: actionMemoryDisclosureLabel(record),
    }));
}

export function listActionsForDisclosure(
  actionRepository: ActionLedgerRepository,
  audienceEntityId: EntityId | null,
  activeParticipants: readonly ActiveParticipant[] | undefined,
  limit: number,
): ActionRecord[] {
  const records: ActionRecord[] = [...actionRepository.list({ audienceEntityId: null, limit })];
  const activeParticipantIds = new Set(
    (activeParticipants ?? []).map((participant) => participant.entityId),
  );

  if (audienceEntityId !== null) {
    records.push(...actionRepository.list({ audienceEntityId, limit }));
  }

  for (const participant of activeParticipants ?? []) {
    records.push(
      ...actionRepository
        .list({ actor: participant.entityId })
        .filter((action) =>
          isActionVisibleToSession(action, audienceEntityId, activeParticipantIds),
        ),
    );
    records.push(...actionRepository.list({ audienceEntityId: participant.entityId, limit }));
  }

  return [...new Map(records.map((record) => [record.id, record])).values()]
    .sort((left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id))
    .slice(0, limit);
}

export function actionActorDisplay(
  actor: ActionRecord["actor"],
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): string {
  if (actor === "borg") {
    return "assistant";
  }

  if (actor === "user") {
    return "user";
  }

  return entityRepository?.get(actor)?.canonical_name ?? "participant";
}

function findParent(parents: Map<string, string>, id: string): string {
  const parent = parents.get(id);

  if (parent === undefined || parent === id) {
    parents.set(id, id);
    return id;
  }

  const root = findParent(parents, parent);
  parents.set(id, root);
  return root;
}

function unionParents(parents: Map<string, string>, leftId: string, rightId: string): void {
  const leftRoot = findParent(parents, leftId);
  const rightRoot = findParent(parents, rightId);

  if (leftRoot === rightRoot) {
    return;
  }

  const root = leftRoot < rightRoot ? leftRoot : rightRoot;
  const child = root === leftRoot ? rightRoot : leftRoot;
  parents.set(child, root);
}

function actionTimestampForState(action: ActionRecord): number {
  const timestampField: ActionStateTimestampField =
    ACTION_STATE_METADATA[action.state].timestamp_field;

  return action[timestampField] ?? action.updated_at;
}

function combineActionScopes(
  records: readonly ActionRecord[],
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes(records.map((record) => actionScope(record, resolver)));
}

function selectThreadOrigin(records: readonly ActionRecord[]): ActionRecord {
  return [...records].sort(
    (left, right) => left.created_at - right.created_at || left.id.localeCompare(right.id),
  )[0] as ActionRecord;
}

function selectThreadCurrent(records: readonly ActionRecord[]): ActionRecord {
  return [...records].sort(
    (left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id),
  )[0] as ActionRecord;
}

function canThreadActions(left: ActionRecord, right: ActionRecord): boolean {
  return (
    left.goal_id !== null &&
    right.goal_id !== null &&
    left.goal_id === right.goal_id &&
    left.actor === right.actor
  );
}

function sameThreadablePair(
  pair: ActionDescriptionSimilarityPair,
  actionsById: ReadonlyMap<string, ActionRecord>,
  threshold: number,
): [ActionRecord, ActionRecord] | null {
  if (pair.similarity < threshold) {
    return null;
  }

  const left = actionsById.get(pair.leftId);
  const right = actionsById.get(pair.rightId);

  if (left === undefined || right === undefined || !canThreadActions(left, right)) {
    return null;
  }

  return [left, right];
}

export async function buildActionThreads(input: {
  records: readonly ActionRecord[];
  repository: ActionLedgerRepository;
  resolver: ScopeResolver;
  similarityThreshold: number;
}): Promise<ActionThread[]> {
  const parents = new Map<string, string>();
  const actionsById = new Map(input.records.map((record) => [record.id, record]));

  for (const record of input.records) {
    parents.set(record.id, record.id);
  }

  const pairs =
    input.repository.findSimilarDescriptionPairs === undefined
      ? []
      : await input.repository.findSimilarDescriptionPairs(
          input.records.filter((record) => record.goal_id !== null),
          input.similarityThreshold,
        );

  for (const pair of pairs) {
    const records = sameThreadablePair(pair, actionsById, input.similarityThreshold);

    if (records === null) {
      continue;
    }

    unionParents(parents, records[0].id, records[1].id);
  }

  const groups = new Map<string, ActionRecord[]>();

  for (const record of input.records) {
    const root = findParent(parents, record.id);
    groups.set(root, [...(groups.get(root) ?? []), record]);
  }

  return [...groups.entries()]
    .map(([id, records]) => {
      const origin = selectThreadOrigin(records);
      const current = selectThreadCurrent(records);

      return {
        id,
        records: [...records].sort(
          (left, right) => left.updated_at - right.updated_at || left.id.localeCompare(right.id),
        ),
        origin,
        current,
        scope: combineActionScopes(records, input.resolver),
      };
    })
    .sort(
      (left, right) =>
        right.current.updated_at - left.current.updated_at ||
        left.current.id.localeCompare(right.current.id),
    );
}

export function renderActionThreadText(
  thread: ActionThread,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): string {
  const currentAt = new Date(actionTimestampForState(thread.current)).toISOString();
  const actor = actionActorDisplay(thread.current.actor, entityRepository);
  const lines = [
    `actor: ${actor}`,
    `originating_intent: ${thread.origin.description}`,
    `transitions: ${thread.records.length}, current: ${thread.current.state} at ${currentAt}`,
  ];

  if (thread.current.id !== thread.origin.id) {
    lines.push(`current_intent: ${thread.current.description}`);
  }

  return lines.join("\n");
}

export function actionThreadStateMetadata(
  thread: ActionThread,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): Record<string, unknown> {
  return {
    record_ids: thread.records.map((record) => record.id),
    transitions: thread.records.length,
    current_action_id: thread.current.id,
    current_updated_at: thread.current.updated_at,
    current_actor: actionActorDisplay(thread.current.actor, entityRepository),
    goal_id: thread.current.goal_id,
    open_question_id: thread.current.open_question_id,
  };
}

export function actionThreadState(thread: ActionThread): ActionState {
  return thread.current.state;
}

export function isActiveActionState(state: ActionState): boolean {
  return ACTION_STATE_METADATA[state].active;
}

export function isTerminalRenderedActionState(state: ActionState): boolean {
  return state === "completed" || state === "not_done" || state === "expired";
}

function isCurrentTurnAction(
  action: ActionRecord,
  currentUserStreamEntryId: StreamEntryId | undefined,
  currentUserStreamEntryIds: readonly StreamEntryId[] = [],
): boolean {
  return (
    (currentUserStreamEntryId !== undefined &&
      action.provenance_stream_entry_ids.includes(currentUserStreamEntryId)) ||
    currentUserStreamEntryIds.some((entryId) =>
      action.provenance_stream_entry_ids.includes(entryId),
    )
  );
}

function isGroupOwnedAction(action: Pick<ActionRecord, "actor" | "audience_entity_id">): boolean {
  return (
    action.actor !== "user" && action.actor !== "borg" && action.actor === action.audience_entity_id
  );
}

function referencedWithinTurns(input: {
  action: ActionRecord;
  currentTurnCounter: number | undefined;
  windowTurns: number;
}): boolean {
  if (
    input.currentTurnCounter === undefined ||
    input.action.last_referenced_turn_counter === null
  ) {
    return false;
  }

  return input.currentTurnCounter - input.action.last_referenced_turn_counter <= input.windowTurns;
}

export function actionSalienceClass(input: {
  thread: ActionThread;
  currentUserStreamEntryId?: StreamEntryId;
  currentUserStreamEntryIds?: readonly StreamEntryId[];
  currentTurnCounter?: number;
}): EvidenceLedgerActionSalienceClass | null {
  const action = input.thread.current;

  if (action.state === "archived") {
    return null;
  }

  if (isTerminalRenderedActionState(action.state)) {
    if (
      isCurrentTurnAction(action, input.currentUserStreamEntryId, input.currentUserStreamEntryIds)
    ) {
      return "completed_recent";
    }

    return referencedWithinTurns({
      action,
      currentTurnCounter: input.currentTurnCounter,
      windowTurns: PARTICIPANT_RECENT_ACTION_TURN_WINDOW,
    })
      ? "completed_recent"
      : null;
  }

  if (action.actor === "borg") {
    return isCurrentTurnAction(
      action,
      input.currentUserStreamEntryId,
      input.currentUserStreamEntryIds,
    )
      ? "borg_current_turn_action"
      : "borg_memory_tracking_action";
  }

  if (isGroupOwnedAction(action)) {
    return "group_pending";
  }

  return referencedWithinTurns({
    action,
    currentTurnCounter: input.currentTurnCounter,
    windowTurns: PARTICIPANT_RECENT_ACTION_TURN_WINDOW,
  })
    ? "participant_pending_recent"
    : "participant_pending_stale";
}

const ACTION_SALIENCE_ORDER: readonly EvidenceLedgerActionSalienceClass[] = [
  "borg_current_turn_action",
  "borg_memory_tracking_action",
  "participant_pending_recent",
  "group_pending",
  "participant_pending_stale",
  "completed_recent",
];

function salienceRank(salienceClass: EvidenceLedgerActionSalienceClass): number {
  return ACTION_SALIENCE_ORDER.indexOf(salienceClass);
}

export function orderActionThreadsBySalience(
  threads: readonly ActionThreadWithSalience[],
): ActionThreadWithSalience[] {
  return [...threads].sort(
    (left, right) =>
      salienceRank(left.salienceClass) - salienceRank(right.salienceClass) ||
      right.current.updated_at - left.current.updated_at ||
      left.current.id.localeCompare(right.current.id),
  );
}

export function isPromptSalientActionSalienceClass(
  salienceClass: EvidenceLedgerActionSalienceClass,
): boolean {
  return PROMPT_SALIENT_ACTION_SALIENCE_CLASS_SET.has(salienceClass);
}

export function summarizeActionPromptSalience(
  threads: readonly ActionThreadWithSalience[],
): ActionPromptSalienceSummary {
  const staleParticipantThreadCount = threads.filter(
    (thread) => thread.salienceClass === "participant_pending_stale",
  ).length;
  let promptSalientActionsTotal = 0;
  let borgOwnedSalientActiveActions = 0;
  let participantOwnedSalientActiveActions = 0;

  for (const thread of threads) {
    if (!isPromptSalientActionSalienceClass(thread.salienceClass)) {
      continue;
    }

    promptSalientActionsTotal += 1;

    if (!isActiveActionState(thread.current.state)) {
      continue;
    }

    if (thread.current.actor === "borg") {
      borgOwnedSalientActiveActions += 1;
      continue;
    }

    if (!isGroupOwnedAction(thread.current)) {
      participantOwnedSalientActiveActions += 1;
    }
  }

  return {
    promptSalientActionsTotal,
    borgOwnedSalientActiveActions,
    participantOwnedSalientActiveActions,
    staleActionsOmittedFromPrompt: Math.max(
      0,
      staleParticipantThreadCount - STALE_PARTICIPANT_ACTION_RENDER_LIMIT,
    ),
  };
}

function truncateOlderActionThreadSample(text: string): string {
  if (text.length <= OLDER_ACTION_THREAD_SAMPLE_MAX_CHARS) {
    return text;
  }

  return `${text.slice(0, OLDER_ACTION_THREAD_SAMPLE_MAX_CHARS - 3)}...`;
}

export function renderOlderActionThreadsSummary(olderThreads: readonly ActionThread[]): string {
  const olderRecordCount = olderThreads.reduce((count, thread) => count + thread.records.length, 0);
  const stateCounts = new Map<ActionState, number>(ACTION_STATES.map((state) => [state, 0]));

  for (const thread of olderThreads) {
    stateCounts.set(thread.current.state, (stateCounts.get(thread.current.state) ?? 0) + 1);
  }

  const stateSummary = ACTION_STATES.map((state) => {
    const count = stateCounts.get(state) ?? 0;
    return count > 0 ? `${state}=${count}` : null;
  })
    .filter((entry): entry is string => entry !== null)
    .join(" ");
  const samples = olderThreads
    .slice(0, OLDER_ACTION_THREAD_SAMPLE_LIMIT)
    .map(
      (thread) =>
        `${thread.current.state}: ${JSON.stringify(
          truncateOlderActionThreadSample(thread.current.description),
        )}`,
    )
    .join(" | ");

  return `Older action threads omitted from this section: threads=${olderThreads.length}, records=${olderRecordCount}, states=${stateSummary}, recent_samples=${samples}.`;
}
