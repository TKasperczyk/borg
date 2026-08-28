import {
  ACTION_STATE_METADATA,
  ACTION_STATES,
  lastReferencedActionLifecycleTurn,
  type ActionDescriptionSimilarityPair,
  type ActionRecord,
  type ActionState,
  type ActionStateTimestampField,
} from "../../memory/actions/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import {
  memoryDisclosureInternalUseNote,
  renderMemoryDisclosureLabelFieldsForModel,
  type MemoryDisclosureLabel,
} from "../../retrieval/index.js";
import { DisjointSet } from "../../util/disjoint-set.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { actionMemoryDisclosureLabel } from "../../memory/common/disclosure-serializers.js";
import type { ActiveParticipant } from "../participants.js";
import type { ActionLedgerRepository } from "./builder-types.js";
import { isActionVisibleForCurrentAudienceStanding } from "./audience-visibility.js";
import { actionScope, combineScopes, type ScopeResolver } from "./scope-resolver.js";
import type { EvidenceLedgerActionSalienceClass, EvidenceLedgerSessionScope } from "./types.js";

export const DEFAULT_ACTION_THREAD_RENDER_LIMIT = 12;
export const DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD = 0.85;
export const DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT = 256;
export const DEFAULT_ACTION_THREAD_SALIENCE_CLASS_RESERVED_SLOTS = 1;
export const DEFAULT_ACTION_THREAD_AUDIENCE_RESERVED_SLOTS = 1;
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
          isActionVisibleForCurrentAudienceStanding(action, audienceEntityId, activeParticipantIds),
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
  // Exact lexicographic (`<`) tie-break: union-by-min root, matching the prior
  // hand-rolled union-find (localeCompare could reorder for non-ActionId strings).
  const parents = new DisjointSet<string>((leftRoot, rightRoot) =>
    leftRoot < rightRoot ? -1 : leftRoot > rightRoot ? 1 : 0,
  );
  const actionsById = new Map(input.records.map((record) => [record.id, record]));

  for (const record of input.records) {
    parents.add(record.id);
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

    parents.union(records[0].id, records[1].id);
  }

  const groups = new Map<string, ActionRecord[]>();

  for (const record of input.records) {
    const root = parents.find(record.id);
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

// The state name is the only thing this section says about a thread's disposition, and two of the
// eight names point the wrong way. `not_done` reads as a pending item and is `terminal: true` -- the
// final outcome vocabulary, a thread the harness will not revisit. `expired` reads as a settled
// ending and is `terminal: false` -- a lifecycle retirement that recorded no outcome at all. A
// reader holding one page cannot recover either fact from the name, so a count of `not_done` reads
// as a to-do list when it is the opposite. Rendered in words rather than as a token: a bare flag
// would be one more enumeration member whose contrast set never appears on the same page. Derived
// from the metadata table for every state, so a new state inherits the phrasing instead of falling
// through to the misreading.
function renderActionStateDisposition(state: ActionState): string {
  const metadata = ACTION_STATE_METADATA[state];

  if (metadata.active) {
    return "open";
  }

  return metadata.terminal ? "closed by outcome" : "closed without an outcome";
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
    `transitions: ${thread.records.length}, current: ${thread.current.state} (${renderActionStateDisposition(
      thread.current.state,
    )}) at ${currentAt}`,
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
  currentTurnGlobal: number | undefined;
  windowTurns: number;
}): boolean {
  const lastReferencedTurnGlobal = lastReferencedActionLifecycleTurn(input.action);

  if (input.currentTurnGlobal === undefined || lastReferencedTurnGlobal === null) {
    return false;
  }

  return input.currentTurnGlobal - lastReferencedTurnGlobal <= input.windowTurns;
}

export function actionSalienceClass(input: {
  thread: ActionThread;
  currentUserStreamEntryId?: StreamEntryId;
  currentUserStreamEntryIds?: readonly StreamEntryId[];
  currentTurnGlobal?: number;
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
      currentTurnGlobal: input.currentTurnGlobal,
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
    currentTurnGlobal: input.currentTurnGlobal,
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

export type ActionThreadAudienceBucket = EntityId | "global";

export function actionThreadAudienceBucket(
  thread: Pick<ActionThread, "current">,
): ActionThreadAudienceBucket {
  return thread.current.audience_entity_id ?? "global";
}

export function allocateActionThreadRenderSlots(input: {
  threads: readonly ActionThreadWithSalience[];
  limit: number;
  salienceClassReservedSlots: number;
  audienceReservedSlots: number;
  audienceOrder?: readonly ActionThreadAudienceBucket[];
}): ActionThreadWithSalience[] {
  const ordered = orderActionThreadsBySalience(input.threads);
  const selectedIds = new Set<string>();
  const selected: ActionThreadWithSalience[] = [];

  function reserve(
    groups: readonly (readonly ActionThreadWithSalience[])[],
    reservedSlots: number,
  ): void {
    for (const group of groups) {
      if (selected.length >= input.limit) {
        return;
      }

      let remaining = reservedSlots;

      for (const thread of group) {
        if (remaining === 0 || selected.length >= input.limit) {
          break;
        }

        if (selectedIds.has(thread.id)) {
          continue;
        }

        selectedIds.add(thread.id);
        selected.push(thread);
        remaining -= 1;
      }
    }
  }

  const salienceGroups = ACTION_SALIENCE_ORDER.map((salienceClass) =>
    ordered.filter((thread) => thread.salienceClass === salienceClass),
  ).filter((group) => group.length > 0);
  reserve(salienceGroups, input.salienceClassReservedSlots);

  const audienceOrder = [
    ...new Set([
      ...(input.audienceOrder ?? []),
      ...ordered.map((thread) => actionThreadAudienceBucket(thread)),
    ]),
  ].filter((audience) => ordered.some((thread) => actionThreadAudienceBucket(thread) === audience));
  const audienceGroups = audienceOrder.map((audience) =>
    ordered.filter((thread) => actionThreadAudienceBucket(thread) === audience),
  );
  reserve(audienceGroups, input.audienceReservedSlots);

  for (const thread of ordered) {
    if (selected.length >= input.limit) {
      break;
    }

    if (!selectedIds.has(thread.id)) {
      selectedIds.add(thread.id);
      selected.push(thread);
    }
  }

  return orderActionThreadsBySalience(selected);
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

export type OlderActionThreadSummaryGroup = {
  audienceScope: ActionThreadAudienceBucket;
  salienceClass: EvidenceLedgerActionSalienceClass;
  threads: readonly ActionThread[];
  disclosureLabel: MemoryDisclosureLabel;
};

function actionThreadSummaryDetails(
  threads: readonly ActionThread[],
  disclosureLabel: MemoryDisclosureLabel,
): string {
  const recordCount = threads.reduce((count, thread) => count + thread.records.length, 0);
  const stateCounts = new Map<ActionState, number>(ACTION_STATES.map((state) => [state, 0]));

  for (const thread of threads) {
    stateCounts.set(thread.current.state, (stateCounts.get(thread.current.state) ?? 0) + 1);
  }

  const stateSummary = ACTION_STATES.map((state) => {
    const count = stateCounts.get(state) ?? 0;
    return count > 0 ? `${state}=${count}` : null;
  })
    .filter((entry): entry is string => entry !== null)
    .join(" ");
  const samples = threads
    .slice(0, OLDER_ACTION_THREAD_SAMPLE_LIMIT)
    .map(
      (thread) =>
        `${thread.current.state}: ${JSON.stringify(
          truncateOlderActionThreadSample(thread.current.description),
        )}`,
    )
    .join(" | ");

  // Fields only: the internal-use sentence that would follow them is a byte-identical constant on
  // every non-public label, so it is hoisted to the summary's own line once instead of copied onto
  // each group. What varies -- and what the reader actually decides disclosure from -- is the class
  // and the private-to binding, which stay here.
  return `threads=${threads.length} records=${recordCount} states=${stateSummary} disclosure_label=${renderMemoryDisclosureLabelFieldsForModel(disclosureLabel)} recent_samples=${samples}`;
}

export function renderOlderActionThreadsSummary(input: {
  groups: readonly OlderActionThreadSummaryGroup[];
  renderedThreadCount: number;
  threadsBuiltCount: number;
  consideredRecordCount: number;
  sourceRecordLimit: number;
  sourceRecordTotal: number | null;
  salienceDroppedThreadCount: number;
}): string {
  const { groups } = input;
  const olderThreads = groups.flatMap((group) => group.threads);
  const olderRecordCount = olderThreads.reduce((count, thread) => count + thread.records.length, 0);
  // The omitted-thread counts describe the render pool only. Two populations never enter it:
  // threads whose salience class resolved to null, and records below the source draw floor
  // (never counted, because the draw stops at the limit). Naming both keeps "omitted" from
  // reading as a complete accounting of everything this section did not show.
  //
  // Three of these counts are threads and one -- `records_considered` -- is records,
  // distinguished only by a field-name prefix, and the total that would let the thread counts
  // be checked against each other was never printed. Without it the three populations read as
  // summable against the record total, which they are not: every considered record lands in
  // exactly one thread (`buildActionThreads` groups the drawn records and nothing else), so
  // the record total exceeds the thread total by the merge surplus and the sum never closes.
  // `threads_built` comes from the builder's own thread count rather than from adding the
  // three, so the printed identity is falsifiable instead of tautological.
  //
  // `records_below_draw_floor` used to be an enumeration -- a count when the draw exhausted the
  // source, a refusal when it stopped at the limit -- and in production only the refusal has ever
  // rendered, because the store has never been smaller than the draw. A token whose contrast set
  // never appears is indistinguishable from a constant to a reader holding one page, and naming
  // the condition it refuses under only lengthens the constant: the named condition is the
  // comparison of the two numbers printed beside it, so the token restates its own operands and
  // can never disagree with them.
  //
  // The exit is to stop enumerating and measure. The source total is one COUNT away, so the field
  // prints how many records the draw never looked at. It is rendered as the stated difference of
  // the two totals rather than as a bare number, so it cannot be mistaken for an independent
  // count of the below-floor rows -- it is derived from them, and says so.
  const recordsBelowDrawFloor =
    input.sourceRecordTotal === null
      ? "unknown_count_source_total_unavailable"
      : `${Math.max(0, input.sourceRecordTotal - input.consideredRecordCount)}`;
  const sourceRecordTotalField =
    input.sourceRecordTotal === null
      ? "source_record_total=unavailable"
      : `source_record_total=${input.sourceRecordTotal}; records_below_draw_floor is that total minus records_considered, not a separate count`;
  // The internal-use sentence for the group lines below, hoisted out of them. It stays a separate
  // line rather than joining the counts: this section truncates from the tail, and a note attached
  // to the counts line would survive at the cost of the group lines it describes.
  const disclosureNote = groups.some((group) => group.disclosureLabel.disclosureClass !== "public")
    ? [`Groups below whose disclosure_class is not public: ${memoryDisclosureInternalUseNote()}.`]
    : [];

  return [
    `Older action threads omitted from this section: threads=${olderThreads.length}, records=${olderRecordCount}.`,
    `Not counted above: salience_dropped_threads=${input.salienceDroppedThreadCount}, records_below_draw_floor=${recordsBelowDrawFloor} (threads_built=${input.threadsBuiltCount} = rendered ${input.renderedThreadCount} + omitted ${olderThreads.length} + dropped ${input.salienceDroppedThreadCount}; records_considered=${input.consideredRecordCount} records, source_record_limit=${input.sourceRecordLimit}, ${sourceRecordTotalField}).`,
    ...disclosureNote,
    ...groups.map(
      (group) =>
        `- audience_scope=${group.audienceScope} salience_class=${group.salienceClass} ${actionThreadSummaryDetails(group.threads, group.disclosureLabel)}`,
    ),
  ].join("\n");
}
