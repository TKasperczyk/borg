import {
  ACTION_STATE_METADATA,
  ACTION_STATES,
  lastReferencedActionLifecycleTurn,
  type ActionUpdateHistory,
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
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { actionMemoryDisclosureLabel } from "../../memory/common/disclosure-serializers.js";
import type { ActiveParticipant } from "../participants.js";
import type { ActionLedgerRepository } from "./builder-types.js";
import { actionScope, combineScopes, type ScopeResolver } from "./scope-resolver.js";
import type { EvidenceLedgerActionSalienceClass, EvidenceLedgerSessionScope } from "./types.js";

export const DEFAULT_ACTION_THREAD_RENDER_LIMIT = 12;
export const DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT = 256;
export const DEFAULT_ACTION_THREAD_SALIENCE_CLASS_RESERVED_SLOTS = 1;
export const DEFAULT_ACTION_THREAD_AUDIENCE_RESERVED_SLOTS = 1;
export const PARTICIPANT_RECENT_ACTION_TURN_WINDOW = 3;
export const ACTION_UPDATE_HISTORY_LIMIT = 5;
export const ACTION_UPDATE_HISTORY_MAX_CHARS = 4_000;
export const ACTION_ACTOR_RESERVED_RENDER_SLOTS = 1;

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
    prioritizeActive: true,
    reserveActors: ["borg", ...participantEntityIds],
    rankAudienceEntityIds,
    rankActorEntityIds: participantEntityIds,
    limit: input.limit,
  });

  return records
    .sort(
      (left, right) =>
        Number(!isActiveActionState(left.state)) - Number(!isActiveActionState(right.state)) ||
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

export async function buildActionThreads(input: {
  records: readonly ActionRecord[];
  repository: ActionLedgerRepository;
  resolver: ScopeResolver;
  /** Retained for callers; similarity never establishes action identity. */
  similarityThreshold?: number;
}): Promise<ActionThread[]> {
  return [...new Map(input.records.map((record) => [record.id, record])).values()]
    .map((record) => ({
      id: record.id,
      records: [record],
      origin: record,
      current: record,
      scope: combineActionScopes([record], input.resolver),
    }))
    .sort(
      (left, right) =>
        Number(!isActiveActionState(left.current.state)) -
          Number(!isActiveActionState(right.current.state)) ||
        right.current.updated_at - left.current.updated_at ||
        left.id.localeCompare(right.id),
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
  if (state === "declined") return "closed by choice to decline";
  const metadata = ACTION_STATE_METADATA[state];

  if (metadata.active) {
    return "open";
  }

  return metadata.terminal ? "closed by outcome" : "closed without an outcome";
}

export function renderActionThreadText(
  thread: ActionThread,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
  history?: ActionUpdateHistory,
): string {
  const action = thread.current;
  const updates: string[] = [];
  let chars = 0;
  for (const update of history?.updates ?? []) {
    const text = `${new Date(update.timestamp).toISOString()} ${update.from_state} -> ${update.to_state}; turn=${update.turn_id ?? "unknown"}; sources=${update.source_stream_entry_ids.join(",")}: ${update.text}`;
    if (
      updates.length >= ACTION_UPDATE_HISTORY_LIMIT ||
      chars + text.length > ACTION_UPDATE_HISTORY_MAX_CHARS
    )
      break;
    updates.push(text);
    chars += text.length;
  }
  const goalLabel =
    action.goal_id === null
      ? "goal: none"
      : action.goal_link_basis === "extractor"
        ? `goal: ${action.goal_id}; goal_link_basis: extractor selected`
        : `goal: ${action.goal_id}; goal_link_basis: legacy turn-selected goal, not an action-specific link`;
  return [
    `actor: ${actionActorDisplay(action.actor, entityRepository)}`,
    goalLabel,
    `action: ${action.description}`,
    `records: 1, current: ${action.state} (${renderActionStateDisposition(action.state)}) since ${new Date(actionTimestampForState(action)).toISOString()}`,
    ...(history === undefined
      ? []
      : [
          `updates: total=${history.total}, shown=${updates.length}, omitted=${Math.max(0, history.total - updates.length)}; newest first`,
          ...updates,
        ]),
  ].join("\n");
}

export function actionThreadStateMetadata(
  thread: ActionThread,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): Record<string, unknown> {
  return {
    record_ids: thread.records.map((record) => record.id),
    records: thread.records.length,
    origin_action_id: thread.origin.id,
    origin_state: thread.origin.state,
    current_action_id: thread.current.id,
    current_updated_at: thread.current.updated_at,
    current_actor: actionActorDisplay(thread.current.actor, entityRepository),
    goal_id: thread.current.goal_id,
    goal_link_basis: thread.current.goal_link_basis ?? "legacy_turn_selected",
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
  return ACTION_STATE_METADATA[state].terminal || state === "expired";
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
      Number(!isActiveActionState(left.current.state)) -
        Number(!isActiveActionState(right.current.state)) ||
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
  reservedActors?: readonly ActionRecord["actor"][];
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

      // An actor reservation also satisfies the row's salience/audience minimum.
      let remaining = Math.max(
        0,
        reservedSlots - group.filter((thread) => selectedIds.has(thread.id)).length,
      );

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

  for (const active of [true, false]) {
    const pool = ordered.filter((thread) => isActiveActionState(thread.current.state) === active);
    if (active)
      reserve(
        [...new Set(input.reservedActors ?? ["borg"])].map((actor) =>
          pool
            .filter((thread) => thread.current.actor === actor)
            .sort((a, b) => b.current.updated_at - a.current.updated_at),
        ),
        ACTION_ACTOR_RESERVED_RENDER_SLOTS,
      );
    const salienceGroups = ACTION_SALIENCE_ORDER.map((salienceClass) =>
      pool.filter((thread) => thread.salienceClass === salienceClass),
    ).filter((group) => group.length > 0);
    reserve(salienceGroups, input.salienceClassReservedSlots);

    const audienceOrder = [
      ...new Set([
        ...(input.audienceOrder ?? []),
        ...pool.map((thread) => actionThreadAudienceBucket(thread)),
      ]),
    ].filter((audience) => pool.some((thread) => actionThreadAudienceBucket(thread) === audience));
    const audienceGroups = audienceOrder.map((audience) =>
      pool.filter((thread) => actionThreadAudienceBucket(thread) === audience),
    );
    reserve(audienceGroups, input.audienceReservedSlots);

    for (const thread of pool) {
      if (selected.length >= input.limit) {
        break;
      }

      if (!selectedIds.has(thread.id)) {
        selectedIds.add(thread.id);
        selected.push(thread);
      }
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
    staleActionsOmittedFromPrompt: 0,
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
  renderLimit: number;
  threadsBuiltCount: number;
  consideredRecordCount: number;
  sourceRecordLimit: number;
  sourceRecordTotal: number | null;
  salienceDroppedThreadCount: number;
}): string {
  const { groups } = input;
  const olderThreads = groups.flatMap((group) => group.threads);
  const olderRecordCount = olderThreads.reduce((count, thread) => count + thread.records.length, 0);
  // Threads now correspond one-to-one to action IDs. Keep the populations and
  // source-floor count explicit so past similarity-grouped counts are not conflated.
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
    `Action threads omitted from this section: threads=${olderThreads.length}, records=${olderRecordCount}.`,
    `Not counted above: salience_dropped_threads=${input.salienceDroppedThreadCount}, records_below_draw_floor=${recordsBelowDrawFloor} (threads_built=${input.threadsBuiltCount} = rendered ${input.renderedThreadCount} + omitted ${olderThreads.length} + dropped ${input.salienceDroppedThreadCount}; render_slots=${input.renderLimit}; records_considered=${input.consideredRecordCount} records, source_record_limit=${input.sourceRecordLimit}, ${sourceRecordTotalField}).`,
    ...disclosureNote,
    ...groups.map(
      (group) =>
        `- audience_scope=${group.audienceScope} salience_class=${group.salienceClass} ${actionThreadSummaryDetails(group.threads, group.disclosureLabel)}`,
    ),
  ].join("\n");
}
