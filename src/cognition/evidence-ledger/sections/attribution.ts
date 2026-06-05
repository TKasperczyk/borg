import type { ActionRecord } from "../../../memory/actions/index.js";
import type { CommitmentRecord } from "../../../memory/commitments/index.js";
import type { GoalRecord } from "../../../memory/self/index.js";
import {
  activeSessionTranscriptEntries,
  collectInactiveStreamEntryRefs,
  streamEntryIsActive,
  type StreamEntry,
  type TranscriptStreamEntry,
} from "../../../stream/index.js";
import type { EntityId } from "../../../util/ids.js";
import { stringifyPromptContent } from "../../../util/token-estimate.js";
import type { ActiveParticipant } from "../../participants.js";
import {
  actionBelongsToGroupChannel,
  commitmentBelongsToGroupChannel,
  goalBelongsToGroupChannel,
  isActionVisibleForCurrentAudienceStanding,
  isCommitmentVisibleToSession,
  isGoalVisibleToSession,
  scopedCommitmentsForEntity,
  scopedGoalsForEntity,
  visibleAudienceEntityIds,
} from "../audience-visibility.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  ACTION_TRUST_RANK,
  COMMITMENT_TRUST_RANK,
  OPEN_QUESTION_TRUST_RANK,
  TRANSCRIPT_TRUST_RANK,
  addEntry,
} from "../section-buckets.js";

const MATRIX_IDS_PER_ROW = 12;
const MATRIX_RECORD_SCAN_LIMIT = 64;
const SIDEBAR_ENTRIES_PER_BUCKET = 5;
const SIDEBAR_EXCERPT_MAX_CHARS = 120;
const ASSISTANT_REASONING_ID_LIMIT = 12;
const GROUP_RECORD_ID_LIMIT = 24;

type ParticipantAttribution = {
  participant: ActiveParticipant;
  saidStreamIds: string[];
  commitmentIds: string[];
  actionIds: string[];
  goalIds: string[];
};

function isTranscriptEntry(entry: StreamEntry): entry is TranscriptStreamEntry {
  return (
    entry.kind === "user_msg" ||
    entry.kind === "agent_msg" ||
    entry.kind === "agent_suppressed" ||
    entry.kind === "agent_observed"
  );
}

function transcriptEntries(context: BuilderSectionContext): TranscriptStreamEntry[] {
  const entries = activeSessionTranscriptEntries(context.streamEntries);
  const currentUserEntry = context.input.currentUserEntry;

  if (
    currentUserEntry !== undefined &&
    currentUserEntry.session_id === context.input.sessionId &&
    isTranscriptEntry(currentUserEntry) &&
    streamEntryIsActive(
      currentUserEntry,
      collectInactiveStreamEntryRefs([...context.streamEntries, currentUserEntry]),
    ) &&
    !entries.some((entry) => entry.id === currentUserEntry.id)
  ) {
    return [...entries, currentUserEntry];
  }

  return entries;
}

function displayNameForParticipant(participant: ActiveParticipant): string {
  return participant.displayName ?? "Participant";
}

function headingForParticipant(participant: ActiveParticipant): string {
  return `${displayNameForParticipant(participant)} <${participant.entityId}>`;
}

function lastItems<T>(items: readonly T[], limit: number): T[] {
  return items.slice(Math.max(0, items.length - limit));
}

function formatIds(ids: readonly string[]): string {
  return ids.length === 0 ? "none" : ids.join(", ");
}

function uniqueById<T extends { id: string }>(records: readonly T[]): T[] {
  return [...new Map(records.map((record) => [record.id, record])).values()];
}

function attributionParticipants(context: BuilderSectionContext): ActiveParticipant[] {
  return (context.input.activeParticipants ?? []).filter((participant) => {
    if (participant.role === "audience") {
      return false;
    }

    return context.repos.entities?.get(participant.entityId)?.kind !== "group";
  });
}

function participantUserEntries(
  entries: readonly TranscriptStreamEntry[],
  participant: ActiveParticipant,
): TranscriptStreamEntry[] {
  return entries.filter(
    (entry) => entry.kind === "user_msg" && entry.sender_entity_id === participant.entityId,
  );
}

function participantCommitments(
  context: BuilderSectionContext,
  participant: ActiveParticipant,
  activeParticipantIds: ReadonlySet<EntityId>,
): CommitmentRecord[] {
  return uniqueById(
    context.repos.commitments?.list({
      activeOnly: true,
      committedByEntity: participant.entityId,
    }) ?? [],
  )
    .filter((commitment) =>
      isCommitmentVisibleToSession(
        commitment,
        context.input.audienceEntityId,
        activeParticipantIds,
      ),
    )
    .slice(0, MATRIX_IDS_PER_ROW);
}

function participantActions(
  context: BuilderSectionContext,
  participant: ActiveParticipant,
  activeParticipantIds: ReadonlySet<EntityId>,
): ActionRecord[] {
  return uniqueById(
    context.repos.actions.list({
      actor: participant.entityId,
      limit: MATRIX_RECORD_SCAN_LIMIT,
    }),
  )
    .filter((action) =>
      isActionVisibleForCurrentAudienceStanding(
        action,
        context.input.audienceEntityId,
        activeParticipantIds,
      ),
    )
    .slice(0, MATRIX_IDS_PER_ROW);
}

function participantGoals(
  context: BuilderSectionContext,
  participant: ActiveParticipant,
  activeParticipantIds: ReadonlySet<EntityId>,
): GoalRecord[] {
  return uniqueById(
    context.repos.goals?.list({
      status: "active",
      ownerEntityId: participant.entityId,
    }) ?? [],
  )
    .filter((goal) =>
      isGoalVisibleToSession(goal, context.input.audienceEntityId, activeParticipantIds),
    )
    .slice(0, MATRIX_IDS_PER_ROW);
}

function participantAttributionRows(context: BuilderSectionContext): ParticipantAttribution[] {
  const participants = attributionParticipants(context);
  const entries = transcriptEntries(context);
  const activeParticipantIds = visibleAudienceEntityIds(
    context.input.audienceEntityId,
    participants,
  );

  return participants.flatMap((participant) => {
    const saidStreamIds = lastItems(
      participantUserEntries(entries, participant).map((entry) => entry.id),
      MATRIX_IDS_PER_ROW,
    );
    const commitmentIds = participantCommitments(context, participant, activeParticipantIds).map(
      (commitment) => commitment.id,
    );
    const actionIds = participantActions(context, participant, activeParticipantIds).map(
      (action) => action.id,
    );
    const goalIds = participantGoals(context, participant, activeParticipantIds).map(
      (goal) => goal.id,
    );

    if (
      saidStreamIds.length === 0 &&
      commitmentIds.length === 0 &&
      actionIds.length === 0 &&
      goalIds.length === 0
    ) {
      return [];
    }

    return [
      {
        participant,
        saidStreamIds,
        commitmentIds,
        actionIds,
        goalIds,
      },
    ];
  });
}

function matrixParticipantText(row: ParticipantAttribution): string {
  return [
    `### ${headingForParticipant(row.participant)}`,
    `- said this session: ${formatIds(row.saidStreamIds)}`,
    `- commitments: ${formatIds(row.commitmentIds)}`,
    `- assigned actions: ${formatIds(row.actionIds)}`,
    `- owned goals: ${formatIds(row.goalIds)}`,
  ].join("\n");
}

function assistantMessageEntries(
  entries: readonly TranscriptStreamEntry[],
): TranscriptStreamEntry[] {
  return entries.filter((entry) => entry.kind === "agent_msg");
}

function groupRecordIds(context: BuilderSectionContext): string[] {
  const audienceEntityId = context.input.audienceEntityId;

  if (audienceEntityId === null) {
    return [];
  }

  const commitments = scopedCommitmentsForEntity(
    context.repos.commitments?.list({
      activeOnly: true,
      audience: audienceEntityId,
    }) ?? context.input.applicableCommitments,
    audienceEntityId,
  )
    .filter((commitment) =>
      commitmentBelongsToGroupChannel(commitment, audienceEntityId, context.repos.entities),
    )
    .map((commitment) => commitment.id);
  const goals = scopedGoalsForEntity(
    context.repos.goals?.list({
      status: "active",
      visibleToAudienceEntityId: audienceEntityId,
    }) ?? [],
    audienceEntityId,
  )
    .filter((goal) => goalBelongsToGroupChannel(goal, audienceEntityId, context.repos.entities))
    .map((goal) => goal.id);
  const actions = uniqueById([
    ...context.repos.actions.list({
      audienceEntityId,
      limit: MATRIX_RECORD_SCAN_LIMIT,
    }),
    ...context.repos.actions.list({
      actor: audienceEntityId,
      limit: MATRIX_RECORD_SCAN_LIMIT,
    }),
  ])
    .filter((action) =>
      actionBelongsToGroupChannel(action, audienceEntityId, context.repos.entities),
    )
    .map((action) => action.id);

  return [...commitments, ...goals, ...actions].slice(0, GROUP_RECORD_ID_LIMIT);
}

function matrixShouldRender(context: BuilderSectionContext): boolean {
  return attributionParticipants(context).length > 1;
}

export function addAttributionMatrixSection(context: BuilderSectionContext): void {
  if (!matrixShouldRender(context)) {
    return;
  }

  const rows = participantAttributionRows(context);
  const assistantStreamIds = lastItems(
    assistantMessageEntries(transcriptEntries(context)).map((entry) => entry.id),
    ASSISTANT_REASONING_ID_LIMIT,
  );
  const groupIds = groupRecordIds(context);

  for (const row of rows) {
    addEntry(context.buckets, "attribution_matrix", {
      id: `attribution_matrix:participant:${row.participant.entityId}`,
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: TRANSCRIPT_TRUST_RANK,
      value: headingForParticipant(row.participant),
      text: matrixParticipantText(row),
      state: "attribution_matrix_participant",
      taint: "none",
    });
  }

  if (assistantStreamIds.length > 0) {
    addEntry(context.buckets, "attribution_matrix", {
      id: "attribution_matrix:assistant",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "assistant",
      trust_rank: TRANSCRIPT_TRUST_RANK,
      value: "Borg / Assistant rationale",
      text: [
        "### Borg / Assistant rationale",
        `- prior reasoning: ${formatIds(assistantStreamIds)}`,
      ].join("\n"),
      state: "attribution_matrix_assistant",
      taint: "none",
    });
  }

  if (groupIds.length > 0) {
    addEntry(context.buckets, "attribution_matrix", {
      id: "attribution_matrix:group_channel",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: Math.max(ACTION_TRUST_RANK, COMMITMENT_TRUST_RANK, OPEN_QUESTION_TRUST_RANK),
      value: "Group / Channel",
      text: ["### Group / Channel", `- group decisions/records: ${formatIds(groupIds)}`].join("\n"),
      state: "attribution_matrix_group_channel",
      taint: "none",
    });
  }
}

function excerpt(content: unknown): string {
  const text = stringifyPromptContent(content);

  if (text.length <= SIDEBAR_EXCERPT_MAX_CHARS) {
    return text;
  }

  return `${text.slice(0, SIDEBAR_EXCERPT_MAX_CHARS - 3)}...`;
}

function sidebarLine(entry: TranscriptStreamEntry): string {
  return `- ${entry.id} [${new Date(entry.timestamp).toISOString()}]: ${excerpt(entry.content)}`;
}

function sidebarParticipantText(
  participant: ActiveParticipant,
  entries: readonly TranscriptStreamEntry[],
): string {
  return [`### ${headingForParticipant(participant)}`, ...entries.map(sidebarLine)].join("\n");
}

function sidebarAssistantText(entries: readonly TranscriptStreamEntry[]): string {
  return ["### Borg / Assistant", ...entries.map(sidebarLine)].join("\n");
}

export function addCurrentSessionAttributionSidebarSection(context: BuilderSectionContext): void {
  const participants = attributionParticipants(context);

  if (participants.length <= 1) {
    return;
  }

  const entries = transcriptEntries(context);

  for (const participant of participants) {
    const participantEntries = lastItems(
      participantUserEntries(entries, participant),
      SIDEBAR_ENTRIES_PER_BUCKET,
    );

    if (participantEntries.length === 0) {
      continue;
    }

    addEntry(context.buckets, "current_session_attribution_sidebar", {
      id: `current_session_attribution_sidebar:participant:${participant.entityId}`,
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "user",
      trust_rank: TRANSCRIPT_TRUST_RANK,
      value: headingForParticipant(participant),
      text: sidebarParticipantText(participant, participantEntries),
      state: "current_session_attribution_sidebar_participant",
      taint: "none",
    });
  }

  const assistantEntries = lastItems(assistantMessageEntries(entries), SIDEBAR_ENTRIES_PER_BUCKET);

  if (assistantEntries.length === 0) {
    return;
  }

  addEntry(context.buckets, "current_session_attribution_sidebar", {
    id: "current_session_attribution_sidebar:assistant",
    source_type: "system_metadata",
    session_scope: "current_session",
    actor: "assistant",
    trust_rank: TRANSCRIPT_TRUST_RANK,
    value: "Borg / Assistant",
    text: sidebarAssistantText(assistantEntries),
    state: "current_session_attribution_sidebar_assistant",
    taint: "none",
  });
}
