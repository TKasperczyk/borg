import { existsSync, readdirSync } from "node:fs";

import { isActionVisibleToSession } from "../../evidence-ledger/audience-visibility.js";
import {
  estimateEvidenceLedgerPromptTokens,
  renderEvidenceLedger,
  type EvidenceLedger,
  type EvidenceLedgerEntry,
} from "../../evidence-ledger/index.js";
import { isFrameAnomaly, type FrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { ActiveParticipant } from "../../participants.js";
import type { ClosureLoopAssessment } from "../../generation/closure-loop.js";
import type { CognitiveMode } from "../../types.js";
import {
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
  type SharedStateCanonicalizationCandidates,
  type SharedStateCommitmentCanonicalizationType,
  type SharedStateLedgerMode,
  type SharedStateUnsettledReconciliationSummary,
} from "../../shared-state/index.js";
import type { ActionRecord, ActionRepository, ActionState } from "../../../memory/actions/index.js";
import type { CommitmentRecord } from "../../../memory/commitments/index.js";
import type {
  SharedStateArtifact,
  SharedStateEntryKind,
  SharedStateRepository,
  SharedStateSourceTrustValidator,
} from "../../../memory/decision-artifacts/index.js";
import {
  collectInactiveStreamEntryRefs,
  getStreamDirectory,
  isQuarantinedUserEntryMarker,
  StreamReader,
  streamEntryIsActive,
  type StreamEntry,
  type StreamEntryIndexRepository,
} from "../../../stream/index.js";
import {
  parseSessionId,
  streamEntryIdHelpers,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../../util/ids.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

const SHARED_STATE_LEDGER_STREAM_METADATA_KEYS = [
  "stream_ids",
  "source_stream_ids",
  "evidence_stream_entry_ids",
] as const;

function addSharedStateArtifactAllowedStreamId(ids: Set<StreamEntryId>, value: unknown): void {
  if (typeof value === "string" && streamEntryIdHelpers.is(value)) {
    ids.add(value);
  }
}

function addSharedStateArtifactAllowedStreamIds(ids: Set<StreamEntryId>, value: unknown): void {
  if (typeof value === "string") {
    addSharedStateArtifactAllowedStreamId(ids, value);
    return;
  }

  if (!Array.isArray(value)) {
    return;
  }

  for (const item of value) {
    addSharedStateArtifactAllowedStreamId(ids, item);
  }
}

function isSharedStateArtifactStreamContentRecord(
  value: unknown,
): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function collectQuarantinedSharedStateArtifactStreamEntryIds(
  entries: readonly StreamEntry[],
  ids: Set<StreamEntryId> = new Set<StreamEntryId>(),
): Set<StreamEntryId> {
  for (const entry of entries) {
    if (!isQuarantinedUserEntryMarker(entry)) {
      continue;
    }

    const content = isSharedStateArtifactStreamContentRecord(entry.content) ? entry.content : {};

    addSharedStateArtifactAllowedStreamId(ids, content.source_stream_entry_id);
    addSharedStateArtifactAllowedStreamIds(ids, content.cited_stream_entry_ids);
  }

  return ids;
}

const SHARED_STATE_STREAM_SESSION_FILE_SUFFIX = ".jsonl";

function listSharedStateArtifactStreamSessionIds(dataDir: string): SessionId[] {
  const streamDir = getStreamDirectory(dataDir);

  if (!existsSync(streamDir)) {
    return [];
  }

  return readdirSync(streamDir)
    .map((filename) => {
      if (!filename.endsWith(SHARED_STATE_STREAM_SESSION_FILE_SUFFIX)) {
        return null;
      }

      try {
        return parseSessionId(filename.slice(0, -SHARED_STATE_STREAM_SESSION_FILE_SUFFIX.length));
      } catch {
        return null;
      }
    })
    .filter((sessionId): sessionId is SessionId => sessionId !== null);
}

async function collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIdsFromStream(
  dataDir: string,
): Promise<ReadonlySet<StreamEntryId>> {
  const ids = new Set<StreamEntryId>();

  for (const sessionId of listSharedStateArtifactStreamSessionIds(dataDir)) {
    const reader = new StreamReader({
      dataDir,
      sessionId,
    });

    for await (const entry of reader.iterate({ kinds: ["internal_event"] })) {
      collectQuarantinedSharedStateArtifactStreamEntryIds([entry], ids);
    }
  }

  return ids;
}

export async function collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds(
  source: string | Pick<StreamEntryIndexRepository, "quarantinedSharedStateArtifactRefs">,
): Promise<ReadonlySet<StreamEntryId>> {
  if (typeof source === "string") {
    return collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIdsFromStream(source);
  }

  return source.quarantinedSharedStateArtifactRefs();
}

export function buildSharedStateSourceTrustValidator(input: {
  currentSessionEntries: readonly StreamEntry[];
  quarantinedStreamEntryIds: ReadonlySet<StreamEntryId>;
}): SharedStateSourceTrustValidator {
  const inactiveRefs = collectInactiveStreamEntryRefs(input.currentSessionEntries);
  const entriesById = new Map(input.currentSessionEntries.map((entry) => [entry.id, entry]));

  return (streamEntryId) => {
    if (input.quarantinedStreamEntryIds.has(streamEntryId)) {
      return {
        allowed: false,
        reason: "quarantined",
      };
    }

    const entry = entriesById.get(streamEntryId);

    if (entry !== undefined) {
      return streamEntryIsActive(entry, inactiveRefs)
        ? { allowed: true }
        : {
            allowed: false,
            reason: "inactive",
          };
    }

    if (inactiveRefs.streamEntryIds.has(streamEntryId)) {
      return {
        allowed: false,
        reason: "inactive",
      };
    }

    return { allowed: true };
  };
}

function sourceStreamEntryIdIsTrusted(
  streamEntryId: StreamEntryId,
  validator: SharedStateSourceTrustValidator | undefined,
): boolean {
  return validator?.(streamEntryId).allowed !== false;
}

function filterTrustedSharedStateArtifactSourceStreamEntryIds(
  streamEntryIds: readonly StreamEntryId[],
  validator: SharedStateSourceTrustValidator | undefined,
): StreamEntryId[] {
  return streamEntryIds.filter((streamEntryId) =>
    sourceStreamEntryIdIsTrusted(streamEntryId, validator),
  );
}

function collectOffLimitsSharedStateArtifactSourceStreamEntryIds(
  streamEntryIds: readonly StreamEntryId[],
  validator: SharedStateSourceTrustValidator | undefined,
): StreamEntryId[] {
  if (validator === undefined) {
    return [];
  }

  return streamEntryIds.filter((streamEntryId) => validator(streamEntryId).allowed === false);
}

export function compactSharedStateArtifactCandidateText(value: string, maxLength = 180): string {
  const trimmed = value.trim();

  return trimmed.length <= maxLength ? trimmed : `${trimmed.slice(0, maxLength - 3)}...`;
}

const SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPE_SET = new Set<string>(
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
);

type SharedStateCommitmentCanonicalizationRecord = CommitmentRecord & {
  type: SharedStateCommitmentCanonicalizationType;
};

export function isSharedStateCommitmentCanonicalizationRecord(
  commitment: CommitmentRecord,
): commitment is SharedStateCommitmentCanonicalizationRecord {
  return SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPE_SET.has(commitment.type);
}

function addScopedActionCandidates(input: {
  target: Map<string, ScopedSharedStateArtifactActionCandidate>;
  actions: readonly ActionRecord[];
  scope: SharedStateArtifactActionCandidateScope;
  audienceEntityId: EntityId | null;
  activeParticipantIds: ReadonlySet<EntityId>;
}): void {
  for (const action of input.actions) {
    if (!isActionVisibleToSession(action, input.audienceEntityId, input.activeParticipantIds)) {
      continue;
    }

    if (!input.target.has(action.id)) {
      input.target.set(action.id, {
        action,
        scope: input.scope,
      });
    }
  }
}

export function selectSharedStateArtifactActionCandidates(input: {
  actionRepository: TurnPhaseCoordinatorOptions["actionRepository"];
  audienceEntityId: EntityId | null;
  activeParticipants: readonly ActiveParticipant[] | undefined;
}): {
  candidates: SharedStateCanonicalizationCandidates["actions"];
  countByScope: Record<SharedStateArtifactActionCandidateScope, number>;
} {
  const activeParticipantIds = new Set(
    (input.activeParticipants ?? []).map((participant) => participant.entityId),
  );
  const scoped = new Map<string, ScopedSharedStateArtifactActionCandidate>();

  if (input.audienceEntityId !== null) {
    addScopedActionCandidates({
      target: scoped,
      actions: input.actionRepository.list({
        states: SHARED_STATE_ACTION_CANDIDATE_STATES,
        audienceEntityId: input.audienceEntityId,
        limit: SHARED_STATE_ACTION_CANDIDATE_LIMIT,
      }),
      scope: "audience",
      audienceEntityId: input.audienceEntityId,
      activeParticipantIds,
    });
  }

  addScopedActionCandidates({
    target: scoped,
    actions: input.actionRepository.list({
      states: SHARED_STATE_ACTION_CANDIDATE_STATES,
      audienceEntityId: null,
      limit: SHARED_STATE_ACTION_CANDIDATE_LIMIT,
    }),
    scope: "global",
    audienceEntityId: input.audienceEntityId,
    activeParticipantIds,
  });

  for (const participant of input.activeParticipants ?? []) {
    addScopedActionCandidates({
      target: scoped,
      actions: input.actionRepository.list({
        states: SHARED_STATE_ACTION_CANDIDATE_STATES,
        actor: participant.entityId,
        limit: SHARED_STATE_ACTION_CANDIDATE_LIMIT,
      }),
      scope: "actor",
      audienceEntityId: input.audienceEntityId,
      activeParticipantIds,
    });
  }

  const selected = [...scoped.values()]
    .sort(
      (left, right) =>
        right.action.updated_at - left.action.updated_at ||
        left.action.id.localeCompare(right.action.id),
    )
    .slice(0, SHARED_STATE_ACTION_CANDIDATE_LIMIT);
  const countByScope: Record<SharedStateArtifactActionCandidateScope, number> = {
    audience: 0,
    global: 0,
    actor: 0,
  };

  for (const candidate of selected) {
    countByScope[candidate.scope] += 1;
  }

  return {
    candidates: selected.map(({ action }) => ({
      id: action.id,
      text: compactSharedStateArtifactCandidateText(action.description),
      actor: action.actor,
      state: action.state,
      session_scope: action.session_scope,
    })),
    countByScope,
  };
}

function addSharedStateEntryIdStreamHandle(ids: Set<StreamEntryId>, entryId: string): void {
  const currentSessionPrefix = "current_session_stream:";
  const currentUserPrefix = "current_user_message:";
  const source = entryId.startsWith(currentSessionPrefix)
    ? entryId.slice(currentSessionPrefix.length)
    : entryId.startsWith(currentUserPrefix)
      ? entryId.slice(currentUserPrefix.length)
      : null;

  addSharedStateArtifactAllowedStreamId(ids, source);
}

function collectSharedStateEntryStreamEntryIds(entry: EvidenceLedgerEntry): StreamEntryId[] {
  const ids = new Set<StreamEntryId>();

  addSharedStateEntryIdStreamHandle(ids, entry.id);
  addSharedStateArtifactAllowedStreamIds(ids, entry.citations);

  for (const key of SHARED_STATE_LEDGER_STREAM_METADATA_KEYS) {
    addSharedStateArtifactAllowedStreamIds(ids, entry.state_metadata?.[key]);
  }

  return [...ids];
}

function collectSharedStateArtifactLedgerVisibleStreamEntryIds(
  ledger: EvidenceLedger,
): StreamEntryId[] {
  const ids = new Set<StreamEntryId>();

  for (const section of ledger.sections) {
    for (const entry of section.entries) {
      for (const streamEntryId of collectSharedStateEntryStreamEntryIds(entry)) {
        ids.add(streamEntryId);
      }
    }
  }

  return [...ids];
}

const SHARED_STATE_IN_FLIGHT_KINDS = [
  "live",
  "low_salience_live",
  "dormant_live",
  "pending",
  "tentative",
] as const satisfies readonly SharedStateEntryKind[];
const SHARED_STATE_ACTION_CANDIDATE_LIMIT = 80;
const SHARED_STATE_ACTION_CANDIDATE_STATES: readonly ActionState[] = [
  "considering",
  "committed_to_do",
  "scheduled",
  "unknown",
];
type SharedStateArtifactActionCandidateScope = "audience" | "global" | "actor";
type ScopedSharedStateArtifactActionCandidate = {
  action: ActionRecord;
  scope: SharedStateArtifactActionCandidateScope;
};

type SharedStateCompileSkipReason =
  | "quarantined_current_turn"
  | "closure_shaped"
  | "idle_no_active_decisions";

export type SharedStateCompileSkip = {
  reason: SharedStateCompileSkipReason;
  previousActiveEntryCount: number;
  perceptionMode: CognitiveMode;
  closureShaped?: boolean;
  hasStateDelta?: boolean;
};

function previousSharedStateArtifactActiveEntryCount(
  artifact: SharedStateArtifact | null | undefined,
): number {
  return (artifact?.entries ?? []).filter((entry) => entry.superseded_by_id === null).length;
}

function previousSharedStateArtifactInFlightEntryCount(
  artifact: SharedStateArtifact | null | undefined,
): number {
  return (artifact?.entries ?? []).filter(
    (entry) =>
      entry.superseded_by_id === null &&
      SHARED_STATE_IN_FLIGHT_KINDS.some((kind) => kind === entry.kind),
  ).length;
}

export function shouldSkipSharedStateCompile(input: {
  enabled: boolean;
  previousArtifact: SharedStateArtifact | null | undefined;
  perceptionMode: CognitiveMode;
  frameAnomaly: FrameAnomalyClassification | null | undefined;
  closureLoopAssessment: ClosureLoopAssessment | null | undefined;
  unsettledReconciliation?: SharedStateUnsettledReconciliationSummary | null;
}): SharedStateCompileSkip | null {
  const previousActiveEntryCount = previousSharedStateArtifactActiveEntryCount(
    input.previousArtifact,
  );

  if (isFrameAnomaly(input.frameAnomaly)) {
    return {
      reason: "quarantined_current_turn",
      previousActiveEntryCount,
      perceptionMode: input.perceptionMode,
    };
  }

  if (!input.enabled) {
    return null;
  }

  if (input.unsettledReconciliation !== null && input.unsettledReconciliation !== undefined) {
    return null;
  }

  if (
    input.closureLoopAssessment?.currentUserClosureShaped === true &&
    input.closureLoopAssessment.currentUserHasSubstantiveStateDelta === false
  ) {
    return {
      reason: "closure_shaped",
      previousActiveEntryCount,
      perceptionMode: input.perceptionMode,
      closureShaped: true,
      hasStateDelta: false,
    };
  }

  if (
    input.perceptionMode === "idle" &&
    previousSharedStateArtifactInFlightEntryCount(input.previousArtifact) === 0
  ) {
    return {
      reason: "idle_no_active_decisions",
      previousActiveEntryCount,
      perceptionMode: input.perceptionMode,
    };
  }

  return null;
}

export function advanceSharedStateCompileSkipAnchor(input: {
  repository: Pick<SharedStateRepository, "upsert">;
  audienceEntityId: EntityId;
  previousArtifact: SharedStateArtifact | null | undefined;
  currentUserStreamEntryId: StreamEntryId;
  nowMs: number;
}): {
  artifact: SharedStateArtifact | null;
  advanced: boolean;
} {
  const artifact = input.repository.upsert(input.audienceEntityId, [], {
    expectedVersion: input.previousArtifact?.record_version,
    now: input.nowMs,
    lastCompiledAt: input.nowMs,
    lastCompiledStreamEntryId: input.currentUserStreamEntryId,
  });

  return {
    artifact,
    advanced: artifact?.last_compiled_stream_entry_id === input.currentUserStreamEntryId,
  };
}

function buildLedgerStreamOrder(ledger: EvidenceLedger): Map<StreamEntryId, number> {
  const streamOrderById = new Map<StreamEntryId, number>();

  for (const section of ledger.sections) {
    for (const entry of section.entries) {
      if (entry.stream_index === undefined) {
        continue;
      }

      for (const streamEntryId of collectSharedStateEntryStreamEntryIds(entry)) {
        const currentIndex = streamOrderById.get(streamEntryId);

        if (currentIndex === undefined || entry.stream_index < currentIndex) {
          streamOrderById.set(streamEntryId, entry.stream_index);
        }
      }
    }
  }

  return streamOrderById;
}

function buildRetainedLedgerWindowStreamOrder(ledger: EvidenceLedger): Map<StreamEntryId, number> {
  const streamOrderById = new Map<StreamEntryId, number>();

  for (const section of ledger.sections) {
    if (section.id !== "current_user_message" && section.id !== "current_session_transcript") {
      continue;
    }

    for (const entry of section.entries) {
      if (entry.stream_index === undefined) {
        continue;
      }

      for (const streamEntryId of collectSharedStateEntryStreamEntryIds(entry)) {
        const currentIndex = streamOrderById.get(streamEntryId);

        if (currentIndex === undefined || entry.stream_index < currentIndex) {
          streamOrderById.set(streamEntryId, entry.stream_index);
        }
      }
    }
  }

  return streamOrderById;
}

function earliestLedgerStreamIndex(
  streamOrderById: ReadonlyMap<StreamEntryId, number>,
): number | null {
  let earliest: number | null = null;

  for (const streamIndex of streamOrderById.values()) {
    if (earliest === null || streamIndex < earliest) {
      earliest = streamIndex;
    }
  }

  return earliest;
}

function isSharedStateArtifactLedgerDeltaEntry(input: {
  entry: EvidenceLedgerEntry;
  streamOrderById: ReadonlyMap<StreamEntryId, number>;
  anchorStreamIndex: number;
}): boolean {
  for (const streamEntryId of collectSharedStateEntryStreamEntryIds(input.entry)) {
    const streamIndex = input.streamOrderById.get(streamEntryId);

    if (streamIndex !== undefined && streamIndex > input.anchorStreamIndex) {
      return true;
    }
  }

  return false;
}

export function buildSharedStateLedgerPromptContext(input: {
  ledger: EvidenceLedger;
  previousArtifact: SharedStateArtifact | null | undefined;
  fullPromptVisibleLedger: string;
  enabled: boolean;
  minTailPerSection: number;
  sourceTrustValidator?: SharedStateSourceTrustValidator;
}): {
  promptVisibleLedger: string;
  ledgerMode: SharedStateLedgerMode;
  visibleStreamEntryIds: StreamEntryId[];
  offLimitsSourceStreamEntryIds: StreamEntryId[];
} {
  const anchorStreamEntryId = input.previousArtifact?.last_compiled_stream_entry_id ?? null;
  const fullVisibleStreamEntryIds = collectSharedStateArtifactLedgerVisibleStreamEntryIds(
    input.ledger,
  );
  const fullTrustedVisibleStreamEntryIds = filterTrustedSharedStateArtifactSourceStreamEntryIds(
    fullVisibleStreamEntryIds,
    input.sourceTrustValidator,
  );
  const fullOffLimitsSourceStreamEntryIds = collectOffLimitsSharedStateArtifactSourceStreamEntryIds(
    fullVisibleStreamEntryIds,
    input.sourceTrustValidator,
  );

  if (!input.enabled || anchorStreamEntryId === null) {
    return {
      promptVisibleLedger: input.fullPromptVisibleLedger,
      ledgerMode: "full_fallback",
      visibleStreamEntryIds: fullTrustedVisibleStreamEntryIds,
      offLimitsSourceStreamEntryIds: fullOffLimitsSourceStreamEntryIds,
    };
  }

  const retainedWindowStreamOrderById = buildRetainedLedgerWindowStreamOrder(input.ledger);
  const windowFloorStreamIndex = earliestLedgerStreamIndex(retainedWindowStreamOrderById);
  const anchorStreamIndex = retainedWindowStreamOrderById.get(anchorStreamEntryId);

  if (
    windowFloorStreamIndex === null ||
    anchorStreamIndex === undefined ||
    anchorStreamIndex < windowFloorStreamIndex
  ) {
    return {
      promptVisibleLedger: input.fullPromptVisibleLedger,
      ledgerMode: "full_fallback",
      visibleStreamEntryIds: fullTrustedVisibleStreamEntryIds,
      offLimitsSourceStreamEntryIds: fullOffLimitsSourceStreamEntryIds,
    };
  }

  const streamOrderById = buildLedgerStreamOrder(input.ledger);
  const minTailPerSection = Math.max(0, Math.floor(input.minTailPerSection));
  const sections = input.ledger.sections.map((section) => {
    const tailStartIndex = Math.max(0, section.entries.length - minTailPerSection);
    const entries = section.entries.filter(
      (entry, index) =>
        index >= tailStartIndex ||
        isSharedStateArtifactLedgerDeltaEntry({
          entry,
          streamOrderById,
          anchorStreamIndex,
        }),
    );

    return {
      ...section,
      entries,
    };
  });
  const deltaLedgerForEstimate = {
    ...input.ledger,
    sections,
  };
  const deltaLedger = {
    ...deltaLedgerForEstimate,
    estimatedTokens: estimateEvidenceLedgerPromptTokens(deltaLedgerForEstimate),
  };
  const deltaVisibleStreamEntryIds =
    collectSharedStateArtifactLedgerVisibleStreamEntryIds(deltaLedger);

  return {
    promptVisibleLedger: renderEvidenceLedger(deltaLedger) ?? "",
    ledgerMode: "delta",
    visibleStreamEntryIds: filterTrustedSharedStateArtifactSourceStreamEntryIds(
      deltaVisibleStreamEntryIds,
      input.sourceTrustValidator,
    ),
    offLimitsSourceStreamEntryIds: collectOffLimitsSharedStateArtifactSourceStreamEntryIds(
      deltaVisibleStreamEntryIds,
      input.sourceTrustValidator,
    ),
  };
}
