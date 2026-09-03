import { SuppressionSet } from "../attention/index.js";
import { formatAutonomyTriggerContext, type AutonomyTriggerContext } from "../autonomy-trigger.js";
import { ContradictionRoutingCooldown } from "../deliberation/contradiction-routing-cooldown.js";
import {
  GenerationGate,
  type GenerationGateResult,
  type GenerationGateStructuralSignals,
} from "../generation/generation-gate.js";
import {
  buildStreamBacklogResponseTo,
  hydrateStreamBacklogBatch,
} from "../ingestion/backlog-terminal.js";
import { buildParticipantRosterFromRepositories } from "../perception/index.js";
import {
  resolveActiveParticipants,
  resolveParticipantProfiles,
  scanRecentParticipantStreamEntries,
} from "../participants.js";
import type {
  StreamEntry,
  StreamEntryIndexRecord,
  StreamResponseTo,
  StreamWriter,
} from "../../stream/index.js";
import { buildObservedEventEmission } from "../../memory/observed-events/index.js";
import { CognitionError } from "../../util/errors.js";
import type { AttachmentId, EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import { isCreatorInOperatorContext } from "../authority.js";
import { isUserTurnOrigin, persistsPerception } from "../types.js";
import {
  isInboundBatchTurnInput,
  orderedInboundBatchEntries,
  renderInboundBatch,
  type CurrentTurnUserInput,
  type CurrentTurnUserInputSenderAttribution,
  type HydratedInboundAttachment,
  type HydratedInboundMessage,
} from "../turn-input.js";
import {
  classifyClosureLoopPhase,
  classifyFrameAnomalyPhase,
} from "./turn-phase/perception-phase.js";
export {
  advanceSharedStateCompileSkipAnchor,
  buildSharedStateLedgerPromptContext,
  shouldSkipSharedStateCompile,
  type SharedStateCompileSkip,
} from "./turn-phase/shared-state-phase.js";
import {
  audienceProfileForParticipants,
  buildFrameAnomalyConversationContext,
} from "./turn-phase/context-build.js";
import {
  buildOperatorSessionSnapshot,
  OPERATOR_SESSION_SNAPSHOT_CAP,
} from "./turn-phase/session-snapshot.js";
export {
  buildContradictionRoutingOverride,
  type BuildContradictionRoutingOverrideInput,
} from "./turn-phase/context-build.js";
import { runExtractionPhase } from "./turn-phase/extraction-phase.js";
import { runRetrievalPhase } from "./turn-phase/retrieval-phase.js";
import { runDeliberationPhase } from "./turn-phase/deliberation-phase.js";
import {
  runPostGenerationPhase,
  suppressFromClosureLoopPhase,
  suppressFromGenerationGatePhase,
} from "./turn-phase/post-generation-phase.js";
import { traceTurnPhase } from "./turn-phase/phase-trace.js";
import { appendHookFailureEvent, catchUpStreamIngestion } from "./turn-phase/utils.js";
import type {
  RunTurnPhasesInput,
  TurnPhaseCoordinatorInput,
  TurnPhaseCoordinatorOptions,
  TurnPhaseInput,
  TurnPhaseResult,
} from "./turn-phase/types.js";
import type { TurnExtractionPhaseResult } from "./turn-phase/extraction-phase.js";
import type { TurnRetrievalPhaseResult } from "./turn-phase/retrieval-phase.js";
import type { TurnDeliberationPhaseResult } from "./turn-phase/deliberation-phase.js";
export type {
  RunTurnPhasesInput,
  TurnPhaseCoordinatorOptions,
  TurnPhaseInput,
  TurnPhaseResult,
} from "./turn-phase/types.js";

const ZEROED_SIGNALS: GenerationGateStructuralSignals = {
  minimalUserInput: false,
  activeDiscourseStop: false,
  recentMinimalUserRun: 0,
  repeatedMinimalSimilarity: null,
  repeatedMinimalExchange: false,
  hardCapDue: false,
  hardCapActiveTurns: 0,
};

type PriorSelfThought = {
  text: string;
  updatedAt: number | null;
  disclosure: string | null;
};

// The scheduler carries text, updated_at and a disclosure label on this payload field; a
// reader that destructures only `text` drops the other two silently, and nothing fails --
// the anchor just renders thinner. Read the whole record here so a field the producer adds
// is a visible omission rather than an invisible one.
function priorSelfThought(
  context: AutonomyTriggerContext | null | undefined,
): PriorSelfThought | null {
  const value = context?.payload.prior_self_thought;

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const record = value as { text?: unknown; updated_at?: unknown; disclosure?: unknown };

  if (typeof record.text !== "string" || record.text.trim().length === 0) {
    return null;
  }

  return {
    text: record.text,
    updatedAt: typeof record.updated_at === "number" ? record.updated_at : null,
    disclosure: typeof record.disclosure === "string" ? record.disclosure : null,
  };
}

// The anchor is a stored record, not the thought currently being had, and it is selected by
// recency alone: `TrainOfThoughtRepository.get()` returns the single latest journal entry
// from whichever wake wrote last, with no filter on the subject this wake fired for. Hoisted
// bare it reads as present thought of unknown age -- so name the age, the depth, the
// selection, and the disclosure class the producer attached. Scope the depth claim to the
// automatic read: this one row is the whole of what the harness fetches, but
// `tool.ownRecords.list` is registered for both the autonomous and deliberator origins, so a
// wider journal read is available in the same turn. Saying "the only journal read in this
// turn" unscoped reads as a capability bound and points away from the one tool that answers
// "have I already done this".
function priorSelfThoughtProvenance(thought: PriorSelfThought): string {
  const written =
    thought.updatedAt === null ? "unknown time" : new Date(thought.updatedAt).toISOString();

  return [
    `Journal anchor (written ${written}`,
    thought.disclosure === null ? "" : `, disclosure: ${thought.disclosure}`,
    "). This is my latest journal entry by recency alone -- the last one written from any wake, ",
    "not one selected for this wake's subject -- and it is one row deep and the only journal read ",
    "the harness performs for me, here or on any other turn. Anything else I have already done or ",
    "already sent is not in it and is not implied absent by it. A wider read exists and is mine to ",
    "make rather than the wake's: tool.ownRecords.list browses my thoughts and journal over a time ",
    "range I choose, on this turn and on every live turn.",
  ].join("");
}

export function cognitionInputForTurnInput(
  turnInput: Pick<TurnPhaseInput, "autonomyTrigger" | "userMessage">,
): string {
  const trigger = turnInput.autonomyTrigger;

  if (trigger === null || trigger === undefined) {
    return turnInput.userMessage;
  }

  const priorThought = priorSelfThought(trigger);

  if (priorThought === null) {
    return formatAutonomyTriggerContext(trigger);
  }

  // Anchor autonomous cognition on where the thinking left off, and keep the
  // structured wake context (goal ids, reasons) the anchor used to replace --
  // now that the journal reaches every wake and not only reflection ones.
  const { prior_self_thought: _hoisted, ...remainingPayload } = trigger.payload;

  return [
    priorSelfThoughtProvenance(priorThought),
    "",
    priorThought.text,
    "",
    formatAutonomyTriggerContext({
      ...trigger,
      payload: remainingPayload,
    }),
  ].join("\n");
}

function previewItems(items: readonly string[], limit = 4): string {
  const head = items.slice(0, limit).join(",");
  return items.length > limit ? `${head},+${items.length - limit}` : head;
}

function summarizePerceptionResult(
  result: Awaited<
    ReturnType<
      ReturnType<TurnPhaseCoordinatorOptions["perceptionGateway"]["beginTurn"]>["perceive"]
    >
  >,
): string {
  return `mode=${result.perception.mode} entities=[${previewItems(result.perception.entities)}]`;
}

function summarizeFrameClassification(
  result: Awaited<ReturnType<typeof classifyFrameAnomalyPhase>>,
): string {
  const classification = result.classification;
  if (classification === null) {
    return "skipped";
  }

  if (classification.status === "degraded") {
    return `degraded reason=${classification.reason} disposition=${result.disposition}`;
  }

  return `kind=${classification.kind} conf=${classification.confidence} disposition=${result.disposition}`;
}

function summarizeExtraction(result: TurnExtractionPhaseResult): string {
  return `actions=${result.createdActionIds.length} goals=${result.persistedPromotions.goalIds.length} steps=${result.persistedPromotions.executiveStepIds.length} creator_directives=${result.creatorDirectives.length} commitment=${result.correctiveCommitment === null ? "none" : "candidate"}`;
}

function summarizeRetrieval(result: TurnRetrievalPhaseResult): string {
  const ledgerEntries =
    result.evidenceLedgerContext.ledger?.sections.reduce(
      (sum, section) => sum + section.entries.length,
      0,
    ) ?? 0;

  return `episodes=${result.retrievedEpisodes.length} evidence=${result.retrieval.evidence.length} ledger=${ledgerEntries}`;
}

function summarizeDeliberation(result: TurnDeliberationPhaseResult): string {
  return `path=${result.deliberation.path} recommendation=${result.deliberation.emissionRecommendation ?? "emit"} stop=${result.deliberation.usage.stop_reason ?? "none"}`;
}

type HydratedInboundBatch = {
  entries: readonly HydratedInboundMessage[];
  sourceEntries: readonly StreamEntry[];
  records: readonly StreamEntryIndexRecord[];
};

function batchEntryAsStreamEntry(entry: HydratedInboundMessage): StreamEntry {
  return {
    id: entry.id,
    session_id: entry.session_id,
    entry_index: entry.entry_index,
    timestamp: entry.timestamp,
    kind: "user_msg",
    content: entry.content,
    sender_entity_id: entry.sender_entity_id ?? null,
    reply_target_entity_id: null,
    compressed: false,
    ...(entry.audience === undefined ? {} : { audience: entry.audience }),
    ...(entry.source_message_key === undefined
      ? {}
      : { source_message_key: entry.source_message_key }),
    ...(entry.observed_at === undefined ? {} : { observed_at: entry.observed_at }),
    ...(entry.conversation === undefined ? {} : { conversation: entry.conversation }),
    ...(entry.metadata === undefined ? {} : { metadata: entry.metadata }),
  };
}

function distinctSenderIds(
  entries: readonly Pick<HydratedInboundMessage, "sender_entity_id">[],
): EntityId[] {
  return [
    ...new Set(
      entries.flatMap((entry) => {
        const senderEntityId = entry.sender_entity_id ?? null;
        return senderEntityId === null ? [] : [senderEntityId];
      }),
    ),
  ];
}

function distinctSenderCount(
  entries: readonly Pick<HydratedInboundMessage, "sender_entity_id">[],
): number {
  const senderIds = distinctSenderIds(entries);
  const hasKnownSender = senderIds.length > 0;
  const hasUnknownSender = entries.some((entry) => {
    const senderEntityId = entry.sender_entity_id ?? null;
    return senderEntityId === null;
  });

  return senderIds.length + (hasKnownSender && hasUnknownSender ? 1 : 0);
}

function singleNonNullBatchSenderId(
  entries: readonly Pick<HydratedInboundMessage, "sender_entity_id">[],
): EntityId | null {
  let senderEntityId: EntityId | null = null;

  for (const entry of entries) {
    const currentSenderEntityId = entry.sender_entity_id ?? null;
    if (currentSenderEntityId === null) {
      return null;
    }
    if (senderEntityId === null) {
      senderEntityId = currentSenderEntityId;
      continue;
    }
    if (senderEntityId !== currentSenderEntityId) {
      return null;
    }
  }

  return senderEntityId;
}

function senderAttributionForEntries(input: {
  entries: readonly Pick<HydratedInboundMessage, "id" | "sender_entity_id">[];
  entityRepository: TurnPhaseCoordinatorOptions["entityRepository"];
}): CurrentTurnUserInputSenderAttribution[] {
  return input.entries.map((entry) => {
    const senderEntityId = entry.sender_entity_id ?? null;
    const senderDisplayName =
      senderEntityId === null
        ? null
        : (input.entityRepository.get(senderEntityId)?.canonical_name ?? null);

    return {
      entryId: entry.id,
      senderEntityId,
      ...(senderDisplayName === null ? {} : { senderDisplayName }),
    };
  });
}

function sourceUserEntriesFromBatch(entries: readonly HydratedInboundMessage[]): StreamEntry[] {
  return entries.map((entry) => batchEntryAsStreamEntry(entry));
}

function validateInboundBatchRequest(input: {
  rawTurnInput: TurnPhaseCoordinatorInput;
  isUserTurn: boolean;
}): readonly StreamEntryId[] | null {
  if (!isInboundBatchTurnInput(input.rawTurnInput)) {
    return null;
  }

  if (!input.isUserTurn) {
    throw new CognitionError("Inbound batch turns must use user origin", {
      code: "INBOUND_BATCH_REQUIRES_USER_ORIGIN",
    });
  }

  const unsafeInput = input.rawTurnInput as {
    userMessage?: unknown;
    attachments?: readonly unknown[];
  };
  const unsafeBatch = input.rawTurnInput.inboundBatch as {
    entries?: unknown;
  };

  if (
    unsafeInput.userMessage !== undefined ||
    unsafeInput.attachments !== undefined ||
    unsafeBatch.entries !== undefined
  ) {
    throw new CognitionError("Inbound batch turns can only include durable entry identifiers", {
      code: "INBOUND_BATCH_INPUT_CONFLICT",
    });
  }

  const entryIds = [...input.rawTurnInput.inboundBatch.entryIds];

  if (entryIds.length === 0) {
    throw new CognitionError("Inbound batch turns require at least one source entry", {
      code: "INBOUND_BATCH_EMPTY",
    });
  }

  if (new Set(entryIds).size !== entryIds.length) {
    throw new CognitionError("Inbound batch cannot contain duplicate source entries", {
      code: "INBOUND_BATCH_DUPLICATE_ENTRY",
    });
  }

  return entryIds;
}

async function hydrateInboundBatch(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  rawTurnInput: TurnPhaseCoordinatorInput;
  entryIds: readonly StreamEntryId[];
}): Promise<HydratedInboundBatch> {
  const entryIndex = input.options.entryIndex;

  if (entryIndex === undefined) {
    throw new CognitionError("Inbound batch turns require the stream entry index", {
      code: "INBOUND_BATCH_INDEX_REQUIRED",
    });
  }

  const batch = await hydrateStreamBacklogBatch({
    dataDir: input.options.config.dataDir,
    entryIndex,
    createStreamReader: input.options.createStreamReader,
    sessionId: input.sessionId,
    entryIds: input.entryIds,
    throughCursorInclusive: isInboundBatchTurnInput(input.rawTurnInput)
      ? input.rawTurnInput.inboundBatch.throughCursorInclusive
      : undefined,
  });
  const hydrated: HydratedInboundMessage[] = [];
  const attachmentsByParentEntry = new Map<StreamEntryId, HydratedInboundAttachment[]>();
  const perceptionByAttachmentId = new Map(
    [...input.options.imagePerceptionRepository.listByParentEntries(input.entryIds).values()]
      .flat()
      .map((perception) => [perception.attachment_id, perception] as const),
  );

  for (const entryId of input.entryIds) {
    const attachments = input.options.attachmentRepository.listByParentEntry(entryId);
    if (attachments.length === 0) {
      continue;
    }

    attachmentsByParentEntry.set(
      entryId,
      attachments.map((attachment) => {
        const perception = perceptionByAttachmentId.get(attachment.attachment_id) ?? null;

        return {
          attachment_id: attachment.attachment_id,
          media_type: attachment.media_type,
          width: attachment.width,
          height: attachment.height,
          perception:
            perception === null
              ? null
              : {
                  perception_id: perception.perception_id,
                  caption: perception.caption,
                  image_kind: perception.image_kind,
                  visible_text: perception.visible_text,
                  search_terms: perception.search_terms,
                },
        };
      }),
    );
  }

  for (const sourceEntry of batch.sourceEntries) {
    const hydratedAttachments = attachmentsByParentEntry.get(sourceEntry.id);

    hydrated.push({
      id: sourceEntry.id,
      session_id: sourceEntry.session_id,
      entry_index: sourceEntry.entry_index!,
      timestamp: sourceEntry.timestamp,
      kind: "user_msg",
      content: sourceEntry.content as string,
      sender_entity_id: sourceEntry.sender_entity_id ?? null,
      ...(hydratedAttachments === undefined ? {} : { attachments: hydratedAttachments }),
      ...(sourceEntry.audience === undefined ? {} : { audience: sourceEntry.audience }),
      ...(sourceEntry.source_message_key === undefined
        ? {}
        : { source_message_key: sourceEntry.source_message_key }),
      ...(sourceEntry.observed_at === undefined ? {} : { observed_at: sourceEntry.observed_at }),
      ...(sourceEntry.conversation === undefined ? {} : { conversation: sourceEntry.conversation }),
      ...(sourceEntry.metadata === undefined ? {} : { metadata: sourceEntry.metadata }),
    });
  }

  return {
    entries: orderedInboundBatchEntries(hydrated),
    sourceEntries: batch.sourceEntries,
    records: batch.records,
  };
}

function initialCurrentTurnUserInput(input: {
  rawTurnInput: TurnPhaseCoordinatorInput;
  batchEntries: readonly HydratedInboundMessage[] | null;
  batchSourceEntries?: readonly StreamEntry[];
  sessionId: SessionId;
  entityRepository: TurnPhaseCoordinatorOptions["entityRepository"];
  responseTo?: StreamResponseTo;
}): CurrentTurnUserInput {
  if (input.batchEntries !== null) {
    const sourceUserEntryIds = input.batchEntries.map((entry) => entry.id);
    const sourceUserEntries =
      input.batchSourceEntries ?? sourceUserEntriesFromBatch(input.batchEntries);
    const scalarSenderEntityId = singleNonNullBatchSenderId(input.batchEntries);
    const renderedText = renderInboundBatch({
      entries: input.batchEntries,
      senderDisplayNameById: (entityId) =>
        input.entityRepository.get(entityId)?.canonical_name ?? null,
    });

    return {
      renderedText,
      currentUserContent: [{ type: "text", text: renderedText }],
      sourceUserEntries,
      sourceUserEntryIds,
      senderAttribution: senderAttributionForEntries({
        entries: input.batchEntries,
        entityRepository: input.entityRepository,
      }),
      effectiveSenderEntityId: scalarSenderEntityId,
      ...(input.responseTo === undefined ? {} : { responseTo: input.responseTo }),
      recencyBeforeEntryIdExclusive: input.batchEntries[0]?.id,
      persistUserMessage: false,
    };
  }

  const userMessage = isInboundBatchTurnInput(input.rawTurnInput)
    ? ""
    : input.rawTurnInput.userMessage;

  return {
    renderedText: userMessage,
    currentUserContent: [],
    sourceUserEntries: [],
    sourceUserEntryIds: [],
    senderAttribution: [],
    effectiveSenderEntityId: input.rawTurnInput.senderEntityId ?? null,
    persistUserMessage: true,
  };
}

function withPersistedCurrentUserEntry(input: {
  currentUserInput: CurrentTurnUserInput;
  persistedUserEntry: StreamEntry | null;
  currentUserContent: CurrentTurnUserInput["currentUserContent"];
  entityRepository: TurnPhaseCoordinatorOptions["entityRepository"];
  fallbackEffectiveSenderEntityId: EntityId | null;
}): CurrentTurnUserInput {
  if (!input.currentUserInput.persistUserMessage) {
    return input.currentUserInput;
  }

  const persistedUserEntry = input.persistedUserEntry;

  if (persistedUserEntry === null) {
    return {
      ...input.currentUserInput,
      currentUserContent: input.currentUserContent,
      effectiveSenderEntityId: input.fallbackEffectiveSenderEntityId,
    };
  }

  return {
    ...input.currentUserInput,
    currentUserContent: input.currentUserContent,
    sourceUserEntries: [persistedUserEntry],
    sourceUserEntryIds: [persistedUserEntry.id],
    senderAttribution: senderAttributionForEntries({
      entries: [
        {
          id: persistedUserEntry.id,
          sender_entity_id: persistedUserEntry.sender_entity_id,
        },
      ],
      entityRepository: input.entityRepository,
    }),
    effectiveSenderEntityId: input.fallbackEffectiveSenderEntityId,
  };
}

export class TurnPhaseCoordinator {
  private readonly contradictionRoutingCooldown = new ContradictionRoutingCooldown();

  constructor(private readonly options: TurnPhaseCoordinatorOptions) {}

  async run(input: RunTurnPhasesInput): Promise<TurnPhaseResult> {
    const rawTurnInput: TurnPhaseCoordinatorInput = {
      ...input.input,
      globalTurnCounter: input.globalTurnCounter,
    };
    const sessionId = input.sessionId;
    const turnId = input.turnId;
    const streamWriter = input.streamWriter;
    const lifecycleTracker = input.lifecycleTracker;
    const appendHookFailure = (
      targetStreamWriter: StreamWriter,
      hook: string,
      error: unknown,
      details?: Record<string, unknown>,
    ) => appendHookFailureEvent(targetStreamWriter, hook, error, details);
    const isUserTurn = isUserTurnOrigin(rawTurnInput.origin);
    const batchEntryIds = validateInboundBatchRequest({
      rawTurnInput,
      isUserTurn,
    });
    const sessionRecord = this.options.sessionsRepository?.get(sessionId) ?? null;
    const effectiveAudience =
      batchEntryIds !== null &&
      rawTurnInput.audience === undefined &&
      sessionRecord?.audience_entity_id !== null &&
      sessionRecord?.audience_entity_id !== undefined
        ? sessionRecord.audience_label
        : rawTurnInput.audience;
    const isSelfAudience = effectiveAudience === "self";

    if (batchEntryIds !== null && rawTurnInput.audience !== undefined && sessionRecord !== null) {
      const expectedAudienceEntityId = sessionRecord.audience_entity_id;
      const inputAudienceEntityId =
        rawTurnInput.audience === "self"
          ? null
          : this.options.entityRepository.findByName(rawTurnInput.audience);

      if (
        expectedAudienceEntityId !== null
          ? inputAudienceEntityId !== expectedAudienceEntityId
          : rawTurnInput.audience !== sessionRecord.audience_label
      ) {
        throw new CognitionError("Inbound batch audience must match its session audience", {
          code: "INBOUND_BATCH_AUDIENCE_MISMATCH",
        });
      }
    }

    const hydratedBatch =
      batchEntryIds === null
        ? null
        : await hydrateInboundBatch({
            options: this.options,
            sessionId,
            rawTurnInput,
            entryIds: batchEntryIds,
          });
    const batchEntries = hydratedBatch?.entries ?? null;
    const preflightAudienceEntityId =
      effectiveAudience === undefined || isSelfAudience
        ? null
        : (sessionRecord?.audience_entity_id ??
          this.options.entityRepository.findByName(effectiveAudience));
    const preflightAudienceEntity =
      preflightAudienceEntityId === null
        ? null
        : this.options.entityRepository.get(preflightAudienceEntityId);
    if (
      preflightAudienceEntity?.kind === "group" &&
      isUserTurn &&
      batchEntries !== null &&
      batchEntries.some(
        (entry) => entry.sender_entity_id === null || entry.sender_entity_id === undefined,
      )
    ) {
      throw new CognitionError("Group-audience inbound batch entries require sender_entity_id", {
        code: "GROUP_BATCH_SENDER_REQUIRED",
      });
    }

    if (
      preflightAudienceEntity?.kind === "group" &&
      isUserTurn &&
      batchEntries === null &&
      (rawTurnInput.senderEntityId === null || rawTurnInput.senderEntityId === undefined)
    ) {
      throw new CognitionError("Group-audience user turns require senderEntityId", {
        code: "GROUP_SENDER_REQUIRED",
      });
    }

    const batchSourceUserEntryIds = hydratedBatch?.entries.map((entry) => entry.id) ?? [];
    const batchSingleNonNullSenderEntityId =
      batchEntries === null ? null : singleNonNullBatchSenderId(batchEntries);
    if (hydratedBatch !== null && this.options.chatResponseWatermarkCoordinator === undefined) {
      throw new CognitionError("Inbound batch turns require chat response watermark coordination", {
        code: "CHAT_RESPONSE_WATERMARK_COORDINATOR_REQUIRED",
      });
    }
    const batchResponseTo =
      hydratedBatch === null
        ? undefined
        : buildStreamBacklogResponseTo({
            coordinator: this.options.chatResponseWatermarkCoordinator!,
            entryIndex: this.options.entryIndex!,
            sessionId,
            sourceEntries: hydratedBatch.sourceEntries,
            records: hydratedBatch.records,
            sourceEntryIds: batchSourceUserEntryIds,
          });
    let currentTurnUserInput = initialCurrentTurnUserInput({
      rawTurnInput,
      batchEntries,
      batchSourceEntries: hydratedBatch?.sourceEntries,
      sessionId,
      entityRepository: this.options.entityRepository,
      responseTo: batchResponseTo,
    });
    const scalarSenderEntityId =
      batchEntries === null
        ? rawTurnInput.senderEntityId
        : (batchSingleNonNullSenderEntityId ?? undefined);
    const turnInput: TurnPhaseInput = {
      userMessage: currentTurnUserInput.renderedText,
      ...(batchEntries !== null ||
      !("attachments" in rawTurnInput) ||
      rawTurnInput.attachments === undefined
        ? {}
        : { attachments: rawTurnInput.attachments }),
      ...(effectiveAudience === undefined ? {} : { audience: effectiveAudience }),
      ...(scalarSenderEntityId === undefined ? {} : { senderEntityId: scalarSenderEntityId }),
      ...(rawTurnInput.stakes === undefined ? {} : { stakes: rawTurnInput.stakes }),
      sessionId,
      ...(rawTurnInput.globalTurnCounter === undefined
        ? {}
        : { globalTurnCounter: rawTurnInput.globalTurnCounter }),
      ...(rawTurnInput.origin === undefined ? {} : { origin: rawTurnInput.origin }),
      ...(rawTurnInput.autonomyTrigger === undefined
        ? {}
        : { autonomyTrigger: rawTurnInput.autonomyTrigger }),
    };

    await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "ingest",
      sub: "pre_turn_catchup",
      run: () =>
        catchUpStreamIngestion({
          coordinator: this.options.streamIngestionCoordinator,
          sessionId,
          streamWriter,
          maxEntries: this.options.config.streamIngestion.preTurnCatchup.maxEntries,
          clampToChatResponseWatermark: currentTurnUserInput.responseTo !== undefined,
          appendHookFailureEvent: appendHookFailure,
        }),
      completedSub: () => "pre_turn_catchup",
    });
    let workingMemory = this.options.workingMemoryStore.load(sessionId);
    lifecycleTracker.captureInitialWorkingMemory(workingMemory);
    const turnPerception = this.options.perceptionGateway.beginTurn({
      turnId,
      onHookFailure: (hook, error, details) =>
        appendHookFailure(streamWriter, hook, error, details),
    });
    const llmClient = this.options.llmFactory();
    const cognitionInput = cognitionInputForTurnInput(turnInput);
    const audienceResolution = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "audience",
      sub: "resolve_profile",
      run: async () => {
        const resolvedAudienceEntityId =
          turnInput.audience === undefined || isSelfAudience
            ? null
            : this.options.entityRepository.resolve(turnInput.audience, {
                provenance: "transport_audience_label",
              });
        const resolvedAudienceEntity =
          resolvedAudienceEntityId === null
            ? null
            : this.options.entityRepository.get(resolvedAudienceEntityId);
        const resolvedAudienceProfile =
          resolvedAudienceEntityId === null
            ? null
            : this.options.socialRepository.getProfile(resolvedAudienceEntityId);

        // In a group channel the social exchange belongs to the current speaker,
        // not to the abstract channel entity. Updating the group too is deferred.
        const resolvedSocialInteractionEntityId =
          resolvedAudienceEntity?.kind === "group"
            ? (turnInput.senderEntityId ?? null)
            : resolvedAudienceEntityId;
        const resolvedGroupSpeakerEntityId =
          resolvedAudienceEntity?.kind === "group" ? (turnInput.senderEntityId ?? null) : null;
        const resolvedGroupSpeakerDisplayName =
          resolvedGroupSpeakerEntityId === null
            ? null
            : (this.options.entityRepository.get(resolvedGroupSpeakerEntityId)?.canonical_name ??
              null);

        return {
          audienceEntityId: resolvedAudienceEntityId,
          audienceEntity: resolvedAudienceEntity,
          audienceProfile: resolvedAudienceProfile,
          socialInteractionEntityId: resolvedSocialInteractionEntityId,
          groupSpeakerEntityId: resolvedGroupSpeakerEntityId,
          groupSpeakerDisplayName: resolvedGroupSpeakerDisplayName,
        };
      },
      completedSub: (result) =>
        `entity=${result.audienceEntityId ?? "self"} kind=${result.audienceEntity?.kind ?? "self"}`,
    });
    const audienceEntityId = audienceResolution.audienceEntityId;
    const audienceEntity = audienceResolution.audienceEntity;
    let audienceProfile = audienceResolution.audienceProfile;
    const socialInteractionEntityId = audienceResolution.socialInteractionEntityId;
    const groupSpeakerEntityId = audienceResolution.groupSpeakerEntityId;
    const groupSpeakerDisplayName = audienceResolution.groupSpeakerDisplayName;
    const sessionAudienceRole = sessionRecord?.audience_role ?? "participant";
    const participationPolicy = sessionRecord?.participation_policy ?? "active";
    const autonomousOutbound =
      turnInput.origin === "autonomous"
        ? (this.options.autonomousOutboundPolicy?.promptContext(sessionId) ?? null)
        : null;
    const currentSenderEntityId =
      audienceEntity?.kind === "group" ? groupSpeakerEntityId : audienceEntityId;
    const effectiveDistinctSenderCount =
      batchEntries === null
        ? currentSenderEntityId === null
          ? 0
          : 1
        : distinctSenderCount(batchEntries);
    const batchAuthoritySenderEntityId =
      batchEntries === null ? null : batchSingleNonNullSenderEntityId;
    const authoritySenderEntityId =
      batchEntries === null ? currentSenderEntityId : batchAuthoritySenderEntityId;
    const authoritySenderEntity =
      authoritySenderEntityId === null
        ? null
        : this.options.entityRepository.get(authoritySenderEntityId);
    const authorityAllowedForSingleSender =
      batchEntries === null || batchAuthoritySenderEntityId !== null;
    const creator = this.options.entityRepository.getCreator();
    const creatorIdentity = creator === null ? null : { displayName: creator.canonical_name };
    const creatorContext = {
      currentSenderEntityId: authoritySenderEntityId,
      currentSenderDisplayName: authoritySenderEntity?.canonical_name ?? null,
      currentSenderBorgRole: authoritySenderEntity?.borg_role ?? null,
      sessionAudienceRole,
    };
    const outboundSourceTypes = new Set(this.options.outboundSourceTypes ?? []);
    // Cross-session AWARENESS (the snapshot) renders only for operator-role
    // turns with single-sender authority. OUTBOUND TARGETING is the separate,
    // narrower concern: a session is targetable only when the sender is a
    // creator-in-operator AND a connector is wired for its source_type. Only
    // targetable sessions expose their session_id to the model, so non-outbound
    // operator turns keep the id-free awareness view.
    const snapshotRepository =
      sessionAudienceRole === "operator" && authorityAllowedForSingleSender
        ? this.options.sessionsRepository
        : undefined;
    const activeOtherSessions =
      snapshotRepository === undefined
        ? []
        : snapshotRepository.list({
            status: "active",
            excludeSessionId: sessionId,
            limit: OPERATOR_SESSION_SNAPSHOT_CAP,
          });
    const totalActiveOtherSessionCount =
      snapshotRepository === undefined
        ? 0
        : snapshotRepository.count({ status: "active", excludeSessionId: sessionId });
    const outboundTargetableSessionIds: ReadonlySet<SessionId> =
      isCreatorInOperatorContext({
        currentSenderBorgRole: creatorContext.currentSenderBorgRole,
        sessionAudienceRole,
      }) && outboundSourceTypes.size > 0
        ? new Set<SessionId>(
            activeOtherSessions
              .filter((session) => outboundSourceTypes.has(session.source_type))
              .map((session) => session.session_id),
          )
        : new Set<SessionId>();
    const operatorSessionSnapshot =
      activeOtherSessions.length > 0
        ? buildOperatorSessionSnapshot({
            sessions: activeOtherSessions,
            totalActiveOtherSessionCount,
            currentSessionId: sessionId,
            nowMs: this.options.clock.now(),
            cap: OPERATOR_SESSION_SNAPSHOT_CAP,
            outboundTargetableSessionIds,
          })
        : null;
    const perceptionResult = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "perception",
      run: () =>
        turnPerception.perceive({
          sessionId,
          isSelfAudience,
          origin: turnInput.origin,
          cognitionInput,
          workingMemory,
          recencyBeforeEntryIdExclusive: currentTurnUserInput.recencyBeforeEntryIdExclusive,
        }),
      completedSub: summarizePerceptionResult,
    });
    const perception = perceptionResult.perception;
    if (persistsPerception(turnInput.origin)) {
      for (const userIdentityName of perception.userIdentityNames ?? []) {
        this.options.entityRepository.resolve(userIdentityName, {
          kind: "person",
          provenance: "user_declared",
        });
      }
    }
    const recencyWindow = perceptionResult.recencyWindow;
    const workingMood = perceptionResult.workingMood;
    workingMemory = perceptionResult.workingMemory;
    const suppressionSet = SuppressionSet.fromEntries(
      workingMemory.suppressed,
      workingMemory.turn_counter,
    );
    const attributionResult = await this.options.attributionLifecycleService.settle({
      isUserTurn,
      audienceEntityId,
      socialEntityId: socialInteractionEntityId,
      perception,
      pendingSocialAttribution: workingMemory.pending_social_attribution,
      pendingTraitAttribution: workingMemory.pending_trait_attribution,
      audienceProfile,
      streamWriter,
      onHookFailure: (hook, error) => appendHookFailure(streamWriter, hook, error),
    });
    const pendingSocialAttribution = attributionResult.pendingSocialAttribution;
    const pendingTraitAttribution = attributionResult.pendingTraitAttribution;
    audienceProfile = attributionResult.audienceProfile;

    const openingPersistence = await this.options.turnOpeningPersistence.persist({
      streamWriter,
      turnId,
      userMessage: turnInput.userMessage,
      attachments: turnInput.attachments,
      persistAttachments: (attachmentInput) =>
        this.options.attachmentService.persistTurnAttachments({
          ...attachmentInput,
          createdTurnGlobal: input.globalTurnCounter,
        }),
      persistUserMessage: isUserTurn && currentTurnUserInput.persistUserMessage,
      persistPerception: persistsPerception(turnInput.origin),
      audience: turnInput.audience,
      senderEntityId: turnInput.senderEntityId,
      speakerEntityId: currentSenderEntityId,
      audienceEntityId,
      workingMemory,
      pendingSocialAttribution,
      pendingTraitAttribution,
      suppressionSet,
      perception,
      now: () => this.options.clock.now(),
    });
    const persistedUserEntry = openingPersistence.persistedUserEntry;
    const persistedUserEntryId = persistedUserEntry?.id;
    const persistedPerceptionEntry = openingPersistence.persistedPerceptionEntry;
    workingMemory = openingPersistence.workingMemory;
    currentTurnUserInput = withPersistedCurrentUserEntry({
      currentUserInput: currentTurnUserInput,
      persistedUserEntry,
      currentUserContent: openingPersistence.currentUserContent,
      entityRepository: this.options.entityRepository,
      fallbackEffectiveSenderEntityId: currentSenderEntityId,
    });
    const sourceUserEntries = currentTurnUserInput.sourceUserEntries;
    const sourceUserEntryIds = currentTurnUserInput.sourceUserEntryIds;
    const currentUserContent = currentTurnUserInput.currentUserContent;
    const currentTurnAttachmentIds: AttachmentId[] = [
      ...openingPersistence.persistedAttachments.map((attachment) => attachment.attachmentId),
      ...(batchEntries ?? []).flatMap((entry) =>
        (entry.attachments ?? []).map((attachment) => attachment.attachment_id),
      ),
    ];

    for (const attachment of openingPersistence.persistedAttachments) {
      await this.options.imagePerceptionService?.perceiveAttachment({
        attachmentId: attachment.attachmentId,
        turnId,
      });
    }

    // Audience tracing happens before perception because that is the clean
    // resolution boundary. Group participant rostering stays here: it relies
    // on current-turn persistence and feeds the frame/extraction context.
    const activeParticipantLimit = this.options.config.generation.activeParticipantLimit;
    const participantScan =
      audienceEntity?.kind === "group"
        ? scanRecentParticipantStreamEntries(
            this.options.createStreamReader(sessionId),
            activeParticipantLimit,
          )
        : null;

    if (
      participantScan !== null &&
      participantScan.capReached !== null &&
      participantScan.foundUniqueParticipants < activeParticipantLimit &&
      this.options.tracer.enabled
    ) {
      this.options.tracer.emit("participant_scan.skipped", {
        turnId,
        session_id: sessionId,
        reason: "cap_reached",
        cap: participantScan.capReached,
        scanned_entries: participantScan.scannedEntries,
        scanned_bytes: participantScan.scannedBytes,
        found_unique_participants: participantScan.foundUniqueParticipants,
        requested_limit: activeParticipantLimit,
      });
    }

    const activeParticipants = resolveActiveParticipants({
      audienceEntityId,
      senderEntityId: turnInput.senderEntityId ?? null,
      streamEntries: participantScan?.entries ?? [],
      entityRepository: this.options.entityRepository,
      limit: activeParticipantLimit,
    });
    let participantRoster = buildParticipantRosterFromRepositories({
      activeParticipants,
      audienceEntityId,
      entityRepository: this.options.entityRepository,
      relationalSlotRepository: this.options.relationalSlotRepository,
    });
    const participantProfiles = resolveParticipantProfiles(
      activeParticipants,
      this.options.socialRepository,
    );
    if (activeParticipants.length > 0) {
      audienceProfile = audienceProfileForParticipants(participantProfiles, audienceEntityId);
    }

    const frameAnomalyConversationContext = buildFrameAnomalyConversationContext({
      audienceEntityId,
      audienceEntity,
      currentUserEntry: persistedUserEntry ?? sourceUserEntries[0],
      activeParticipants,
      participantStreamEntries: participantScan?.entries ?? [],
      entityRepository: this.options.entityRepository,
      currentSenderEntityId,
      currentSenderBorgRole: creatorContext.currentSenderBorgRole,
      sessionAudienceRole,
    });
    const frameAnomalyPhase = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "frame",
      run: () =>
        classifyFrameAnomalyPhase({
          options: this.options,
          appendHookFailureEvent: appendHookFailure,
          llmClient,
          turnId,
          sessionId,
          isUserTurn,
          userMessage: turnInput.userMessage,
          recentHistory: recencyWindow.messages,
          conversationContext: frameAnomalyConversationContext,
          currentSenderBorgRole: creatorContext.currentSenderBorgRole,
          sessionAudienceRole,
          sessionSourceType: sessionRecord?.source_type ?? null,
          persistedUserEntryId,
          sourceUserEntryIds,
          streamWriter,
        }),
      completedSub: summarizeFrameClassification,
    });
    const currentTurnFrameAnomaly = frameAnomalyPhase.actionableFrameAnomaly;
    if (
      frameAnomalyPhase.disposition === "quarantine" &&
      frameAnomalyPhase.actionableFrameAnomaly !== null
    ) {
      const observedEventEmission = buildObservedEventEmission({
        occurredAt: this.options.clock.now(),
        sessionId,
        disposition: frameAnomalyPhase.disposition,
        actionableFrameAnomaly: frameAnomalyPhase.actionableFrameAnomaly,
        speakerEntityId: currentSenderEntityId,
        audienceEntityId,
        sourceUserEntryIds,
      });

      if (observedEventEmission !== null) {
        try {
          this.options.observedEventRepository?.record(observedEventEmission);
        } catch (error) {
          await appendHookFailure(streamWriter, "observed_event_emission", error, {
            turnId,
            kind: frameAnomalyPhase.actionableFrameAnomaly.kind,
            sourceStreamEntryIds: [...sourceUserEntryIds],
          });
        }
      }
    }

    const extraction = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "extract",
      run: () =>
        runExtractionPhase({
          options: this.options,
          appendHookFailureEvent: appendHookFailure,
          llmClient,
          turnId,
          sessionId,
          turnInput,
          isUserTurn,
          cognitionInput,
          perception,
          workingMemory,
          recentHistory: recencyWindow.messages,
          audienceEntityId,
          groupSpeakerEntityId,
          groupSpeakerDisplayName,
          currentSenderEntityId: authoritySenderEntityId,
          currentSenderDisplayName: creatorContext.currentSenderDisplayName,
          currentSenderBorgRole: creatorContext.currentSenderBorgRole,
          sessionAudienceRole,
          participantRoster,
          persistedUserEntryId,
          sourceUserEntryIds,
          senderAttribution: currentTurnUserInput.senderAttribution,
          distinctSenderCount: effectiveDistinctSenderCount,
          currentTurnFrameAnomaly,
          streamWriter,
          trackAppliedSlotNegation: (slot) => lifecycleTracker.trackAppliedSlotNegation(slot),
        }),
      completedSub: summarizeExtraction,
    });
    const correctiveCommitment = extraction.correctiveCommitment;
    const correctiveCommitmentSupersession = extraction.correctiveCommitmentSupersession;
    const correctiveCommitmentRetirement = extraction.correctiveCommitmentRetirement;
    workingMemory = extraction.workingMemory;
    participantRoster = buildParticipantRosterFromRepositories({
      activeParticipants,
      audienceEntityId,
      entityRepository: this.options.entityRepository,
      relationalSlotRepository: this.options.relationalSlotRepository,
    });
    lifecycleTracker.trackCreatedActionIds(extraction.createdActionIds);
    lifecycleTracker.trackCreatedGoalIds(extraction.persistedPromotions.goalIds);
    lifecycleTracker.trackCreatedExecutiveStepIds(extraction.persistedPromotions.executiveStepIds);

    const closureLoopAssessment = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "closure_loop",
      run: () =>
        classifyClosureLoopPhase({
          options: this.options,
          appendHookFailureEvent: appendHookFailure,
          llmClient,
          turnId,
          sessionId,
          isUserTurn,
          userMessage: turnInput.userMessage,
          recentHistory: recencyWindow.messages,
          persistedUserEntryId,
          sourceUserEntryIds,
          sourceUserEntries,
          workingMemory,
          streamWriter,
        }),
      completedSub: (result) =>
        result === null
          ? "skipped"
          : result.closureLoopDetected === true
            ? "closure-loop detected"
            : result.currentUserSubstantive === true
              ? "substantive"
              : result.currentUserClosureShaped === true
                ? "closure-shaped"
                : "no-op",
    });

    if (closureLoopAssessment?.currentUserSubstantive === true) {
      workingMemory = this.options.discourseStateService.clearClosureLoop({
        workingMemory,
        reason: closureLoopAssessment.reason,
        turnId,
        sessionId,
      });
      workingMemory = this.options.discourseStateService.clearStopState({
        workingMemory,
        reason: `Closure-loop classifier marked the current turn substantive: ${closureLoopAssessment.reason}`,
        turnId,
        sessionId,
      });
    } else if (
      closureLoopAssessment?.currentUserClosureShaped === true &&
      workingMemory.discourse_state?.closure_loop?.status === "named"
    ) {
      return suppressFromClosureLoopPhase({
        turnId,
        sessionId,
        turnInput,
        streamWriter,
        appendHookFailureEvent: appendHookFailure,
        options: this.options,
        workingMemory,
        persistedUserEntryId,
        sourceUserEntryIds,
        responseTo: currentTurnUserInput.responseTo,
        correctiveCommitment,
        correctiveCommitmentSupersession,
        correctiveCommitmentRetirement,
        perceptionMode: perception.mode,
        reason: closureLoopAssessment.reason,
      });
    } else if (closureLoopAssessment?.closureLoopDetected === true) {
      workingMemory = this.options.discourseStateService.setClosureLoopDetected({
        workingMemory,
        sourceStreamEntryIds: closureLoopAssessment.sourceStreamEntryIds,
        reason: closureLoopAssessment.reason,
        turnId,
        sessionId,
      });
    }

    const generationGate = new GenerationGate({
      llmClient,
      embeddingClient: this.options.embeddingClient,
      model: this.options.config.anthropic.models.background,
      hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
      onDegraded: (reason, error) =>
        appendHookFailure(streamWriter, "generation_gate", error ?? reason, {
          reason,
        }),
    });
    const gateResult: GenerationGateResult = isUserTurn
      ? await traceTurnPhase({
          tracer: this.options.tracer,
          clock: this.options.clock,
          turnId,
          sessionId,
          phase: "generation_gate",
          run: () =>
            generationGate.evaluate({
              userMessage: turnInput.userMessage,
              workingMemory,
              recencyMessages: recencyWindow.messages,
            }),
          completedSub: (result) =>
            result.action === "suppress" ? `suppress: ${result.explanation ?? ""}` : "allow",
        })
      : {
          action: "proceed",
          explanation: "non-user turn: generation gate not applicable",
          clearDiscourseStop: false,
          classified: false,
          signals: ZEROED_SIGNALS,
        };

    if (gateResult.signals.hardCapDue) {
      await this.options.discourseStateService.appendHardCapEvent({
        streamWriter,
        turnId,
        sessionId,
        activeTurns: gateResult.signals.hardCapActiveTurns,
        hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
        stateReason:
          workingMemory.discourse_state?.stop_until_substantive_content?.reason ?? "unknown",
      });
    }

    if (gateResult.clearDiscourseStop) {
      workingMemory = this.options.discourseStateService.clearStopState({
        workingMemory,
        reason: gateResult.explanation,
        turnId,
        sessionId,
      });
    }

    if (gateResult.action === "suppress") {
      return suppressFromGenerationGatePhase({
        turnId,
        sessionId,
        turnInput,
        streamWriter,
        appendHookFailureEvent: appendHookFailure,
        options: this.options,
        workingMemory,
        persistedUserEntryId,
        sourceUserEntryIds,
        responseTo: currentTurnUserInput.responseTo,
        gateResult,
        correctiveCommitment,
        correctiveCommitmentSupersession,
        correctiveCommitmentRetirement,
        perceptionMode: perception.mode,
      });
    }

    const retrievalPhase = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "retrieval",
      run: () =>
        runRetrievalPhase({
          options: this.options,
          sessionId,
          turnId,
          turnInput,
          isSelfAudience,
          isUserTurn,
          cognitionInput,
          llmClient,
          recencyMessages: recencyWindow.messages,
          audienceEntityId,
          audienceEntity,
          currentSenderBorgRole: creatorContext.currentSenderBorgRole,
          operatorOnlyDirectivesAllowed: authorityAllowedForSingleSender,
          audienceProfile,
          sessionAudienceRole,
          perception,
          workingMemory,
          suppressionSet,
          actionLinkSelfContext: extraction.actionLinkSelfContext,
          persistedPromotions: extraction.persistedPromotions,
          correctiveCommitment,
          activeParticipants,
          participantRoster,
          participantProfiles,
          persistedUserEntry: persistedUserEntry ?? undefined,
          currentUserEntries: currentTurnUserInput.persistUserMessage
            ? undefined
            : sourceUserEntries,
          currentTurnAttachmentIds,
          currentTurnFrameAnomaly,
          closureLoopAssessment,
        }),
      completedSub: summarizeRetrieval,
    });
    const deliberationPhase = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "delib",
      run: () =>
        runDeliberationPhase({
          options: this.options,
          llmClient,
          sessionId,
          turnId,
          turnInput,
          streamWriter,
          isSelfAudience,
          audienceEntityId,
          participationPolicy,
          creatorIdentity,
          creatorContext,
          autonomousOutbound,
          operatorSessionSnapshot,
          persistedUserEntryId,
          sourceUserEntryIds,
          currentUserContent,
          perception,
          workingMemory,
          activeParticipants,
          participantRoster,
          participantProfiles,
          audienceProfile,
          recencyMessages: recencyWindow.messages,
          currentTurnFrameAnomaly,
          retrievalPhase,
          contradictionRoutingCooldown: this.contradictionRoutingCooldown,
        }),
      completedSub: summarizeDeliberation,
    });
    const deliberation = deliberationPhase.deliberation;
    workingMemory = deliberationPhase.workingMemory;
    const knownInternalIdentifiers = [
      ...(operatorSessionSnapshot?.sessions.map((session) => session.session_id) ?? []),
      ...(autonomousOutbound?.targets.map((target) => target.session_id) ?? []),
    ];

    return runPostGenerationPhase({
      options: this.options,
      appendHookFailureEvent: appendHookFailure,
      llmClient,
      sessionId,
      sessionSourceType: sessionRecord?.source_type ?? null,
      sessionAudienceRole,
      turnId,
      turnInput,
      streamWriter,
      lifecycleTracker,
      cognitionInput,
      perception,
      workingMemory,
      workingMood,
      persistedUserEntry: persistedUserEntry ?? undefined,
      sourceUserEntries,
      persistedPerceptionEntry,
      persistedUserEntryId,
      sourceUserEntryIds,
      senderAttribution: currentTurnUserInput.senderAttribution,
      responseTo: currentTurnUserInput.responseTo,
      correctiveCommitment,
      correctiveCommitmentSupersession,
      correctiveCommitmentRetirement,
      deliberation,
      origin: turnInput.origin,
      autonomyTrigger: turnInput.autonomyTrigger,
      retrievalPhase,
      closureLoopCurrentUserAct: closureLoopAssessment?.currentUserAct ?? null,
      audienceEntityId,
      audienceIsGroup: audienceEntity?.kind === "group",
      senderEntityId: turnInput.senderEntityId ?? null,
      socialInteractionEntityId,
      pendingSocialAttribution,
      suppressionSet,
      isUserTurn,
      currentTurnFrameAnomaly,
      closureLoopAssessment,
      activeParticipants,
      knownInternalIdentifiers,
    });
  }
}
