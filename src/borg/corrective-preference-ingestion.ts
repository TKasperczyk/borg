import { CorrectivePreferenceExtractorDegradedError } from "../cognition/commitments/corrective-preference-extractor.js";
import type { CorrectivePreferenceTurnService } from "../cognition/commitments/corrective-preference-service.js";
import { resolveActiveParticipants } from "../cognition/participants.js";
import { buildParticipantRosterFromRepositories } from "../cognition/perception/index.js";
import { TurnContextCompiler } from "../cognition/recency/index.js";
import type { LLMClient } from "../llm/index.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import { getBudgetErrorTokens, withBudget } from "../offline/budget.js";
import {
  StreamReader,
  filterActiveStreamEntries,
  type StreamEntry,
  type StreamEntryIndexRepository,
} from "../stream/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { BudgetExceededError } from "../util/errors.js";
import type { SessionId } from "../util/ids.js";

const CORRECTIVE_PREFERENCE_BUDGET_NAME = "corrective-preference-extractor";
const DEFAULT_MAX_ENTRY_FAILURES = 3;
const MAX_RECEIPT_ERROR_LENGTH = 2_048;

function formatReceiptError(error: unknown): string {
  const rendered =
    error instanceof Error ? `${error.name}: ${error.message}` : `NonError: ${String(error)}`;

  return rendered.slice(0, MAX_RECEIPT_ERROR_LENGTH);
}

function isBudgetExceededFailure(error: unknown, seen = new Set<unknown>()): boolean {
  if (seen.has(error)) {
    return false;
  }

  seen.add(error);

  if (error instanceof BudgetExceededError) {
    return true;
  }

  if (error instanceof CorrectivePreferenceExtractorDegradedError) {
    return isBudgetExceededFailure(error.degradationCause, seen);
  }

  if (error instanceof Error && "cause" in error) {
    return isBudgetExceededFailure(error.cause, seen);
  }

  return false;
}

export type CorrectivePreferenceIngestionOptions = {
  service: Pick<CorrectivePreferenceTurnService, "extractAndApply" | "persistCommitment">;
  llmClient: LLMClient;
  dataDir: string;
  entryIndex: Pick<
    StreamEntryIndexRepository,
    | "lookup"
    | "getCorrectivePreferenceIngestionReceipt"
    | "recordCorrectivePreferenceIngestionProcessed"
    | "recordCorrectivePreferenceIngestionFailure"
  >;
  entityRepository: Pick<EntityRepository, "get" | "findByName">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list">;
  tracer: TurnTracer;
  budget: number | null;
  clock?: Clock;
  logger?: Pick<Console, "error">;
  maxEntryFailures?: number;
  onHookFailure: (
    sessionId: SessionId,
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ) => Promise<void>;
};

type IngestionOutcome =
  | "corrective_preference"
  | "retire_commitment"
  | "none"
  | "skipped"
  | "failed";

export class CorrectivePreferenceIngestion {
  private readonly contextCompiler = new TurnContextCompiler();
  private readonly clock: Clock;
  private readonly logger: Pick<Console, "error">;
  private readonly maxEntryFailures: number;

  constructor(private readonly options: CorrectivePreferenceIngestionOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.logger = options.logger ?? console;
    this.maxEntryFailures = options.maxEntryFailures ?? DEFAULT_MAX_ENTRY_FAILURES;

    if (!Number.isInteger(this.maxEntryFailures) || this.maxEntryFailures < 1) {
      throw new Error("maxEntryFailures must be a positive integer");
    }
  }

  async process(input: { sessionId: SessionId; entries: readonly StreamEntry[] }): Promise<void> {
    for (const entry of filterActiveStreamEntries(input.entries)) {
      if (entry.kind !== "user_msg") {
        continue;
      }

      const existingReceipt =
        this.options.entryIndex.getCorrectivePreferenceIngestionReceipt(entry.id);

      if (
        existingReceipt?.status === "processed" ||
        existingReceipt?.status === "dead_letter"
      ) {
        continue;
      }

      try {
        await this.processUserEntry(input.sessionId, entry);
        this.options.entryIndex.recordCorrectivePreferenceIngestionProcessed({
          sourceEntryId: entry.id,
          sessionId: input.sessionId,
          updatedAt: this.clock.now(),
        });
      } catch (error) {
        const budgetExceeded = isBudgetExceededFailure(error);
        const receipt = this.options.entryIndex.recordCorrectivePreferenceIngestionFailure({
          sourceEntryId: entry.id,
          sessionId: input.sessionId,
          error: formatReceiptError(error),
          updatedAt: this.clock.now(),
          maxFailures: this.maxEntryFailures,
          // The cap is checked after provider usage is reported. Retrying the
          // same configured overflow would only spend the same tokens again,
          // so budget overflow is immediately and loudly dead-lettered.
          deadLetterImmediately: budgetExceeded,
        });

        if (receipt.status === "processed") {
          // A concurrent worker committed the durable success receipt first.
          continue;
        }

        if (receipt.status === "dead_letter") {
          this.traceFailureDisposition({
            sessionId: input.sessionId,
            entry,
            event: "corrective_preference.ingestion.dead_lettered",
            failureCount: receipt.failure_count,
            error: receipt.last_error ?? formatReceiptError(error),
            reason: budgetExceeded ? "budget_exceeded" : "retry_ceiling_reached",
          });
          this.logger.error("Corrective-preference ingestion entry dead-lettered", {
            session_id: input.sessionId,
            source_stream_entry_id: entry.id,
            failure_count: receipt.failure_count,
            reason: budgetExceeded ? "budget_exceeded" : "retry_ceiling_reached",
            error: receipt.last_error ?? formatReceiptError(error),
          });
          continue;
        }

        this.traceFailureDisposition({
          sessionId: input.sessionId,
          entry,
          event: "corrective_preference.ingestion.retry_scheduled",
          failureCount: receipt.failure_count,
          error: receipt.last_error ?? formatReceiptError(error),
          reason: "retryable_failure",
        });
        throw error;
      }
    }
  }

  private traceFailureDisposition(input: {
    sessionId: SessionId;
    entry: StreamEntry;
    event:
      | "corrective_preference.ingestion.retry_scheduled"
      | "corrective_preference.ingestion.dead_lettered";
    failureCount: number;
    error: string;
    reason: "retryable_failure" | "retry_ceiling_reached" | "budget_exceeded";
  }): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    this.options.tracer.emit(input.event, {
      turnId: `sidecar_ingestion:${input.entry.id}`,
      session_id: input.sessionId,
      source_stream_entry_id: input.entry.id,
      failure_count: input.failureCount,
      max_failures: this.maxEntryFailures,
      reason: input.reason,
      error: input.error,
    });
  }

  private traceOutcome(input: {
    sessionId: SessionId;
    entry: StreamEntry;
    audienceEntityId: StreamEntry["sender_entity_id"];
    outcome: IngestionOutcome;
    reason?: string;
    tokensUsed: number;
    commitmentId?: string;
    retiredCommitmentId?: string;
  }): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    this.options.tracer.emit("corrective_preference.ingestion.completed", {
      turnId: `sidecar_ingestion:${input.entry.id}`,
      session_id: input.sessionId,
      source_stream_entry_id: input.entry.id,
      audience_entity_id: input.audienceEntityId ?? null,
      outcome: input.outcome,
      ...(input.reason === undefined ? {} : { reason: input.reason }),
      tokens_used: input.tokensUsed,
      budget: this.options.budget,
      budget_exhausted: this.options.budget !== null && input.tokensUsed > this.options.budget,
      ...(input.commitmentId === undefined ? {} : { commitment_id: input.commitmentId }),
      ...(input.retiredCommitmentId === undefined
        ? {}
        : { retired_commitment_id: input.retiredCommitmentId }),
    });
  }

  private async processUserEntry(sessionId: SessionId, entry: StreamEntry): Promise<void> {
    const senderEntityId = entry.sender_entity_id ?? null;

    if (senderEntityId === null) {
      this.traceOutcome({
        sessionId,
        entry,
        audienceEntityId: null,
        outcome: "skipped",
        reason: "missing_sender_entity_id",
        tokensUsed: 0,
      });
      return;
    }

    const sender = this.options.entityRepository.get(senderEntityId);

    if (sender === null || sender.kind !== "person") {
      this.traceOutcome({
        sessionId,
        entry,
        audienceEntityId: senderEntityId,
        outcome: "skipped",
        reason: sender === null ? "unknown_sender_entity_id" : "sender_entity_not_person",
        tokensUsed: 0,
      });
      return;
    }

    if (typeof entry.content !== "string" || entry.content.trim().length === 0) {
      this.traceOutcome({
        sessionId,
        entry,
        audienceEntityId: senderEntityId,
        outcome: "skipped",
        reason: "unsupported_user_content",
        tokensUsed: 0,
      });
      return;
    }

    const userMessage = entry.content;

    const turnId = `sidecar_ingestion:${entry.id}`;
    const recentHistory = this.contextCompiler.compile(
      new StreamReader({
        dataDir: this.options.dataDir,
        sessionId,
        entryIndex: this.options.entryIndex,
      }),
      { beforeEntryIdExclusive: entry.id },
    ).messages;
    const activeParticipants = resolveActiveParticipants({
      audienceEntityId: senderEntityId,
      senderEntityId,
      streamEntries: [],
      entityRepository: this.options.entityRepository,
    });
    const participantRoster = buildParticipantRosterFromRepositories({
      activeParticipants,
      audienceEntityId: senderEntityId,
      entityRepository: this.options.entityRepository,
      relationalSlotRepository: this.options.relationalSlotRepository,
    });

    // Sidecar audience identity is deliberately the stamped sender person, not
    // a session group entity. Consequently the cognition path's recent
    // group-participant scan has no group audience to expand here. The sender
    // remains a participant (not operator), and cross-audience targets stay
    // disabled so one human's rule cannot be rebound onto another human.

    try {
      const budgeted = await withBudget(
        CORRECTIVE_PREFERENCE_BUDGET_NAME,
        this.options.budget,
        async ({ wrapClient }) => {
          const result = await this.options.service.extractAndApply({
            llmClient: wrapClient(this.options.llmClient),
            turnId,
            isUserTurn: true,
            userMessage,
            persistedUserEntryId: entry.id,
            sourceUserEntryIds: [entry.id],
            recentHistory,
            audienceEntityId: senderEntityId,
            committedByEntityId: senderEntityId,
            currentSenderEntityId: senderEntityId,
            currentSenderBorgRole: sender.borg_role,
            sessionAudienceRole: "participant",
            speakerDisplayName: sender.canonical_name,
            participantRoster,
            relationshipEvidenceStreamEntries: [entry],
            crossAudienceTargeting: {
              allowed: false,
              candidateAudiences: [],
            },
            sessionId,
            onHookFailure: (hook, error, details) =>
              this.options.onHookFailure(sessionId, hook, error, details),
            trackAppliedSlotNegation: () => undefined,
          });

          await this.options.service.persistCommitment({
            commitment: result.commitment,
            supersession: result.commitmentSupersession,
            retirement: result.commitmentRetirement,
            turnId,
            sessionId,
            onHookFailure: (hook, error, details) =>
              this.options.onHookFailure(sessionId, hook, error, details),
          });

          return result;
        },
      );
      const result = budgeted.result;
      const outcome: IngestionOutcome =
        result.commitment !== null
          ? "corrective_preference"
          : result.commitmentRetirement !== null
            ? "retire_commitment"
            : "none";

      this.traceOutcome({
        sessionId,
        entry,
        audienceEntityId: senderEntityId,
        outcome,
        tokensUsed: budgeted.tokens_used,
        ...(result.commitment === null ? {} : { commitmentId: result.commitment.id }),
        ...(result.commitmentRetirement === null
          ? {}
          : { retiredCommitmentId: result.commitmentRetirement.retiredId }),
      });
    } catch (error) {
      this.traceOutcome({
        sessionId,
        entry,
        audienceEntityId: senderEntityId,
        outcome: "failed",
        reason: "ingestion_failed",
        tokensUsed: getBudgetErrorTokens(error),
      });
      throw error;
    }
  }
}
