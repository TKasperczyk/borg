import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { PostGenerationGuardMode } from "../../config/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { ClosureLoopState, ClosurePressureHistoryEntry } from "../../memory/working/index.js";
import type { SessionId } from "../../util/ids.js";
import type { TurnTraceData, TurnTracer } from "../../tracing/tracer.js";
import type { ClosureLoopDialogueAct } from "./closure-loop.js";
import type { PendingTurnEmission } from "./types.js";

export const CLOSURE_RESPONSE_AUDIT_TOOL_NAME = "EmitClosureResponseAudit";

export const CLOSURE_RESPONSE_SPAN_KINDS = [
  "imperative_closer",
  "aphoristic_valediction",
  "quotable_closing_tail",
] as const;

export const CLOSURE_RESPONSE_SHAPES = ["no_closure", "mixed", "closure_only"] as const;

export const CLOSURE_PRESSURE_HISTORY_ACTIVE_TTL_MS = 600_000;
export const CLOSURE_PRESSURE_HISTORY_ACTIVE_TURN_WINDOW = 5;

const closureResponseSpanSchema = z
  .object({
    text: z.string().min(1),
    kind: z.enum(CLOSURE_RESPONSE_SPAN_KINDS),
    rationale: z.string().min(1).optional().default("closure-function span"),
  })
  .strict();

const closureResponseAuditSchema = z
  .object({
    spans: z.array(closureResponseSpanSchema).default([]),
    response_shape: z.enum(CLOSURE_RESPONSE_SHAPES),
    reason: z.string().min(1),
  })
  .strict();

const CLOSURE_RESPONSE_AUDIT_TOOL = {
  name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
  description:
    "Classify closure-function spans in a just-generated assistant response without rewriting it.",
  inputSchema: toToolInputSchema(closureResponseAuditSchema),
} satisfies LLMToolDefinition;

const CLOSURE_RESPONSE_AUDIT_SYSTEM_PROMPT = [
  "You audit a just-generated assistant response for closure-pressure spans.",
  "A closure-pressure span pushes the user toward ending, pausing, sleeping, leaving, validating a terminal beat, or converting an open transition into a quotable closing tag.",
  "Classify spans only in the candidate response. Do not classify the user's message.",
  "Look for the conversational function of the span, not any fixed phrase list.",
  "Closure shapes include imperative send-offs, aphoristic or valedictory tails, ritual holding phrases, and taglines that make the exchange feel complete rather than answer the user's substantive need.",
  "Return spans as exact text copied from the response.",
  'Set response_shape to "no_closure" when no span has closure function.',
  'Set response_shape to "mixed" when the response has substantive content plus one or more closure-function spans.',
  'Set response_shape to "closure_only" when removing closure-function spans would leave no substantive content.',
  "Do not treat ordinary concise substantive answers as closure-pressure just because they are short.",
].join("\n");

export type ClosureResponseSpan = z.infer<typeof closureResponseSpanSchema>;
export type ClosureResponseAudit = z.infer<typeof closureResponseAuditSchema>;

type ClosurePressureGuardEmission = Extract<
  PendingTurnEmission,
  { kind: "message" | "suppressed" }
>;

export type ClosurePressureGuardResult = {
  emission: ClosurePressureGuardEmission;
  verdict: "passed" | "suppressed";
  removed_spans: string[];
  active_closure_commitments: string[];
  reason: string;
  audit: ClosureResponseAudit | null;
};

export type ClosurePressureGuardOptions = {
  llmClient: LLMClient;
  auditModel: string;
  mode?: PostGenerationGuardMode;
  tracer?: TurnTracer;
};

export type ClosurePressureGuardInput = {
  turnId: string;
  sessionId?: SessionId;
  response: string;
  activeCommitments: readonly CommitmentRecord[];
  closureLoop: ClosureLoopState | null;
  closurePressureHistory?: readonly ClosurePressureHistoryEntry[];
  currentUserClosureKind?: ClosureLoopDialogueAct | null;
  currentTurn?: number;
  nowMs?: number;
};

function parseAuditResponse(input: unknown): ClosureResponseAudit {
  const parsed = closureResponseAuditSchema.safeParse(input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
}

function closureAuditStructuredError(error: unknown): unknown {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return new Error(`Closure response auditor did not emit ${CLOSURE_RESPONSE_AUDIT_TOOL_NAME}`);
  }

  return isStructuredToolCallError(error) ? (error.cause ?? error) : error;
}

function buildAuditMessages(response: string): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        response,
      }),
    },
  ];
}

function activeClosureCommitmentFamilies(
  commitments: readonly CommitmentRecord[],
): CommitmentRecord[] {
  return commitments.filter((commitment) => commitment.closure_pressure_relevance === "no_closure");
}

function activeClosureCommitmentLabels(commitments: readonly CommitmentRecord[]): string[] {
  return commitments.map((commitment) => `${commitment.id}:${commitment.directive_family}`);
}

function activeClosurePressureHistoryEntries(input: {
  entries: readonly ClosurePressureHistoryEntry[];
  nowMs?: number;
  currentTurn?: number;
}): ClosurePressureHistoryEntry[] {
  return input.entries.filter((entry) => {
    const withinTime =
      input.nowMs === undefined ||
      Math.max(0, input.nowMs - entry.ts) <= CLOSURE_PRESSURE_HISTORY_ACTIVE_TTL_MS;
    const withinTurns =
      input.currentTurn === undefined ||
      entry.turn === undefined ||
      Math.max(0, input.currentTurn - entry.turn) <= CLOSURE_PRESSURE_HISTORY_ACTIVE_TURN_WINDOW;

    return withinTime && withinTurns;
  });
}

export function shouldEnforceClosurePressure(input: {
  activeClosureCommitments: readonly CommitmentRecord[];
  closureLoop: ClosureLoopState | null;
  closureHistoryActive: boolean;
}): boolean {
  return (
    input.closureHistoryActive ||
    input.activeClosureCommitments.length > 0 ||
    input.closureLoop?.status === "named"
  );
}

function traceClosureGuard(input: {
  tracer?: TurnTracer;
  turnId: string;
  sessionId?: SessionId;
  verdict: "passed" | "suppressed";
  mode?: PostGenerationGuardMode;
  wouldHaveVerdict?: "passed" | "suppressed";
  wouldHaveSuppressionReason?: string;
  removedSpans: readonly string[];
  activeClosureCommitments: readonly string[];
  reason: string;
  audit: ClosureResponseAudit | null;
  originalResponse?: string;
  auditError?: string;
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  const includePayloads = input.tracer.includePayloads === true;
  const mode = input.mode ?? "enforce";
  const wouldHaveVerdict = input.wouldHaveVerdict ?? input.verdict;
  const payload: TurnTraceData = {
    turnId: input.turnId,
    ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
    mode,
    verdict: mode === "shadow" ? "passed" : input.verdict,
    wouldHaveVerdict,
    removed_spans: [...input.removedSpans],
    active_closure_commitments: [...input.activeClosureCommitments],
    reason: input.reason,
    spans_detected: input.audit?.spans.length ?? 0,
    response_shape: input.audit?.response_shape ?? null,
    ...(input.auditError === undefined ? {} : { audit_error: input.auditError }),
    ...(input.wouldHaveSuppressionReason === undefined
      ? {}
      : { wouldHaveSuppressionReason: input.wouldHaveSuppressionReason }),
  };

  const audit = input.audit;
  const includeSpans =
    audit !== null && (includePayloads || input.reason === "mixed_closure_observed");

  if (includeSpans) {
    payload.spans = audit.spans.map((span) => ({
      text: span.text,
      kind: span.kind,
      rationale: span.rationale,
    }));
  }

  if (includePayloads && input.originalResponse !== undefined) {
    payload.original_response = input.originalResponse;
  }

  input.tracer.emit("closure_response_guard.completed", payload);
}

function traceClosureAuditInconsistent(input: {
  tracer?: TurnTracer;
  turnId: string;
  sessionId?: SessionId;
  reason: string;
  audit: ClosureResponseAudit;
  activeClosureCommitments: readonly string[];
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  input.tracer.emit("closure_pressure_audit.degraded", {
    turnId: input.turnId,
    ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
    reason: input.reason,
    spans_detected: input.audit.spans.length,
    response_shape: input.audit.response_shape,
    active_closure_commitments: [...input.activeClosureCommitments],
    spans: input.audit.spans.map((span) => ({
      text: span.text,
      kind: span.kind,
      rationale: span.rationale,
    })),
  });
}

function formatAuditError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

export class ClosurePressureGuard {
  constructor(private readonly options: ClosurePressureGuardOptions) {}

  private mode(): PostGenerationGuardMode {
    return this.options.mode ?? "enforce";
  }

  private applyMode(
    input: ClosurePressureGuardInput,
    result: ClosurePressureGuardResult,
    mode: PostGenerationGuardMode,
  ): ClosurePressureGuardResult {
    if (mode === "enforce" || result.verdict === "passed") {
      return result;
    }

    return {
      ...result,
      emission: {
        kind: "message",
        content: input.response,
      },
      verdict: "passed",
    };
  }

  private async audit(response: string): Promise<ClosureResponseAudit> {
    try {
      return (
        await callStructuredTool({
          llmClient: this.options.llmClient,
          request: {
            model: this.options.auditModel,
            system: CLOSURE_RESPONSE_AUDIT_SYSTEM_PROMPT,
            messages: buildAuditMessages(response),
            tools: [CLOSURE_RESPONSE_AUDIT_TOOL],
            tool_choice: { type: "tool", name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME },
            max_tokens: 512,
            budget: "closure-response-auditor",
          },
          toolName: CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
          parse: parseAuditResponse,
        })
      ).parsed;
    } catch (error) {
      throw closureAuditStructuredError(error);
    }
  }

  async run(input: ClosurePressureGuardInput): Promise<ClosurePressureGuardResult> {
    const activeCommitments = activeClosureCommitmentFamilies(input.activeCommitments);
    const closureHistoryExplicitlyAllowed =
      input.currentUserClosureKind === "user_requests_closure";
    const activeClosureHistory = activeClosurePressureHistoryEntries({
      entries: input.closurePressureHistory ?? [],
      nowMs: input.nowMs,
      currentTurn: input.currentTurn,
    });
    const closureHistoryActive =
      activeClosureHistory.length > 0 && !closureHistoryExplicitlyAllowed;
    const activeCommitmentLabels = [
      ...activeClosureCommitmentLabels(activeCommitments),
      ...(closureHistoryActive ? ["closure_pressure_history"] : []),
    ];
    const closureLoopNamed = input.closureLoop?.status === "named";
    const effectiveMode: PostGenerationGuardMode =
      this.mode() === "enforce" &&
      shouldEnforceClosurePressure({
        activeClosureCommitments: activeCommitments,
        closureLoop: input.closureLoop,
        closureHistoryActive,
      })
        ? "enforce"
        : "shadow";
    let audit: ClosureResponseAudit;

    try {
      audit = await this.audit(input.response);
    } catch (error) {
      const auditError = formatAuditError(error);
      const reason = "closure_response_audit_failed_open";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit: null,
        auditError,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "message",
            content: input.response,
          },
          verdict: "passed",
          removed_spans: [],
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit: null,
        },
        effectiveMode,
      );
    }

    if (audit.spans.length === 0 && audit.response_shape === "no_closure") {
      const reason = "no_closure_spans";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "message",
            content: input.response,
          },
          verdict: "passed",
          removed_spans: [],
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit,
        },
        effectiveMode,
      );
    }

    if (audit.spans.length === 0) {
      const reason = "closure_pressure_audit.degraded_no_spans";

      traceClosureAuditInconsistent({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        reason,
        audit,
        activeClosureCommitments: activeCommitmentLabels,
      });

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "message",
            content: input.response,
          },
          verdict: "passed",
          removed_spans: [],
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit,
        },
        effectiveMode,
      );
    }

    if (audit.response_shape === "no_closure") {
      const reason = "closure_pressure_audit.degraded_with_spans";

      // A no_closure shape with non-empty spans is self-inconsistent. Trace the audit as
      // degraded, but do not enforce: as with mixed responses, a non-critical ambiguous
      // LLM signal must not decide what reaches the user.
      traceClosureAuditInconsistent({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        reason,
        audit,
        activeClosureCommitments: activeCommitmentLabels,
      });

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "message",
            content: input.response,
          },
          verdict: "passed",
          removed_spans: [],
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit,
        },
        effectiveMode,
      );
    }

    const removedSpans = audit.spans.map((span) => span.text);

    if (audit.response_shape === "closure_only" && !closureLoopNamed) {
      const reason = "closure_only_observed";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "passed",
        wouldHaveVerdict: "suppressed",
        wouldHaveSuppressionReason: "closure_pressure_only",
        removedSpans,
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "message",
            content: input.response,
          },
          verdict: "passed",
          removed_spans: removedSpans,
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit,
        },
        effectiveMode,
      );
    }

    if (activeCommitments.length === 0 && !closureLoopNamed && !closureHistoryActive) {
      const reason = "no_active_closure_preference";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "message",
            content: input.response,
          },
          verdict: "passed",
          removed_spans: [],
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit,
        },
        effectiveMode,
      );
    }

    if (audit.response_shape === "closure_only" && closureLoopNamed) {
      const reason = "closure_pressure_only";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        sessionId: input.sessionId,
        mode: effectiveMode,
        verdict: "suppressed",
        wouldHaveSuppressionReason: reason,
        removedSpans,
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(
        input,
        {
          emission: {
            kind: "suppressed",
            reason,
            closure_pressure_history_reason: "span_removed",
          },
          verdict: "suppressed",
          removed_spans: removedSpans,
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit,
        },
        effectiveMode,
      );
    }

    const reason = "mixed_closure_observed";

    traceClosureGuard({
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId: input.sessionId,
      mode: effectiveMode,
      verdict: "passed",
      wouldHaveVerdict: "suppressed",
      removedSpans,
      activeClosureCommitments: activeCommitmentLabels,
      reason,
      audit,
    });

    return this.applyMode(
      input,
      {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: removedSpans,
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      },
      effectiveMode,
    );
  }
}
