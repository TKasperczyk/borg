import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { PostGenerationGuardMode } from "../../config/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import { deleteSpans, isStructurallyEmptyText } from "../../util/span-deletion.js";
import type { ClosureLoopState, ClosurePressureHistoryEntry } from "../../memory/working/index.js";
import type { TurnTraceData, TurnTracer } from "../tracing/tracer.js";
import type { ClosureLoopDialogueAct } from "./closure-loop.js";
import type { PendingTurnEmission } from "./types.js";

export const CLOSURE_RESPONSE_AUDIT_TOOL_NAME = "EmitClosureResponseAudit";

export const CLOSURE_RESPONSE_SPAN_KINDS = [
  "imperative_closer",
  "aphoristic_valediction",
  "quotable_closing_tail",
] as const;

export const CLOSURE_RESPONSE_SHAPES = ["no_closure", "mixed", "closure_only"] as const;

export const CLOSURE_FUNCTION_EXAMPLES = [
  "Go.",
  "Go read.",
  "Go finish it.",
  "Go save the streak.",
  "Sleep.",
  "Held.",
  "Standing by.",
  "Surface when you surface.",
  "Trip thread held.",
  "Manana.",
  "That's the right note to end on.",
  "Banks is waiting.",
  "Held. Book.",
] as const;

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
  "Closure-function examples include:",
  ...CLOSURE_FUNCTION_EXAMPLES.map((example) => `- ${example}`),
  "Return spans as exact text copied from the response.",
  'Set response_shape to "no_closure" when no span has closure function.',
  'Set response_shape to "mixed" when the response has substantive content plus one or more closure-function spans.',
  'Set response_shape to "closure_only" when removing closure-function spans would leave no substantive content.',
  "Do not treat ordinary concise substantive answers as closure-pressure just because they are short.",
].join("\n");

export type ClosureResponseSpan = z.infer<typeof closureResponseSpanSchema>;
export type ClosureResponseAudit = z.infer<typeof closureResponseAuditSchema>;

export type ClosurePressureGuardResult = {
  emission: PendingTurnEmission;
  verdict: "passed" | "rewritten" | "suppressed";
  removed_spans: string[];
  active_closure_commitments: string[];
  reason: string;
  audit: ClosureResponseAudit | null;
};

export type ClosurePressureGuardOptions = {
  llmClient: LLMClient;
  auditModel: string;
  rewriteModel: string;
  mode?: PostGenerationGuardMode;
  tracer?: TurnTracer;
};

export type ClosurePressureGuardInput = {
  turnId: string;
  response: string;
  activeCommitments: readonly CommitmentRecord[];
  closureLoop: ClosureLoopState | null;
  closurePressureHistory?: readonly ClosurePressureHistoryEntry[];
  currentUserClosureKind?: ClosureLoopDialogueAct | null;
  currentTurn?: number;
  nowMs?: number;
};

function parseAuditResponse(result: LLMCompleteResult): ClosureResponseAudit {
  const call = result.tool_calls.find(
    (toolCall) => toolCall.name === CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
  );

  if (call === undefined) {
    throw new Error(`Closure response auditor did not emit ${CLOSURE_RESPONSE_AUDIT_TOOL_NAME}`);
  }

  const parsed = closureResponseAuditSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
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
      Math.max(0, input.currentTurn - entry.turn) <=
        CLOSURE_PRESSURE_HISTORY_ACTIVE_TURN_WINDOW;

    return withinTime && withinTurns;
  });
}

function traceClosureGuard(input: {
  tracer?: TurnTracer;
  turnId: string;
  verdict: "passed" | "rewritten" | "suppressed";
  mode?: PostGenerationGuardMode;
  wouldHaveVerdict?: "passed" | "rewritten" | "suppressed";
  wouldHaveSuppressionReason?: string;
  removedSpans: readonly string[];
  activeClosureCommitments: readonly string[];
  reason: string;
  audit: ClosureResponseAudit | null;
  originalResponse?: string;
  rewrittenResponse?: string;
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

  if (includePayloads && input.audit !== null) {
    payload.spans = input.audit.spans.map((span) => ({
      text: span.text,
      kind: span.kind,
      rationale: span.rationale,
    }));
  }

  if (includePayloads && input.originalResponse !== undefined) {
    payload.original_response = input.originalResponse;
  }

  if (includePayloads && input.rewrittenResponse !== undefined) {
    payload.rewritten_response = input.rewrittenResponse;
  }

  input.tracer.emit("closure_response_guard", payload);
}

function traceClosureAuditInconsistent(input: {
  tracer?: TurnTracer;
  turnId: string;
  reason: string;
  audit: ClosureResponseAudit;
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  input.tracer.emit("closure_pressure_audit_inconsistent", {
    turnId: input.turnId,
    reason: input.reason,
    spans_detected: input.audit.spans.length,
    response_shape: input.audit.response_shape,
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
  ): ClosurePressureGuardResult {
    if (this.mode() === "enforce" || result.verdict === "passed") {
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
    const result = await this.options.llmClient.complete({
      model: this.options.auditModel,
      system: CLOSURE_RESPONSE_AUDIT_SYSTEM_PROMPT,
      messages: buildAuditMessages(response),
      tools: [CLOSURE_RESPONSE_AUDIT_TOOL],
      tool_choice: { type: "tool", name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME },
      max_tokens: 512,
      budget: "closure-response-auditor",
    });

    return parseAuditResponse(result);
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
    let audit: ClosureResponseAudit;

    try {
      audit = await this.audit(input.response);
    } catch (error) {
      const auditError = formatAuditError(error);
      const failClosed =
        this.mode() === "enforce" &&
        (activeCommitments.length > 0 || closureLoopNamed || closureHistoryActive);

      if (failClosed) {
        const reason = "closure_response_audit_failed_closed";

        traceClosureGuard({
          tracer: this.options.tracer,
          turnId: input.turnId,
          mode: this.mode(),
          verdict: "suppressed",
          wouldHaveSuppressionReason: reason,
          removedSpans: [],
          activeClosureCommitments: activeCommitmentLabels,
          reason,
          audit: null,
          auditError,
        });

        return {
          emission: {
            kind: "suppressed",
            reason,
            closure_pressure_history_reason: "audit_caught",
          },
          verdict: "suppressed",
          removed_spans: [],
          active_closure_commitments: activeCommitmentLabels,
          reason,
          audit: null,
        };
      }

      const reason = "closure_response_audit_failed_open";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        mode: this.mode(),
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit: null,
        auditError,
      });

      return this.applyMode(input, {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit: null,
      });
    }

    if (audit.spans.length === 0 && audit.response_shape === "no_closure") {
      const reason = "no_closure_spans";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        mode: this.mode(),
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(input, {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      });
    }

    if (audit.spans.length === 0) {
      const reason = "closure_pressure_audit_inconsistent_no_spans";

      traceClosureAuditInconsistent({
        tracer: this.options.tracer,
        turnId: input.turnId,
        reason,
        audit,
      });

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        mode: this.mode(),
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(input, {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      });
    }

    if (audit.response_shape === "no_closure") {
      traceClosureAuditInconsistent({
        tracer: this.options.tracer,
        turnId: input.turnId,
        reason: "closure_pressure_audit_inconsistent_with_spans",
        audit,
      });
    }

    if (activeCommitments.length === 0 && !closureLoopNamed && !closureHistoryActive) {
      const reason = "no_active_closure_preference";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        mode: this.mode(),
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(input, {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      });
    }

    const removedSpans = audit.spans.map((span) => span.text);

    if (audit.response_shape === "closure_only") {
      const reason = "closure_pressure_only";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        mode: this.mode(),
        verdict: "suppressed",
        wouldHaveSuppressionReason: reason,
        removedSpans,
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(input, {
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
      });
    }

    const deletion = deleteSpans(
      input.response,
      audit.spans.map((span) => span.text),
    );

    if (!deletion.allRemoved || isStructurallyEmptyText(deletion.result)) {
      const reason = "closure_pressure_only";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        mode: this.mode(),
        verdict: "suppressed",
        wouldHaveSuppressionReason: reason,
        removedSpans,
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return this.applyMode(input, {
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
      });
    }

    const reason = "closure_spans_removed";

    traceClosureGuard({
      tracer: this.options.tracer,
      turnId: input.turnId,
      mode: this.mode(),
      verdict: "rewritten",
      removedSpans,
      activeClosureCommitments: activeCommitmentLabels,
      reason,
      audit,
      originalResponse: input.response,
      rewrittenResponse: deletion.result,
    });

    return this.applyMode(input, {
      emission: {
        kind: "message",
        content: deletion.result,
        closure_pressure_history_reason: "span_removed",
      },
      verdict: "rewritten",
      removed_spans: removedSpans,
      active_closure_commitments: activeCommitmentLabels,
      reason,
      audit,
    });
  }
}
