import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import { valueAppearsIn } from "../../util/text-presence.js";
import type { ClosureLoopState } from "../../memory/working/index.js";
import type { TurnTraceData, TurnTracer } from "../tracing/tracer.js";
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

const STRUCTURALLY_EMPTY_TEXT_PATTERN = /^[\s\p{P}\p{S}]*$/u;
const SPAN_WHITESPACE_PATTERN = /\s/u;
const TRAILING_REMOVAL_JUNK_PATTERN = /[\s,;:]+$/u;

const SMART_QUOTE_REPLACEMENTS: ReadonlyArray<readonly [string, string]> = [
  ["\u2018", "'"],
  ["\u2019", "'"],
  ["\u201a", "'"],
  ["\u201b", "'"],
  ["\u2032", "'"],
  ["\u201c", '"'],
  ["\u201d", '"'],
  ["\u201e", '"'],
  ["\u201f", '"'],
  ["\u2033", '"'],
];

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
  tracer?: TurnTracer;
};

export type ClosurePressureGuardInput = {
  turnId: string;
  response: string;
  activeCommitments: readonly CommitmentRecord[];
  closureLoop: ClosureLoopState | null;
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

function traceClosureGuard(input: {
  tracer?: TurnTracer;
  turnId: string;
  verdict: "passed" | "rewritten" | "suppressed";
  removedSpans: readonly string[];
  activeClosureCommitments: readonly string[];
  reason: string;
  audit: ClosureResponseAudit | null;
  originalResponse?: string;
  rewrittenResponse?: string;
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  const includePayloads = input.tracer.includePayloads === true;
  const payload: TurnTraceData = {
    turnId: input.turnId,
    verdict: input.verdict,
    removed_spans: [...input.removedSpans],
    active_closure_commitments: [...input.activeClosureCommitments],
    reason: input.reason,
    spans_detected: input.audit?.spans.length ?? 0,
    response_shape: input.audit?.response_shape ?? null,
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

type TextRange = {
  start: number;
  end: number;
};

type NormalizedTextIndex = {
  text: string;
  ranges: TextRange[];
};

function normalizeSpanCharacter(value: string): string {
  let normalized = value;

  for (const [from, to] of SMART_QUOTE_REPLACEMENTS) {
    normalized = normalized.replaceAll(from, to);
  }

  return normalized.normalize("NFC").toLocaleLowerCase();
}

function buildNormalizedTextIndex(value: string): NormalizedTextIndex {
  let text = "";
  const ranges: TextRange[] = [];
  let offset = 0;

  for (const character of value) {
    const start = offset;
    offset += character.length;

    if (SPAN_WHITESPACE_PATTERN.test(character)) {
      if (text[text.length - 1] === " ") {
        const lastRange = ranges[ranges.length - 1];

        if (lastRange !== undefined) {
          lastRange.end = offset;
        }
      } else {
        text += " ";
        ranges.push({ start, end: offset });
      }

      continue;
    }

    for (const normalizedCharacter of normalizeSpanCharacter(character)) {
      text += normalizedCharacter;
      ranges.push({ start, end: offset });
    }
  }

  return { text, ranges };
}

function normalizedSpanText(value: string): string {
  return buildNormalizedTextIndex(value).text.trim();
}

function findNormalizedSpanRanges(response: string, spanText: string): TextRange[] {
  const normalized = buildNormalizedTextIndex(response);
  const needle = normalizedSpanText(spanText);
  const matches: TextRange[] = [];

  if (needle.length === 0) {
    return matches;
  }

  let index = normalized.text.indexOf(needle);

  while (index >= 0) {
    const first = normalized.ranges[index];
    const last = normalized.ranges[index + needle.length - 1];

    if (first !== undefined && last !== undefined) {
      matches.push({
        start: first.start,
        end: last.end,
      });
    }

    index = normalized.text.indexOf(needle, index + 1);
  }

  return matches;
}

function rangesOverlap(left: TextRange, right: TextRange): boolean {
  return left.start < right.end && right.start < left.end;
}

function findUniqueRemovalRange(input: {
  response: string;
  span: ClosureResponseSpan;
  selectedRanges: readonly TextRange[];
}): TextRange | null {
  if (!valueAppearsIn(input.response, input.span.text)) {
    return null;
  }

  const matches = findNormalizedSpanRanges(input.response, input.span.text).filter(
    (range) => !input.selectedRanges.some((selected) => rangesOverlap(range, selected)),
  );

  return matches.length === 1 ? (matches[0] ?? null) : null;
}

function structurallyEmptyText(value: string): boolean {
  return STRUCTURALLY_EMPTY_TEXT_PATTERN.test(value);
}

function cleanRemovedClosureText(value: string): string {
  return value.replace(TRAILING_REMOVAL_JUNK_PATTERN, "").trim();
}

function removeClosureSpansDeterministically(input: {
  response: string;
  spans: readonly ClosureResponseSpan[];
}): string | null {
  const ranges: TextRange[] = [];

  for (const span of input.spans) {
    const range = findUniqueRemovalRange({
      response: input.response,
      span,
      selectedRanges: ranges,
    });

    if (range === null) {
      return null;
    }

    ranges.push(range);
  }

  let rewritten = input.response;

  for (const range of [...ranges].sort((left, right) => right.start - left.start)) {
    rewritten = `${rewritten.slice(0, range.start)}${rewritten.slice(range.end)}`;
  }

  return cleanRemovedClosureText(rewritten);
}

export class ClosurePressureGuard {
  constructor(private readonly options: ClosurePressureGuardOptions) {}

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
    const activeCommitmentLabels = activeClosureCommitmentLabels(activeCommitments);
    const closureLoopNamed = input.closureLoop?.status === "named";
    let audit: ClosureResponseAudit;

    try {
      audit = await this.audit(input.response);
    } catch {
      const reason = "closure_response_audit_failed_open";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit: null,
      });

      return {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit: null,
      };
    }

    if (audit.spans.length === 0 && audit.response_shape === "no_closure") {
      const reason = "no_closure_spans";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      };
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
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      };
    }

    if (audit.response_shape === "no_closure") {
      traceClosureAuditInconsistent({
        tracer: this.options.tracer,
        turnId: input.turnId,
        reason: "closure_pressure_audit_inconsistent_with_spans",
        audit,
      });
    }

    if (activeCommitments.length === 0 && !closureLoopNamed) {
      const reason = "no_active_closure_preference";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        verdict: "passed",
        removedSpans: [],
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return {
        emission: {
          kind: "message",
          content: input.response,
        },
        verdict: "passed",
        removed_spans: [],
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      };
    }

    const removedSpans = audit.spans.map((span) => span.text);

    if (audit.response_shape === "closure_only") {
      const reason = "closure_pressure_only";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        verdict: "suppressed",
        removedSpans,
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return {
        emission: {
          kind: "suppressed",
          reason,
        },
        verdict: "suppressed",
        removed_spans: removedSpans,
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      };
    }

    const rewritten = removeClosureSpansDeterministically({
      response: input.response,
      spans: audit.spans,
    });

    if (rewritten === null || structurallyEmptyText(rewritten)) {
      const reason = "closure_pressure_only";

      traceClosureGuard({
        tracer: this.options.tracer,
        turnId: input.turnId,
        verdict: "suppressed",
        removedSpans,
        activeClosureCommitments: activeCommitmentLabels,
        reason,
        audit,
      });

      return {
        emission: {
          kind: "suppressed",
          reason,
        },
        verdict: "suppressed",
        removed_spans: removedSpans,
        active_closure_commitments: activeCommitmentLabels,
        reason,
        audit,
      };
    }

    const reason = "closure_spans_removed";

    traceClosureGuard({
      tracer: this.options.tracer,
      turnId: input.turnId,
      verdict: "rewritten",
      removedSpans,
      activeClosureCommitments: activeCommitmentLabels,
      reason,
      audit,
      originalResponse: input.response,
      rewrittenResponse: rewritten,
    });

    return {
      emission: {
        kind: "message",
        content: rewritten,
      },
      verdict: "rewritten",
      removed_spans: removedSpans,
      active_closure_commitments: activeCommitmentLabels,
      reason,
      audit,
    };
  }
}
