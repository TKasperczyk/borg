import { z } from "zod";

import type { ToolLoopResultObservation, ToolLoopResultObserver } from "../turn-action/index.js";
import { assertJsonValue, type JsonValue } from "../../util/json-value.js";
import {
  createContentAddressedCaptureSidecar,
  type PendingContentAddressedCaptureSidecar,
} from "./context-capture-storage.js";
import {
  fingerprintCanonicalValue,
  serializeCanonicalValue,
  type CanonicalRequestFingerprint,
} from "./request-fingerprint.js";

const FINALIZER_TOOL_TRANSCRIPT_SCHEMA_VERSION = 1 as const;
const FINALIZER_TOOL_TRANSCRIPT_SUBDIRECTORY = "finalizer-tool-transcripts";
export const FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES = 8 * 1024 * 1024;

const sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const canonicalFingerprintSchema = z
  .object({
    canonicalChars: z.number().int().nonnegative(),
    canonicalSha256: sha256Schema,
  })
  .strict();
const jsonValueSchema = z.custom<JsonValue>((value) => {
  try {
    assertJsonValue(value);
    return true;
  } catch {
    return false;
  }
});
const transcriptResultSchema = z.discriminatedUnion("ok", [
  z.object({ ok: z.literal(true), output: jsonValueSchema }).strict(),
  z.object({ ok: z.literal(false), error: z.string() }).strict(),
]);
const transcriptEventSchema = z
  .object({
    ordinal: z.number().int().positive(),
    iteration: z.number().int().positive(),
    batch_position: z.number().int().positive(),
    call_id: z.string().min(1),
    tool_name: z.string().min(1),
    raw_arguments: jsonValueSchema,
    arguments_fingerprint: canonicalFingerprintSchema,
    disposition: z.enum(["dispatched", "skipped_unavailable", "skipped_iteration_cap"]),
    result: transcriptResultSchema,
    duration_ms: z.number().finite().nonnegative(),
  })
  .strict();
const incompleteReasonSchema = z.enum([
  "observation_failed",
  "event_count_mismatch",
  "source_incomplete",
]);

const finalizerToolTranscriptBaseSchema = z
  .object({
    schema_version: z.literal(FINALIZER_TOOL_TRANSCRIPT_SCHEMA_VERSION),
    request_binding: canonicalFingerprintSchema.nullable(),
    complete: z.boolean(),
    incomplete_reasons: z.array(incompleteReasonSchema),
    event_count: z.number().int().nonnegative(),
    dispatched_count: z.number().int().nonnegative(),
    events: z.array(transcriptEventSchema),
  })
  .strict();
type ParsedFinalizerToolTranscript = z.infer<typeof finalizerToolTranscriptBaseSchema>;

function validateTranscriptCounts(
  transcript: ParsedFinalizerToolTranscript,
  context: z.RefinementCtx,
): void {
  if (
    transcript.event_count < transcript.events.length ||
    (transcript.complete && transcript.event_count !== transcript.events.length)
  ) {
    context.addIssue({
      code: "custom",
      path: ["event_count"],
      message: "Tool transcript event count is inconsistent with its captured events",
    });
  }
  const capturedDispatchedCount = transcript.events.filter(
    (event) => event.disposition === "dispatched",
  ).length;
  if (
    transcript.dispatched_count > transcript.event_count ||
    transcript.dispatched_count < capturedDispatchedCount ||
    (transcript.complete && transcript.dispatched_count !== capturedDispatchedCount)
  ) {
    context.addIssue({
      code: "custom",
      path: ["dispatched_count"],
      message: "Tool transcript dispatched count is inconsistent with its captured events",
    });
  }
}

function validateTranscriptCompleteness(
  transcript: ParsedFinalizerToolTranscript,
  context: z.RefinementCtx,
): void {
  if (transcript.complete === (transcript.incomplete_reasons.length === 0)) return;
  context.addIssue({
    code: "custom",
    path: ["complete"],
    message: "Tool transcript completeness does not match its reasons",
  });
}

function validateTranscriptOrdinals(
  transcript: ParsedFinalizerToolTranscript,
  context: z.RefinementCtx,
): void {
  let previousOrdinal = 0;
  for (const [index, event] of transcript.events.entries()) {
    if (
      event.ordinal <= previousOrdinal ||
      event.ordinal > transcript.event_count ||
      (transcript.complete && event.ordinal !== index + 1)
    ) {
      context.addIssue({
        code: "custom",
        path: ["events", index, "ordinal"],
        message: "Tool transcript ordinals are inconsistent with its event sequence",
      });
    }
    previousOrdinal = event.ordinal;
  }
}

const finalizerToolTranscriptSchema = finalizerToolTranscriptBaseSchema.superRefine(
  (transcript, context) => {
    validateTranscriptCounts(transcript, context);
    validateTranscriptCompleteness(transcript, context);
    validateTranscriptOrdinals(transcript, context);
  },
);

const manifestBaseShape = {
  event_count: z.number().int().nonnegative(),
  dispatched_count: z.number().int().nonnegative(),
  payload_bytes: z.number().int().nonnegative(),
  request_binding: canonicalFingerprintSchema.nullable(),
  replay_eligible: z.boolean(),
};

const finalizerToolTranscriptManifestBaseSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("none"),
      ...manifestBaseShape,
      canonical_sha256: z.null(),
      relative_path: z.null(),
      incomplete_reasons: z.array(incompleteReasonSchema).length(0),
    })
    .strict(),
  z
    .object({
      status: z.literal("complete"),
      ...manifestBaseShape,
      canonical_sha256: sha256Schema,
      relative_path: z.string().regex(/^finalizer-tool-transcripts\/[a-f0-9]{64}$/),
      incomplete_reasons: z.array(incompleteReasonSchema).length(0),
    })
    .strict(),
  z
    .object({
      status: z.literal("incomplete"),
      ...manifestBaseShape,
      canonical_sha256: sha256Schema,
      relative_path: z.null(),
      incomplete_reasons: z.array(incompleteReasonSchema).min(1),
    })
    .strict(),
  z
    .object({
      status: z.literal("omitted_oversized"),
      ...manifestBaseShape,
      canonical_sha256: sha256Schema,
      relative_path: z.null(),
      incomplete_reasons: z.array(incompleteReasonSchema),
    })
    .strict(),
]);
type ParsedFinalizerToolTranscriptManifest = z.infer<
  typeof finalizerToolTranscriptManifestBaseSchema
>;

function validateManifestDispatchedCount(
  manifest: ParsedFinalizerToolTranscriptManifest,
  context: z.RefinementCtx,
): void {
  if (manifest.dispatched_count > manifest.event_count) {
    context.addIssue({
      code: "custom",
      path: ["dispatched_count"],
      message: "Tool transcript dispatched count exceeds its event count",
    });
  }
}

function validateManifestStatusCounts(
  manifest: ParsedFinalizerToolTranscriptManifest,
  context: z.RefinementCtx,
): void {
  if (
    manifest.status === "none" &&
    (manifest.event_count !== 0 || manifest.dispatched_count !== 0 || manifest.payload_bytes !== 0)
  ) {
    context.addIssue({
      code: "custom",
      path: ["status"],
      message: "A none tool transcript manifest must have zero counts and bytes",
    });
  }
  if (manifest.status === "complete" && manifest.event_count === 0) {
    context.addIssue({
      code: "custom",
      path: ["event_count"],
      message: "A complete sidecar manifest must contain at least one event",
    });
  }
}

function validateManifestReplayEligibility(
  manifest: ParsedFinalizerToolTranscriptManifest,
  context: z.RefinementCtx,
): void {
  const hasReplayMaterial = manifest.status === "complete" || manifest.status === "none";
  if (manifest.replay_eligible === (hasReplayMaterial && manifest.request_binding !== null)) return;
  context.addIssue({
    code: "custom",
    path: ["replay_eligible"],
    message: "Tool transcript replay eligibility does not match its material status",
  });
}

export const finalizerToolTranscriptManifestSchema =
  finalizerToolTranscriptManifestBaseSchema.superRefine((manifest, context) => {
    validateManifestDispatchedCount(manifest, context);
    validateManifestReplayEligibility(manifest, context);
    validateManifestStatusCounts(manifest, context);
  });

export type FinalizerToolTranscript = z.infer<typeof finalizerToolTranscriptSchema>;
export type FinalizerToolTranscriptManifest = z.infer<typeof finalizerToolTranscriptManifestSchema>;

export type FinalizerToolTranscriptSnapshot = {
  transcript: FinalizerToolTranscript;
};

type PreparedFinalizerToolTranscript = {
  manifest: FinalizerToolTranscriptManifest;
  pendingSidecar: PendingContentAddressedCaptureSidecar | null;
};

function cloneJsonValue(value: unknown): JsonValue {
  assertJsonValue(value);
  return JSON.parse(JSON.stringify(value)) as JsonValue;
}

function transcriptResult(
  result: ToolLoopResultObservation["result"],
): FinalizerToolTranscript["events"][number]["result"] {
  return result.ok
    ? { ok: true, output: cloneJsonValue(result.output) }
    : { ok: false, error: result.error };
}

/**
 * A best-effort observer for a sampled finalizer call. Its public methods are
 * deliberately no-throw so capture can never change the live tool loop.
 */
export class FinalizerToolTranscriptCollector implements ToolLoopResultObserver {
  private readonly events: FinalizerToolTranscript["events"] = [];
  private readonly incompleteReasons = new Set<z.infer<typeof incompleteReasonSchema>>();
  private observationCount = 0;
  private dispatchedObservationCount = 0;

  observe(observation: ToolLoopResultObservation): void {
    this.observationCount += 1;
    if (observation.disposition === "dispatched") this.dispatchedObservationCount += 1;
    try {
      if (observation.ordinal !== this.observationCount) {
        throw new TypeError("Tool transcript observation ordinal is not contiguous");
      }
      const rawArguments = cloneJsonValue(observation.rawArguments);
      this.events.push(
        transcriptEventSchema.parse({
          ordinal: observation.ordinal,
          iteration: observation.iteration,
          batch_position: observation.batchPosition,
          call_id: observation.callId,
          tool_name: observation.toolName,
          raw_arguments: rawArguments,
          arguments_fingerprint: fingerprintCanonicalValue(rawArguments),
          disposition: observation.disposition,
          result: transcriptResult(observation.result),
          duration_ms: observation.durationMs,
        }),
      );
    } catch {
      this.incompleteReasons.add("observation_failed");
    }
  }

  markIncomplete(_error: unknown): void {
    this.incompleteReasons.add("observation_failed");
  }

  finish(input: {
    requestBinding: CanonicalRequestFingerprint | null;
    expectedEventCount: number | null;
    sourceCompleted: boolean;
  }): FinalizerToolTranscriptSnapshot {
    if (!input.sourceCompleted) this.incompleteReasons.add("source_incomplete");
    if (input.expectedEventCount !== null && input.expectedEventCount !== this.observationCount) {
      this.incompleteReasons.add("event_count_mismatch");
    }
    return {
      transcript: {
        schema_version: FINALIZER_TOOL_TRANSCRIPT_SCHEMA_VERSION,
        request_binding: input.requestBinding,
        complete: this.incompleteReasons.size === 0,
        incomplete_reasons: [...this.incompleteReasons],
        event_count: this.observationCount,
        dispatched_count: this.dispatchedObservationCount,
        events: [...this.events],
      },
    };
  }
}

export function parseFinalizerToolTranscript(value: unknown): FinalizerToolTranscript {
  return finalizerToolTranscriptSchema.parse(value);
}

export function prepareFinalizerToolTranscript(input: {
  snapshot: FinalizerToolTranscriptSnapshot;
  maxBytes?: number;
}): PreparedFinalizerToolTranscript {
  const transcript = input.snapshot.transcript;
  const requestBinding = transcript.request_binding;
  if (transcript.complete && transcript.event_count === 0) {
    return {
      manifest: {
        status: "none",
        event_count: 0,
        dispatched_count: 0,
        payload_bytes: 0,
        canonical_sha256: null,
        relative_path: null,
        request_binding: requestBinding,
        replay_eligible: requestBinding !== null,
        incomplete_reasons: [],
      },
      pendingSidecar: null,
    };
  }

  const bytes = Buffer.from(serializeCanonicalValue(transcript));
  const sidecar = createContentAddressedCaptureSidecar({
    subdirectory: FINALIZER_TOOL_TRANSCRIPT_SUBDIRECTORY,
    bytes,
  });
  const common = {
    event_count: transcript.event_count,
    dispatched_count: transcript.dispatched_count,
    payload_bytes: sidecar.byteSize,
    canonical_sha256: sidecar.sha256,
    request_binding: requestBinding,
    incomplete_reasons: transcript.incomplete_reasons,
  };

  if (!transcript.complete) {
    return {
      manifest: {
        status: "incomplete",
        ...common,
        relative_path: null,
        replay_eligible: false,
      },
      pendingSidecar: null,
    };
  }

  const maxBytes = Math.min(
    input.maxBytes ?? FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
    FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
  );
  if (sidecar.byteSize > maxBytes) {
    return {
      manifest: {
        status: "omitted_oversized",
        ...common,
        relative_path: null,
        replay_eligible: false,
      },
      pendingSidecar: null,
    };
  }

  return {
    manifest: {
      status: "complete",
      ...common,
      relative_path: sidecar.relativePath,
      replay_eligible: requestBinding !== null,
    },
    pendingSidecar: sidecar,
  };
}
