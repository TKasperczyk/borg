import { randomUUID } from "node:crypto";
import { z } from "zod";

import type { LLMConverseOptions } from "../../llm/index.js";
import type { ToolLoopUsage } from "../turn-action/index.js";
import { NOOP_TRACER, type TurnTracer } from "../../tracing/tracer.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { AttachmentId, SessionId } from "../../util/ids.js";
import {
  appendBoundedContextCapture,
  commitStagedContentAddressedCaptureSidecars,
  contentAddressedCaptureSidecarStorageBytes,
  createContentAddressedCaptureSidecar,
  discardStagedContentAddressedCaptureSidecars,
  pendingNewContentAddressedCaptureSidecarBytes,
  stageContentAddressedCaptureSidecars,
  type PendingContentAddressedCaptureSidecar,
  type StagedContentAddressedCaptureSidecar,
} from "./context-capture-storage.js";
import {
  FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
  FinalizerToolTranscriptCollector,
  finalizerToolTranscriptManifestSchema,
  prepareFinalizerToolTranscript,
  type FinalizerToolTranscriptManifest,
  type FinalizerToolTranscriptSnapshot,
} from "./finalizer-tool-transcript.js";
import {
  fingerprintCanonicalRequest,
  fingerprintSystemSurface,
  type CanonicalRequestFingerprint,
  type RequestSurfaceFingerprint,
} from "./request-fingerprint.js";
import type {
  FinalizerResolvedSurfaceVariant,
  FinalizerSurfaceVariant,
} from "./prompt/finalizer-context.js";
import type { DeliberationContext } from "./types.js";

const FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION = 2 as const;
const LEGACY_FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION = 1 as const;
// Sized to real captured records: the paired-replay projection (both rendered
// surfaces + render input + exact request) runs 42-65 MB per record on live
// finalizer contexts; the previous 32 MB cap silently skipped every one
// (trace: finalizer_context_capture.skipped record_oversized, 2026-08-15).
const DEFAULT_FINALIZER_CONTEXT_CAPTURE_MAX_RECORD_BYTES = 96 * 1024 * 1024;
const DEFAULT_FINALIZER_CONTEXT_CAPTURE_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024;
const DEFAULT_FINALIZER_CONTEXT_CAPTURE_MAX_SIDECAR_BYTES = 512 * 1024 * 1024;
const FINALIZER_CAPTURE_FILE_NAME = "finalizer-contexts.jsonl";

export type FinalizerCaptureOutcome =
  | {
      status: "completed";
      attempts: number;
      structuralReason: "terminal_emission" | "nonterminal_tool_loop" | "no_terminal_emission";
      decisionKind: string;
      decision: unknown;
      terminalToolCalls: readonly unknown[];
      reasoningText: string;
      usage: ToolLoopUsage;
    }
  | {
      status: "threw";
      attempts: number;
      structuralReason: "finalizer_error";
      error: { name: string; message: string; code?: string };
    };

export type FinalizerImageSidecar = {
  attachment_id: string;
  media_type: string;
  sha256: string;
  byte_size: number;
  relative_path: string;
};

type FinalizerContextCaptureRecordBase = {
  capture_id: string;
  captured_at: number;
  turn_id: string | null;
  session_id: SessionId;
  path: "system_1" | "system_2";
  attempt_kind: "initial" | "regenerate";
  /** Configured policy is absent on schema-v1 records written before scoped routing. */
  configured_surface_variant?: FinalizerSurfaceVariant;
  /** Concrete surface placed on the live request. */
  live_surface_variant: FinalizerResolvedSurfaceVariant;
  turn_origin: DeliberationContext["turnOrigin"];
  projected_context: Record<string, unknown>;
  evidence_ledger: DeliberationContext["evidenceLedger"];
  surfaces: {
    legacy: {
      system: NonNullable<LLMConverseOptions["system"]>;
      fingerprint: RequestSurfaceFingerprint;
    };
    compact: {
      system: NonNullable<LLMConverseOptions["system"]>;
      fingerprint: RequestSurfaceFingerprint;
    };
  };
  live_request: LLMConverseOptions | null;
  fidelity: {
    verified: boolean;
    request: CanonicalRequestFingerprint | null;
    surfaceMatchesRequest: boolean;
  };
  image_sidecars: readonly FinalizerImageSidecar[];
  live_outcome: FinalizerCaptureOutcome;
};

type FinalizerContextCaptureRecordV1 = FinalizerContextCaptureRecordBase & {
  schema_version: typeof LEGACY_FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION;
  replay: {
    eligible: boolean;
    exclusion_reason:
      | "autonomous"
      | "nonterminal_tools"
      | "nonterminal_outcome"
      | "source_threw"
      | "missing_request"
      | null;
  };
};

type FinalizerContextCaptureRecordV2 = FinalizerContextCaptureRecordBase & {
  schema_version: typeof FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION;
  tool_transcript: FinalizerToolTranscriptManifest;
  replay: FinalizerContextCaptureRecordV1["replay"] & {
    /** Material readiness for the stage-two recorded-result dispatcher. */
    recorded_results_eligible: boolean;
  };
};

export type FinalizerContextCaptureRecord =
  | FinalizerContextCaptureRecordV1
  | FinalizerContextCaptureRecordV2;

export type FinalizerContextCaptureOptions = {
  dataDir: string;
  sampleRate: number;
  clock?: Clock;
  tracer?: TurnTracer;
  random?: () => number;
  maxRecordBytes?: number;
  maxFileBytes?: number;
  maxSidecarBytes?: number;
  attachmentResolver?: (attachmentId: AttachmentId) => {
    mediaType: string;
    bytes: Buffer | Uint8Array;
  };
};

export type BuildFinalizerContextCaptureRecordInput = {
  capturedAt: number;
  turnId?: string;
  sessionId: SessionId;
  path: "system_1" | "system_2";
  attemptKind: "initial" | "regenerate";
  configuredSurfaceVariant: FinalizerSurfaceVariant;
  liveSurfaceVariant: FinalizerResolvedSurfaceVariant;
  context: DeliberationContext;
  legacySystem: NonNullable<LLMConverseOptions["system"]>;
  compactSystem: NonNullable<LLMConverseOptions["system"]>;
  liveRequest: LLMConverseOptions | null;
  liveRequestFingerprint?: CanonicalRequestFingerprint | null;
  outcome: FinalizerCaptureOutcome;
  usedNonTerminalTools: boolean;
  toolTranscript?: FinalizerToolTranscriptSnapshot;
  captureId?: string;
};

export type FinalizerContextCaptureWriteResult =
  | { status: "captured"; path: string; bytes: number; record: FinalizerContextCaptureRecord }
  | { status: "skipped"; reason: "record_oversized" | "file_full"; bytes: number }
  | { status: "failed"; reason: string };

type PendingImageSidecar = {
  record: FinalizerImageSidecar;
  sidecar: PendingContentAddressedCaptureSidecar;
};

const captureRecordBaseShape = {
  capture_id: z.string().min(1),
  captured_at: z.number().finite(),
  turn_id: z.string().nullable(),
  session_id: z.string().min(1),
  path: z.enum(["system_1", "system_2"]),
  attempt_kind: z.enum(["initial", "regenerate"]),
  configured_surface_variant: z.enum(["compact", "compact_conversational", "legacy"]).optional(),
  live_surface_variant: z.enum(["compact", "legacy"]),
  turn_origin: z.unknown().optional(),
  projected_context: z.record(z.string(), z.unknown()),
  evidence_ledger: z.unknown().nullish(),
  surfaces: z
    .object({
      legacy: z.object({ system: z.unknown(), fingerprint: z.record(z.string(), z.unknown()) }),
      compact: z.object({ system: z.unknown(), fingerprint: z.record(z.string(), z.unknown()) }),
    })
    .strict(),
  live_request: z.unknown().nullable(),
  fidelity: z
    .object({
      verified: z.boolean(),
      request: z.unknown().nullable(),
      surfaceMatchesRequest: z.boolean(),
    })
    .strict(),
  image_sidecars: z.array(
    z
      .object({
        attachment_id: z.string().min(1),
        media_type: z.string().min(1),
        sha256: z.string().regex(/^[a-f0-9]{64}$/),
        byte_size: z.number().int().nonnegative(),
        relative_path: z.string().regex(/^finalizer-images\/[a-f0-9]{64}$/),
      })
      .strict(),
  ),
  live_outcome: z.record(z.string(), z.unknown()),
};
const replayBaseShape = {
  eligible: z.boolean(),
  exclusion_reason: z
    .enum([
      "autonomous",
      "nonterminal_tools",
      "nonterminal_outcome",
      "source_threw",
      "missing_request",
    ])
    .nullable(),
};
const captureRecordV1Schema = z
  .object({
    schema_version: z.literal(LEGACY_FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION),
    ...captureRecordBaseShape,
    replay: z.object(replayBaseShape).strict(),
  })
  .strict();
const captureRecordV2Schema = z
  .object({
    schema_version: z.literal(FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION),
    ...captureRecordBaseShape,
    tool_transcript: finalizerToolTranscriptManifestSchema,
    replay: z
      .object({
        ...replayBaseShape,
        recorded_results_eligible: z.boolean(),
      })
      .strict(),
  })
  .strict();
const captureRecordSchema = z.discriminatedUnion("schema_version", [
  captureRecordV1Schema,
  captureRecordV2Schema,
]);

function jsonRoundTrip<T>(value: T): T {
  const text = JSON.stringify(value);
  if (text === undefined) throw new TypeError("Finalizer capture value is not JSON serializable");
  return JSON.parse(text) as T;
}

/** Exact renderer closure minus repositories, callbacks, raw user payloads, and image bytes. */
function projectFinalizerContext(context: DeliberationContext): Record<string, unknown> {
  const {
    entityRepository: _entityRepository,
    reRetrieve: _reRetrieve,
    currentUserContent: _currentUserContent,
    evidenceLedger: _evidenceLedger,
    userMessage: _userMessage,
    perception,
    ...serializable
  } = context;
  return jsonRoundTrip({
    ...serializable,
    perception: {
      mode: perception.mode,
      affectiveSignal: {
        valence: perception.affectiveSignal.valence,
        arousal: perception.affectiveSignal.arousal,
        dominant_emotion: perception.affectiveSignal.dominant_emotion,
      },
    },
  }) as Record<string, unknown>;
}

function imageAttachmentIds(request: LLMConverseOptions | null): AttachmentId[] {
  if (request === null) return [];
  const ids: AttachmentId[] = [];
  for (const message of request.messages) {
    if (typeof message.content === "string") continue;
    for (const block of message.content) {
      if (block.type === "image_ref") ids.push(block.attachment_id);
    }
  }
  return [...new Set(ids)];
}

function buildImageSidecars(
  request: LLMConverseOptions | null,
  resolver: FinalizerContextCaptureOptions["attachmentResolver"],
): PendingImageSidecar[] {
  if (resolver === undefined) return [];
  return imageAttachmentIds(request).map((attachmentId) => {
    const resolved = resolver(attachmentId);
    const sidecar = createContentAddressedCaptureSidecar({
      subdirectory: "finalizer-images",
      bytes: resolved.bytes,
    });
    return {
      record: {
        attachment_id: attachmentId,
        media_type: resolved.mediaType,
        sha256: sidecar.sha256,
        byte_size: sidecar.byteSize,
        relative_path: sidecar.relativePath,
      },
      sidecar,
    };
  });
}

function resolveToolTranscriptSnapshot(
  input: BuildFinalizerContextCaptureRecordInput,
  requestFingerprint: CanonicalRequestFingerprint | null,
): FinalizerToolTranscriptSnapshot {
  return (
    input.toolTranscript ??
    new FinalizerToolTranscriptCollector().finish({
      requestBinding: requestFingerprint,
      expectedEventCount: input.usedNonTerminalTools ? 1 : 0,
      sourceCompleted: input.outcome.status === "completed",
    })
  );
}

function toolTranscriptTraceDetails(
  manifest: FinalizerToolTranscriptManifest,
): Record<string, string | number> {
  return {
    tool_transcript_status: manifest.status,
    tool_transcript_events: manifest.event_count,
    tool_transcript_dispatched: manifest.dispatched_count,
    tool_transcript_bytes: manifest.payload_bytes,
    tool_transcript_incomplete_reasons: manifest.incomplete_reasons.length,
    tool_transcript_replay_eligible: manifest.replay_eligible ? 1 : 0,
  };
}

function replayExclusionReason(
  input: BuildFinalizerContextCaptureRecordInput,
): FinalizerContextCaptureRecordV1["replay"]["exclusion_reason"] {
  if (input.context.turnOrigin === "autonomous") return "autonomous";
  if (input.usedNonTerminalTools) return "nonterminal_tools";
  if (input.outcome.status === "threw") return "source_threw";
  if (input.outcome.structuralReason !== "terminal_emission") return "nonterminal_outcome";
  return input.liveRequest === null ? "missing_request" : null;
}

function transcriptRequestMatches(
  transcript: FinalizerToolTranscriptManifest,
  requestFingerprint: CanonicalRequestFingerprint | null,
): boolean {
  return (
    requestFingerprint !== null &&
    transcript.request_binding !== null &&
    transcript.request_binding.canonicalChars === requestFingerprint.canonicalChars &&
    transcript.request_binding.canonicalSha256 === requestFingerprint.canonicalSha256
  );
}

function recordedResultsReplayEligible(input: {
  source: BuildFinalizerContextCaptureRecordInput;
  surfaceMatchesRequest: boolean;
  requestFingerprint: CanonicalRequestFingerprint | null;
  transcript: FinalizerToolTranscriptManifest;
}): boolean {
  return (
    input.surfaceMatchesRequest &&
    transcriptRequestMatches(input.transcript, input.requestFingerprint) &&
    input.transcript.replay_eligible &&
    input.source.outcome.status === "completed" &&
    input.source.outcome.structuralReason !== "no_terminal_emission" &&
    (!input.source.usedNonTerminalTools || input.transcript.event_count > 0)
  );
}

export function buildFinalizerContextCaptureRecord(
  input: BuildFinalizerContextCaptureRecordInput,
  sidecars: readonly FinalizerImageSidecar[] = [],
  preparedToolTranscript?: FinalizerToolTranscriptManifest,
): FinalizerContextCaptureRecord {
  const liveSurface =
    input.liveSurfaceVariant === "compact" ? input.compactSystem : input.legacySystem;
  const liveSurfaceFingerprint = fingerprintSystemSurface(liveSurface);
  const requestFingerprint =
    input.liveRequestFingerprint ??
    (input.liveRequest === null ? null : fingerprintCanonicalRequest(input.liveRequest));
  const requestSurfaceFingerprint =
    input.liveRequest?.system === undefined
      ? null
      : fingerprintSystemSurface(input.liveRequest.system);
  const surfaceMatchesRequest =
    requestSurfaceFingerprint !== null &&
    requestSurfaceFingerprint.transportSha256 === liveSurfaceFingerprint.transportSha256;
  const toolTranscript =
    preparedToolTranscript ??
    prepareFinalizerToolTranscript({
      snapshot: resolveToolTranscriptSnapshot(input, requestFingerprint),
    }).manifest;
  const exclusionReason = replayExclusionReason(input);
  const recordedResultsEligible = recordedResultsReplayEligible({
    source: input,
    surfaceMatchesRequest,
    requestFingerprint,
    transcript: toolTranscript,
  });
  return parseFinalizerContextCaptureRecord(
    jsonRoundTrip({
      schema_version: FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION,
      capture_id: input.captureId ?? randomUUID(),
      captured_at: input.capturedAt,
      turn_id: input.turnId ?? null,
      session_id: input.sessionId,
      path: input.path,
      attempt_kind: input.attemptKind,
      configured_surface_variant: input.configuredSurfaceVariant,
      live_surface_variant: input.liveSurfaceVariant,
      ...(input.context.turnOrigin === undefined ? {} : { turn_origin: input.context.turnOrigin }),
      projected_context: projectFinalizerContext(input.context),
      evidence_ledger: input.context.evidenceLedger ?? null,
      surfaces: {
        legacy: {
          system: input.legacySystem,
          fingerprint: fingerprintSystemSurface(input.legacySystem),
        },
        compact: {
          system: input.compactSystem,
          fingerprint: fingerprintSystemSurface(input.compactSystem),
        },
      },
      live_request: input.liveRequest,
      fidelity: {
        verified: surfaceMatchesRequest && requestFingerprint !== null,
        request: requestFingerprint,
        surfaceMatchesRequest,
      },
      image_sidecars: sidecars,
      tool_transcript: toolTranscript,
      replay: {
        eligible: exclusionReason === null,
        exclusion_reason: exclusionReason,
        recorded_results_eligible: recordedResultsEligible,
      },
      live_outcome: input.outcome,
    }),
  );
}

export function parseFinalizerContextCaptureRecord(value: unknown): FinalizerContextCaptureRecord {
  return captureRecordSchema.parse(value) as unknown as FinalizerContextCaptureRecord;
}

export class FinalizerContextCapture {
  private readonly clock: Clock;
  private readonly tracer: TurnTracer;
  private readonly random: () => number;
  private readonly maxRecordBytes: number;
  private readonly maxFileBytes: number;
  private readonly maxSidecarBytes: number;

  constructor(private readonly options: FinalizerContextCaptureOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.random = options.random ?? Math.random;
    this.maxRecordBytes =
      options.maxRecordBytes ?? DEFAULT_FINALIZER_CONTEXT_CAPTURE_MAX_RECORD_BYTES;
    this.maxFileBytes = options.maxFileBytes ?? DEFAULT_FINALIZER_CONTEXT_CAPTURE_MAX_FILE_BYTES;
    this.maxSidecarBytes =
      options.maxSidecarBytes ?? DEFAULT_FINALIZER_CONTEXT_CAPTURE_MAX_SIDECAR_BYTES;
  }

  shouldCapture(): boolean {
    return this.options.sampleRate > 0 && this.random() < this.options.sampleRate;
  }

  capturedAt(): number {
    return this.clock.now();
  }

  recordAssemblyFailure(
    input: Pick<BuildFinalizerContextCaptureRecordInput, "turnId" | "sessionId">,
    error: unknown,
  ): void {
    this.emit(input, "failed", {
      phase: "alternate_surface_assembly",
      reason: error instanceof Error ? error.message : String(error),
    });
  }

  private emit(
    input: Pick<BuildFinalizerContextCaptureRecordInput, "turnId" | "sessionId">,
    status: "captured" | "skipped" | "failed",
    details: Record<string, string | number>,
  ): void {
    if (!this.tracer.enabled || input.turnId === undefined) return;
    try {
      this.tracer.emit(`deliberation.finalizer_context_capture.${status}`, {
        turnId: input.turnId,
        session_id: input.sessionId,
        ...details,
      });
    } catch {
      // Capture telemetry is observational and cannot become a live-turn
      // dependency if a custom tracer fails.
    }
  }

  async capture(
    input: BuildFinalizerContextCaptureRecordInput,
  ): Promise<FinalizerContextCaptureWriteResult> {
    let stagedSidecars: StagedContentAddressedCaptureSidecar[] = [];
    let transcriptTraceDetails: Record<string, string | number> = {};
    try {
      const requestFingerprint =
        input.liveRequestFingerprint ??
        (input.liveRequest === null ? null : fingerprintCanonicalRequest(input.liveRequest));
      const imageSidecars = buildImageSidecars(input.liveRequest, this.options.attachmentResolver);
      const preparedToolTranscript = prepareFinalizerToolTranscript({
        snapshot: resolveToolTranscriptSnapshot(input, requestFingerprint),
        maxBytes: FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
      });
      transcriptTraceDetails = toolTranscriptTraceDetails(preparedToolTranscript.manifest);
      const pendingSidecars = [
        ...imageSidecars.map((sidecar) => sidecar.sidecar),
        ...(preparedToolTranscript.pendingSidecar === null
          ? []
          : [preparedToolTranscript.pendingSidecar]),
      ];
      const record = buildFinalizerContextCaptureRecord(
        input,
        imageSidecars.map((sidecar) => sidecar.record),
        preparedToolTranscript.manifest,
      );
      const bytes = Buffer.byteLength(`${JSON.stringify(record)}\n`);
      if (bytes > this.maxRecordBytes) {
        this.emit(input, "skipped", {
          reason: "record_oversized",
          record_bytes: bytes,
          ...transcriptTraceDetails,
        });
        return { status: "skipped", reason: "record_oversized", bytes };
      }
      if (
        contentAddressedCaptureSidecarStorageBytes(this.options.dataDir, [
          "finalizer-images",
          "finalizer-tool-transcripts",
        ]) +
          pendingNewContentAddressedCaptureSidecarBytes(this.options.dataDir, pendingSidecars) >
        this.maxSidecarBytes
      ) {
        this.emit(input, "skipped", {
          reason: "file_full",
          record_bytes: bytes,
          ...transcriptTraceDetails,
        });
        return { status: "skipped", reason: "file_full", bytes };
      }
      stagedSidecars = stageContentAddressedCaptureSidecars(this.options.dataDir, pendingSidecars);
      const result = await appendBoundedContextCapture({
        dataDir: this.options.dataDir,
        fileName: FINALIZER_CAPTURE_FILE_NAME,
        record,
        maxFileBytes: this.maxFileBytes,
      });
      if (result.status === "file_full") {
        discardStagedContentAddressedCaptureSidecars(stagedSidecars);
        stagedSidecars = [];
        this.emit(input, "skipped", {
          reason: "file_full",
          record_bytes: bytes,
          ...transcriptTraceDetails,
        });
        return { status: "skipped", reason: "file_full", bytes };
      }
      commitStagedContentAddressedCaptureSidecars(stagedSidecars);
      stagedSidecars = [];
      this.emit(input, "captured", {
        record_bytes: bytes,
        ...transcriptTraceDetails,
      });
      return { status: "captured", path: result.path, bytes, record };
    } catch (error) {
      let reason = error instanceof Error ? error.message : String(error);
      try {
        discardStagedContentAddressedCaptureSidecars(stagedSidecars);
      } catch (cleanupError) {
        const cleanupReason =
          cleanupError instanceof Error ? cleanupError.message : String(cleanupError);
        reason = `${reason}; staged sidecar cleanup failed: ${cleanupReason}`;
      }
      this.emit(input, "failed", { reason, ...transcriptTraceDetails });
      return { status: "failed", reason };
    }
  }
}
