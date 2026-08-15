import { randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  existsSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, join } from "node:path";
import { z } from "zod";

import type { LLMConverseOptions } from "../../llm/index.js";
import type { ToolLoopUsage } from "../turn-action/index.js";
import { NOOP_TRACER, type TurnTracer } from "../../tracing/tracer.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { AttachmentId, SessionId } from "../../util/ids.js";
import {
  appendBoundedContextCapture,
  resolveContextCaptureSubdirectory,
} from "./context-capture-storage.js";
import {
  fingerprintCanonicalRequest,
  fingerprintSystemSurface,
  sha256Bytes,
  type CanonicalRequestFingerprint,
  type RequestSurfaceFingerprint,
} from "./request-fingerprint.js";
import type { FinalizerSurfaceVariant } from "./prompt/finalizer-context.js";
import type { DeliberationContext } from "./types.js";

const FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION = 1 as const;
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

export type FinalizerContextCaptureRecord = {
  schema_version: typeof FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION;
  capture_id: string;
  captured_at: number;
  turn_id: string | null;
  session_id: SessionId;
  path: "system_1" | "system_2";
  attempt_kind: "initial" | "regenerate";
  live_surface_variant: FinalizerSurfaceVariant;
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
  live_outcome: FinalizerCaptureOutcome;
};

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
  liveSurfaceVariant: FinalizerSurfaceVariant;
  context: DeliberationContext;
  legacySystem: NonNullable<LLMConverseOptions["system"]>;
  compactSystem: NonNullable<LLMConverseOptions["system"]>;
  liveRequest: LLMConverseOptions | null;
  liveRequestFingerprint?: CanonicalRequestFingerprint | null;
  outcome: FinalizerCaptureOutcome;
  usedNonTerminalTools: boolean;
  captureId?: string;
};

export type FinalizerContextCaptureWriteResult =
  | { status: "captured"; path: string; bytes: number; record: FinalizerContextCaptureRecord }
  | { status: "skipped"; reason: "record_oversized" | "file_full"; bytes: number }
  | { status: "failed"; reason: string };

type PendingSidecar = FinalizerImageSidecar & { bytes: Uint8Array };
type StagedSidecar = {
  sha256: string;
  stagedPath: string | null;
  finalPath: string;
  directory: string;
};

const captureRecordSchema = z
  .object({
    schema_version: z.literal(FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION),
    capture_id: z.string().min(1),
    captured_at: z.number().finite(),
    turn_id: z.string().nullable(),
    session_id: z.string().min(1),
    path: z.enum(["system_1", "system_2"]),
    attempt_kind: z.enum(["initial", "regenerate"]),
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
    replay: z
      .object({
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
      })
      .strict(),
    live_outcome: z.record(z.string(), z.unknown()),
  })
  .strict();

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

function buildSidecars(
  request: LLMConverseOptions | null,
  resolver: FinalizerContextCaptureOptions["attachmentResolver"],
): PendingSidecar[] {
  if (resolver === undefined) return [];
  return imageAttachmentIds(request).map((attachmentId) => {
    const resolved = resolver(attachmentId);
    const bytes = Buffer.from(resolved.bytes);
    const sha256 = sha256Bytes(bytes);
    return {
      attachment_id: attachmentId,
      media_type: resolved.mediaType,
      sha256,
      byte_size: bytes.byteLength,
      relative_path: join("finalizer-images", sha256),
      bytes,
    };
  });
}

function stagePrivateSidecars(
  dataDir: string,
  sidecars: readonly PendingSidecar[],
): StagedSidecar[] {
  if (sidecars.length === 0) return [];
  const sidecarDirectory = resolveContextCaptureSubdirectory(dataDir, "finalizer-images");
  mkdirSync(sidecarDirectory, { recursive: true, mode: 0o700 });
  chmodSync(sidecarDirectory, 0o700);
  const staged: StagedSidecar[] = [];
  try {
    for (const sidecar of sidecars) {
      const finalPath = join(sidecarDirectory, basename(sidecar.relative_path));
      if (existsSync(finalPath)) {
        const existingStat = lstatSync(finalPath);
        if (existingStat.isSymbolicLink() || !existingStat.isFile()) {
          throw new Error(`Finalizer capture sidecar is not a regular file: ${finalPath}`);
        }
        if (sha256Bytes(readFileSync(finalPath)) !== sidecar.sha256) {
          throw new Error(`Finalizer capture sidecar hash mismatch: ${sidecar.sha256}`);
        }
        chmodSync(finalPath, 0o600);
        staged.push({
          sha256: sidecar.sha256,
          stagedPath: null,
          finalPath,
          directory: sidecarDirectory,
        });
        continue;
      }
      const stagedPath = join(
        sidecarDirectory,
        `.staged-${randomUUID()}-${basename(sidecar.relative_path)}`,
      );
      const fd = openSync(stagedPath, "wx", 0o600);
      try {
        writeFileSync(fd, sidecar.bytes);
        fsyncSync(fd);
      } catch (error) {
        try {
          closeSync(fd);
        } finally {
          unlinkSync(stagedPath);
        }
        throw error;
      }
      closeSync(fd);
      chmodSync(stagedPath, 0o600);
      staged.push({ sha256: sidecar.sha256, stagedPath, finalPath, directory: sidecarDirectory });
    }
    return staged;
  } catch (error) {
    for (const sidecar of staged) {
      if (sidecar.stagedPath !== null && existsSync(sidecar.stagedPath)) {
        unlinkSync(sidecar.stagedPath);
      }
    }
    throw error;
  }
}

function discardStagedSidecars(sidecars: readonly StagedSidecar[]): void {
  for (const sidecar of sidecars) {
    if (sidecar.stagedPath !== null && existsSync(sidecar.stagedPath)) {
      unlinkSync(sidecar.stagedPath);
    }
  }
}

function commitStagedSidecars(sidecars: readonly StagedSidecar[]): void {
  const syncedDirectories = new Set<string>();
  for (const sidecar of sidecars) {
    if (sidecar.stagedPath === null) continue;
    if (existsSync(sidecar.finalPath)) {
      if (sha256Bytes(readFileSync(sidecar.finalPath)) !== sidecar.sha256) {
        throw new Error(`Finalizer capture sidecar hash mismatch: ${sidecar.sha256}`);
      }
      unlinkSync(sidecar.stagedPath);
    } else {
      renameSync(sidecar.stagedPath, sidecar.finalPath);
      chmodSync(sidecar.finalPath, 0o600);
    }
    syncedDirectories.add(sidecar.directory);
  }
  for (const directory of syncedDirectories) {
    const directoryFd = openSync(directory, "r");
    try {
      fsyncSync(directoryFd);
    } finally {
      closeSync(directoryFd);
    }
  }
}

function sidecarStorageBytes(dataDir: string): number {
  const directory = resolveContextCaptureSubdirectory(dataDir, "finalizer-images");
  if (!existsSync(directory)) return 0;
  return readdirSync(directory).reduce((sum, name) => {
    const path = join(directory, name);
    const stats = lstatSync(path);
    if (stats.isSymbolicLink()) {
      throw new Error(`Finalizer capture sidecar must not be a symlink: ${path}`);
    }
    return sum + (stats.isFile() ? stats.size : 0);
  }, 0);
}

function pendingNewSidecarBytes(dataDir: string, sidecars: readonly PendingSidecar[]): number {
  const directory = resolveContextCaptureSubdirectory(dataDir, "finalizer-images");
  const unique = new Map(sidecars.map((sidecar) => [sidecar.sha256, sidecar]));
  return [...unique.values()].reduce(
    (sum, sidecar) => sum + (existsSync(join(directory, sidecar.sha256)) ? 0 : sidecar.byte_size),
    0,
  );
}

export function buildFinalizerContextCaptureRecord(
  input: BuildFinalizerContextCaptureRecordInput,
  sidecars: readonly FinalizerImageSidecar[] = [],
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
  const exclusionReason =
    input.context.turnOrigin === "autonomous"
      ? ("autonomous" as const)
      : input.usedNonTerminalTools
        ? ("nonterminal_tools" as const)
        : input.outcome.status === "threw"
          ? ("source_threw" as const)
          : input.outcome.structuralReason !== "terminal_emission"
            ? ("nonterminal_outcome" as const)
            : input.liveRequest === null
              ? ("missing_request" as const)
              : null;
  return parseFinalizerContextCaptureRecord(
    jsonRoundTrip({
      schema_version: FINALIZER_CONTEXT_CAPTURE_SCHEMA_VERSION,
      capture_id: input.captureId ?? randomUUID(),
      captured_at: input.capturedAt,
      turn_id: input.turnId ?? null,
      session_id: input.sessionId,
      path: input.path,
      attempt_kind: input.attemptKind,
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
      replay: { eligible: exclusionReason === null, exclusion_reason: exclusionReason },
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
    this.tracer.emit(`deliberation.finalizer_context_capture.${status}`, {
      turnId: input.turnId,
      session_id: input.sessionId,
      ...details,
    });
  }

  async capture(
    input: BuildFinalizerContextCaptureRecordInput,
  ): Promise<FinalizerContextCaptureWriteResult> {
    let stagedSidecars: StagedSidecar[] = [];
    try {
      const pendingSidecars = buildSidecars(input.liveRequest, this.options.attachmentResolver);
      const sidecarRecords = pendingSidecars.map(({ bytes: _bytes, ...record }) => record);
      const record = buildFinalizerContextCaptureRecord(input, sidecarRecords);
      const bytes = Buffer.byteLength(`${JSON.stringify(record)}\n`);
      if (bytes > this.maxRecordBytes) {
        this.emit(input, "skipped", { reason: "record_oversized", record_bytes: bytes });
        return { status: "skipped", reason: "record_oversized", bytes };
      }
      if (
        sidecarStorageBytes(this.options.dataDir) +
          pendingNewSidecarBytes(this.options.dataDir, pendingSidecars) >
        this.maxSidecarBytes
      ) {
        this.emit(input, "skipped", { reason: "file_full", record_bytes: bytes });
        return { status: "skipped", reason: "file_full", bytes };
      }
      stagedSidecars = stagePrivateSidecars(this.options.dataDir, pendingSidecars);
      const result = await appendBoundedContextCapture({
        dataDir: this.options.dataDir,
        fileName: FINALIZER_CAPTURE_FILE_NAME,
        record,
        maxFileBytes: this.maxFileBytes,
      });
      if (result.status === "file_full") {
        discardStagedSidecars(stagedSidecars);
        stagedSidecars = [];
        this.emit(input, "skipped", { reason: "file_full", record_bytes: bytes });
        return { status: "skipped", reason: "file_full", bytes };
      }
      commitStagedSidecars(stagedSidecars);
      stagedSidecars = [];
      this.emit(input, "captured", { record_bytes: bytes });
      return { status: "captured", path: result.path, bytes, record };
    } catch (error) {
      discardStagedSidecars(stagedSidecars);
      const reason = error instanceof Error ? error.message : String(error);
      this.emit(input, "failed", { reason });
      return { status: "failed", reason };
    }
  }
}
