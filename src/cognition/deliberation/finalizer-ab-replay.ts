import { performance } from "node:perf_hooks";

import type { LLMClient, LLMConverseOptions } from "../../llm/index.js";
import type { FinalizerContextCaptureRecord } from "./finalizer-context-capture.js";
import {
  fingerprintCanonicalRequest,
  fingerprintSystemSurface,
  type CanonicalRequestFingerprint,
  type RequestSurfaceFingerprint,
} from "./request-fingerprint.js";

export type FinalizerAbReplayMode = "dry" | "live";

export type FinalizerAbLiveOutcome = {
  status: "completed" | "threw";
  durationMs: number;
  requestFingerprint: CanonicalRequestFingerprint;
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
  messageBlocks?: unknown;
  error?: { name: string; message: string };
};

export type FinalizerAbReplayResult = {
  schema_version: 1;
  capture_id: string;
  source_turn_id: string | null;
  source_path: "system_1" | "system_2";
  source_attempt_kind: "initial" | "regenerate";
  source_live_surface_variant: "compact" | "legacy";
  replayed_at: number;
  mode: FinalizerAbReplayMode;
  pairing_status:
    | "paired"
    | "excluded_autonomous"
    | "excluded_nonterminal"
    | "excluded_source_outcome"
    | "skipped_fidelity";
  execution_order: readonly ["compact" | "legacy", "compact" | "legacy"];
  fidelity: {
    storedVerified: boolean;
    currentSourceSystemMatchesCapture: boolean;
    currentSourceRequestMatchesCapture: boolean;
  };
  surfaces: Record<"compact" | "legacy", RequestSurfaceFingerprint>;
  size_delta: { compact_minus_legacy_chars: number };
  live?: Record<"compact" | "legacy", FinalizerAbLiveOutcome>;
};

export type ReplayFinalizerContextCaptureOptions =
  | { mode: "dry"; pairIndex?: number; now?: () => number }
  | { mode: "live"; llmClient: LLMClient; pairIndex?: number; now?: () => number };

const TERMINAL_FINALIZER_TOOL_NAMES = new Set([
  "EmitAnswer",
  "EmitObserve",
  "EmitNoOutput",
  "EmitSelfReport",
  "EmitContinueThought",
]);

function requestForVariant(
  record: FinalizerContextCaptureRecord,
  variant: "compact" | "legacy",
): LLMConverseOptions {
  if (record.live_request === null) {
    throw new TypeError("Finalizer capture has no live request");
  }
  const terminalTools = record.live_request.tools?.filter((tool) =>
    TERMINAL_FINALIZER_TOOL_NAMES.has(tool.name),
  );
  // Replay receives fake terminal schemas only. No dispatcher, repository,
  // stream writer, retrieval callback, or working-memory service is reachable.
  return {
    ...record.live_request,
    system: record.surfaces[variant].system,
    ...(terminalTools === undefined ? {} : { tools: terminalTools }),
  };
}

async function executeVariant(
  record: FinalizerContextCaptureRecord,
  variant: "compact" | "legacy",
  llmClient: LLMClient,
): Promise<FinalizerAbLiveOutcome> {
  const request = requestForVariant(record, variant);
  const requestFingerprint = fingerprintCanonicalRequest(request);
  const started = performance.now();
  try {
    const response = await llmClient.converse(request);
    return {
      status: "completed",
      durationMs: performance.now() - started,
      requestFingerprint,
      usage: {
        input_tokens: response.input_tokens,
        output_tokens: response.output_tokens,
        ...(response.cache_creation_input_tokens === undefined
          ? {}
          : { cache_creation_input_tokens: response.cache_creation_input_tokens }),
        ...(response.cache_read_input_tokens === undefined
          ? {}
          : { cache_read_input_tokens: response.cache_read_input_tokens }),
      },
      messageBlocks: response.messageBlocks,
    };
  } catch (error) {
    return {
      status: "threw",
      durationMs: performance.now() - started,
      requestFingerprint,
      error: {
        name: error instanceof Error ? error.name : "UnknownThrownValue",
        message: error instanceof Error ? error.message : String(error),
      },
    };
  }
}

export async function replayFinalizerContextCapture(
  record: FinalizerContextCaptureRecord,
  options: ReplayFinalizerContextCaptureOptions,
): Promise<FinalizerAbReplayResult> {
  const pairIndex = options.pairIndex ?? 0;
  const executionOrder = (
    pairIndex % 2 === 0 ? ["compact", "legacy"] : ["legacy", "compact"]
  ) as readonly ["compact" | "legacy", "compact" | "legacy"];
  const compact = fingerprintSystemSurface(record.surfaces.compact.system);
  const legacy = fingerprintSystemSurface(record.surfaces.legacy.system);
  const sourceSurface = record.surfaces[record.live_surface_variant].fingerprint;
  const currentSource = record.live_request?.system;
  const currentRequestFingerprint =
    record.live_request === null ? null : fingerprintCanonicalRequest(record.live_request);
  const currentSourceSystemMatchesCapture =
    currentSource !== undefined &&
    fingerprintSystemSurface(currentSource).transportSha256 === sourceSurface.transportSha256;
  const currentSourceRequestMatchesCapture =
    currentRequestFingerprint !== null &&
    record.fidelity.request !== null &&
    currentRequestFingerprint.canonicalChars === record.fidelity.request.canonicalChars &&
    currentRequestFingerprint.canonicalSha256 === record.fidelity.request.canonicalSha256;
  const fidelityMatches =
    record.fidelity.verified &&
    currentSourceSystemMatchesCapture &&
    currentSourceRequestMatchesCapture;
  const pairingStatus =
    record.replay.exclusion_reason === "autonomous"
      ? ("excluded_autonomous" as const)
      : record.replay.exclusion_reason === "nonterminal_tools"
        ? ("excluded_nonterminal" as const)
        : record.replay.exclusion_reason === "nonterminal_outcome" ||
            record.replay.exclusion_reason === "source_threw"
          ? ("excluded_source_outcome" as const)
          : !fidelityMatches
            ? ("skipped_fidelity" as const)
            : ("paired" as const);
  const base = {
    schema_version: 1 as const,
    capture_id: record.capture_id,
    source_turn_id: record.turn_id,
    source_path: record.path,
    source_attempt_kind: record.attempt_kind,
    source_live_surface_variant: record.live_surface_variant,
    replayed_at: (options.now ?? Date.now)(),
    mode: options.mode,
    pairing_status: pairingStatus,
    execution_order: executionOrder,
    fidelity: {
      storedVerified: record.fidelity.verified,
      currentSourceSystemMatchesCapture,
      currentSourceRequestMatchesCapture,
    },
    surfaces: { compact, legacy },
    size_delta: { compact_minus_legacy_chars: compact.systemChars - legacy.systemChars },
  } satisfies Omit<FinalizerAbReplayResult, "live">;

  if (options.mode === "dry" || pairingStatus !== "paired") return base;
  const live = {} as Record<"compact" | "legacy", FinalizerAbLiveOutcome>;
  for (const variant of executionOrder) {
    live[variant] = await executeVariant(record, variant, options.llmClient);
  }
  return { ...base, live };
}
