import { performance } from "node:perf_hooks";

import type { LLMClient, LLMSystemBlock } from "../../llm/index.js";
import {
  fingerprintPlannerRequest,
  renderCapturedPlannerSurfacePair,
  type CapturedPlannerSurface,
  type PlannerContextCaptureRecord,
  type PlannerRequestFingerprint,
  type PlannerSurfaceFingerprint,
} from "./planner-context-capture.js";
import {
  createS2PlannerRequestSnapshot,
  runS2Planner,
  type S2PlannerOutcome,
  type S2PlannerResult,
} from "./s2-planner.js";

export type PlannerAbReplayMode = "dry" | "live";

export type PlannerAbSurfaceSummary = {
  fingerprint: PlannerSurfaceFingerprint;
  expectedFingerprint: PlannerSurfaceFingerprint;
  byteFaithfulToCapture: boolean;
  traceSummary: CapturedPlannerSurface["rendered"]["traceSummary"];
};

type PlannerAbLiveResultPayload = {
  durationMs: number;
  plan: S2PlannerResult["plan"];
  reasoning: string;
  usage: S2PlannerResult["usage"];
  requestFingerprint: PlannerRequestFingerprint | null;
};

export type PlannerAbLiveOutcome =
  | (Extract<S2PlannerOutcome, { status: "completed" }> & PlannerAbLiveResultPayload)
  | (Extract<S2PlannerOutcome, { status: "degraded" }> & PlannerAbLiveResultPayload)
  | {
      status: "threw";
      attempts: number;
      structuralReason: "non_retryable_planner_error";
      durationMs: number;
      error: Extract<S2PlannerOutcome, { status: "threw" }>["error"];
      requestFingerprint: PlannerRequestFingerprint | null;
    };

export type PlannerAbReplayResultRecord = {
  schema_version: 2;
  capture_id: string;
  source_turn_id: string | null;
  source_session_id: string;
  source_captured_at: number;
  source_live_surface_variant: "compact" | "legacy";
  source_outcome: PlannerContextCaptureRecord["live_outcome"];
  replayed_at: number;
  mode: PlannerAbReplayMode;
  pairing_status: "paired" | "excluded_source_outcome" | "skipped_fidelity";
  fidelity: {
    storedVerified: boolean;
    currentSourceRequestMatchesCapture: boolean;
  };
  execution_order: readonly ["compact" | "legacy", "compact" | "legacy"];
  messages: { count: number; chars: number };
  surfaces: { compact: PlannerAbSurfaceSummary; legacy: PlannerAbSurfaceSummary };
  size_delta: {
    compact_minus_legacy_chars: number;
    compact_minus_legacy_estimated_tokens: number;
  };
  live?: { compact: PlannerAbLiveOutcome; legacy: PlannerAbLiveOutcome };
};

export type ReplayPlannerContextCaptureOptions =
  | {
      mode: "dry";
      pairIndex?: number;
      now?: () => number;
      includeNonCompleted?: boolean;
    }
  | {
      mode: "live";
      llmClient: LLMClient;
      pairIndex?: number;
      now?: () => number;
      includeNonCompleted?: boolean;
    };

function fingerprintsEqual(
  left: PlannerSurfaceFingerprint,
  right: PlannerSurfaceFingerprint,
): boolean {
  return (
    left.systemChars === right.systemChars &&
    left.systemSha256 === right.systemSha256 &&
    left.transportSha256 === right.transportSha256 &&
    left.systemBlockCount === right.systemBlockCount &&
    left.cacheBreakpointCount === right.cacheBreakpointCount
  );
}

function requestFingerprintsEqual(
  left: PlannerRequestFingerprint | null,
  right: PlannerRequestFingerprint,
): boolean {
  return (
    left !== null &&
    left.canonicalChars === right.canonicalChars &&
    left.canonicalSha256 === right.canonicalSha256
  );
}

function surfaceSummary(
  surface: CapturedPlannerSurface,
  expectedFingerprint: PlannerSurfaceFingerprint,
): PlannerAbSurfaceSummary {
  return {
    fingerprint: surface.fingerprint,
    expectedFingerprint,
    byteFaithfulToCapture: fingerprintsEqual(surface.fingerprint, expectedFingerprint),
    traceSummary: surface.rendered.traceSummary,
  };
}

function compactSystemBlocks(surface: CapturedPlannerSurface): readonly LLMSystemBlock[] {
  if (typeof surface.rendered.system === "string") {
    throw new TypeError("Compact planner replay unexpectedly rendered a string system prompt");
  }
  return surface.rendered.system;
}

async function executeVariant(
  record: PlannerContextCaptureRecord,
  surface: CapturedPlannerSurface,
  variant: "compact" | "legacy",
  llmClient: LLMClient,
): Promise<PlannerAbLiveOutcome> {
  const startedAt = performance.now();
  let outcome: S2PlannerOutcome | undefined;
  let requestAttempt = 0;
  let requestFingerprint: PlannerRequestFingerprint | null = null;

  try {
    // This boundary can only perform the unary planner request and schema
    // extraction. No substrate writer, repository, retrieval callback, tool
    // dispatcher, or working-memory service is reachable from replay.
    const result = await runS2Planner({
      llmClient,
      model: record.render_input.model,
      baseSystemPrompt: record.render_input.legacyBaseSystemPrompt,
      dialogueMessages: record.render_input.dialogueMessages,
      selfSnapshot: record.render_input.compactContext.selfSnapshot as unknown as Parameters<
        typeof runS2Planner
      >[0]["selfSnapshot"],
      additionalPromptSections: record.render_input.additionalPromptSections,
      maxTokens: record.render_input.maxTokens,
      ...(record.render_input.thinking === undefined
        ? {}
        : { thinking: record.render_input.thinking }),
      ...(record.render_input.effort === undefined ? {} : { effort: record.render_input.effort }),
      turnOrigin: record.render_input.compactContext.turnOrigin,
      plannerSurface:
        variant === "compact"
          ? {
              variant: "compact",
              system: compactSystemBlocks(surface),
              traceSummary: surface.rendered.traceSummary,
            }
          : { variant: "legacy" },
      onRequestPrepared: (prepared) => {
        if (requestFingerprint === null) {
          requestAttempt = prepared.attempt;
          requestFingerprint = fingerprintPlannerRequest(prepared);
        }
      },
      onOutcome: (plannerOutcome) => {
        outcome = plannerOutcome;
      },
    });
    if (outcome?.status === "threw") {
      throw new Error("Planner returned after reporting a threw outcome");
    }
    const finished: Exclude<S2PlannerOutcome, { status: "threw" }> =
      outcome ??
      (result.plan === null
        ? {
            status: "degraded",
            attempts: requestAttempt || 1,
            structuralReason: "missing_emit_turn_plan_tool_use",
          }
        : {
            status: "completed",
            attempts: requestAttempt || 1,
            structuralReason: "emit_turn_plan",
          });
    const payload = {
      durationMs: performance.now() - startedAt,
      plan: result.plan,
      reasoning: result.reasoning,
      usage: result.usage,
      requestFingerprint,
    };
    if (finished.status === "completed") {
      return { ...finished, ...payload };
    }
    return { ...finished, ...payload };
  } catch (error) {
    const thrown =
      outcome?.status === "threw"
        ? outcome
        : {
            status: "threw" as const,
            attempts: requestAttempt,
            structuralReason: "non_retryable_planner_error" as const,
            error:
              error instanceof Error
                ? { name: error.name, message: error.message }
                : { name: "UnknownThrownValue", message: String(error) },
          };
    return {
      ...thrown,
      durationMs: performance.now() - startedAt,
      requestFingerprint,
    };
  }
}

export async function replayPlannerContextCapture(
  record: PlannerContextCaptureRecord,
  options: ReplayPlannerContextCaptureOptions,
): Promise<PlannerAbReplayResultRecord> {
  const pair = renderCapturedPlannerSurfacePair(record.render_input);
  const pairIndex = options.pairIndex ?? 0;
  const executionOrder = (
    pairIndex % 2 === 0 ? ["compact", "legacy"] : ["legacy", "compact"]
  ) as readonly ["compact" | "legacy", "compact" | "legacy"];
  const now = options.now ?? Date.now;
  const compactSummary = surfaceSummary(pair.compact, record.expected_surfaces.compact);
  const legacySummary = surfaceSummary(pair.legacy, record.expected_surfaces.legacy);
  const currentSourceRequest = createS2PlannerRequestSnapshot({
    attempt: 1,
    system: pair[record.live_surface_variant].rendered.system,
    messages: record.render_input.dialogueMessages,
    model: record.render_input.model,
    maxTokens: record.render_input.maxTokens,
    ...(record.render_input.thinking === undefined
      ? {}
      : { thinking: record.render_input.thinking }),
    ...(record.render_input.effort === undefined ? {} : { effort: record.render_input.effort }),
    ...(record.render_input.compactContext.turnOrigin === undefined
      ? {}
      : { turnOrigin: record.render_input.compactContext.turnOrigin }),
  });
  const currentSourceRequestMatchesCapture = requestFingerprintsEqual(
    record.fidelity.liveRequest,
    fingerprintPlannerRequest(currentSourceRequest),
  );
  const pairingStatus =
    record.live_outcome.status !== "completed" && options.includeNonCompleted !== true
      ? ("excluded_source_outcome" as const)
      : options.mode === "live" &&
          (record.fidelity.verified !== true || !currentSourceRequestMatchesCapture)
        ? ("skipped_fidelity" as const)
        : ("paired" as const);
  const base = {
    schema_version: 2 as const,
    capture_id: record.capture_id,
    source_turn_id: record.turn_id,
    source_session_id: record.session_id,
    source_captured_at: record.captured_at,
    source_live_surface_variant: record.live_surface_variant,
    source_outcome: record.live_outcome,
    replayed_at: now(),
    mode: options.mode,
    pairing_status: pairingStatus,
    fidelity: {
      storedVerified: record.fidelity.verified,
      currentSourceRequestMatchesCapture,
    },
    execution_order: executionOrder,
    messages: {
      count: record.render_input.dialogueMessages.length,
      chars: record.render_input.dialogueMessages.reduce(
        (sum, message) => sum + message.content.length,
        0,
      ),
    },
    surfaces: { compact: compactSummary, legacy: legacySummary },
    size_delta: {
      compact_minus_legacy_chars:
        compactSummary.fingerprint.systemChars - legacySummary.fingerprint.systemChars,
      compact_minus_legacy_estimated_tokens:
        compactSummary.traceSummary.totalEstimatedTokens -
        legacySummary.traceSummary.totalEstimatedTokens,
    },
  } satisfies Omit<PlannerAbReplayResultRecord, "live">;

  if (options.mode === "dry" || pairingStatus !== "paired") {
    return base;
  }

  const outcomes = {} as Record<"compact" | "legacy", PlannerAbLiveOutcome>;
  for (const variant of executionOrder) {
    outcomes[variant] = await executeVariant(record, pair[variant], variant, options.llmClient);
  }
  return { ...base, live: outcomes };
}
