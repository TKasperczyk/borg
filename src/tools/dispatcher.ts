import { z } from "zod";
import type { RecordIdResolver } from "./record-id-resolver.js";
import {
  resolveToolRecordIds,
  missingToolIdOutput,
  type RecordIdInput,
  type ToolIdResolution,
} from "./record-id-inputs.js";
import { isPlainRecord } from "../util/guards.js";

import type { TurnOrigin } from "../contracts/cognitive-contracts.js";
import type { BorgRole } from "../memory/commitments/index.js";
import type { SessionAudienceRole } from "../sessions/index.js";
import { StreamWriter } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { BorgError, ToolError } from "../util/errors.js";
import type { RetrievalDegradation } from "../retrieval/degradation.js";
import type { TurnTracer } from "../tracing/tracer.js";
import {
  runWithTimeoutPolicy,
  TimeoutBudgetExceeded,
  type TimeoutBudget,
  type TimeoutLoadSignals,
} from "../util/timeout-policy.js";
import {
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";

export type ToolOrigin = "autonomous" | "deliberator";

export type ToolInvocationContext = {
  sessionId: SessionId;
  origin: ToolOrigin;
  turnOrigin?: TurnOrigin;
  turnId?: string;
  audienceEntityId?: EntityId | null;
  currentSenderBorgRole?: BorgRole | null;
  sessionAudienceRole?: SessionAudienceRole;
  toolCallEntryId?: StreamEntryId;
  provenance?: unknown;
  signal?: AbortSignal;
  onRetrievalDegraded?: (degradation: RetrievalDegradation) => void;
};

export type ToolDefinition<Input = unknown, Output = unknown> = {
  name: string;
  description: string;
  menuSummary?: string;
  allowedOrigins: readonly ToolOrigin[];
  writeScope: "read" | "write";
  inputSchema: z.ZodType<Input>;
  outputSchema: z.ZodType<Output>;
  recordIdInputs?: readonly RecordIdInput[];
  // Per-tool ceiling for tools whose backing store legitimately needs longer
  // than the dispatcher default (vector search over a fragmented index).
  // Structural capacity, not a judgment knob; call-level timeoutMs still wins.
  timeoutMs?: number;
  timeoutClass?: "search";
  invoke(input: Input, context: ToolInvocationContext): Promise<Output>;
};

export type ToolDispatchCall = {
  callId?: string;
  toolName: string;
  input: unknown;
  sessionId?: SessionId;
  turnId?: string;
  origin: ToolOrigin;
  turnOrigin?: TurnOrigin;
  audienceEntityId?: EntityId | null;
  currentSenderBorgRole?: BorgRole | null;
  sessionAudienceRole?: SessionAudienceRole;
  provenance?: unknown;
  timeoutMs?: number;
  timeoutPolicy?: TimeoutBudget;
  deadlineAt?: number;
  signal?: AbortSignal;
  onRetrievalDegraded?: (degradation: RetrievalDegradation) => void;
};

export type ToolSkippedCall = {
  callId?: string;
  toolName: string;
  input: unknown;
  sessionId?: SessionId;
  turnId?: string;
  origin: ToolOrigin;
  turnOrigin?: TurnOrigin;
  audienceEntityId?: EntityId | null;
  currentSenderBorgRole?: BorgRole | null;
  sessionAudienceRole?: SessionAudienceRole;
  provenance?: unknown;
  skipReason: string;
  error?: string;
};

export type ToolDispatchResult<Output = unknown> =
  | {
      callId: string;
      toolName: string;
      ok: true;
      output: Output;
      durationMs: number;
    }
  | {
      callId: string;
      toolName: string;
      ok: false;
      error: string;
      errorCode?: string;
      durationMs: number;
    };

export type ToolDispatcherOptions = {
  recordIdResolver?: RecordIdResolver;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
  clock?: Clock;
  defaultTimeoutMs?: number;
  searchTimeoutMs?: number;
  searchMaxTimeoutMs?: number;
  loadSignals?: (call: ToolDispatchCall) => TimeoutLoadSignals;
  tracer?: Pick<TurnTracer, "emit">;
};

type SettledInvocation<Output> =
  | {
      kind: "resolved";
      value: Output;
    }
  | {
      kind: "rejected";
      error: unknown;
    };

function formatToolError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export class ToolDispatcher {
  private readonly tools = new Map<string, ToolDefinition>();
  private readonly clock: Clock;
  private readonly defaultTimeoutMs: number;

  constructor(private readonly options: ToolDispatcherOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? 5_000;
  }

  register(tool: ToolDefinition): this {
    if (this.tools.has(tool.name)) {
      throw new ToolError(`Tool already registered: ${tool.name}`, {
        code: "TOOL_ALREADY_REGISTERED",
      });
    }

    this.tools.set(tool.name, tool);
    return this;
  }

  listTools(origin: ToolOrigin): ToolDefinition[] {
    return [...this.tools.values()].filter((tool) => tool.allowedOrigins.includes(origin));
  }

  getDefinition(name: string): ToolDefinition | null {
    return this.tools.get(name) ?? null;
  }

  async recordSkippedCall(call: ToolSkippedCall): Promise<ToolDispatchResult> {
    const sessionId = call.sessionId ?? DEFAULT_SESSION_ID;
    const callId = call.callId ?? createStreamEntryId();
    const writer = this.options.createStreamWriter(sessionId);
    const error = call.error ?? call.skipReason;

    try {
      await writer.append({
        kind: "tool_call",
        ...(call.turnId === undefined ? {} : { turn_id: call.turnId }),
        content: {
          call_id: callId,
          tool_name: call.toolName,
          input: call.input,
          origin: call.origin,
          ...(call.turnOrigin === undefined ? {} : { turn_origin: call.turnOrigin }),
          ...(call.provenance === undefined ? {} : { provenance: call.provenance }),
          skipped: true,
          skip_reason: call.skipReason,
        },
      });

      await writer.append({
        kind: "tool_result",
        ...(call.turnId === undefined ? {} : { turn_id: call.turnId }),
        content: {
          call_id: callId,
          ok: false,
          error,
          duration_ms: 0,
        },
      });

      return {
        callId,
        toolName: call.toolName,
        ok: false,
        error,
        durationMs: 0,
      };
    } finally {
      writer.close();
    }
  }

  async dispatch(call: ToolDispatchCall): Promise<ToolDispatchResult> {
    const tool = this.tools.get(call.toolName);
    const sessionId = call.sessionId ?? DEFAULT_SESSION_ID;
    const callId = call.callId ?? createStreamEntryId();
    const timeoutMs =
      call.timeoutMs ??
      call.timeoutPolicy?.timeoutMs ??
      tool?.timeoutMs ??
      (tool?.timeoutClass === "search"
        ? (this.options.searchTimeoutMs ?? 60_000)
        : this.defaultTimeoutMs);
    const maxTimeoutMs =
      call.timeoutMs !== undefined ||
      (call.timeoutPolicy === undefined && tool?.timeoutMs !== undefined)
        ? timeoutMs
        : (call.timeoutPolicy?.maxTimeoutMs ??
          (tool?.timeoutClass === "search"
            ? (this.options.searchMaxTimeoutMs ?? 120_000)
            : timeoutMs));
    const startedAt = this.clock.now();
    const writer = this.options.createStreamWriter(sessionId);

    try {
      const idResolutions: ToolIdResolution[] = [];
      let resolvedInput: ReturnType<typeof resolveToolRecordIds> = {
        input: call.input,
        missing: [],
      };
      let resolutionFailure: { error: unknown } | undefined;
      // The origin guard still precedes all lookup. Persist canonical addresses
      // before invocation so downstream accounting (including outbound limits)
      // observes the same target for full IDs and handles. Retain the mapping.
      if (tool !== undefined && tool.allowedOrigins.includes(call.origin)) {
        try {
          resolvedInput = resolveToolRecordIds(
            tool,
            call.input,
            this.options.recordIdResolver,
            idResolutions,
          );
        } catch (error) {
          resolutionFailure = { error };
        }
      }
      const toolCallEntry = await writer.append({
        kind: "tool_call",
        ...(call.turnId === undefined ? {} : { turn_id: call.turnId }),
        content: {
          call_id: callId,
          tool_name: call.toolName,
          input: resolvedInput.input,
          ...(idResolutions.length === 0 ? {} : { id_resolutions: idResolutions }),
          origin: call.origin,
          ...(call.turnOrigin === undefined ? {} : { turn_origin: call.turnOrigin }),
          ...(call.provenance === undefined ? {} : { provenance: call.provenance }),
        },
      });

      if (tool === undefined) {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          "Unknown tool",
          startedAt,
          call.turnId,
        );
      }

      if (!tool.allowedOrigins.includes(call.origin)) {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          `Tool ${call.toolName} is not allowed for origin ${call.origin}`,
          startedAt,
          call.turnId,
        );
      }

      if (resolutionFailure !== undefined) {
        const { error } = resolutionFailure;
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          formatToolError(error),
          startedAt,
          call.turnId,
          error instanceof BorgError ? error.code : "TOOL_INPUT_INVALID",
          idResolutions,
        );
      }
      const parsedInput = tool.inputSchema.safeParse(resolvedInput.input);

      // A missing prefix has valid ID structure but intentionally cannot pass
      // a full-ID refinement. Other input errors still win over a missing result.
      const remainingIssues = parsedInput.success
        ? []
        : parsedInput.error.issues.filter(
            (issue) =>
              !resolvedInput.missing.some((missing) => issue.path.join(".") === missing.field),
          );
      if (
        !parsedInput.success &&
        (resolvedInput.missing.length === 0 || remainingIssues.length > 0)
      ) {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          parsedInput.error.message,
          startedAt,
          call.turnId,
          "TOOL_INPUT_INVALID",
          idResolutions,
        );
      }

      const context = {
        sessionId,
        origin: call.origin,
        ...(call.turnOrigin === undefined ? {} : { turnOrigin: call.turnOrigin }),
        ...(call.turnId === undefined ? {} : { turnId: call.turnId }),
        ...(call.audienceEntityId === undefined ? {} : { audienceEntityId: call.audienceEntityId }),
        ...(call.currentSenderBorgRole === undefined
          ? {}
          : { currentSenderBorgRole: call.currentSenderBorgRole }),
        ...(call.sessionAudienceRole === undefined
          ? {}
          : { sessionAudienceRole: call.sessionAudienceRole }),
        toolCallEntryId: toolCallEntry.id,
        provenance: call.provenance,
        onRetrievalDegraded: call.onRetrievalDegraded,
      };
      const invocation = await (async (): Promise<SettledInvocation<unknown>> => {
        // Write-scoped tools are not raced against the dispatcher timeout:
        // a timed-out write can still commit after the model sees failure.
        // Write tools must keep their own execution bounded internally.
        try {
          const missing = resolvedInput.missing.find((item) => item.spec.notFound !== "invoke");
          if (missing !== undefined)
            return { kind: "resolved", value: missingToolIdOutput(missing) };
          // Only read filters use 'invoke' for a missing prefix. Their original
          // implementation owns the empty-range semantics (including unfiltered lanes).
          const input = parsedInput.success ? parsedInput.data : resolvedInput.input;
          return {
            kind: "resolved",
            value:
              tool.writeScope === "read"
                ? await runWithTimeoutPolicy({
                    timeoutMs,
                    maxTimeoutMs,
                    deadlineAt: call.deadlineAt,
                    signal: call.signal,
                    clock: this.clock,
                    loadSignals: () => this.options.loadSignals?.(call) ?? {},
                    observe: (observation) =>
                      this.options.tracer?.emit("tool_call.budget", {
                        turnId: call.turnId ?? callId,
                        session_id: sessionId,
                        callId,
                        toolName: call.toolName,
                        ...observation,
                      }),
                    run: (signal) => tool.invoke(input, { ...context, signal }),
                  })
                : await tool.invoke(input, context),
          };
        } catch (error) {
          return {
            kind: "rejected",
            error,
          };
        }
      })();

      if (invocation.kind === "rejected") {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          invocation.error instanceof TimeoutBudgetExceeded
            ? invocation.error.message
            : formatToolError(invocation.error),
          startedAt,
          call.turnId,
          invocation.error instanceof BorgError ? invocation.error.code : "TOOL_INVOCATION_FAILED",
          idResolutions,
        );
      }

      const parsedOutput = tool.outputSchema.safeParse(invocation.value);

      if (!parsedOutput.success) {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          parsedOutput.error.message,
          startedAt,
          call.turnId,
          undefined,
          idResolutions,
        );
      }

      const output =
        idResolutions.length > 0 && isPlainRecord(parsedOutput.data)
          ? { ...parsedOutput.data, id_resolutions: idResolutions }
          : parsedOutput.data;
      const durationMs = Math.max(0, this.clock.now() - startedAt);
      await writer.append({
        kind: "tool_result",
        ...(call.turnId === undefined ? {} : { turn_id: call.turnId }),
        content: {
          call_id: callId,
          ok: true,
          output,
          duration_ms: durationMs,
        },
      });

      return {
        callId,
        toolName: call.toolName,
        ok: true,
        output,
        durationMs,
      };
    } finally {
      writer.close();
    }
  }

  private async appendErrorResult(
    writer: StreamWriter,
    callId: string,
    toolName: string,
    error: string,
    startedAt: number,
    turnId?: string,
    errorCode?: string,
    idResolutions: readonly ToolIdResolution[] = [],
  ): Promise<ToolDispatchResult> {
    if (idResolutions.length > 0) error += `\n${JSON.stringify({ id_resolutions: idResolutions })}`;
    const durationMs = Math.max(0, this.clock.now() - startedAt);

    await writer.append({
      kind: "tool_result",
      ...(turnId === undefined ? {} : { turn_id: turnId }),
      content: {
        call_id: callId,
        ok: false,
        error,
        duration_ms: durationMs,
      },
    });

    return {
      callId,
      toolName,
      ok: false,
      error,
      ...(errorCode === undefined ? {} : { errorCode }),
      durationMs,
    };
  }
}
