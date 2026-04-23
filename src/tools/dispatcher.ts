import { z } from "zod";

import { StreamWriter } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { ToolError } from "../util/errors.js";
import { createStreamEntryId, DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";

export type ToolOrigin = "autonomous" | "deliberator";

export type ToolInvocationContext = {
  sessionId: SessionId;
  origin: ToolOrigin;
  provenance?: unknown;
};

export type ToolDefinition<Input = unknown, Output = unknown> = {
  name: string;
  description: string;
  allowedOrigins: readonly ToolOrigin[];
  writeScope: "read" | "write";
  inputSchema: z.ZodType<Input>;
  outputSchema: z.ZodType<Output>;
  invoke(input: Input, context: ToolInvocationContext): Promise<Output>;
};

export type ToolDispatchCall = {
  callId?: string;
  toolName: string;
  input: unknown;
  sessionId?: SessionId;
  origin: ToolOrigin;
  provenance?: unknown;
  timeoutMs?: number;
};

export type ToolSkippedCall = {
  callId?: string;
  toolName: string;
  input: unknown;
  sessionId?: SessionId;
  origin: ToolOrigin;
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
      durationMs: number;
    };

export type ToolDispatcherOptions = {
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
  clock?: Clock;
  defaultTimeoutMs?: number;
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

function withTimeout<Output>(
  promise: Promise<Output>,
  timeoutMs: number,
): Promise<SettledInvocation<Output> | { kind: "timeout" }> {
  const settledPromise = promise.then<SettledInvocation<Output>, SettledInvocation<Output>>(
    (value) => ({
      kind: "resolved",
      value,
    }),
    (error) => ({
      kind: "rejected",
      error,
    }),
  );
  let timeoutHandle: ReturnType<typeof setTimeout> | undefined;

  return Promise.race([
    settledPromise,
    new Promise<{ kind: "timeout" }>((resolve) => {
      timeoutHandle = setTimeout(() => resolve({ kind: "timeout" }), timeoutMs);
    }),
  ]).finally(() => {
    if (timeoutHandle !== undefined) {
      clearTimeout(timeoutHandle);
    }
  });
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
        content: {
          call_id: callId,
          tool_name: call.toolName,
          input: call.input,
          origin: call.origin,
          ...(call.provenance === undefined ? {} : { provenance: call.provenance }),
          skipped: true,
          skip_reason: call.skipReason,
        },
      });

      await writer.append({
        kind: "tool_result",
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
    const timeoutMs = call.timeoutMs ?? this.defaultTimeoutMs;
    const startedAt = this.clock.now();
    const writer = this.options.createStreamWriter(sessionId);

    try {
      await writer.append({
        kind: "tool_call",
        content: {
          call_id: callId,
          tool_name: call.toolName,
          input: call.input,
          origin: call.origin,
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
        );
      }

      if (!tool.allowedOrigins.includes(call.origin)) {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          `Tool ${call.toolName} is not allowed for origin ${call.origin}`,
          startedAt,
        );
      }

      const parsedInput = tool.inputSchema.safeParse(call.input);

      if (!parsedInput.success) {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          parsedInput.error.message,
          startedAt,
        );
      }

      const context = {
        sessionId,
        origin: call.origin,
        provenance: call.provenance,
      };
      const invocation =
        tool.writeScope === "read"
          ? await withTimeout(tool.invoke(parsedInput.data, context), timeoutMs)
          : await (async (): Promise<SettledInvocation<unknown>> => {
              // Write-scoped tools are not raced against the dispatcher timeout:
              // a timed-out write can still commit after the model sees failure.
              // Write tools must keep their own execution bounded internally.
              try {
                return {
                  kind: "resolved",
                  value: await tool.invoke(parsedInput.data, context),
                };
              } catch (error) {
                return {
                  kind: "rejected",
                  error,
                };
              }
            })();

      if (invocation.kind === "timeout") {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          `Timed out after ${timeoutMs}ms`,
          startedAt,
        );
      }

      if (invocation.kind === "rejected") {
        return await this.appendErrorResult(
          writer,
          callId,
          call.toolName,
          formatToolError(invocation.error),
          startedAt,
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
        );
      }

      const durationMs = Math.max(0, this.clock.now() - startedAt);
      await writer.append({
        kind: "tool_result",
        content: {
          call_id: callId,
          ok: true,
          output: parsedOutput.data,
          duration_ms: durationMs,
        },
      });

      return {
        callId,
        toolName: call.toolName,
        ok: true,
        output: parsedOutput.data,
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
  ): Promise<ToolDispatchResult> {
    const durationMs = Math.max(0, this.clock.now() - startedAt);

    await writer.append({
      kind: "tool_result",
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
      durationMs,
    };
  }
}
