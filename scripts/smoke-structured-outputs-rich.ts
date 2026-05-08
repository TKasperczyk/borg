// Larger smoke: builds a multi-section EvidenceLedger and asks the model to
// produce claim-bearing prose. Surfaces the actual failure mode the simulator
// is hitting (every turn currently fails with manifest_finalizer_failed).

import {
  runManifestFinalizer,
  type ManifestFinalizerResult,
} from "../src/cognition/deliberation/manifest-finalizer.ts";
import type { EvidenceLedger } from "../src/cognition/evidence-ledger/index.ts";
import type {
  TurnTraceData,
  TurnTraceEventName,
  TurnTracer,
} from "../src/cognition/tracing/tracer.ts";
import { selectScriptClients } from "./_clients.ts";

class CapturingTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;
  readonly events: { event: TurnTraceEventName; data: TurnTraceData }[] = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

const SMOKE_MODEL = "claude-opus-4-7";

const richEvidenceLedger: EvidenceLedger = {
  sections: [
    {
      id: "current_user_message",
      label: "1. Current User Message",
      entries: [
        {
          id: "current_user_message:strm_smokerichcurrn",
          source_type: "current_user_message",
          session_scope: "current_session",
          actor: "user",
          trust_rank: 1,
          text: "Hey, I'm working on the embedding pipeline again. Did we settle on 4096 dimensions last time, or was that for a different project?",
        },
      ],
    },
    {
      id: "current_session_transcript",
      label: "2. Current Session Transcript",
      entries: [
        {
          id: "current_session_transcript:strm_smokerichprio1",
          source_type: "current_session_stream",
          session_scope: "current_session",
          actor: "user",
          trust_rank: 2,
          text: "I picked text-embedding-qwen3-embedding-8b for the embedding model. 4096 dims.",
        },
        {
          id: "current_session_transcript:strm_smokerichprio2",
          source_type: "current_session_stream",
          session_scope: "current_session",
          actor: "assistant",
          trust_rank: 3,
          text: "Acknowledged. Qwen3 embedding 8b at 4096 dims, written to working memory.",
        },
      ],
    },
  ],
  transcriptIncluded: false,
  transcriptOmittedReason: "over_budget",
  estimatedTokens: 320,
};

function describeError(error: unknown): string {
  const lines: string[] = [];
  let cursor: unknown = error;
  let depth = 0;

  while (cursor !== undefined && cursor !== null && depth < 8) {
    if (cursor instanceof Error) {
      lines.push(`${cursor.name}: ${cursor.message}`);
      const status = (cursor as { status?: unknown }).status;
      if (typeof status === "number") {
        lines.push(`  status=${status}`);
      }
      const code = (cursor as { code?: unknown }).code;
      if (typeof code === "string") {
        lines.push(`  code=${code}`);
      }
      const errorBody = (cursor as { error?: unknown }).error;
      if (errorBody !== undefined) {
        lines.push(`  error_body=${JSON.stringify(errorBody)}`);
      }
      cursor = (cursor as { cause?: unknown }).cause;
    } else {
      lines.push(String(cursor));
      break;
    }
    depth += 1;
  }

  return lines.join("\n");
}

function fail(message: string): never {
  process.stderr.write(`SMOKE FAIL: ${message}\n`);
  process.exit(1);
}

async function main(): Promise<void> {
  const start = Date.now();
  const clients = await selectScriptClients({
    mode: "real",
    warn: (message) => process.stderr.write(`WARN ${message}\n`),
  });

  if (clients.llmMode !== "real") {
    fail("real LLM unavailable -- cannot run live smoke. See WARN above.");
  }

  process.stdout.write(`smoke: model=${SMOKE_MODEL} auth=${clients.config.anthropic.auth}\n`);

  const tracer = new CapturingTracer();
  let result: ManifestFinalizerResult;
  try {
    result = await runManifestFinalizer({
      llmClient: clients.llm,
      model: SMOKE_MODEL,
      baseSystemPrompt: [
        "You are Borg, an agent with explicit memory and identity.",
        "Use retrieved evidence to answer. Cite ledger entry IDs.",
      ].join("\n"),
      dialogueMessages: [
        {
          role: "user",
          content:
            "Hey, I'm working on the embedding pipeline again. Did we settle on 4096 dimensions last time, or was that for a different project?",
        },
      ],
      evidenceLedger: richEvidenceLedger,
      maxTokens: 1024,
      path: "system_1",
      tracer,
      turnId: "rich_smoke",
    });
  } catch (error) {
    const parseFailed = tracer.events.find(
      (entry) => entry.event === "manifest_finalizer_parse_failed",
    );
    process.stderr.write(`SMOKE FAIL: runManifestFinalizer threw:\n${describeError(error)}\n`);
    if (parseFailed !== undefined) {
      process.stderr.write(
        `\nparse_failed trace event:\n${JSON.stringify(parseFailed.data, null, 2)}\n`,
      );
    }
    process.exit(1);
  }

  const { manifest, usage } = result;
  const elapsedMs = Date.now() - start;

  process.stdout.write(
    [
      "SMOKE PASS",
      `elapsed_ms=${elapsedMs}`,
      `input_tokens=${usage.input_tokens}`,
      `output_tokens=${usage.output_tokens}`,
      `discourse_act=${manifest.discourse_act}`,
      `claim_count=${manifest.claims.length}`,
      `claim_kinds=${manifest.claims.map((c) => c.kind).join(",")}`,
      `final_text=${JSON.stringify(manifest.final_text).slice(0, 200)}`,
      "claims:",
      JSON.stringify(manifest.claims, null, 2),
      "",
    ].join("\n"),
  );
}

main().catch((error) => {
  fail(error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error));
});
