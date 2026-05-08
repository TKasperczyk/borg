// Live smoke test for the manifest finalizer's structured-outputs path.
// Hits the real Anthropic API once and verifies the response parses against
// emitManifestResponseSchema. Exit 0 on PASS, non-zero on FAIL. Used as a
// precondition gate before launching simulator runs.

import {
  runManifestFinalizer,
  type ManifestFinalizerResult,
} from "../src/cognition/deliberation/manifest-finalizer.ts";
import type { EvidenceLedger } from "../src/cognition/evidence-ledger/index.ts";
import { selectScriptClients } from "./_clients.ts";

const SMOKE_MODEL = "claude-opus-4-7";

const minimalEvidenceLedger: EvidenceLedger = {
  sections: [
    {
      id: "current_user_message",
      label: "1. Current User Message",
      entries: [
        {
          id: "current_user_message:strm_smoke00000000",
          source_type: "current_user_message",
          session_scope: "current_session",
          actor: "user",
          trust_rank: 1,
          text: "Say 'Acknowledged.' and nothing else.",
        },
      ],
    },
  ],
  transcriptIncluded: false,
  transcriptOmittedReason: "over_budget",
  estimatedTokens: 16,
};

function describeError(error: unknown): string {
  const lines: string[] = [];
  let cursor: unknown = error;
  let depth = 0;

  while (cursor !== undefined && cursor !== null && depth < 6) {
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

  let result: ManifestFinalizerResult;
  try {
    result = await runManifestFinalizer({
      llmClient: clients.llm,
      model: SMOKE_MODEL,
      baseSystemPrompt:
        "You are running a smoke test. Respond with the literal word 'Acknowledged.' and nothing more.",
      dialogueMessages: [
        { role: "user", content: "Say 'Acknowledged.' and nothing else." },
      ],
      evidenceLedger: minimalEvidenceLedger,
      maxTokens: 512,
      path: "system_1",
    });
  } catch (error) {
    fail(`runManifestFinalizer threw:\n${describeError(error)}`);
  }

  const { manifest, usage } = result;
  const elapsedMs = Date.now() - start;

  if (manifest.final_text.length === 0) {
    fail("manifest.final_text was empty");
  }

  if (!Array.isArray(manifest.claims)) {
    fail("manifest.claims was not an array");
  }

  process.stdout.write(
    [
      "SMOKE PASS",
      `elapsed_ms=${elapsedMs}`,
      `input_tokens=${usage.input_tokens}`,
      `output_tokens=${usage.output_tokens}`,
      `discourse_act=${manifest.discourse_act}`,
      `claim_count=${manifest.claims.length}`,
      `final_text=${JSON.stringify(manifest.final_text).slice(0, 120)}`,
      "",
    ].join("\n"),
  );
}

main().catch((error) => {
  fail(error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error));
});
