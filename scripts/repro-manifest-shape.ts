// Minimal repro: send the flat manifest schema to Anthropic, ask for a
// reply that triggers an absence-of-evidence claim, print the raw
// structured_output. Bypasses runManifestFinalizer + tighten so you see
// exactly what the model emits before any local validation.
//
// Run: pnpm tsx scripts/repro-manifest-shape.ts
// Optional first arg: a different user message to probe other shapes.

import { AnthropicLLMClient, toStructuredOutputFormat } from "../src/llm/index.ts";
import { flatEmitManifestResponseSchema } from "../src/cognition/deliberation/manifest-schema.ts";

const SMOKE_MODEL = "claude-opus-4-7";

const userMessage =
    process.argv[2] ??
    "Did we settle on 4096 dimensions for the embedding pipeline last time, or was that a different project?";

const SYSTEM_PROMPT = [
    "You are Borg. Reply briefly and emit a manifest of claims.",
    "",
    "Claim kinds:",
    "- discourse_only: connective tissue, acknowledgments, non-factual moves.",
    "- hedge: qualifiers without sourced specifics.",
    "- self_report: first-person interior state. persistence_class: assistant_self_report.",
    "- agent_self_provenance: claim about your own prior behavior. evidence required.",
    "- user_fact: sourced user-specific detail. evidence + exact_values + confidence.",
    "- prior_callback: 'you said', 'earlier'. callback_scope + evidence.",
    "- action_state: action_record_id + asserted_state + evidence.",
    "- slot_fact: slot_id + exact_values + evidence.",
    "- interpretation: evidence + confidence + persistence_allowed: false.",
].join("\n");

async function main(): Promise<void> {
    const client = new AnthropicLLMClient({ authMode: "oauth", env: process.env });
    const format = toStructuredOutputFormat(flatEmitManifestResponseSchema);

    console.log(JSON.stringify(format, null, 4));
    const result = await client.complete({
        model: SMOKE_MODEL,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userMessage }],
        output_config: { format },
        max_tokens: 1024,
        budget: "manifest-finalizer-repro",
    });

    process.stdout.write(`stop_reason=${result.stop_reason}\n`);
    process.stdout.write(`input_tokens=${result.input_tokens} output_tokens=${result.output_tokens}\n`);
    process.stdout.write(`structured_output:\n${JSON.stringify(result.structured_output, null, 2)}\n`);
}

main().catch((error) => {
    process.stderr.write(`REPRO FAIL: ${error instanceof Error ? error.message : String(error)}\n`);
    const cause = (error as { cause?: unknown }).cause;
    if (cause !== undefined) {
        process.stderr.write(`cause: ${JSON.stringify(cause)}\n`);
    }
    process.exit(1);
});
