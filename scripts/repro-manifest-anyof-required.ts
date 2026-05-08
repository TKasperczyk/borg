// Test whether Anthropic structured-outputs accepts a flat schema with
// per-kind `required` constraints layered on top via anyOf.
//
// The shape: ONE property bag at the parent (no duplication), with a thin
// anyOf below where each branch only sets `kind: {const}` and `required:
// [...]`. This is grammar-additive, not multiplicative like the original
// 9-branch discriminated union with full per-branch property bags.
//
// If this works:
//  - Anthropic enforces "user_fact must have evidence + exact_values + confidence"
//  - "agent_self_provenance must have evidence"
//  - etc.
// And our local tighten step still strips per-kind unrelated fields.
//
// Run: pnpm tsx scripts/repro-manifest-anyof-required.ts ["custom message"]

import { AnthropicLLMClient } from "../src/llm/index.ts";

const SMOKE_MODEL = "claude-opus-4-7";

const userMessage =
  process.argv[2] ??
  "Did we settle on 4096 dimensions for the embedding pipeline last time, or was that a different project?";

const SYSTEM_PROMPT = [
  "You are Borg. Reply briefly and emit a manifest of claims.",
  "",
  "Claim kinds and what each requires:",
  "- discourse_only: connective tissue, acknowledgments, non-factual moves. No required fields.",
  "- hedge: qualifiers without sourced specifics. No required fields.",
  "- self_report: first-person interior state. Requires persistence_class: assistant_self_report.",
  "- agent_self_provenance: claim about your own prior behavior, anchored in cited evidence. Requires evidence.",
  "- user_fact: sourced user-specific detail. Requires evidence + exact_values + confidence.",
  "- prior_callback: 'you said', 'earlier'. Requires callback_scope + evidence.",
  "- action_state: action_record_id + asserted_state + evidence (all required).",
  "- slot_fact: slot_id + exact_values + evidence (all required).",
  "- interpretation: evidence + confidence + persistence_allowed: false (all required).",
  "",
  "If you don't have evidence for a span, use discourse_only or hedge -- not agent_self_provenance.",
].join("\n");

const evidenceItem = {
  type: "object",
  properties: {
    id: { type: "string" },
    source_type: {
      type: "string",
      enum: [
        "current_user_message",
        "current_session_stream",
        "prior_session_stream",
        "episode",
        "semantic_node",
        "semantic_edge",
        "action_record",
        "relational_slot",
        "commitment",
        "assistant_stream",
        "system_metadata",
      ],
    },
  },
  required: ["id", "source_type"],
  additionalProperties: false,
} as const;

const claimSchema = {
  type: "object",
  properties: {
    kind: {
      type: "string",
      enum: [
        "discourse_only",
        "user_fact",
        "prior_callback",
        "action_state",
        "slot_fact",
        "agent_self_provenance",
        "self_report",
        "interpretation",
        "hedge",
      ],
    },
    rendered_span: { type: "string" },
    addresses_audience_by_name: { type: "boolean" },
    exact_values: { type: "array", items: { type: "string" } },
    evidence: { type: "array", items: evidenceItem },
    confidence: {
      type: "string",
      enum: ["direct", "inferred", "uncertain", "low", "medium", "high"],
    },
    scope_disclosure_span: { type: "string" },
    callback_scope: {
      type: "string",
      enum: ["current_turn", "current_session_prior", "prior_session"],
    },
    action_record_id: { type: "string" },
    asserted_state: {
      type: "string",
      enum: [
        "considering",
        "committed_to_do",
        "scheduled",
        "completed",
        "not_done",
        "unknown",
      ],
    },
    slot_id: { type: "string" },
    persistence_class: { type: "string", const: "assistant_self_report" },
    persistence_allowed: { type: "boolean", const: false },
  },
  required: ["kind", "rendered_span"],
  additionalProperties: false,
  allOf: [
    {
      if: { properties: { kind: { const: "user_fact" } } },
      then: { required: ["evidence", "exact_values", "confidence"] },
    },
    {
      if: { properties: { kind: { const: "prior_callback" } } },
      then: { required: ["evidence", "callback_scope"] },
    },
    {
      if: { properties: { kind: { const: "action_state" } } },
      then: { required: ["evidence", "action_record_id", "asserted_state"] },
    },
    {
      if: { properties: { kind: { const: "slot_fact" } } },
      then: { required: ["evidence", "exact_values", "slot_id"] },
    },
    {
      if: { properties: { kind: { const: "agent_self_provenance" } } },
      then: { required: ["evidence"] },
    },
    {
      if: { properties: { kind: { const: "self_report" } } },
      then: { required: ["persistence_class"] },
    },
    {
      if: { properties: { kind: { const: "interpretation" } } },
      then: { required: ["evidence", "confidence", "persistence_allowed"] },
    },
  ],
} as const;

const responseSchema = {
  type: "object",
  properties: {
    final_text: { type: "string" },
    discourse_act: {
      type: "string",
      enum: [
        "answer",
        "clarify",
        "challenge_frame",
        "acknowledge",
        "continue_task",
        "boundary",
        "no_output",
      ],
    },
    claims: { type: "array", items: claimSchema },
    no_output_reason: { type: "string" },
  },
  required: ["final_text", "discourse_act", "claims"],
  additionalProperties: false,
} as const;

async function main(): Promise<void> {
  const client = new AnthropicLLMClient({ authMode: "oauth", env: process.env });

  const result = await client.complete({
    model: SMOKE_MODEL,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userMessage }],
    output_config: { format: { type: "json_schema", schema: responseSchema } },
    max_tokens: 1024,
    budget: "manifest-finalizer-anyof-repro",
  });

  process.stdout.write(`stop_reason=${result.stop_reason}\n`);
  process.stdout.write(
    `input_tokens=${result.input_tokens} output_tokens=${result.output_tokens}\n`,
  );
  process.stdout.write(
    `structured_output:\n${JSON.stringify(result.structured_output, null, 2)}\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`REPRO FAIL: ${error instanceof Error ? error.message : String(error)}\n`);
  const cause = (error as { cause?: unknown }).cause;
  if (cause !== undefined) {
    process.stderr.write(`cause: ${JSON.stringify(cause)}\n`);
  }
  process.exit(1);
});
