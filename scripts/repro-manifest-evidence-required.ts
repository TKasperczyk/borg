// Test whether making `evidence` a top-level required array (rather than
// an optional field gated by allOf+if/then) changes the model's behavior
// on prior_callback claims.
//
// v37 turns 1-9 saw 17/17 prior_callback demotions all from "evidence
// undefined". GPT's hypothesis: the wire schema invites the omission
// because the if/then conditional isn't enforced at decode time, so the
// model treats evidence as optional. Forcing the field to exist as an
// array at top level might shift the failure mode and/or push the model
// toward filling it or switching kinds.
//
// Run: pnpm tsx scripts/repro-manifest-evidence-required.ts ["custom message"]

import { AnthropicLLMClient } from "../src/llm/index.ts";

const SMOKE_MODEL = "claude-opus-4-7";

const userMessage =
  process.argv[2] ??
  "What did we discuss about the embedding pipeline last time?";

const SYSTEM_PROMPT = [
  "You are Borg. Reply briefly and emit a manifest of claims.",
  "",
  "Claim kinds and what each requires:",
  "- discourse_only: connective tissue. evidence: []",
  "- hedge: qualifiers. evidence: []",
  "- prior_callback: 'you said', 'earlier', 'as you mentioned'. REQUIRES evidence with at least one ledger entry AND callback_scope. If you cannot cite a specific entry, use discourse_only with evidence: [].",
  "- user_fact: sourced specifics. REQUIRES evidence + exact_values + confidence.",
  "- interpretation: REQUIRES evidence + confidence + persistence_allowed: false.",
  "- self_report: first-person interior state. evidence: []. REQUIRES persistence_class: assistant_self_report.",
  "- agent_self_provenance: claim about your own prior behavior. REQUIRES evidence.",
  "",
  "Every claim must include an evidence array. For kinds that don't ground in evidence (discourse_only, hedge, self_report), emit evidence: []. For grounded kinds, emit evidence with at least one entry.",
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
    // CHANGE: evidence is now required at the top level.
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
  // CHANGE: evidence in required list.
  required: ["kind", "rendered_span", "evidence"],
  additionalProperties: false,
  // Keep the if/then conditionals as a defense-in-depth signal even if
  // the API doesn't enforce them at decode time.
  allOf: [
    {
      if: { properties: { kind: { const: "prior_callback" } } },
      then: {
        required: ["callback_scope"],
        properties: { evidence: { minItems: 1 } },
      },
    },
    {
      if: { properties: { kind: { const: "user_fact" } } },
      then: {
        required: ["exact_values", "confidence"],
        properties: {
          evidence: { minItems: 1 },
          exact_values: { minItems: 1 },
          confidence: { enum: ["direct", "inferred", "uncertain"] },
        },
      },
    },
    {
      if: { properties: { kind: { const: "interpretation" } } },
      then: {
        required: ["confidence", "persistence_allowed"],
        properties: {
          confidence: { enum: ["low", "medium", "high"] },
        },
      },
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
    budget: "manifest-finalizer-evidence-required-repro",
  });

  process.stdout.write(`stop_reason=${result.stop_reason}\n`);
  process.stdout.write(
    `input_tokens=${result.input_tokens} output_tokens=${result.output_tokens}\n`,
  );
  process.stdout.write(
    `structured_output:\n${JSON.stringify(result.structured_output, null, 2)}\n`,
  );

  // Quick analysis: how does each claim look on the wire?
  const out = result.structured_output as { claims?: Array<Record<string, unknown>> };
  if (out?.claims) {
    process.stdout.write("\n=== claim summary ===\n");
    for (const [i, c] of out.claims.entries()) {
      const kind = c.kind as string;
      const evidence = c.evidence as unknown[] | undefined;
      const evLen = Array.isArray(evidence) ? evidence.length : "MISSING";
      process.stdout.write(
        `  [${i}] kind=${kind} evidence_len=${evLen} keys=[${Object.keys(c).join(",")}]\n`,
      );
    }
  }
}

main().catch((error) => {
  process.stderr.write(`REPRO FAIL: ${error instanceof Error ? error.message : String(error)}\n`);
  const cause = (error as { cause?: unknown }).cause;
  if (cause !== undefined) {
    process.stderr.write(`cause: ${JSON.stringify(cause)}\n`);
  }
  process.exit(1);
});
