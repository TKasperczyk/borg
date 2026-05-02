import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { commitmentIdSchema, commitmentTypeSchema } from "../../memory/commitments/index.js";
import type { CommitmentId, EntityId } from "../../util/ids.js";
import type { RecencyMessage } from "../recency/index.js";

const CONFIDENCE_THRESHOLD = 0.8;
const CORRECTIVE_PREFERENCE_TOOL_NAME = "EmitCorrectivePreference";

const correctivePreferenceSchema = z
  .object({
    classification: z
      .enum(["corrective_preference", "none"])
      .describe(
        "Use corrective_preference only when the user is asking Borg to change durable future response behavior; use none for ordinary conversation, venting, task requests, or one-turn remarks.",
      ),
    type: commitmentTypeSchema
      .exclude(["promise"])
      .nullable()
      .describe(
        "Classify the durable response-behavior change as preference, rule, or boundary. Use null when classification is none.",
      ),
    directive: z
      .string()
      .nullable()
      .describe(
        "A concise first-person operational directive Borg can enforce when drafting or revising responses. Use null when classification is none.",
      ),
    priority: z
      .number()
      .int()
      .nullable()
      .describe(
        "Relative enforcement priority. Use higher values for explicit prohibitions or boundaries, lower values for softer style preferences. Use null when classification is none.",
      ),
    reason: z
      .string()
      .min(1)
      .describe("Brief semantic reason for the classification, grounded in the current user turn."),
    confidence: z
      .number()
      .min(0)
      .max(1)
      .describe("Confidence that the current user turn is making a durable correction."),
    supersedes_commitment_id: commitmentIdSchema
      .nullable()
      .optional()
      .describe(
        "Existing commitment id this correction replaces or tightens, if one was clearly selected from the supplied active commitments.",
      ),
  })
  .strict();

const CORRECTIVE_PREFERENCE_TOOL = {
  name: CORRECTIVE_PREFERENCE_TOOL_NAME,
  description:
    "Classify whether the current user turn creates a durable correction to Borg's future response behavior.",
  inputSchema: toToolInputSchema(correctivePreferenceSchema),
} satisfies LLMToolDefinition;

type CorrectivePreferenceToolInput = z.infer<typeof correctivePreferenceSchema>;

class MissingCorrectivePreferenceToolCallError extends Error {}

export type CorrectivePreferenceCandidate = {
  type: Exclude<z.infer<typeof commitmentTypeSchema>, "promise">;
  directive: string;
  priority: number;
  reason: string;
  confidence: number;
  supersedes_commitment_id?: CommitmentId | null;
};

export type CorrectivePreferenceExtractorDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type CorrectivePreferenceExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (
    reason: CorrectivePreferenceExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractCorrectivePreferenceInput = {
  userMessage: string;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  activeCommitments: readonly {
    id: CommitmentId;
    type: string;
    directive: string;
    priority: number;
  }[];
};

function toCandidate(input: CorrectivePreferenceToolInput): CorrectivePreferenceCandidate | null {
  if (input.classification !== "corrective_preference" || input.confidence < CONFIDENCE_THRESHOLD) {
    return null;
  }

  if (input.type === null || input.directive === null || input.priority === null) {
    return null;
  }

  const directive = input.directive.trim();
  const reason = input.reason.trim();

  if (directive.length === 0 || reason.length === 0) {
    return null;
  }

  return {
    type: input.type,
    directive,
    priority: input.priority,
    reason,
    confidence: input.confidence,
    supersedes_commitment_id: input.supersedes_commitment_id ?? null,
  };
}

function parseResponse(result: LLMCompleteResult): CorrectivePreferenceCandidate | null {
  const call = result.tool_calls.find(
    (toolCall) => toolCall.name === CORRECTIVE_PREFERENCE_TOOL_NAME,
  );

  if (call === undefined) {
    throw new MissingCorrectivePreferenceToolCallError(
      `Corrective preference extractor did not emit ${CORRECTIVE_PREFERENCE_TOOL_NAME}`,
    );
  }

  const parsed = correctivePreferenceSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return toCandidate(parsed.data);
}

export class CorrectivePreferenceExtractor {
  constructor(private readonly options: CorrectivePreferenceExtractorOptions = {}) {}

  private async degraded(
    reason: CorrectivePreferenceExtractorDegradedReason,
    error?: unknown,
  ): Promise<null> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return null;
  }

  async extract(
    input: ExtractCorrectivePreferenceInput,
  ): Promise<CorrectivePreferenceCandidate | null> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    try {
      return parseResponse(
        await this.options.llmClient.complete({
          model: this.options.model,
          system: [
            "Classify whether the user is making a durable correction to Borg's future response behavior.",
            "Return corrective_preference only when the user is directing Borg to change how it should answer in future turns, such as a recurring style, boundary, interaction rule, or response pattern.",
            "Return none for ordinary task requests, emotional disclosure, venting, disagreement, one-turn instructions, or discussion about a behavior without asking Borg to adopt a lasting change.",
            "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
            "When uncertain, return none. The directive must be enforceable by a later response checker without needing to remember the current phrasing.",
          ].join("\n"),
          messages: [
            {
              role: "user",
              content: JSON.stringify({
                current_user_message: input.userMessage,
                recent_history: input.recentHistory.slice(-8).map((message) => ({
                  role: message.role,
                  content: message.content,
                })),
                audience_entity_id: input.audienceEntityId,
                active_commitments: input.activeCommitments.map((commitment) => ({
                  id: commitment.id,
                  type: commitment.type,
                  directive: commitment.directive,
                  priority: commitment.priority,
                })),
              }),
            },
          ],
          tools: [CORRECTIVE_PREFERENCE_TOOL],
          tool_choice: { type: "tool", name: CORRECTIVE_PREFERENCE_TOOL_NAME },
          max_tokens: 512,
          budget: "corrective-preference-extractor",
        }),
      );
    } catch (error) {
      return this.degraded(
        error instanceof MissingCorrectivePreferenceToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
      );
    }
  }
}

export { CORRECTIVE_PREFERENCE_TOOL_NAME };
