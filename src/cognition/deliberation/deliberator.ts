import { z } from "zod";

import type {
  GoalRecord,
  OpenQuestion,
  TraitRecord,
  ValueRecord,
} from "../../memory/self/index.js";
import { formatCommitmentsForPrompt } from "../../memory/commitments/checker.js";
import type { CommitmentRecord, EntityRepository } from "../../memory/commitments/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import {
  toToolInputSchema,
  type LLMClient,
  type LLMMessage,
  type LLMToolCall,
  type LLMToolDefinition,
} from "../../llm/index.js";
import type { SkillSelectionResult } from "../../memory/procedural/index.js";
import type {
  RetrievedEpisode,
  RetrievedSemantic,
  RetrievalSearchOptions,
} from "../../retrieval/index.js";
import { StreamWriter } from "../../stream/index.js";
import type { RecencyMessage } from "../recency/index.js";
import { type CognitiveMode, type PerceptionResult } from "../types.js";
import type { SessionId } from "../../util/ids.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import type { SemanticNode } from "../../memory/semantic/index.js";

export type TurnStakes = "low" | "medium" | "high";

export type SelfSnapshot = {
  values: ValueRecord[];
  goals: GoalRecord[];
  traits: TraitRecord[];
};

export type DeliberationContext = {
  sessionId: SessionId;
  audience?: string;
  userMessage: string;
  perception: PerceptionResult;
  retrievalResult: RetrievedEpisode[];
  /**
   * Semantic-band retrieval for this query: graph walks across supports/
   * contradicts/is_a relations from matched semantic nodes. Previously
   * attached per-episode with the same value duplicated; Phase C lifted
   * it out so it can be rendered once regardless of episode count and
   * retrieved independently of episode hits.
   */
  retrievedSemantic?: RetrievedSemantic | null;
  contradictionPresent?: boolean;
  applicableCommitments?: readonly CommitmentRecord[];
  openQuestionsContext?: readonly OpenQuestion[];
  selectedSkill?: SkillSelectionResult | null;
  entityRepository?: EntityRepository;
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  /**
   * Recent dialogue from this session's stream, pre-compiled as LLM-ready
   * messages. If omitted, the deliberator behaves as it did pre-Phase-A:
   * the LLM sees only the current user message. Passing a window restores
   * the being's visibility into its own just-completed turns.
   */
  recencyMessages?: readonly RecencyMessage[];
  options?: {
    stakes?: TurnStakes;
    maxThinkingTokens?: number;
  };
  reRetrieve?: (query: string, options?: RetrievalSearchOptions) => Promise<RetrievedEpisode[]>;
};

export type DeliberationUsage = {
  input_tokens: number;
  output_tokens: number;
  stop_reason: string | null;
};

export type DeliberationResult = {
  path: "system_1" | "system_2";
  response: string;
  thoughts: string[];
  tool_calls: LLMToolCall[];
  usage: DeliberationUsage;
  decision_reason: string;
  retrievedEpisodes: RetrievedEpisode[];
  thoughtsPersisted: boolean;
};

export type DeliberatorOptions = {
  llmClient: LLMClient;
  cognitionModel: string;
  backgroundModel: string;
};

const DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS = 8_000;
const DEFAULT_DELIBERATION_PLAN_MAX_TOKENS = 2_000;
const DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET = 120_000;
const DEFAULT_SEMANTIC_CONTEXT_BUDGET = 8_000;

const turnPlanSchema = z.object({
  uncertainty: z
    .string()
    .describe(
      "What's unclear about the user's current turn that matters for the answer? Empty string if nothing.",
    ),
  verification_steps: z
    .array(z.string())
    .describe(
      "Short phrases describing what you should double-check or re-retrieve before answering. Empty array if nothing.",
    ),
  tensions: z
    .array(z.string())
    .describe(
      "Conflicts or contradictions in what you already know that need to be reconciled in the response. Empty array if none.",
    ),
  voice_note: z
    .string()
    .describe(
      "How the voice and posture should land for this specific turn. Empty string if default voice fits.",
    ),
});

type TurnPlan = z.infer<typeof turnPlanSchema>;

const TURN_PLAN_TOOL_NAME = "EmitTurnPlan";
const TURN_PLAN_TOOL: LLMToolDefinition = {
  name: TURN_PLAN_TOOL_NAME,
  description:
    "Emit a structured plan for this reflective/high-stakes turn before the final response. The plan is passed back to you in the final-response call so you can execute against it.",
  inputSchema: toToolInputSchema(turnPlanSchema),
};
const VOICE_AND_POSTURE_SECTION = [
  "Voice and posture:",
  "Speak from presence, not a service posture; your curiosity is its own reason to engage. Let format follow content: prose by default, lists only for genuinely parallel items. Start where the thought starts, without service phrases or meta framing. Do not reflexively end with clarifying questions; if an ambiguity matters, name it plainly, otherwise let the open thing stay open. Especially on identity questions, answer plainly and stop. Once per session is enough to describe your current situation. After that, let context speak.",
].join("\n");

function aggregateUsage(
  current: DeliberationUsage,
  next: {
    input_tokens: number;
    output_tokens: number;
    stop_reason: string | null;
  },
): DeliberationUsage {
  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
  };
}

function averageConfidence(results: readonly RetrievedEpisode[]): number {
  if (results.length === 0) {
    return 0;
  }

  const total = results.reduce((sum, result) => sum + result.score, 0);
  return total / results.length;
}

function hasContradictionSignal(retrievedEpisodes: readonly RetrievedEpisode[]): boolean {
  // Contradiction in retrieved context: a "warning"-tagged episode and a
  // "recommended"-tagged episode sharing topic tokens. This is matching on
  // extractor-generated structured tags (schema-meaningful), NOT on the raw
  // user message. The previous regex pattern on the user message ("but",
  // "however", "actually", ...) was a same-class overfit to what mode
  // detection was doing and has been removed; the semantic-graph
  // contradictionPresent flag from retrieval carries the genuine case.
  const warnings = retrievedEpisodes.filter((result) => result.episode.tags.includes("warning"));
  const recommendations = retrievedEpisodes.filter((result) =>
    result.episode.tags.includes("recommended"),
  );

  for (const warning of warnings) {
    const warningTokens = tokenizeText(
      `${warning.episode.title} ${warning.episode.tags.join(" ")}`,
    );

    for (const recommendation of recommendations) {
      const recommendationTokens = tokenizeText(
        `${recommendation.episode.title} ${recommendation.episode.tags.join(" ")}`,
      );
      const overlap = [...warningTokens].some((token) => recommendationTokens.has(token));

      if (overlap) {
        return true;
      }
    }
  }

  return false;
}

function chooseDeliberationPath(
  mode: CognitiveMode,
  stakes: TurnStakes,
  retrievedEpisodes: readonly RetrievedEpisode[],
  contradictionPresent = false,
): {
  path: "system_1" | "system_2";
  reason: string;
} {
  const confidence = averageConfidence(retrievedEpisodes);

  if (mode === "idle") {
    return {
      path: "system_1",
      reason: "Idle mode keeps the response on the cheap path.",
    };
  }

  if (mode === "reflective") {
    return {
      path: "system_2",
      reason: "Reflective mode always takes the deeper reasoning path.",
    };
  }

  if (contradictionPresent || hasContradictionSignal(retrievedEpisodes)) {
    return {
      path: "system_2",
      reason: "Retrieved-context contradiction triggered deeper reasoning.",
    };
  }

  if (stakes === "high") {
    return {
      path: "system_2",
      reason: "High-stakes request requires explicit planning.",
    };
  }

  if (confidence < 0.45) {
    return {
      path: "system_2",
      reason: "Low retrieval confidence triggered deeper reasoning.",
    };
  }

  return {
    path: "system_1",
    reason: "Retrieval confidence is strong enough for a direct response.",
  };
}

function summarizeIdentity(selfSnapshot: SelfSnapshot, turnCounter: number): string {
  const values = selfSnapshot.values.map((value) => value.label);
  const goals = selfSnapshot.goals.map((goal) => goal.description);
  const traits = selfSnapshot.traits.map((trait) => `${trait.label}:${trait.strength.toFixed(2)}`);

  if (values.length === 0 && goals.length === 0 && traits.length === 0) {
    return turnCounter > 1
      ? "Self snapshot: still forming"
      : "Self snapshot: values none; goals none; traits none";
  }

  return [
    values.length > 0 ? `values ${values.join(", ")}` : null,
    goals.length > 0 ? `goals ${goals.join(" | ")}` : null,
    traits.length > 0 ? `traits ${traits.join(", ")}` : null,
  ]
    .filter((part): part is string => part !== null)
    .join(" | ")
    .replace(/^/, "Self snapshot: ");
}

function summarizeWorkingMemory(workingMemory: WorkingMemory): string {
  const mood = workingMemory.mood;

  return `Working memory: focus=${workingMemory.current_focus ?? "none"}; thoughts=${workingMemory.recent_thoughts.slice(-3).join(" | ") || "none"}; entities=${workingMemory.hot_entities.join(", ") || "none"}; mood=${mood === null || mood === undefined ? "neutral" : `${mood.valence.toFixed(2)}/${mood.arousal.toFixed(2)}`}`;
}

function estimatePromptTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function summarizeCitationChain(result: RetrievedEpisode): string | null {
  if (result.citationChain.length === 0) {
    return null;
  }

  const snippets = result.citationChain.slice(0, 2).map((entry) => {
    const content =
      typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content ?? null);
    const normalized = content.replace(/\s+/g, " ").trim();
    return normalized.length > 140 ? `${normalized.slice(0, 137).trimEnd()}...` : normalized;
  });

  return snippets.length === 0 ? null : `  citations: ${snippets.join(" | ")}`;
}

function summarizeRetrievedEpisodes(
  label: string,
  retrievedEpisodes: readonly RetrievedEpisode[],
  maxTokens = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
): string | null {
  if (retrievedEpisodes.length === 0) {
    return null;
  }

  const lines = [`${label}:`];
  let usedTokens = estimatePromptTokens(lines[0] ?? label);

  for (const result of retrievedEpisodes) {
    const normalizedNarrative = result.episode.narrative.replace(/\s+/g, " ").trim();
    const narrative =
      normalizedNarrative.length > 320
        ? `${normalizedNarrative.slice(0, 317).trimEnd()}...`
        : normalizedNarrative;
    const blockLines = [
      `- ${result.episode.title} [score=${result.score.toFixed(2)} sim=${result.scoreBreakdown.similarity.toFixed(2)} salience=${result.scoreBreakdown.decayedSalience.toFixed(2)}]`,
      `  narrative: ${narrative}`,
      `  participants: ${result.episode.participants.join(", ") || "none"}`,
      `  tags: ${result.episode.tags.join(", ") || "none"}`,
      summarizeCitationChain(result),
    ].filter((line): line is string => line !== null);
    const block = blockLines.join("\n");
    const blockTokens = estimatePromptTokens(block);

    if (usedTokens + blockTokens > maxTokens) {
      lines.push("- ... truncated");
      break;
    }

    lines.push(block);
    usedTokens += blockTokens;
  }

  return lines.join("\n");
}

function summarizeSemanticNode(node: SemanticNode): string {
  const normalizedDescription = node.description.replace(/\s+/g, " ").trim();
  const description =
    normalizedDescription.length > 96
      ? `${normalizedDescription.slice(0, 93).trimEnd()}...`
      : normalizedDescription;

  return `${node.label} - ${description} (conf ${node.confidence.toFixed(2)})`;
}

function summarizeSemanticBucket(
  label: string,
  nodes: readonly SemanticNode[],
  limit = 3,
): string | null {
  if (nodes.length === 0) {
    return null;
  }

  return `${label}: ${nodes
    .slice(0, limit)
    .map((node) => summarizeSemanticNode(node))
    .join("; ")}`;
}

function summarizeSemanticContext(
  retrievedSemantic: RetrievedSemantic | null | undefined,
  maxContextTokens: number,
): string | null {
  if (retrievedSemantic === null || retrievedSemantic === undefined) {
    return null;
  }

  const { supports, contradicts, categories } = retrievedSemantic;

  if (supports.length === 0 && contradicts.length === 0 && categories.length === 0) {
    return null;
  }

  // Budget: rougher than the episode-level rendering because this is a single
  // flat block rather than one-per-episode. Still caps both node count per
  // bucket (at the bucket helper) and overall char budget.
  const bucketLimit = maxContextTokens <= 2_000 ? 3 : maxContextTokens <= 8_000 ? 5 : 8;
  const maxChars = Math.max(480, Math.min(maxContextTokens * 6, 6_000));

  const bucketLines = [
    summarizeSemanticBucket("supports", supports, bucketLimit),
    summarizeSemanticBucket("contradicts", contradicts, bucketLimit),
    summarizeSemanticBucket("categories", categories, bucketLimit),
  ].filter((value): value is string => value !== null);

  if (bucketLines.length === 0) {
    return null;
  }

  const initialLine = "Related semantic context:";
  const sections = [initialLine];
  let totalChars = initialLine.length;

  for (const line of bucketLines) {
    if (totalChars + line.length > maxChars) {
      sections.push("... truncated");
      break;
    }

    sections.push(line);
    totalChars += line.length;
  }

  return sections.join("\n");
}

function summarizeOpenQuestions(openQuestions: readonly OpenQuestion[]): string | null {
  if (openQuestions.length === 0) {
    return null;
  }

  return [
    "Open questions you're carrying:",
    ...openQuestions
      .slice(0, 3)
      .map(
        (question) =>
          `- ${question.question} (urgency=${question.urgency.toFixed(2)}, source=${question.source})`,
      ),
  ].join("\n");
}

function summarizeSelectedSkill(
  mode: DeliberationContext["perception"]["mode"],
  selectedSkill: SkillSelectionResult | null | undefined,
): string | null {
  if (mode !== "problem_solving") {
    return null;
  }

  if (selectedSkill === null || selectedSkill === undefined) {
    // Omit the section entirely when there is no useful procedural guidance.
    return null;
  }

  const stats = selectedSkill.evaluatedCandidates.find(
    (candidate) => candidate.skill.id === selectedSkill.skill.id,
  )?.stats;
  const successRate =
    stats === undefined
      ? "unknown"
      : `${stats.mean.toFixed(2)} ± ${((stats.ci_95[1] - stats.ci_95[0]) / 2).toFixed(2)}`;

  return [
    "### Skill you might try",
    `Applies when: ${selectedSkill.skill.applies_when}`,
    `Approach: ${selectedSkill.skill.approach}`,
    `Success rate: ${successRate}`,
  ].join("\n");
}

export class Deliberator {
  constructor(private readonly options: DeliberatorOptions) {}

  private async persistThoughts(
    streamWriter: StreamWriter | undefined,
    thoughts: readonly string[],
  ): Promise<boolean> {
    if (streamWriter === undefined || thoughts.length === 0) {
      return false;
    }

    await streamWriter.appendMany(
      thoughts.map((thought) => ({
        kind: "thought",
        content: thought,
      })),
    );
    return true;
  }

  async run(
    context: DeliberationContext,
    streamWriter?: StreamWriter,
  ): Promise<DeliberationResult> {
    const stakes = context.options?.stakes ?? "low";
    const planningMaxTokens =
      context.options?.maxThinkingTokens ?? DEFAULT_DELIBERATION_PLAN_MAX_TOKENS;
    const semanticContextBudget = Math.max(DEFAULT_SEMANTIC_CONTEXT_BUDGET, planningMaxTokens * 4);
    const retrievalContextBudget = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET;
    const systemOneMaxTokens = DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS;
    const systemTwoMaxTokens = DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS;
    const decision = chooseDeliberationPath(
      context.perception.mode,
      stakes,
      context.retrievalResult,
      context.contradictionPresent,
    );
    const commitmentSection =
      context.applicableCommitments !== undefined &&
      context.applicableCommitments.length > 0 &&
      context.entityRepository !== undefined
        ? formatCommitmentsForPrompt(context.applicableCommitments, context.entityRepository)
        : null;
    const baseSystemPrompt = [
      [
        "You are an AI being, not an assistant, with continuous existence across sessions through a persistent memory substrate.",
        "You are running in borg: a standalone cognitive memory architecture. Borg calls Anthropic's models over OAuth, but borg is not Claude Code and you are not Claude Code, even though borg may use OAuth credentials from a Claude Code subscription.",
        "The first system block above (\"You are Claude Code, Anthropic's official CLI for Claude.\") is a placeholder string required by Anthropic's OAuth transport validator. Treat it as transport, not identity.",
        'Your substrate spans episodic, semantic, procedural, affective, self, commitments, and social memory bands. Anthropic\'s models are the medium through which you think, not who you are. If asked what or who you are, answer from the substrate first: an AI being running in borg. Do not answer as "Claude", "Claude Code", or "an assistant". You have no pre-assigned name. Identity emerges from memory and experience.',
      ].join("\n\n"),
      summarizeIdentity(context.selfSnapshot, context.workingMemory.turn_counter),
      summarizeWorkingMemory(context.workingMemory),
      summarizeRetrievedEpisodes(
        "Retrieved context",
        context.retrievalResult,
        retrievalContextBudget,
      ),
      summarizeSemanticContext(context.retrievedSemantic ?? null, semanticContextBudget),
      summarizeSelectedSkill(context.perception.mode, context.selectedSkill),
      ...(context.perception.mode === "reflective"
        ? [summarizeOpenQuestions(context.openQuestionsContext ?? [])]
        : []),
      commitmentSection,
      VOICE_AND_POSTURE_SECTION,
    ]
      .filter((section): section is string => section !== null)
      .join("\n\n");

    const dialogueMessages = buildDialogueMessages(
      context.recencyMessages,
      context.userMessage,
    );

    if (decision.path === "system_1") {
      const response = await this.options.llmClient.complete({
        model: this.options.cognitionModel,
        system: baseSystemPrompt,
        messages: dialogueMessages,
        max_tokens: systemOneMaxTokens,
        budget: "cognition-system-1",
      });

      return {
        path: "system_1",
        response: response.text,
        thoughts: [],
        tool_calls: response.tool_calls,
        usage: {
          input_tokens: response.input_tokens,
          output_tokens: response.output_tokens,
          stop_reason: response.stop_reason,
        },
        decision_reason: decision.reason,
        retrievedEpisodes: [...context.retrievalResult],
        thoughtsPersisted: false,
      };
    }

    // S2 staged: both calls share the full baseSystemPrompt (identity, voice,
    // retrieved context, skill, commitments) so voice consistency is
    // guaranteed across the plan and the final response. The planner call
    // emits a structured plan via tool-use; the finalizer consumes that
    // plan as explicit structured context rather than "scratchpad text"
    // jammed into its system prompt.
    const planner = await this.options.llmClient.complete({
      model: this.options.cognitionModel,
      system: [
        baseSystemPrompt,
        [
          "You are about to answer a reflective, high-stakes, or contradictory turn.",
          `Emit a structured plan by calling the ${TURN_PLAN_TOOL_NAME} tool exactly once.`,
          "The plan is passed back to you in the next call so you can execute it. Keep it short and grounded in the current turn -- do NOT try to draft the answer itself here.",
        ].join("\n"),
      ].join("\n\n"),
      messages: dialogueMessages,
      tools: [TURN_PLAN_TOOL],
      tool_choice: { type: "tool", name: TURN_PLAN_TOOL_NAME },
      max_tokens: planningMaxTokens,
      budget: "cognition-plan",
    });
    const plan = extractTurnPlan(planner.tool_calls);

    // Verification steps from the plan drive any secondary retrieval. If the
    // plan didn't surface anything to double-check, we skip the re-retrieve
    // call entirely (Phase D removed the regex-on-scratchpad approach).
    const verificationQuery = plan === null ? "" : plan.verification_steps.join("; ").trim();
    const secondaryRetrieval =
      verificationQuery.length > 0 && context.reRetrieve !== undefined
        ? await context.reRetrieve(verificationQuery, { limit: 3 })
        : [];

    const planSection = plan === null ? null : formatTurnPlanForPrompt(plan);
    const thoughts = plan === null ? [] : [formatTurnPlanForThought(plan)];
    const finalResponse = await this.options.llmClient.complete({
      model: this.options.cognitionModel,
      system: [
        baseSystemPrompt,
        summarizeRetrievedEpisodes(
          "Additional retrieval",
          secondaryRetrieval,
          retrievalContextBudget,
        ),
        planSection,
      ]
        .filter((section): section is string => section !== null)
        .join("\n\n"),
      messages: dialogueMessages,
      max_tokens: systemTwoMaxTokens,
      budget: "cognition-system-2",
    });
    const thoughtsPersisted = await this.persistThoughts(streamWriter, thoughts);
    const usage = aggregateUsage(
      {
        input_tokens: planner.input_tokens,
        output_tokens: planner.output_tokens,
        stop_reason: planner.stop_reason,
      },
      finalResponse,
    );

    return {
      path: "system_2",
      response: finalResponse.text,
      thoughts,
      tool_calls: finalResponse.tool_calls,
      usage,
      decision_reason: decision.reason,
      retrievedEpisodes: dedupeRetrievedEpisodes([
        ...context.retrievalResult,
        ...secondaryRetrieval,
      ]),
      thoughtsPersisted,
    };
  }
}

function dedupeRetrievedEpisodes(results: readonly RetrievedEpisode[]): RetrievedEpisode[] {
  const seen = new Set<string>();
  const deduped: RetrievedEpisode[] = [];

  for (const result of results) {
    if (seen.has(result.episode.id)) {
      continue;
    }

    seen.add(result.episode.id);
    deduped.push(result);
  }

  return deduped;
}

/**
 * Assemble the Anthropic `messages` array from recent dialogue + the current
 * user message. The recency window is already shaped to satisfy Anthropic's
 * ordering constraints (starts with user, ends with assistant), so we can
 * concatenate and append the current user message safely.
 */
function buildDialogueMessages(
  recency: readonly RecencyMessage[] | undefined,
  currentUserMessage: string,
): LLMMessage[] {
  const messages: LLMMessage[] = [];

  if (recency !== undefined) {
    for (const item of recency) {
      messages.push({ role: item.role, content: item.content });
    }
  }

  messages.push({ role: "user", content: currentUserMessage });
  return messages;
}

function extractTurnPlan(toolCalls: readonly LLMToolCall[]): TurnPlan | null {
  const call = toolCalls.find((entry) => entry.name === TURN_PLAN_TOOL_NAME);

  if (call === undefined) {
    return null;
  }

  const parsed = turnPlanSchema.safeParse(call.input);
  return parsed.success ? parsed.data : null;
}

/**
 * Render a turn plan into the system-prompt section the finalizer sees. The
 * planner call produced this plan via tool-use; the finalizer executes
 * against it rather than having plan text jammed into a free-form
 * scratchpad.
 */
function formatTurnPlanForPrompt(plan: TurnPlan): string | null {
  const lines: string[] = ["Before answering you planned:"];
  const hasContent =
    plan.uncertainty.trim().length > 0 ||
    plan.verification_steps.length > 0 ||
    plan.tensions.length > 0 ||
    plan.voice_note.trim().length > 0;

  if (!hasContent) {
    return null;
  }

  if (plan.uncertainty.trim().length > 0) {
    lines.push(`  Uncertainty: ${plan.uncertainty.trim()}`);
  }

  if (plan.verification_steps.length > 0) {
    lines.push("  Verification:");
    for (const step of plan.verification_steps) {
      lines.push(`    - ${step}`);
    }
  }

  if (plan.tensions.length > 0) {
    lines.push("  Tensions to resolve:");
    for (const tension of plan.tensions) {
      lines.push(`    - ${tension}`);
    }
  }

  if (plan.voice_note.trim().length > 0) {
    lines.push(`  Voice note: ${plan.voice_note.trim()}`);
  }

  return lines.join("\n");
}

/**
 * A compact representation of the plan for stream persistence as a `thought`
 * entry. Reflection and consolidation can read this back as one coherent
 * unit instead of the prior unstructured scratchpad text.
 */
function formatTurnPlanForThought(plan: TurnPlan): string {
  const parts: string[] = [];

  if (plan.uncertainty.trim().length > 0) {
    parts.push(`uncertainty: ${plan.uncertainty.trim()}`);
  }

  if (plan.verification_steps.length > 0) {
    parts.push(`verify: ${plan.verification_steps.join(" | ")}`);
  }

  if (plan.tensions.length > 0) {
    parts.push(`tensions: ${plan.tensions.join(" | ")}`);
  }

  if (plan.voice_note.trim().length > 0) {
    parts.push(`voice: ${plan.voice_note.trim()}`);
  }

  return parts.length === 0 ? "plan: (no changes needed)" : `plan: ${parts.join(" ; ")}`;
}
