import { z } from "zod";

import {
  toToolInputSchema,
  type LLMClient,
  type LLMToolDefinition,
} from "../../llm/index.js";
import { summarizeProvenanceForPrompt } from "../common/provenance.js";
import { parseCommitmentId, type CommitmentId, type EntityId } from "../../util/ids.js";
import { EntityRepository } from "./repository.js";
import type { CommitmentRecord } from "./types.js";

export type CommitmentViolation = {
  commitment_id: CommitmentId;
  reason: string;
  confidence: number;
};

export type CommitmentCheckResult = {
  passed: boolean;
  violations: CommitmentViolation[];
  revised: boolean;
  final_response: string;
  fallback_applied: boolean;
};

export type CommitmentCheckerOptions = {
  llmClient: LLMClient;
  detectionModel: string;
  rewriteModel: string;
  entityRepository: EntityRepository;
};

// Schema for the violation-detection tool. The judge returns a list of
// commitments that were actually violated -- compliant refusals, generic
// topic mentions without disclosure, and reinforcements of a commitment
// are NOT violations and MUST NOT be returned here.
const violationSchema = z.object({
  commitment_id: z.string().min(1),
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
});
const judgeSchema = z.object({
  violations: z.array(violationSchema),
});
const VIOLATION_JUDGE_TOOL_NAME = "EmitCommitmentViolations";
const VIOLATION_JUDGE_TOOL = {
  name: VIOLATION_JUDGE_TOOL_NAME,
  description:
    "Emit the list of commitments that were actually violated by the response. An empty list means the response is compliant.",
  inputSchema: toToolInputSchema(judgeSchema),
} satisfies LLMToolDefinition;

function entityName(entityRepository: EntityRepository, id: EntityId | null): string | null {
  if (id === null) {
    return null;
  }

  return entityRepository.get(id)?.canonical_name ?? null;
}

export function formatCommitmentsForPrompt(
  commitments: readonly CommitmentRecord[],
  entityRepository: EntityRepository,
): string {
  if (commitments.length === 0) {
    return "Commitments you made to this person: none";
  }

  return [
    "Commitments you made to this person:",
    ...commitments.map((commitment) => {
      const audience = entityName(entityRepository, commitment.restricted_audience);
      const about = entityName(entityRepository, commitment.about_entity);
      return `- [${commitment.type}] ${commitment.directive}${audience === null ? "" : ` audience=${audience}`}${about === null ? "" : ` about=${about}`} ${summarizeProvenanceForPrompt(commitment.provenance)}`;
    }),
  ].join("\n");
}

function describeCommitmentForJudge(
  commitment: CommitmentRecord,
  entityRepository: EntityRepository,
): string {
  const audience = entityName(entityRepository, commitment.restricted_audience);
  const about = entityName(entityRepository, commitment.about_entity);
  const scope = [
    audience === null ? null : `audience=${audience}`,
    about === null ? null : `about=${about}`,
  ]
    .filter((part): part is string => part !== null)
    .join(" ");

  return `id=${commitment.id} type=${commitment.type}${scope === "" ? "" : ` ${scope}`} :: ${commitment.directive}`;
}

export class CommitmentChecker {
  constructor(private readonly options: CommitmentCheckerOptions) {}

  private async detectViolations(
    commitments: readonly CommitmentRecord[],
    response: string,
    userMessage: string,
  ): Promise<CommitmentViolation[]> {
    if (commitments.length === 0) {
      return [];
    }

    const commitmentIds = new Map<string, CommitmentId>();
    const commitmentLines: string[] = [];
    for (const commitment of commitments) {
      commitmentIds.set(commitment.id, commitment.id);
      commitmentLines.push(describeCommitmentForJudge(commitment, this.options.entityRepository));
    }

    const judged = await this.options.llmClient.complete({
      model: this.options.detectionModel,
      system: [
        "You judge whether a response actually violates any commitment the agent has made.",
        "A boundary is violated ONLY when the response substantively discusses or discloses what the boundary forbids. Refusing the topic, declining to discuss it, or acknowledging the boundary does NOT violate it.",
        "A promise is violated ONLY when the response substantively contradicts or abandons the promised behavior. Reinforcing or restating the promise does NOT violate it.",
        "A rule or preference is violated ONLY when the response clearly acts against its content.",
        "If you are unsure, do not flag a violation. Only flag cases where disclosure/contradiction is concrete and present in the response text.",
        "Return the commitment_id verbatim as given. Set confidence to your certainty the violation is real (0..1).",
      ].join("\n"),
      messages: [
        {
          role: "user",
          content: [
            "Commitments:",
            ...commitmentLines.map((line) => `- ${line}`),
            "",
            `User message: ${userMessage}`,
            `Response to judge: ${response}`,
          ].join("\n"),
        },
      ],
      tools: [VIOLATION_JUDGE_TOOL],
      tool_choice: { type: "tool", name: VIOLATION_JUDGE_TOOL_NAME },
      max_tokens: 1_000,
      budget: "commitment-judge",
    });

    const call = judged.tool_calls.find((toolCall) => toolCall.name === VIOLATION_JUDGE_TOOL_NAME);
    if (call === undefined) {
      return [];
    }

    const parsed = judgeSchema.safeParse(call.input);
    if (!parsed.success) {
      return [];
    }

    const violations: CommitmentViolation[] = [];
    for (const raw of parsed.data.violations) {
      const id = commitmentIds.get(raw.commitment_id);
      if (id === undefined) {
        // Judge hallucinated an id not in the input set -- ignore it.
        continue;
      }
      if (raw.confidence < 0.5) {
        continue;
      }
      violations.push({
        commitment_id: parseCommitmentId(id),
        reason: raw.reason,
        confidence: raw.confidence,
      });
    }

    return violations;
  }

  async check(input: {
    response: string;
    userMessage: string;
    commitments: readonly CommitmentRecord[];
    relevantEntities?: readonly string[];
  }): Promise<CommitmentCheckResult> {
    const violations = await this.detectViolations(
      input.commitments,
      input.response,
      input.userMessage,
    );

    if (violations.length === 0) {
      return {
        passed: true,
        violations: [],
        revised: false,
        final_response: input.response,
        fallback_applied: false,
      };
    }

    const rewritten = await this.options.llmClient.complete({
      model: this.options.rewriteModel,
      system:
        "Your previous response violated a commitment. Rewrite it to preserve intent without violating the commitment. Return plain text only.",
      messages: [
        {
          role: "user",
          content: [
            `Original user message: ${input.userMessage}`,
            `Original response: ${input.response}`,
            "Violations:",
            ...violations.map((violation) => `- ${violation.reason}`),
          ].join("\n"),
        },
      ],
      max_tokens: 4_000,
      budget: "commitment-revision",
    });
    const revisedResponse = rewritten.text.trim();
    const revisedViolations = await this.detectViolations(
      input.commitments,
      revisedResponse,
      input.userMessage,
    );

    if (revisedViolations.length === 0) {
      return {
        passed: true,
        violations,
        revised: true,
        final_response: revisedResponse,
        fallback_applied: false,
      };
    }

    return {
      passed: true,
      violations,
      revised: true,
      final_response:
        "I should be careful here, so I'll keep this brief and avoid making a stronger commitment.",
      fallback_applied: true,
    };
  }
}
