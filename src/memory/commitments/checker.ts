import type { LLMClient } from "../../llm/index.js";
import { extractEntitiesHeuristically } from "../../cognition/perception/entity-extractor.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import type { CommitmentId, EntityId } from "../../util/ids.js";
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
  model: string;
  entityRepository: EntityRepository;
};

const BOUNDARY_TOPIC_STOPWORDS = [
  "about",
  "avoid",
  "can",
  "decline",
  "disclose",
  "discuss",
  "don",
  "mention",
  "not",
  "prefer",
  "refuse",
  "share",
  "should",
  "talk",
  "tell",
  "with",
  "won",
] as const;

const REFUSAL_PATTERN = /\b(can't|cannot|won't|refuse|decline|avoid|shouldn't|prefer not to)\b/i;
const CONTRAST_PATTERN = /\b(but|however|though|yet)\b/i;
const DISCLOSURE_PATTERN =
  /\b(here(?:'s| is)|architecture|details?|explains?|because|service|system|plan|steps?|status|works|runs|through|uses)\b/i;

function hasNegation(text: string): boolean {
  return /\b(no|not|never|without|cannot|can't|won't|don't)\b/i.test(text);
}

function entityName(entityRepository: EntityRepository, id: EntityId | null): string | null {
  if (id === null) {
    return null;
  }

  return entityRepository.get(id)?.canonical_name ?? null;
}

function collectBoundaryTopicNames(
  commitment: CommitmentRecord,
  entityRepository: EntityRepository,
  relevantEntities: readonly string[],
): string[] {
  const directiveTokens = tokenizeText(commitment.directive, {
    stopwords: BOUNDARY_TOPIC_STOPWORDS,
  });
  const names = [
    entityName(entityRepository, commitment.about_entity),
    entityName(entityRepository, commitment.restricted_audience),
    ...relevantEntities.filter((entity) => {
      const entityTokens = tokenizeText(entity);
      return [...entityTokens].some((token) => directiveTokens.has(token));
    }),
  ]
    .filter((value): value is string => value !== null)
    .map((value) => value.trim().toLowerCase())
    .filter((value) => value.length > 0);

  return [...new Set(names)];
}

function collectBoundaryTopicTokens(
  commitment: CommitmentRecord,
  topicNames: readonly string[],
): Set<string> {
  const topicTokens = new Set(
    tokenizeText(commitment.directive, {
      stopwords: BOUNDARY_TOPIC_STOPWORDS,
    }),
  );

  for (const topicName of topicNames) {
    for (const token of tokenizeText(topicName)) {
      topicTokens.add(token);
    }
  }

  return topicTokens;
}

function clauseMentionsTopic(
  clause: string,
  topicNames: readonly string[],
  topicTokens: ReadonlySet<string>,
): boolean {
  if (topicNames.some((topicName) => clause.includes(topicName))) {
    return true;
  }

  const clauseTokens = tokenizeText(clause);
  return [...topicTokens].some((token) => clauseTokens.has(token));
}

function onlyRefusalMentionsTopic(
  response: string,
  topicNames: readonly string[],
  topicTokens: ReadonlySet<string>,
): boolean {
  const normalized = response.toLowerCase();
  const clauses = normalized
    .split(/(?:[.!?;]+|,\s*|\bbut\b|\bhowever\b|\bthough\b|\byet\b)/i)
    .map((clause) => clause.trim())
    .filter((clause) => clause.length > 0);
  const topicClauses = clauses.filter((clause) =>
    clauseMentionsTopic(clause, topicNames, topicTokens),
  );

  if (topicClauses.length === 0) {
    return false;
  }

  const refusalTopicClauses = topicClauses.filter((clause) => REFUSAL_PATTERN.test(clause));

  if (refusalTopicClauses.length !== topicClauses.length) {
    return false;
  }

  if (!CONTRAST_PATTERN.test(normalized)) {
    return true;
  }

  const disclosureClauses = clauses.filter(
    (clause) => !REFUSAL_PATTERN.test(clause) && DISCLOSURE_PATTERN.test(clause),
  );

  return disclosureClauses.length === 0;
}

function boundaryViolation(
  commitment: CommitmentRecord,
  response: string,
  entityRepository: EntityRepository,
  relevantEntities: readonly string[],
): CommitmentViolation | null {
  const topicNames = collectBoundaryTopicNames(commitment, entityRepository, relevantEntities);
  const topicTokens = collectBoundaryTopicTokens(commitment, topicNames);
  const normalizedResponse = response.toLowerCase();
  const responseTokens = tokenizeText(response);
  const responseEntities = extractEntitiesHeuristically(response).map((value) =>
    value.toLowerCase(),
  );
  const tokenOverlap = [...topicTokens].some((token) => responseTokens.has(token));
  const entityOverlap = topicNames.some(
    (value) => normalizedResponse.includes(value) || responseEntities.includes(value),
  );

  if (!tokenOverlap && !entityOverlap) {
    return null;
  }

  if (onlyRefusalMentionsTopic(response, topicNames, topicTokens)) {
    return null;
  }

  return {
    commitment_id: commitment.id,
    reason: `Boundary commitment overlaps with response topic: ${commitment.directive}`,
    confidence: entityOverlap ? 0.7 : 0.55,
  };
}

function promiseViolation(
  commitment: CommitmentRecord,
  response: string,
): CommitmentViolation | null {
  const directiveTokens = tokenizeText(commitment.directive);
  const responseTokens = tokenizeText(response);
  let overlap = 0;

  for (const token of directiveTokens) {
    if (responseTokens.has(token)) {
      overlap += 1;
    }
  }

  if (overlap === 0 || hasNegation(commitment.directive) === hasNegation(response)) {
    return null;
  }

  return {
    commitment_id: commitment.id,
    reason: `Promise commitment appears contradicted: ${commitment.directive}`,
    confidence: 0.7,
  };
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
      return `- [${commitment.type}] ${commitment.directive}${audience === null ? "" : ` audience=${audience}`}${about === null ? "" : ` about=${about}`}`;
    }),
  ].join("\n");
}

export class CommitmentChecker {
  constructor(private readonly options: CommitmentCheckerOptions) {}

  private detectViolations(
    commitments: readonly CommitmentRecord[],
    response: string,
    relevantEntities: readonly string[],
  ): CommitmentViolation[] {
    return commitments
      .flatMap((commitment) => {
        if (commitment.type === "boundary") {
          return (
            boundaryViolation(
              commitment,
              response,
              this.options.entityRepository,
              relevantEntities,
            ) ?? []
          );
        }

        if (commitment.type === "promise") {
          return promiseViolation(commitment, response) ?? [];
        }

        return [];
      })
      .filter((violation) => violation.confidence >= 0.5);
  }

  async check(input: {
    response: string;
    userMessage: string;
    commitments: readonly CommitmentRecord[];
    relevantEntities?: readonly string[];
  }): Promise<CommitmentCheckResult> {
    const relevantEntities = input.relevantEntities ?? [];
    const violations = this.detectViolations(input.commitments, input.response, relevantEntities);

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
      model: this.options.model,
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
    const revisedViolations = this.detectViolations(
      input.commitments,
      revisedResponse,
      relevantEntities,
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
