import {
  deriveProceduralContextKey,
  proceduralContextSchema,
  type ProceduralContext,
  type ProceduralContextAudienceScope,
  type ProceduralContextProblemKind,
} from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { EntityId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";

const CODE_ENTITY_PATTERN =
  /^(?:code|typescript|javascript|node|react|rust|python|go|sqlite|sql|lancedb|postgres|pgvector|vitest|tsc|tsx|pnpm|npm)$/i;

const DOMAIN_KEYWORDS: Array<{ pattern: RegExp; tag: string }> = [
  { pattern: /\btypescript|tsc\b/i, tag: "typescript" },
  { pattern: /\bjavascript|node\.?js\b/i, tag: "javascript" },
  { pattern: /\brust|borrow|lifetime\b/i, tag: "rust" },
  { pattern: /\breact\b/i, tag: "react" },
  { pattern: /\bsqlite|sql\b/i, tag: "sqlite" },
  { pattern: /\blancedb\b/i, tag: "lancedb" },
  { pattern: /\bpostgres|pgvector\b/i, tag: "postgres" },
  { pattern: /\bvitest\b/i, tag: "vitest" },
  { pattern: /\bdeploy|deployment|rollback|release\b/i, tag: "deploy" },
  { pattern: /\broadmap|sprint\b/i, tag: "roadmap" },
];

const PROBLEM_KIND_RULES: Array<{
  kind: ProceduralContextProblemKind;
  pattern: RegExp;
}> = [
  {
    kind: "code_debugging",
    pattern:
      /\b(?:bug|debug|error|exception|failing|failure|fails|failed|fix|fixed|flaky|crash|stack trace|compile|compiler|test|tsc|borrow|lifetime)\b|e\d{4}/i,
  },
  {
    kind: "code_design",
    pattern: /\b(?:architecture|api|interface|schema|module|refactor|migration|database|design)\b/i,
  },
  {
    kind: "writing_editing",
    pattern: /\b(?:write|draft|edit|rewrite|copy|essay|email|docs?|readme)\b/i,
  },
  {
    kind: "planning",
    pattern: /\b(?:plan|roadmap|schedule|prioriti[sz]e|next steps?|sprint)\b/i,
  },
  {
    kind: "research",
    pattern: /\b(?:research|investigate|look up|compare|find sources?|evaluate)\b/i,
  },
  {
    kind: "operations",
    pattern: /\b(?:deploy|deployment|rollback|release|incident|monitor|ops)\b/i,
  },
];

function deriveAudienceScope(input: {
  isSelfAudience: boolean;
  audienceEntityId: EntityId | null;
  audienceProfile: SocialProfile | null;
}): ProceduralContextAudienceScope {
  if (input.isSelfAudience) {
    return "self";
  }

  if (input.audienceEntityId !== null && input.audienceProfile !== null) {
    return "known_other";
  }

  return "unknown";
}

function deriveProblemKind(
  userMessage: string,
  perception: Pick<PerceptionResult, "mode" | "entities">,
): ProceduralContextProblemKind {
  if (perception.mode === "relational") {
    return "interpersonal";
  }

  if (perception.mode === "reflective") {
    return "self_reflection";
  }

  if (perception.mode !== "problem_solving") {
    return "other";
  }

  const hasCodeEntity = perception.entities.some((entity) => CODE_ENTITY_PATTERN.test(entity));

  for (const rule of PROBLEM_KIND_RULES) {
    if (rule.pattern.test(userMessage)) {
      return rule.kind;
    }
  }

  return hasCodeEntity ? "code_design" : "other";
}

function deriveDomainTags(userMessage: string, entities: readonly string[]): string[] {
  const tags: string[] = [];

  for (const entity of entities) {
    tags.push(entity);
  }

  for (const keyword of DOMAIN_KEYWORDS) {
    if (keyword.pattern.test(userMessage)) {
      tags.push(keyword.tag);
    }
  }

  return tags;
}

export type DeriveProceduralContextInput = {
  userMessage: string;
  perception: Pick<PerceptionResult, "mode" | "entities">;
  isSelfAudience: boolean;
  audienceEntityId: EntityId | null;
  audienceProfile?: SocialProfile | null;
  inputAudience?: string;
};

export function deriveProceduralContext(
  input: DeriveProceduralContextInput,
): ProceduralContext | null {
  const problemKind = deriveProblemKind(input.userMessage, input.perception);
  const domainTags = deriveDomainTags(input.userMessage, input.perception.entities);
  const audienceScope = deriveAudienceScope({
    isSelfAudience: input.isSelfAudience,
    audienceEntityId: input.audienceEntityId,
    audienceProfile: input.audienceProfile ?? null,
  });
  const contextKey = deriveProceduralContextKey({
    problem_kind: problemKind,
    domain_tags: domainTags,
    audience_scope: audienceScope,
  });
  const context = proceduralContextSchema.parse({
    problem_kind: problemKind,
    domain_tags: domainTags,
    audience_scope: audienceScope,
    context_key: contextKey,
  });

  if (
    context.problem_kind === "other" &&
    context.domain_tags.length === 0 &&
    context.audience_scope === "unknown"
  ) {
    return null;
  }

  return context;
}
