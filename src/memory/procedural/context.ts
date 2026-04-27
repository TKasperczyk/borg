import { z } from "zod";

export const proceduralContextProblemKindSchema = z.enum([
  "code_debugging",
  "code_design",
  "writing_editing",
  "planning",
  "research",
  "interpersonal",
  "self_reflection",
  "operations",
  "other",
]);

export const proceduralContextAudienceScopeSchema = z.enum(["self", "known_other", "unknown"]);

function canonicalizeDomainTag(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[:,\s]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
}

const proceduralContextDomainTagSchema = z
  .string()
  .transform((value) => canonicalizeDomainTag(value))
  .pipe(z.string().min(1).max(64));

export type ProceduralContextProblemKind = z.infer<typeof proceduralContextProblemKindSchema>;
export type ProceduralContextAudienceScope = z.infer<typeof proceduralContextAudienceScopeSchema>;

const GENERIC_DOMAIN_TAGS_BY_PROBLEM_KIND = {
  code_debugging: new Set(["debug", "debugging"]),
  code_design: new Set(["design"]),
  writing_editing: new Set(["edit", "editing", "writing"]),
  planning: new Set(["plan", "planning"]),
  research: new Set(["research"]),
  interpersonal: new Set(["interpersonal"]),
  self_reflection: new Set(["reflection", "self-reflection"]),
  operations: new Set(["operations", "ops"]),
  other: new Set<string>(),
} satisfies Record<ProceduralContextProblemKind, ReadonlySet<string>>;

function canonicalizeDomainTags(
  problemKind: ProceduralContextProblemKind,
  domainTags: readonly string[],
): string[] {
  const genericTags = GENERIC_DOMAIN_TAGS_BY_PROBLEM_KIND[problemKind];

  return [
    ...new Set(
      domainTags
        .map(canonicalizeDomainTag)
        .filter((tag) => tag.length > 0 && !genericTags.has(tag)),
    ),
  ]
    .sort()
    .slice(0, 3);
}

export function deriveProceduralContextKey(input: {
  problem_kind: ProceduralContextProblemKind;
  domain_tags: readonly string[];
  audience_scope: ProceduralContextAudienceScope;
}): string {
  const domainTags = canonicalizeDomainTags(input.problem_kind, input.domain_tags);

  return `${input.problem_kind}:${domainTags.join(",")}:${input.audience_scope}`;
}

export const proceduralContextSchema = z
  .object({
    problem_kind: proceduralContextProblemKindSchema,
    domain_tags: z.array(proceduralContextDomainTagSchema).max(3),
    audience_scope: proceduralContextAudienceScopeSchema,
    context_key: z.string().min(1),
  })
  .transform((context) => {
    const domainTags = canonicalizeDomainTags(context.problem_kind, context.domain_tags);

    return {
      ...context,
      domain_tags: domainTags,
      context_key: deriveProceduralContextKey({
        problem_kind: context.problem_kind,
        domain_tags: domainTags,
        audience_scope: context.audience_scope,
      }),
    };
  });

export type ProceduralContext = z.infer<typeof proceduralContextSchema>;
