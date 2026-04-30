import { createHash } from "node:crypto";

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
  return value.normalize("NFKC").trim().toLowerCase();
}

const proceduralContextDomainTagSchema = z
  .string()
  .transform((value) => canonicalizeDomainTag(value))
  .pipe(z.string().min(1).max(64));

const proceduralContextMetadataBaseSchema = z.object({
  problem_kind: proceduralContextProblemKindSchema,
  domain_tags: z.array(proceduralContextDomainTagSchema).max(32),
  audience_scope: proceduralContextAudienceScopeSchema,
});

export type ProceduralContextProblemKind = z.infer<typeof proceduralContextProblemKindSchema>;
export type ProceduralContextAudienceScope = z.infer<typeof proceduralContextAudienceScopeSchema>;

export function canonicalizeDomainTags(domainTags: readonly string[], maxTags = 3): string[] {
  const selected: string[] = [];
  const seen = new Set<string>();

  for (const value of domainTags) {
    const tag = canonicalizeDomainTag(value);

    if (tag.length === 0 || seen.has(tag)) {
      continue;
    }

    seen.add(tag);
    selected.push(tag);

    if (selected.length === maxTags) {
      break;
    }
  }

  return selected;
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableJson(item)).join(",")}]`;
  }

  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`)
      .join(",")}}`;
  }

  return JSON.stringify(value);
}

function sha1(value: string): string {
  return createHash("sha1").update(value).digest("hex");
}

export function deriveProceduralContextKey(input: {
  problem_kind: ProceduralContextProblemKind;
  domain_tags: readonly string[];
  audience_scope: ProceduralContextAudienceScope;
}): string {
  const domainTags = canonicalizeDomainTags(input.domain_tags);

  return `v2:${sha1(
    stableJson({
      problem_kind: input.problem_kind,
      domain_tags: domainTags,
      audience_scope: input.audience_scope,
    }),
  )}`;
}

export const proceduralContextKeySchema = z.string().regex(/^v2:[0-9a-f]{40}$/u, {
  message: "Invalid procedural context key",
});

export const proceduralContextMetadataSchema = proceduralContextMetadataBaseSchema.transform(
  (context) => ({
    ...context,
    domain_tags: canonicalizeDomainTags(context.domain_tags),
  }),
);

export const proceduralContextSchema = z
  .object({
    ...proceduralContextMetadataBaseSchema.shape,
    context_key: z.string().min(1),
  })
  .transform((context) => {
    const domainTags = canonicalizeDomainTags(context.domain_tags);

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

export type ProceduralContextMetadata = z.infer<typeof proceduralContextMetadataSchema>;
export type ProceduralContext = z.infer<typeof proceduralContextSchema>;
