import { z } from "zod";

import { isPlainRecord } from "../../util/guards.js";
import { provenanceSchema, type Provenance } from "../common/provenance.js";
import {
  commitmentIdHelpers,
  sharedStateEntryIdHelpers,
  entityIdHelpers,
  streamEntryIdHelpers,
  type CommitmentId,
  type SharedStateEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";

export const COMMITMENT_TYPES = ["promise", "boundary", "rule", "preference"] as const;
export const COMMITMENT_KINDS = [
  "assistant_commitment",
  "audience_rule",
  "participant_preference",
  "boundary",
  "process_norm",
] as const;
export const COMMITMENT_ENFORCEMENT_CLASSES = ["critical", "advisory"] as const;
export const COMMITMENT_CRITICAL_DOMAINS = [
  "privacy",
  "audience_scope",
  "safety",
  "explicit_no_disclosure",
  "internal_tool_hygiene",
] as const;
export const CLOSURE_PRESSURE_RELEVANCE = ["no_closure", "neutral", "closure_seeking"] as const;
export const ENTITY_KINDS = ["person", "group", "self", "abstract"] as const;
export const BORG_ROLES = ["creator"] as const;
export const NAME_PROVENANCES = [
  "user_declared",
  "user_confirmed",
  "config_default_user",
  "transport_audience_label",
  "assistant_seeded",
  "creator_directive",
  "unknown",
] as const;

export function normalizeDirectiveFamily(value: string): string {
  const lower = value.trim().toLowerCase();
  const chars: string[] = [];
  let previousWasSeparator = false;

  for (const char of lower) {
    const code = char.codePointAt(0) ?? 0;
    const isAsciiLetter = code >= 97 && code <= 122;
    const isDigit = code >= 48 && code <= 57;

    if (isAsciiLetter || isDigit) {
      chars.push(char);
      previousWasSeparator = false;
      continue;
    }

    if (!previousWasSeparator && chars.length > 0) {
      chars.push("_");
      previousWasSeparator = true;
    }
  }

  while (chars.at(-1) === "_") {
    chars.pop();
  }

  return chars.join("");
}

export const entityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid entity id",
  })
  .transform((value) => value as EntityId);

export const commitmentIdSchema = z
  .string()
  .refine((value) => commitmentIdHelpers.is(value), {
    message: "Invalid commitment id",
  })
  .transform((value) => value as CommitmentId);

export const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const commitmentCanonicalizedByArtifactEntryIdSchema = z
  .string()
  .refine((value) => sharedStateEntryIdHelpers.is(value), {
    message: "Invalid commitment canonicalized shared state entry id",
  })
  .transform((value) => value as SharedStateEntryId);

export const commitmentTypeSchema = z.enum(COMMITMENT_TYPES);
export const commitmentKindSchema = z.enum(COMMITMENT_KINDS);
export const commitmentEnforcementClassSchema = z.enum(COMMITMENT_ENFORCEMENT_CLASSES);
export const commitmentCriticalDomainSchema = z.enum(COMMITMENT_CRITICAL_DOMAINS);
export const closurePressureRelevanceSchema = z.enum(CLOSURE_PRESSURE_RELEVANCE);
export const entityKindSchema = z.enum(ENTITY_KINDS);
export const borgRoleSchema = z.enum(BORG_ROLES);
export const nameProvenanceSchema = z.enum(NAME_PROVENANCES);

export const directiveFamilySchema = z
  .string()
  .min(1)
  .transform((value) => normalizeDirectiveFamily(value))
  .pipe(z.string().min(1).max(64));

export const entityRecordSchema = z.object({
  id: entityIdSchema,
  canonical_name: z.string().min(1),
  aliases: z.array(z.string().min(1)),
  kind: entityKindSchema.nullable(),
  borg_role: borgRoleSchema.nullable(),
  name_provenance: nameProvenanceSchema.optional(),
  created_at: z.number().finite(),
});

export const commitmentSchema = z.object({
  id: commitmentIdSchema,
  record_version: z.number().int().positive().optional(),
  type: commitmentTypeSchema,
  kind: commitmentKindSchema,
  enforcement_class: commitmentEnforcementClassSchema,
  critical_domain: commitmentCriticalDomainSchema.nullable(),
  directive_family: directiveFamilySchema,
  closure_pressure_relevance: closurePressureRelevanceSchema.default("neutral"),
  directive: z.string().min(1),
  priority: z.number().int(),
  made_to_entity: entityIdSchema.nullable(),
  restricted_audience: entityIdSchema.nullable(),
  about_entity: entityIdSchema.nullable(),
  committed_by_entity_id: entityIdSchema.nullable().optional(),
  provenance: provenanceSchema,
  source_stream_entry_ids: z.array(streamEntryIdSchema).min(1).optional(),
  created_at: z.number().finite(),
  expires_at: z.number().finite().nullable(),
  expired_at: z.number().finite().nullable(),
  revoked_at: z.number().finite().nullable(),
  revoked_reason: z.string().nullable(),
  revoke_provenance: provenanceSchema.nullable(),
  superseded_by: commitmentIdSchema.nullable(),
  canonicalized_by_artifact_entry_id: commitmentCanonicalizedByArtifactEntryIdSchema
    .nullable()
    .optional(),
  last_reinforced_at: z.number().finite(),
});

function normalizeLegacyCommitmentValue(value: unknown): unknown {
  if (!isPlainRecord(value)) {
    return value;
  }

  const kind = commitmentKindSchema.safeParse(value.kind ?? "assistant_commitment");

  if (!kind.success) {
    return value;
  }

  const rawEnforcementClass = value.enforcement_class;
  const enforcementClass =
    rawEnforcementClass === null || rawEnforcementClass === undefined
      ? defaultCommitmentEnforcementClass(kind.data)
      : commitmentEnforcementClassSchema.safeParse(rawEnforcementClass);

  if (typeof enforcementClass !== "string" && !enforcementClass.success) {
    return value;
  }

  const effectiveEnforcementClass =
    typeof enforcementClass === "string" ? enforcementClass : enforcementClass.data;
  const rawCriticalDomain = value.critical_domain;
  const criticalDomain =
    effectiveEnforcementClass === "critical"
      ? rawCriticalDomain === null || rawCriticalDomain === undefined
        ? defaultCommitmentCriticalDomain(kind.data, effectiveEnforcementClass)
        : commitmentCriticalDomainSchema.safeParse(rawCriticalDomain)
      : null;

  if (typeof criticalDomain !== "string" && criticalDomain !== null && !criticalDomain.success) {
    return value;
  }

  return {
    ...value,
    kind: kind.data,
    enforcement_class: effectiveEnforcementClass,
    critical_domain:
      typeof criticalDomain === "string"
        ? criticalDomain
        : criticalDomain === null
          ? null
          : criticalDomain.data,
  };
}

export const legacyCommitmentSchema = z.preprocess(
  normalizeLegacyCommitmentValue,
  commitmentSchema,
);

export const commitmentPatchSchema = commitmentSchema
  .omit({
    id: true,
    record_version: true,
    created_at: true,
  })
  .partial()
  .strict();

export type EntityRecord = z.infer<typeof entityRecordSchema>;
export type CommitmentRecord = z.infer<typeof commitmentSchema>;
export type CommitmentPatch = z.infer<typeof commitmentPatchSchema>;
export type CommitmentType = z.infer<typeof commitmentTypeSchema>;
export type CommitmentKind = z.infer<typeof commitmentKindSchema>;
export type CommitmentEnforcementClass = z.infer<typeof commitmentEnforcementClassSchema>;
export type CommitmentCriticalDomain = z.infer<typeof commitmentCriticalDomainSchema>;
export type ClosurePressureRelevance = z.infer<typeof closurePressureRelevanceSchema>;
export type EntityKind = z.infer<typeof entityKindSchema>;
export type BorgRole = z.infer<typeof borgRoleSchema>;
export type NameProvenance = z.infer<typeof nameProvenanceSchema>;
export type CommitmentProvenance = Provenance;

export type CommitmentListOptions = {
  activeOnly?: boolean;
  audience?: EntityId | null;
  aboutEntity?: EntityId | null;
  committedByEntity?: EntityId | null;
  nowMs?: number;
};

export type CommitmentApplicableOptions = {
  audience?: EntityId | null;
  aboutEntity?: EntityId | null;
  nowMs?: number;
};

export function defaultCommitmentEnforcementClass(
  kind: CommitmentKind,
): CommitmentEnforcementClass {
  return kind === "boundary" || kind === "audience_rule" ? "critical" : "advisory";
}

export function defaultCommitmentCriticalDomain(
  kind: CommitmentKind,
  enforcementClass = defaultCommitmentEnforcementClass(kind),
): CommitmentCriticalDomain | null {
  if (enforcementClass !== "critical") {
    return null;
  }

  return kind === "boundary" || kind === "audience_rule" ? "audience_scope" : null;
}

export type LegacyCommitmentEnforcementFields = Pick<CommitmentRecord, "kind"> &
  Partial<Pick<CommitmentRecord, "enforcement_class" | "critical_domain">>;

export function effectiveCommitmentEnforcementClass(
  commitment: LegacyCommitmentEnforcementFields,
): CommitmentEnforcementClass {
  if (commitment.enforcement_class === "critical" || commitment.enforcement_class === "advisory") {
    return commitment.enforcement_class;
  }

  return defaultCommitmentEnforcementClass(commitment.kind);
}

export function effectiveCommitmentCriticalDomain(
  commitment: LegacyCommitmentEnforcementFields,
): CommitmentCriticalDomain | null {
  const enforcementClass = effectiveCommitmentEnforcementClass(commitment);

  if (enforcementClass !== "critical") {
    return null;
  }

  return commitment.critical_domain ?? defaultCommitmentCriticalDomain(commitment.kind);
}
