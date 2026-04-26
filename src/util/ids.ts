import { customAlphabet } from "nanoid";

const ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789";
const HEX_ID_ALPHABET = "abcdef0123456789";
const ID_LENGTH = 16;
const DEFAULT_SESSION_LITERAL = "default";
const createNanoId = customAlphabet(ID_ALPHABET, ID_LENGTH);
const createAutonomyWakeNanoId = customAlphabet(HEX_ID_ALPHABET, ID_LENGTH);

export type BrandedId<BrandName extends string> = string & {
  readonly __brand: BrandName;
};

export type StreamEntryId = BrandedId<"StreamEntryId">;
export type SessionId = BrandedId<"SessionId">;
export type EpisodeId = BrandedId<"EpisodeId">;
export type GoalId = BrandedId<"GoalId">;
export type ValueId = BrandedId<"ValueId">;
export type TraitId = BrandedId<"TraitId">;
export type AutobiographicalPeriodId = BrandedId<"AutobiographicalPeriodId">;
export type GrowthMarkerId = BrandedId<"GrowthMarkerId">;
export type OpenQuestionId = BrandedId<"OpenQuestionId">;
export type SemanticNodeId = BrandedId<"SemanticNodeId">;
export type SemanticEdgeId = BrandedId<"SemanticEdgeId">;
export type CommitmentId = BrandedId<"CommitmentId">;
export type EntityId = BrandedId<"EntityId">;
export type SkillId = BrandedId<"SkillId">;
export type MaintenanceRunId = BrandedId<"MaintenanceRunId">;
export type AutonomyWakeId = BrandedId<"AutonomyWakeId">;
export type AuditId = number & {
  readonly __brand: "AuditId";
};

export const DEFAULT_SESSION_ID = DEFAULT_SESSION_LITERAL as SessionId;

export type IdHelpers<BrandName extends string> = {
  readonly pattern: RegExp;
  create(): BrandedId<BrandName>;
  is(value: string): value is BrandedId<BrandName>;
  parse(value: string): BrandedId<BrandName>;
};

export function createIdHelpers<BrandName extends string>(prefix: string): IdHelpers<BrandName> {
  const pattern = new RegExp(`^${prefix}_[${ID_ALPHABET}]{${ID_LENGTH}}$`);

  return {
    pattern,
    create: () => `${prefix}_${createNanoId()}` as BrandedId<BrandName>,
    is: (value: string): value is BrandedId<BrandName> => pattern.test(value),
    parse: (value: string): BrandedId<BrandName> => {
      if (!pattern.test(value)) {
        throw new TypeError(`Invalid ${prefix} identifier: ${value}`);
      }

      return value as BrandedId<BrandName>;
    },
  };
}

export const streamEntryIdHelpers = createIdHelpers<"StreamEntryId">("strm");
export const sessionIdHelpers = createIdHelpers<"SessionId">("sess");
export const episodeIdHelpers = createIdHelpers<"EpisodeId">("ep");
export const goalIdHelpers = createIdHelpers<"GoalId">("goal");
export const valueIdHelpers = createIdHelpers<"ValueId">("val");
export const traitIdHelpers = createIdHelpers<"TraitId">("trt");
export const autobiographicalPeriodIdHelpers = createIdHelpers<"AutobiographicalPeriodId">("abp");
export const growthMarkerIdHelpers = createIdHelpers<"GrowthMarkerId">("grw");
export const openQuestionIdHelpers = createIdHelpers<"OpenQuestionId">("oq");
export const semanticNodeIdHelpers = createIdHelpers<"SemanticNodeId">("semn");
export const semanticEdgeIdHelpers = createIdHelpers<"SemanticEdgeId">("seme");
export const commitmentIdHelpers = createIdHelpers<"CommitmentId">("cmt");
export const entityIdHelpers = createIdHelpers<"EntityId">("ent");
export const skillIdHelpers = createIdHelpers<"SkillId">("skl");
export const maintenanceRunIdHelpers = createIdHelpers<"MaintenanceRunId">("run");
export const autonomyWakeIdHelpers: IdHelpers<"AutonomyWakeId"> = {
  pattern: new RegExp(`^autonomy_wake_[${HEX_ID_ALPHABET}]{${ID_LENGTH}}$`),
  create: () => `autonomy_wake_${createAutonomyWakeNanoId()}` as AutonomyWakeId,
  is: (value: string): value is AutonomyWakeId => autonomyWakeIdHelpers.pattern.test(value),
  parse: (value: string): AutonomyWakeId => {
    if (!autonomyWakeIdHelpers.pattern.test(value)) {
      throw new TypeError(`Invalid autonomy_wake identifier: ${value}`);
    }

    return value as AutonomyWakeId;
  },
};

export const createStreamEntryId = (): StreamEntryId => streamEntryIdHelpers.create();
export const createSessionId = (): SessionId => sessionIdHelpers.create();
export const createEpisodeId = (): EpisodeId => episodeIdHelpers.create();
export const createGoalId = (): GoalId => goalIdHelpers.create();
export const createValueId = (): ValueId => valueIdHelpers.create();
export const createTraitId = (): TraitId => traitIdHelpers.create();
export const createAutobiographicalPeriodId = (): AutobiographicalPeriodId =>
  autobiographicalPeriodIdHelpers.create();
export const createGrowthMarkerId = (): GrowthMarkerId => growthMarkerIdHelpers.create();
export const createOpenQuestionId = (): OpenQuestionId => openQuestionIdHelpers.create();
export const createSemanticNodeId = (): SemanticNodeId => semanticNodeIdHelpers.create();
export const createSemanticEdgeId = (): SemanticEdgeId => semanticEdgeIdHelpers.create();
export const createCommitmentId = (): CommitmentId => commitmentIdHelpers.create();
export const createEntityId = (): EntityId => entityIdHelpers.create();
export const createSkillId = (): SkillId => skillIdHelpers.create();
export const createMaintenanceRunId = (): MaintenanceRunId => maintenanceRunIdHelpers.create();
export const createAutonomyWakeId = (): AutonomyWakeId => autonomyWakeIdHelpers.create();

export function isSessionId(value: string): value is SessionId {
  return value === DEFAULT_SESSION_LITERAL || sessionIdHelpers.is(value);
}

export function parseSessionId(value: string): SessionId {
  if (value === DEFAULT_SESSION_LITERAL) {
    return DEFAULT_SESSION_ID;
  }

  return sessionIdHelpers.parse(value);
}

export function parseEpisodeId(value: string): EpisodeId {
  return episodeIdHelpers.parse(value);
}

export function parseGoalId(value: string): GoalId {
  return goalIdHelpers.parse(value);
}

export function parseValueId(value: string): ValueId {
  return valueIdHelpers.parse(value);
}

export function parseTraitId(value: string): TraitId {
  return traitIdHelpers.parse(value);
}

export function parseAutobiographicalPeriodId(value: string): AutobiographicalPeriodId {
  return autobiographicalPeriodIdHelpers.parse(value);
}

export function parseGrowthMarkerId(value: string): GrowthMarkerId {
  return growthMarkerIdHelpers.parse(value);
}

export function parseOpenQuestionId(value: string): OpenQuestionId {
  return openQuestionIdHelpers.parse(value);
}

export function parseSemanticNodeId(value: string): SemanticNodeId {
  return semanticNodeIdHelpers.parse(value);
}

export function parseSemanticEdgeId(value: string): SemanticEdgeId {
  return semanticEdgeIdHelpers.parse(value);
}

export function parseCommitmentId(value: string): CommitmentId {
  return commitmentIdHelpers.parse(value);
}

export function parseEntityId(value: string): EntityId {
  return entityIdHelpers.parse(value);
}

export function parseSkillId(value: string): SkillId {
  return skillIdHelpers.parse(value);
}

export function parseMaintenanceRunId(value: string): MaintenanceRunId {
  return maintenanceRunIdHelpers.parse(value);
}

export function parseAutonomyWakeId(value: string): AutonomyWakeId {
  return autonomyWakeIdHelpers.parse(value);
}

export function parseAuditId(value: number | string): AuditId {
  const candidate =
    typeof value === "number"
      ? value
      : /^\d+$/.test(String(value).trim())
        ? Number(String(value).trim())
        : Number.NaN;

  if (!Number.isInteger(candidate) || candidate <= 0) {
    throw new TypeError(`Invalid audit identifier: ${value}`);
  }

  return candidate as AuditId;
}
