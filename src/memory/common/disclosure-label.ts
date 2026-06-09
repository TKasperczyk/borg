import { z } from "zod";

import { entityIdHelpers, type EntityId } from "../../util/ids.js";
import { normalizeEpisodeAccess, type EpisodeAccessLike } from "../episodic/audience-filter.js";

export const MEMORY_DISCLOSURE_CLASSES = [
  "public",
  "relationship_private",
  "operator_private",
  "self_private",
  "sensitive",
  "unknown",
] as const;

export type MemoryDisclosureClass = (typeof MEMORY_DISCLOSURE_CLASSES)[number];

export type MemoryDisclosureLabel = {
  readonly disclosureClass: MemoryDisclosureClass;
  readonly originAudienceEntityIds: EntityId[];
  readonly privateToEntityIds: EntityId[];
  readonly publicToEntityIds: EntityId[];
};

export const MEMORY_DISCLOSURE_INTERNAL_USE_NOTE =
  "I can use this internally; I do not disclose it to the current audience unless authorized";
export const SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE =
  "supported by private source episodes; I can use this internally; I do not reveal source details to the current audience unless authorized";

export type MemoryDisclosureLabelRenderContext = "memory" | "semantic_source";

export function memoryDisclosureInternalUseNote(
  context: MemoryDisclosureLabelRenderContext = "memory",
): string {
  return context === "semantic_source"
    ? SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE
    : MEMORY_DISCLOSURE_INTERNAL_USE_NOTE;
}

export function renderMemoryDisclosureLabelForModel(
  label: MemoryDisclosureLabel,
  options: {
    context?: MemoryDisclosureLabelRenderContext;
  } = {},
): string {
  const fragments = [`disclosure_class=${label.disclosureClass}`];

  if (label.originAudienceEntityIds.length > 0) {
    fragments.push(`origin_audience=${label.originAudienceEntityIds.join(",")}`);
  }

  if (label.disclosureClass !== "public") {
    const privateTo =
      label.privateToEntityIds.length === 0 ? "unknown" : label.privateToEntityIds.join(",");
    fragments.push(`private-to=${privateTo}; ${memoryDisclosureInternalUseNote(options.context)}`);
  }

  return fragments.join(" ");
}

export function renderSemanticSourceDisclosureLabelForModel(label: MemoryDisclosureLabel): string {
  return renderMemoryDisclosureLabelForModel(label, { context: "semantic_source" });
}

const memoryDisclosureEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid memory disclosure entity id",
  })
  .transform((value) => value as EntityId);

export const memoryDisclosureLabelSchema: z.ZodType<MemoryDisclosureLabel> = z.object({
  disclosureClass: z.enum(MEMORY_DISCLOSURE_CLASSES),
  originAudienceEntityIds: z.array(memoryDisclosureEntityIdSchema),
  privateToEntityIds: z.array(memoryDisclosureEntityIdSchema),
  publicToEntityIds: z.array(memoryDisclosureEntityIdSchema),
});

export const memoryDisclosureLabelMetadataSchema = z.object({
  disclosure_class: z.enum(MEMORY_DISCLOSURE_CLASSES),
  origin_audience_entity_ids: z.array(memoryDisclosureEntityIdSchema),
  private_to_entity_ids: z.array(memoryDisclosureEntityIdSchema),
  public_to_entity_ids: z.array(memoryDisclosureEntityIdSchema),
});

export type MemoryDisclosureLabelMetadata = {
  disclosure_class: MemoryDisclosureClass;
  origin_audience_entity_ids: EntityId[];
  private_to_entity_ids: EntityId[];
  public_to_entity_ids: EntityId[];
};

const MEMORY_DISCLOSURE_CLASS_RESTRICTION_RANK = {
  public: 0,
  relationship_private: 1,
  operator_private: 2,
  self_private: 3,
  sensitive: 4,
  unknown: 5,
} as const satisfies Record<MemoryDisclosureClass, number>;

function uniqueEntityIds(entityIds: readonly EntityId[]): EntityId[] {
  return [...new Set(entityIds)];
}

export function publicMemoryDisclosureLabel(): MemoryDisclosureLabel {
  return {
    disclosureClass: "public",
    originAudienceEntityIds: [],
    privateToEntityIds: [],
    publicToEntityIds: [],
  };
}

export function unknownMemoryDisclosureLabel(
  originAudienceEntityIds: readonly EntityId[] = [],
): MemoryDisclosureLabel {
  const uniqueIds = uniqueEntityIds(originAudienceEntityIds);

  return {
    disclosureClass: "unknown",
    originAudienceEntityIds: uniqueIds,
    privateToEntityIds: uniqueIds,
    publicToEntityIds: [],
  };
}

export function relationshipPrivateMemoryDisclosureLabel(
  entityIds: readonly (EntityId | null | undefined)[],
): MemoryDisclosureLabel {
  const uniqueIds = uniqueEntityIds(
    entityIds.filter((entityId): entityId is EntityId => entityId != null),
  );

  if (uniqueIds.length === 0) {
    return unknownMemoryDisclosureLabel();
  }

  return {
    disclosureClass: "relationship_private",
    originAudienceEntityIds: uniqueIds,
    privateToEntityIds: uniqueIds,
    publicToEntityIds: [],
  };
}

export function selfPrivateMemoryDisclosureLabel(
  originAudienceEntityIds: readonly EntityId[] = [],
): MemoryDisclosureLabel {
  const uniqueIds = uniqueEntityIds(originAudienceEntityIds);

  return {
    disclosureClass: "self_private",
    originAudienceEntityIds: uniqueIds,
    privateToEntityIds: uniqueIds,
    publicToEntityIds: [],
  };
}

export function memoryDisclosureLabelMetadata(
  label: MemoryDisclosureLabel,
): MemoryDisclosureLabelMetadata {
  return {
    disclosure_class: label.disclosureClass,
    origin_audience_entity_ids: [...label.originAudienceEntityIds],
    private_to_entity_ids: [...label.privateToEntityIds],
    public_to_entity_ids: [...label.publicToEntityIds],
  };
}

type MemoryDisclosureEntityIdListKey =
  | "originAudienceEntityIds"
  | "privateToEntityIds"
  | "publicToEntityIds";

function mergeEntityIds(
  labels: readonly MemoryDisclosureLabel[],
  key: MemoryDisclosureEntityIdListKey,
) {
  return [...new Set(labels.flatMap((label) => label[key]))].sort();
}

export function combineMemoryDisclosureLabels(
  labels: readonly MemoryDisclosureLabel[],
): MemoryDisclosureLabel {
  if (labels.length === 0) {
    return unknownMemoryDisclosureLabel();
  }

  const disclosureClass = labels.reduce((mostRestrictive, label) => {
    return MEMORY_DISCLOSURE_CLASS_RESTRICTION_RANK[label.disclosureClass] >
      MEMORY_DISCLOSURE_CLASS_RESTRICTION_RANK[mostRestrictive]
      ? label.disclosureClass
      : mostRestrictive;
  }, "public" as MemoryDisclosureClass);

  return {
    disclosureClass,
    originAudienceEntityIds: mergeEntityIds(labels, "originAudienceEntityIds"),
    privateToEntityIds: mergeEntityIds(labels, "privateToEntityIds"),
    publicToEntityIds: mergeEntityIds(labels, "publicToEntityIds"),
  };
}

export function memoryDisclosureLabelFromEpisodeAccess(
  input: EpisodeAccessLike,
): MemoryDisclosureLabel {
  const normalized = normalizeEpisodeAccess(input);
  const originAudienceEntityIds = normalized.origin_audience_entity_ids;

  if (originAudienceEntityIds.length === 0 && normalized.shared) {
    return publicMemoryDisclosureLabel();
  }

  if (originAudienceEntityIds.length === 0) {
    return unknownMemoryDisclosureLabel();
  }

  return {
    disclosureClass: "relationship_private",
    originAudienceEntityIds,
    privateToEntityIds: originAudienceEntityIds,
    publicToEntityIds: [],
  };
}

type EpisodeAccessRecord<TEpisodeId extends string = string> = EpisodeAccessLike & {
  id: TEpisodeId;
};

export async function resolveDisclosureLabelsByEpisodeId<TEpisodeId extends string>(
  episodeIds: readonly TEpisodeId[],
  resolveAccess: (
    episodeIds: readonly TEpisodeId[],
  ) => Promise<readonly EpisodeAccessRecord<TEpisodeId>[]> | readonly EpisodeAccessRecord<TEpisodeId>[],
): Promise<Map<TEpisodeId, MemoryDisclosureLabel>> {
  const uniqueEpisodeIds = [...new Set(episodeIds)];

  if (uniqueEpisodeIds.length === 0) {
    return new Map();
  }

  const episodes = await resolveAccess(uniqueEpisodeIds);

  return new Map(
    episodes.map((episode) => [episode.id, memoryDisclosureLabelFromEpisodeAccess(episode)]),
  );
}

export async function combineDisclosureLabelForEpisodeIds<TEpisodeId extends string>(
  episodeIds: readonly TEpisodeId[],
  resolveAccess: (
    episodeIds: readonly TEpisodeId[],
  ) => Promise<readonly EpisodeAccessRecord<TEpisodeId>[]> | readonly EpisodeAccessRecord<TEpisodeId>[],
): Promise<MemoryDisclosureLabel> {
  const labelsByEpisodeId = await resolveDisclosureLabelsByEpisodeId(episodeIds, resolveAccess);

  return combineMemoryDisclosureLabels(
    episodeIds.map(
      (episodeId) => labelsByEpisodeId.get(episodeId) ?? unknownMemoryDisclosureLabel(),
    ),
  );
}

export function parseMemoryDisclosureLabel(value: unknown): MemoryDisclosureLabel {
  if (value === null || value === undefined || value === "") {
    return unknownMemoryDisclosureLabel();
  }

  const decoded = typeof value === "string" ? (JSON.parse(value) as unknown) : value;
  return memoryDisclosureLabelSchema.parse(decoded);
}

export function memoryDisclosureLabelFromMetadata(value: unknown): MemoryDisclosureLabel | null {
  const parsedLabel = memoryDisclosureLabelSchema.safeParse(value);
  if (parsedLabel.success) {
    return parsedLabel.data;
  }

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const parsedMetadata = memoryDisclosureLabelMetadataSchema.safeParse(value);
  if (!parsedMetadata.success) {
    return unknownMemoryDisclosureLabel();
  }

  return {
    disclosureClass: parsedMetadata.data.disclosure_class,
    originAudienceEntityIds: parsedMetadata.data.origin_audience_entity_ids,
    privateToEntityIds: parsedMetadata.data.private_to_entity_ids,
    publicToEntityIds: parsedMetadata.data.public_to_entity_ids,
  };
}
