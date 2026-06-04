import { z } from "zod";

import { entityIdHelpers, type EntityId } from "../../util/ids.js";

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

export function parseMemoryDisclosureLabel(value: unknown): MemoryDisclosureLabel {
  if (value === null || value === undefined || value === "") {
    return unknownMemoryDisclosureLabel();
  }

  const decoded = typeof value === "string" ? (JSON.parse(value) as unknown) : value;
  return memoryDisclosureLabelSchema.parse(decoded);
}
