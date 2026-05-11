import type { EntityRepository } from "../memory/commitments/index.js";
import type { EntityId } from "../util/ids.js";

export type SpeakerEntityRepository = Pick<EntityRepository, "get">;

export function resolveSpeakerDisplayName(
  entityRepository: SpeakerEntityRepository | undefined,
  senderEntityId: EntityId | null | undefined,
): string | null {
  if (entityRepository === undefined || senderEntityId === null || senderEntityId === undefined) {
    return null;
  }

  return entityRepository.get(senderEntityId)?.canonical_name ?? null;
}
