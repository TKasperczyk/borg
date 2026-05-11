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

function sanitizeSpeakerDisplayName(displayName: string): string {
  const sanitized = displayName
    .replace(/[\r\n\t]+/g, " ")
    .replace(/[\]\\]/g, "")
    .replace(/:+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return sanitized.length === 0 ? "speaker" : sanitized;
}

export function prefixSpeakerTag(content: string, displayName: string | null): string {
  if (displayName === null) {
    return content;
  }

  return `[${sanitizeSpeakerDisplayName(displayName)}]: ${content}`;
}
