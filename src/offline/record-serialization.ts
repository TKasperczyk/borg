import { memoryDisclosurePayloadFields } from "../memory/common/disclosure-serializers.js";
import { unknownMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";

export function serializableRecord(value: unknown): unknown {
  if (value instanceof Float32Array) {
    return {
      embedding_dims: value.length,
    };
  }

  if (Array.isArray(value)) {
    return value.map((entry) => serializableRecord(entry));
  }

  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, serializableRecord(entry)]),
    );
  }

  return value;
}

export function hasDisclosurePayload(record: Record<string, unknown>): boolean {
  return record.disclosure_label !== undefined || record.disclosure !== undefined;
}

export function hasSemanticSourceEpisodeIds(record: Record<string, unknown>): boolean {
  return (
    Array.isArray(record.source_episode_ids) ||
    Array.isArray(record.evidence_episode_ids) ||
    Array.isArray(record.episode_ids)
  );
}

export function serializableRecordWithFallbackDisclosure(value: unknown): unknown {
  const serialized = serializableRecord(value);

  if (Array.isArray(serialized)) {
    return serialized.map((entry) => serializableRecordWithFallbackDisclosure(entry));
  }

  if (serialized !== null && typeof serialized === "object") {
    const record = Object.fromEntries(
      Object.entries(serialized).map(([key, entry]) => [
        key,
        serializableRecordWithFallbackDisclosure(entry),
      ]),
    );

    if (hasSemanticSourceEpisodeIds(record) && !hasDisclosurePayload(record)) {
      return {
        ...record,
        ...memoryDisclosurePayloadFields(unknownMemoryDisclosureLabel()),
      };
    }

    return record;
  }

  return serialized;
}
