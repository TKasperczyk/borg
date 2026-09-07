import type { SemanticObservationMetadata } from "./types.js";

export function buildNodeEmbeddingText(input: {
  label: string;
  description: string;
  aliases: readonly string[];
  observationMetadata: SemanticObservationMetadata | null;
}): string {
  const parts = [input.label, input.description, input.aliases.join(" ")];

  if (input.observationMetadata !== null) {
    parts.push(JSON.stringify(input.observationMetadata));
  }

  return parts.join("\n");
}
