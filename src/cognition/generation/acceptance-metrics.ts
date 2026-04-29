import type { StreamEntry } from "../../stream/index.js";

const ROLE_LABELS = ["Human", "Assistant", "User", "AI", "Borg"] as const;
const NARRATED_NON_GENERATION = new Set([
  "(no response)",
  "(not generating)",
  "(stopping.)",
  "stopping.",
  "understood. stopping.",
]);

export type GenerationDisciplineEntry = Pick<StreamEntry, "kind" | "content">;

export type GenerationDisciplineMetrics = {
  agentMessages: number;
  roleLabelAgentMessages: number;
  emptyAgentMessages: number;
  narratedNonGenerationMessages: number;
  maxConsecutiveMinimalAgentEmissions: number;
};

function contentString(content: unknown): string {
  return typeof content === "string" ? content : "";
}

function startsWithRoleLabel(content: string): boolean {
  const trimmedStart = content.trimStart();

  return ROLE_LABELS.some((label) => trimmedStart.startsWith(`${label}:`));
}

export function isMinimalNonSubstantiveAgentContent(content: string): boolean {
  const normalized = content.trim().toLowerCase();

  if (normalized.length === 0 || NARRATED_NON_GENERATION.has(normalized)) {
    return true;
  }

  if (normalized.length <= 12 && !/[?]/.test(normalized)) {
    return true;
  }

  return false;
}

export function summarizeGenerationDiscipline(
  entries: readonly GenerationDisciplineEntry[],
): GenerationDisciplineMetrics {
  let agentMessages = 0;
  let roleLabelAgentMessages = 0;
  let emptyAgentMessages = 0;
  let narratedNonGenerationMessages = 0;
  let currentMinimalRun = 0;
  let maxConsecutiveMinimalAgentEmissions = 0;

  for (const entry of entries) {
    if (entry.kind === "user_msg") {
      continue;
    }

    if (entry.kind !== "agent_msg") {
      continue;
    }

    const content = contentString(entry.content);
    const normalized = content.trim().toLowerCase();
    agentMessages += 1;

    if (startsWithRoleLabel(content)) {
      roleLabelAgentMessages += 1;
    }

    if (content.trim().length === 0) {
      emptyAgentMessages += 1;
    }

    if (NARRATED_NON_GENERATION.has(normalized)) {
      narratedNonGenerationMessages += 1;
    }

    if (isMinimalNonSubstantiveAgentContent(content)) {
      currentMinimalRun += 1;
      maxConsecutiveMinimalAgentEmissions = Math.max(
        maxConsecutiveMinimalAgentEmissions,
        currentMinimalRun,
      );
    } else {
      currentMinimalRun = 0;
    }
  }

  return {
    agentMessages,
    roleLabelAgentMessages,
    emptyAgentMessages,
    narratedNonGenerationMessages,
    maxConsecutiveMinimalAgentEmissions,
  };
}
