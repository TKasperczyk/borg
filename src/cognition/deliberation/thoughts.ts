// Persists deliberation thoughts and formats S2 plans for the thought stream.
import { StreamWriter, type StreamEntry } from "../../stream/index.js";
import type { TurnPlan } from "./s2-planner.js";

export async function persistDeliberationThoughts(
  streamWriter: StreamWriter | undefined,
  thoughts: readonly string[],
): Promise<StreamEntry[]> {
  if (streamWriter === undefined || thoughts.length === 0) {
    return [];
  }

  return streamWriter.appendMany(
    thoughts.map((thought) => ({
      kind: "thought",
      content: thought,
    })),
  );
}

/**
 * A compact representation of the plan for stream persistence as a `thought`
 * entry. Reflection and consolidation can read this back as one coherent
 * unit instead of the prior unstructured scratchpad text.
 */
export function formatTurnPlanForThought(plan: TurnPlan): string {
  const parts: string[] = [];

  if (plan.uncertainty.trim().length > 0) {
    parts.push(`uncertainty: ${plan.uncertainty.trim()}`);
  }

  if (plan.verification_steps.length > 0) {
    parts.push(`verify: ${plan.verification_steps.join(" | ")}`);
  }

  if (plan.tensions.length > 0) {
    parts.push(`tensions: ${plan.tensions.join(" | ")}`);
  }

  if (plan.voice_note.trim().length > 0) {
    parts.push(`voice: ${plan.voice_note.trim()}`);
  }

  if (plan.emission_recommendation === "no_output") {
    parts.push("emission: no_output");
  }

  if (plan.intents.length > 0) {
    parts.push(
      `intents: ${plan.intents
        .map((intent) =>
          intent.next_action === null
            ? intent.description.trim()
            : `${intent.description.trim()} -> ${intent.next_action.trim()}`,
        )
        .join(" | ")}`,
    );
  }

  return parts.length === 0 ? "plan: (no changes needed)" : `plan: ${parts.join(" ; ")}`;
}
