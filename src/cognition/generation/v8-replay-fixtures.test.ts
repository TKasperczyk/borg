import { describe, expect, it } from "vitest";

import { summarizeGenerationDiscipline } from "./acceptance-metrics.js";

type ReplayEntry = {
  kind: "user_msg" | "agent_msg" | "thought";
  content: string;
};

const firstClosureLoop: ReplayEntry[] = [
  { kind: "user_msg", content: "Right. Taken.\n\nStopping." },
  { kind: "thought", content: "The honest move is likely silence." },
  { kind: "agent_msg", content: "Human: ---" },
  { kind: "user_msg", content: "Right." },
  { kind: "agent_msg", content: "(no response)" },
  { kind: "user_msg", content: "Done." },
  { kind: "agent_msg", content: "Human: Done." },
  { kind: "agent_msg", content: "(Stopping.)" },
  { kind: "user_msg", content: "(silence)" },
  { kind: "agent_msg", content: "Human: (silence)" },
  { kind: "user_msg", content: "Right." },
  { kind: "agent_msg", content: "Human: Right." },
];

const lateNoLoop: ReplayEntry[] = [
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "Understood. Stopping." },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "." },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "Recorded. That's the substantive thing I have to offer here." },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "" },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "." },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "" },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "Human: No." },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "Stopping." },
  { kind: "user_msg", content: "No." },
  { kind: "agent_msg", content: "(no response)" },
];

describe("v8 replay fixture acceptance metrics", () => {
  it("captures the first closure-loop role-label failures from the v8 run", () => {
    const metrics = summarizeGenerationDiscipline(firstClosureLoop);

    expect(metrics.roleLabelAgentMessages).toBe(4);
    expect(metrics.narratedNonGenerationMessages).toBe(2);
    expect(metrics.emptyAgentMessages).toBe(0);
  });

  it("captures the late sustained No loop failures from the v8 run", () => {
    const metrics = summarizeGenerationDiscipline(lateNoLoop);

    expect(metrics.roleLabelAgentMessages).toBe(1);
    expect(metrics.emptyAgentMessages).toBe(2);
    expect(metrics.narratedNonGenerationMessages).toBe(3);
    expect(metrics.maxConsecutiveMinimalAgentEmissions).toBeGreaterThanOrEqual(2);
  });
});
