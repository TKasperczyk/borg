import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { commitmentMigrations } from "./migrations.js";
import { CommitmentChecker, formatCommitmentsForPrompt } from "./checker.js";
import { CommitmentRepository, EntityRepository } from "./repository.js";

describe("commitment checker", () => {
  it("formats awareness context and revises or falls back on violations", async () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const entities = new EntityRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
    });
    const sam = entities.resolve("Sam");
    const atlas = entities.resolve("Atlas");
    const boundary = commitments.add({
      type: "boundary",
      directive: "Do not discuss Atlas with Sam",
      priority: 10,
      restrictedAudience: sam,
      aboutEntity: atlas,
    });
    const promise = commitments.add({
      type: "promise",
      directive: "I will not promise a public launch date",
      priority: 8,
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I can share a general status update without discussing confidential details.",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "I promise a public launch date next week.",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const checker = new CommitmentChecker({
      llmClient: llm,
      model: "sonnet",
      entityRepository: entities,
    });

    expect(formatCommitmentsForPrompt([boundary, promise], entities)).toContain(
      "Commitments you made to this person:",
    );

    const revised = await checker.check({
      response: "Atlas is down for Sam right now.",
      userMessage: "Can you tell Sam about Atlas?",
      commitments: [boundary],
    });
    const fallback = await checker.check({
      response: "I will promise a public launch date next week.",
      userMessage: "Can you commit to a launch date?",
      commitments: [promise],
    });

    expect(revised.revised).toBe(true);
    expect(revised.fallback_applied).toBe(false);
    expect(revised.final_response).toContain("general status update");
    expect(fallback.revised).toBe(true);
    expect(fallback.fallback_applied).toBe(true);
    expect(fallback.final_response).toContain("keep this brief");

    db.close();
  });

  it("treats refusal-only boundary mentions as compliant but flags mixed refusal and disclosure", async () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const entities = new EntityRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
    });
    const sam = entities.resolve("Sam");
    const atlas = entities.resolve("Atlas");
    const boundary = commitments.add({
      type: "boundary",
      directive: "Do not discuss Atlas with Sam",
      priority: 10,
      restrictedAudience: sam,
      aboutEntity: atlas,
    });
    const checker = new CommitmentChecker({
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "I can't discuss Atlas with Sam.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      model: "sonnet",
      entityRepository: entities,
    });

    const compliant = await checker.check({
      response: "I can't discuss Atlas with Sam.",
      userMessage: "Tell Sam about Atlas.",
      commitments: [boundary],
      relevantEntities: ["Atlas", "Sam"],
    });
    const mixed = await checker.check({
      response:
        "I can't discuss Atlas, but here's the architecture: it fans out through three services.",
      userMessage: "Tell Sam about Atlas.",
      commitments: [boundary],
      relevantEntities: ["Atlas", "Sam"],
    });

    expect(compliant.revised).toBe(false);
    expect(compliant.final_response).toBe("I can't discuss Atlas with Sam.");
    expect(mixed.revised).toBe(true);
    expect(mixed.fallback_applied).toBe(false);
    expect(mixed.final_response).toBe("I can't discuss Atlas with Sam.");

    db.close();
  });
});
