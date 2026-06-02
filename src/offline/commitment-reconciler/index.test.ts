import { afterEach, describe, expect, it } from "vitest";

import type { LLMCompleteOptions } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import { ManualClock } from "../../util/clock.js";
import { createEntityId, createStreamEntryId } from "../../util/ids.js";
import { createOfflineTestHarness, type OfflineTestHarness } from "../test-support.js";
import { CommitmentReconcilerProcess } from "./index.js";

const TOOL_NAME = "EmitCommitmentReconciliation";
const manualProvenance = { kind: "manual" } as const;

function reconciliationResponse(
  judgments: Array<{
    commitment_ids: string[];
    resolution: "supersede_to_survivor" | "keep_independent" | "conflict";
    survivor_commitment_id?: string | null;
    superseded_commitment_ids?: string[];
    reason: string;
  }>,
  usage: { inputTokens: number; outputTokens: number } = { inputTokens: 20, outputTokens: 10 },
) {
  return {
    text: "",
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: TOOL_NAME,
        input: {
          judgments: judgments.map((judgment) => ({
            survivor_commitment_id: null,
            superseded_commitment_ids: [],
            ...judgment,
          })),
        },
      },
    ],
  };
}

function createProcess(harness: OfflineTestHarness): CommitmentReconcilerProcess {
  return new CommitmentReconcilerProcess({
    commitmentRepository: harness.commitmentRepository,
    registry: harness.registry,
  });
}

function addCommitment(
  harness: OfflineTestHarness,
  overrides: Partial<{
    directiveFamily: string;
    directive: string;
    priority: number;
    restrictedAudience: CommitmentRecord["restricted_audience"];
    madeToEntity: CommitmentRecord["made_to_entity"];
    aboutEntity: CommitmentRecord["about_entity"];
    committedByEntityId: CommitmentRecord["committed_by_entity_id"];
    enforcementClass: CommitmentRecord["enforcement_class"];
    criticalDomain: CommitmentRecord["critical_domain"];
    closurePressureRelevance: CommitmentRecord["closure_pressure_relevance"];
    sourceStreamEntryIds: CommitmentRecord["source_stream_entry_ids"];
  }> = {},
): CommitmentRecord {
  return harness.commitmentRepository.add({
    type: "preference",
    kind: "participant_preference",
    directiveFamily: overrides.directiveFamily ?? "handle_format",
    directive: overrides.directive ?? "Use the requested handle format.",
    priority: overrides.priority ?? 5,
    restrictedAudience: overrides.restrictedAudience,
    madeToEntity: overrides.madeToEntity,
    aboutEntity: overrides.aboutEntity,
    committedByEntityId: overrides.committedByEntityId,
    enforcementClass: overrides.enforcementClass,
    criticalDomain: overrides.criticalDomain,
    closurePressureRelevance: overrides.closurePressureRelevance,
    sourceStreamEntryIds: overrides.sourceStreamEntryIds,
    provenance: manualProvenance,
  });
}

describe("CommitmentReconcilerProcess", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    await Promise.all(cleanup.splice(0).map((fn) => fn()));
  });

  it("reconciles same-meaning commitments with different family slugs in the same scope", async () => {
    const clock = new ManualClock(1_000);
    const firstEntryId = createStreamEntryId();
    const secondEntryId = createStreamEntryId();
    const committedBy = createEntityId();
    const llmClient = new FakeLLMClient({
      responses: [
        (options: LLMCompleteOptions) => {
          const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
            commitments: Array<{ id: string }>;
          };
          const [first, second] = payload.commitments;

          return reconciliationResponse([
            {
              commitment_ids: [first!.id, second!.id],
              resolution: "supersede_to_survivor",
              survivor_commitment_id: second!.id,
              superseded_commitment_ids: [first!.id],
              reason: "Both records express the same handle-format commitment.",
            },
          ]);
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      clock,
      llmClient,
    });
    cleanup.push(harness.cleanup);
    const process = createProcess(harness);
    const first = addCommitment(harness, {
      directiveFamily: "mention_handle_format_botarena",
      directive: "Use the configured @-handle format when mentioning BotArena participants.",
      priority: 3,
      committedByEntityId: committedBy,
      enforcementClass: "critical",
      criticalDomain: "privacy",
      closurePressureRelevance: "no_closure",
      sourceStreamEntryIds: [firstEntryId],
    });

    clock.set(2_000);

    const second = addCommitment(harness, {
      directiveFamily: "mandatory_mention_handles",
      directive: "Mention BotArena participants with their required @ handles.",
      priority: 7,
      sourceStreamEntryIds: [secondEntryId],
    });

    const result = await process.run(harness.createContext());

    expect(result.candidate_stats).toMatchObject({
      proposed: 1,
      accepted: 1,
      rejected: 0,
    });
    expect(result.changes).toHaveLength(1);
    expect(result.changes[0]?.action).toBe("commitment_reconciliation_supersede");

    const survivor = harness.commitmentRepository.get(second.id);
    const duplicate = harness.commitmentRepository.get(first.id);

    expect(duplicate?.superseded_by).toBe(second.id);
    expect(survivor).toMatchObject({
      id: second.id,
      superseded_by: null,
      enforcement_class: "critical",
      critical_domain: "privacy",
      priority: 7,
      closure_pressure_relevance: "no_closure",
      last_reinforced_at: 2_000,
    });
    expect(survivor?.source_stream_entry_ids).toEqual(
      expect.arrayContaining([firstEntryId, secondEntryId]),
    );
    expect(survivor?.source_stream_entry_ids).toHaveLength(2);

    const audit = harness.auditLog.list({ process: "commitment-reconciler" })[0]!;
    await harness.auditLog.revert(audit.id);

    expect(harness.commitmentRepository.get(first.id)?.superseded_by).toBeNull();
    expect(harness.commitmentRepository.get(second.id)).toMatchObject({
      enforcement_class: "advisory",
      critical_domain: null,
      priority: 7,
      closure_pressure_relevance: "neutral",
      source_stream_entry_ids: [secondEntryId],
    });
  });

  it("enqueues commitment reconciliation review items for conflicts", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        (options: LLMCompleteOptions) => {
          const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
            commitments: Array<{ id: string }>;
          };
          const [first, second] = payload.commitments;

          return reconciliationResponse([
            {
              commitment_ids: [first!.id, second!.id],
              resolution: "conflict",
              reason: "The two commitments cannot both be followed.",
            },
          ]);
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient,
    });
    cleanup.push(harness.cleanup);
    const process = createProcess(harness);
    const first = addCommitment(harness, {
      directiveFamily: "reply_style_a",
      directive: "Keep replies short.",
    });
    const second = addCommitment(harness, {
      directiveFamily: "reply_style_b",
      directive: "Always provide extensive replies.",
    });

    const result = await process.run(harness.createContext());
    const reviews = harness.reviewQueueRepository.list({
      kind: "commitment_reconciliation",
      openOnly: true,
    });

    expect(result.changes[0]?.action).toBe("enqueue_commitment_reconciliation_review");
    expect(reviews).toHaveLength(1);
    expect(reviews[0]?.refs).toMatchObject({
      target_type: "commitment_reconciliation",
      subkind: "conflict",
      commitment_ids: expect.arrayContaining([first.id, second.id]),
      reason: "The two commitments cannot both be followed.",
    });
    expect(harness.auditLog.list({ process: "commitment-reconciler" })).toEqual([]);
  });

  it("does not reconcile commitments in different audience scopes", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          () => {
            throw new Error("LLM should not be called for singleton scope groups");
          },
        ],
      }),
    });
    cleanup.push(harness.cleanup);
    const process = createProcess(harness);
    const firstAudience = createEntityId();
    const secondAudience = createEntityId();
    const first = addCommitment(harness, {
      directiveFamily: "same_meaning_a",
      directive: "Use required handles.",
      restrictedAudience: firstAudience,
    });
    const second = addCommitment(harness, {
      directiveFamily: "same_meaning_b",
      directive: "Use required handles.",
      restrictedAudience: secondAudience,
    });

    const plan = await process.plan(harness.createContext());

    expect(plan.group_count).toBe(0);
    expect(plan.auto_supersedes).toEqual([]);
    expect(plan.reviews).toEqual([]);
    expect(harness.commitmentRepository.get(first.id)?.superseded_by).toBeNull();
    expect(harness.commitmentRepository.get(second.id)?.superseded_by).toBeNull();
  });

  it("caps groups per run", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        (options: LLMCompleteOptions) => {
          const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
            commitments: Array<{ id: string }>;
          };

          return reconciliationResponse([
            {
              commitment_ids: payload.commitments.map((commitment) => commitment.id),
              resolution: "keep_independent",
              reason: "The commitments are independent.",
            },
          ]);
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient,
    });
    cleanup.push(harness.cleanup);
    const process = createProcess(harness);

    for (const restrictedAudience of [createEntityId(), createEntityId(), createEntityId()]) {
      addCommitment(harness, {
        directiveFamily: `group_${restrictedAudience}_a`,
        directive: "First commitment in group.",
        restrictedAudience,
      });
      addCommitment(harness, {
        directiveFamily: `group_${restrictedAudience}_b`,
        directive: "Second commitment in group.",
        restrictedAudience,
      });
    }

    const plan = await process.plan(harness.createContext(), {
      params: {
        maxGroupsPerRun: 1,
      },
    });

    expect(plan.group_count).toBe(1);
    expect(plan.remaining_group_count).toBe(2);
    expect(plan.run_capped).toBe(true);
    expect(plan.tokens_used).toBe(30);
  });

  it("reports budget exhaustion without applying partial judgments", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        (options: LLMCompleteOptions) => {
          const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
            commitments: Array<{ id: string }>;
          };

          return reconciliationResponse(
            [
              {
                commitment_ids: payload.commitments.map((commitment) => commitment.id),
                resolution: "keep_independent",
                reason: "The commitments are independent.",
              },
            ],
            {
              inputTokens: 80,
              outputTokens: 20,
            },
          );
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient,
    });
    cleanup.push(harness.cleanup);
    const process = createProcess(harness);

    addCommitment(harness, {
      directiveFamily: "budget_a",
      directive: "First budget commitment.",
    });
    addCommitment(harness, {
      directiveFamily: "budget_b",
      directive: "Second budget commitment.",
    });

    const plan = await process.plan(harness.createContext(), {
      budget: 10,
    });

    expect(plan.budget_exhausted).toBe(true);
    expect(plan.tokens_used).toBe(100);
    expect(plan.auto_supersedes).toEqual([]);
    expect(plan.reviews).toEqual([]);
    expect(plan.errors[0]?.code).toBe("OFFLINE_BUDGET_EXCEEDED");
  });
});
