import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
} from "../../util/ids.js";
import type {
  ActivationPolicy,
  CreatorDirective,
  CreatorDirectiveQueueInput,
  DisclosurePolicy,
} from "../../memory/creator-directives/index.js";
import { createOfflineTestHarness, type OfflineTestHarness } from "../test-support.js";
import { CreatorDirectiveReconcilerProcess } from "./index.js";

const TOOL_NAME = "EmitDirectiveReconciliation";

function reconciliationResponse(
  judgments: Array<{
    member_ids: string[];
    verdict: "same_intent" | "conflicting" | "independent";
    confidence: "high" | "medium" | "low";
    rationale: string;
  }>,
) {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 10,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: TOOL_NAME,
        input: {
          judgments,
        },
      },
    ],
  };
}

function disclosurePolicy(overrides: Partial<DisclosurePolicy> = {}): DisclosurePolicy {
  return {
    content_scope: "public",
    allowed_entity_ids: [],
    excluded_entity_ids: [],
    subject_may_know: null,
    mention_policy: "answer_if_asked",
    denied_audience_behavior: "omit",
    boundary_prompt: null,
    topic_tags: [],
    ...overrides,
  };
}

function activationPolicy(overrides: Partial<ActivationPolicy> = {}): ActivationPolicy {
  return {
    scope: "same_as_disclosure",
    allowed_entity_ids: [],
    excluded_entity_ids: [],
    ...overrides,
  };
}

function queueDirective(
  harness: OfflineTestHarness,
  overrides: Partial<CreatorDirectiveQueueInput> = {},
): CreatorDirective {
  return harness.creatorDirectiveRepository.queue({
    kind: "subject_fact",
    createdByEntityId: createEntityId(),
    sourceSessionId: createSessionId(),
    authorizationStreamEntryIds: [createStreamEntryId()],
    contentSourceStreamEntryIds: [createStreamEntryId()],
    subjectKind: "unknown",
    canonicalFact: "The creator supplied a durable subject fact.",
    disclosurePolicy: disclosurePolicy(),
    activationPolicy: activationPolicy(),
    priority: 5,
    ...overrides,
  });
}

function createProcess(harness: OfflineTestHarness): CreatorDirectiveReconcilerProcess {
  return new CreatorDirectiveReconcilerProcess({
    creatorDirectiveRepository: harness.creatorDirectiveRepository,
    registry: harness.registry,
  });
}

describe("CreatorDirectiveReconcilerProcess", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("merges redundant same-scope directives and reverses the supersede", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const older = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Alice prefers concise deployment updates.",
      priority: 1,
      createdAt: 1_000,
    });
    const survivor = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Alice wants concise deployment updates.",
      priority: 9,
      createdAt: 2_000,
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [older.id, survivor.id],
          verdict: "same_intent",
          confidence: "high",
          rationale: "Both records express the same creator directive.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(result.changes).toHaveLength(1);
    expect(result.changes[0]).toMatchObject({
      action: "creator_directive_merge",
      targets: {
        survivor_id: survivor.id,
        superseded_ids: [older.id],
      },
    });
    expect(harness.creatorDirectiveRepository.get(survivor.id)).toMatchObject({
      status: "active",
      superseded_by: null,
    });
    expect(harness.creatorDirectiveRepository.get(older.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
      record_version: 2,
    });

    const audit = harness.auditLog.list({ process: "creator-directive-reconciler" })[0];
    expect(audit).toMatchObject({
      action: "creator_directive_merge",
      targets: {
        survivor_id: survivor.id,
        superseded_ids: [older.id],
      },
    });

    await harness.auditLog.revert(audit!.id, "test");

    expect(harness.creatorDirectiveRepository.get(older.id)).toMatchObject({
      status: "active",
      superseded_by: null,
      record_version: 3,
    });
    expect(harness.creatorDirectiveRepository.get(survivor.id)).toMatchObject({
      status: "active",
    });
  });

  it("routes same content with different scope to review without mutation", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const publicDirective = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Mention Atlas rollback status when asked.",
      disclosurePolicy: disclosurePolicy({
        content_scope: "public",
      }),
    });
    const operatorOnlyDirective = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Mention Atlas rollback status when asked.",
      disclosurePolicy: disclosurePolicy({
        content_scope: "operator_only",
      }),
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [publicDirective.id, operatorOnlyDirective.id],
          verdict: "same_intent",
          confidence: "high",
          rationale: "The intent is the same, but scope must be handled outside the LLM.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});
    const review = harness.reviewQueueRepository.getOpen()[0];

    expect(result.errors).toEqual([]);
    expect(result.changes[0]).toMatchObject({
      action: "enqueue_creator_directive_reconciliation_review",
      targets: {
        subkind: "same_content_different_scope",
      },
    });
    expect(review).toMatchObject({
      kind: "creator_directive_reconciliation",
      refs: {
        subkind: "same_content_different_scope",
        directive_ids: [operatorOnlyDirective.id, publicDirective.id].sort(),
      },
    });
    expect(harness.creatorDirectiveRepository.get(publicDirective.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(operatorOnlyDirective.id)?.status).toBe("active");
    await expect(
      harness.reviewQueueRepository.resolve(review!.id, "accept"),
    ).resolves.toMatchObject({
      resolution: "accept",
    });
    expect(harness.creatorDirectiveRepository.get(publicDirective.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(operatorOnlyDirective.id)?.status).toBe("active");
  });

  it("routes conflicts to review", async () => {
    const creator = createEntityId();
    const first = createEntityId();
    const second = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const allowFirst = queueDirective(harness, {
      createdByEntityId: creator,
      subjectKind: "entity",
      subjectEntityId: first,
      canonicalFact: "Route Atlas updates to the first operator.",
    });
    const allowSecond = queueDirective(harness, {
      createdByEntityId: creator,
      subjectKind: "entity",
      subjectEntityId: first,
      canonicalFact: "Route Atlas updates to a different operator.",
      contentSourceStreamEntryIds: [createStreamEntryId()],
      authorizationStreamEntryIds: [createStreamEntryId()],
    });
    const unrelatedFamily = queueDirective(harness, {
      createdByEntityId: creator,
      subjectKind: "entity",
      subjectEntityId: second,
      canonicalFact: "Keep another family out of this judgment.",
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [allowFirst.id, allowSecond.id],
          verdict: "conflicting",
          confidence: "high",
          rationale: "The directives choose incompatible routing outcomes.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});
    const review = harness.reviewQueueRepository.getOpen()[0];

    expect(result.errors).toEqual([]);
    expect(review).toMatchObject({
      refs: {
        subkind: "conflict",
        directive_ids: [allowFirst.id, allowSecond.id].sort(),
      },
    });
    expect(harness.creatorDirectiveRepository.get(allowFirst.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(allowSecond.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(unrelatedFamily.id)?.status).toBe("active");
  });

  it("leaves independent directives active without review", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const first = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Use terse replies for deployment status.",
    });
    const second = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Remember that Atlas has a staging environment.",
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [first.id, second.id],
          verdict: "independent",
          confidence: "high",
          rationale: "The directives can coexist without redundancy.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([]);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
    expect(harness.creatorDirectiveRepository.get(first.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(second.id)?.status).toBe("active");
  });

  it("merges multilingual same-intent directives through the LLM verdict only", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const english = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Always describe Atlas rollback status briefly.",
      priority: 4,
      createdAt: 1_000,
    });
    const spanish = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Describe siempre brevemente el estado de rollback de Atlas.",
      priority: 4,
      createdAt: 2_000,
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [english.id, spanish.id],
          verdict: "same_intent",
          confidence: "high",
          rationale: "The two records express the same directive across languages.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(result.changes[0]).toMatchObject({
      action: "creator_directive_merge",
      targets: {
        survivor_id: spanish.id,
        superseded_ids: [english.id],
      },
    });
  });

  it("aborts an apply-time stale survivor without orphaning any loser", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const firstLoser = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Keep deployment updates concise.",
      priority: 1,
    });
    const secondLoser = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Keep release notes brief.",
      priority: 2,
    });
    const survivor = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Keep deployment updates very concise.",
      priority: 10,
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [firstLoser.id, secondLoser.id, survivor.id],
          verdict: "same_intent",
          confidence: "high",
          rationale: "One directive restates the other.",
        },
      ]),
    );
    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext(), {});

    harness.creatorDirectiveRepository.revoke(survivor.id, "creator withdrew it");
    const result = await process.apply(harness.createContext(), plan);

    expect(result.changes[0]).toMatchObject({
      action: "skip_stale_creator_directive_merge",
      targets: {
        survivor_id: survivor.id,
        superseded_ids: [firstLoser.id, secondLoser.id].sort(),
        reason: "stale_or_concurrent_mutation",
      },
    });
    expect(harness.creatorDirectiveRepository.get(firstLoser.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(secondLoser.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(survivor.id)?.status).toBe("revoked");
    expect(harness.auditLog.list({ process: "creator-directive-reconciler" })).toEqual([]);
  });

  it("does not partially reverse a drifted creator directive merge audit", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const firstLoser = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Prefer brief Atlas status.",
      priority: 1,
    });
    const secondLoser = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Prefer concise Atlas status.",
      priority: 2,
    });
    const survivor = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Prefer very concise Atlas status.",
      priority: 10,
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [firstLoser.id, secondLoser.id, survivor.id],
          verdict: "same_intent",
          confidence: "high",
          rationale: "The records are redundant.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});
    const audit = harness.auditLog.list({ process: "creator-directive-reconciler" })[0]!;
    const firstSupersededVersion = harness.creatorDirectiveRepository.get(
      firstLoser.id,
    )!.record_version;
    const otherReplacement = queueDirective(harness, {
      createdByEntityId: creator,
      subjectKind: "entity",
      subjectEntityId: createEntityId(),
      canonicalFact: "Replacement directive for drift setup.",
      priority: 11,
    });

    expect(result.changes).toHaveLength(1);
    expect(harness.creatorDirectiveRepository.get(firstLoser.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });
    expect(harness.creatorDirectiveRepository.get(secondLoser.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });

    harness.creatorDirectiveRepository.reverseSupersede(
      firstLoser.id,
      survivor.id,
      firstSupersededVersion,
    );
    harness.creatorDirectiveRepository.supersede(firstLoser.id, otherReplacement.id);

    await expect(harness.auditLog.revert(audit.id, "test")).rejects.toThrow(
      "Creator directive merge reversal is stale",
    );
    expect(harness.creatorDirectiveRepository.get(firstLoser.id)).toMatchObject({
      status: "superseded",
      superseded_by: otherReplacement.id,
    });
    expect(harness.creatorDirectiveRepository.get(secondLoser.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });
    expect(harness.auditLog.get(audit.id)).toMatchObject({
      reverted_at: null,
      reverted_by: null,
    });
  });

  it("is idempotent on a second run after merging", async () => {
    const creator = createEntityId();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const first = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Prefer concise answers about Atlas.",
      priority: 1,
    });
    const second = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Prefer concise Atlas answers.",
      priority: 2,
    });
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [first.id, second.id],
          verdict: "same_intent",
          confidence: "high",
          rationale: "The records are redundant.",
        },
      ]),
    );
    const process = createProcess(harness);

    const firstRun = await process.run(harness.createContext(), {});
    const secondRun = await process.run(harness.createContext(), {});

    expect(firstRun.changes).toHaveLength(1);
    expect(secondRun.errors).toEqual([]);
    expect(secondRun.changes).toEqual([]);
    expect(llm.requests).toHaveLength(1);
  });

  it("skips planning a family while any reconciliation review for that family is open", async () => {
    const creator = createEntityId();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const first = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Route Atlas status to the release lead.",
    });
    const second = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Route Atlas status to the platform lead.",
    });
    const third = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Keep Atlas routing notes active.",
    });
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [first.id, second.id],
          verdict: "conflicting",
          confidence: "high",
          rationale: "The directives choose incompatible recipients.",
        },
      ]),
    );
    const process = createProcess(harness);

    const firstRun = await process.run(harness.createContext(), {});
    const secondPlan = await process.plan(harness.createContext(), {});

    expect(firstRun.changes).toHaveLength(1);
    expect(harness.reviewQueueRepository.getOpen()).toHaveLength(1);
    expect(harness.creatorDirectiveRepository.get(third.id)?.status).toBe("active");
    expect(secondPlan.family_count).toBe(0);
    expect(secondPlan.auto_merges).toEqual([]);
    expect(secondPlan.reviews).toEqual([]);
    expect(llm.requests).toHaveLength(1);
  });

  it("reports run_capped and remaining family count when the family cap truncates work", async () => {
    const creator = createEntityId();
    const llm = new FakeLLMClient({
      responses: [reconciliationResponse([])],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: {
        offline: {
          creatorDirectiveReconciler: {
            maxFamiliesPerRun: 1,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    const subjectEntityIds: EntityId[] = [createEntityId(), createEntityId(), createEntityId()];

    for (const subjectEntityId of subjectEntityIds) {
      queueDirective(harness, {
        createdByEntityId: creator,
        subjectKind: "entity",
        subjectEntityId,
        canonicalFact: "One member in a capped family.",
      });
      queueDirective(harness, {
        createdByEntityId: creator,
        subjectKind: "entity",
        subjectEntityId,
        canonicalFact: "Another member in a capped family.",
      });
    }

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext(), {});
    const preview = process.preview(plan);

    expect(plan.run_capped).toBe(true);
    expect(plan.remaining_family_count).toBe(2);
    expect(preview.run_capped).toBe(true);
    expect(preview.pending_family_count).toBe(2);
    expect(llm.requests).toHaveLength(1);
  });
});
