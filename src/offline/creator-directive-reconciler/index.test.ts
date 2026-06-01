import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  createCreatorDirectiveId,
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
import { creatorDirectiveSchema } from "../../memory/creator-directives/index.js";
import { createOfflineTestHarness, type OfflineTestHarness } from "../test-support.js";
import {
  CreatorDirectiveReconcilerProcess,
  mergeWidensDisclosure,
  revokeWidensDisclosure,
} from "./index.js";

const TOOL_NAME = "EmitDirectiveReconciliation";

function reconciliationResponse(
  judgments: Array<{
    member_ids: string[];
    verdict: "same_intent" | "conflicting" | "independent";
    resolution: "supersede_to_survivor" | "revoke_stale" | "keep_independent" | "escalate";
    survivor_id?: string | null;
    loser_ids?: string[];
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
          judgments: judgments.map((judgment) => ({
            survivor_id: null,
            loser_ids: [],
            ...judgment,
          })),
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

const BOUNDARY_PROMPT = "A creator-defined confidentiality boundary applies.";

const ORACLE_RENDER_RANK = {
  omit: 0,
  boundary: 1,
  content: 2,
} as const;

const ORACLE_MENTION_RANK = {
  never_mention: 0,
  only_if_topic_raised: 1,
  answer_if_asked: 2,
  proactive: 3,
} as const;

const ORACLE_DENIED_RANK = {
  omit: 0,
  render_boundary_when_relevant: 1,
} as const;

type OracleRenderMode = keyof typeof ORACLE_RENDER_RANK;
type OracleRealization = {
  disclosure: number;
  activation: number;
  mention: number;
  denied: number;
  subject: number;
};

function makeDirectiveRecord(
  overrides: {
    id?: CreatorDirective["id"];
    createdByEntityId?: EntityId;
    subjectEntityId?: EntityId;
    disclosurePolicy?: DisclosurePolicy;
    activationPolicy?: ActivationPolicy;
    priority?: number;
  } = {},
): CreatorDirective {
  const subjectEntityId = overrides.subjectEntityId ?? createEntityId();

  return creatorDirectiveSchema.parse({
    id: overrides.id ?? createCreatorDirectiveId(),
    record_version: 1,
    status: "active",
    kind: "subject_fact",
    created_by_entity_id: overrides.createdByEntityId ?? createEntityId(),
    source_session_id: createSessionId(),
    authorization_stream_entry_ids: [createStreamEntryId()],
    content_source_stream_entry_ids: [createStreamEntryId()],
    subject_kind: "entity",
    subject_entity_id: subjectEntityId,
    semantic_slot: null,
    canonical_fact: "The creator supplied a durable subject fact.",
    operational_directive: null,
    disclosure_policy: overrides.disclosurePolicy ?? disclosurePolicy(),
    activation_policy: overrides.activationPolicy ?? activationPolicy(),
    priority: overrides.priority ?? 5,
    superseded_by: null,
    revoked_reason: null,
    created_at: 1_000,
    updated_at: 1_000,
  });
}

function seededRandom(seed: number): () => number {
  let state = seed;

  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function pick<T>(random: () => number, values: readonly T[]): T {
  return values[Math.floor(random() * values.length)]!;
}

function randomSubset(
  random: () => number,
  values: readonly EntityId[],
  options: { min: number; max?: number; excluded?: ReadonlySet<EntityId> },
): EntityId[] {
  const excluded = options.excluded ?? new Set<EntityId>();
  const shuffled = [...values].filter((value) => !excluded.has(value)).sort(() => random() - 0.5);
  const max = Math.min(options.max ?? shuffled.length, shuffled.length);
  const length = options.min + Math.floor(random() * (max - options.min + 1));

  return shuffled.slice(0, length);
}

function randomDisclosurePolicy(
  random: () => number,
  pool: readonly EntityId[],
  subject: EntityId,
): DisclosurePolicy {
  const contentScope = pick(random, [
    "operator_only",
    "public",
    "allow_list",
    "subject_only",
    "all_except",
  ] as const);
  const mentionPolicy = pick(random, [
    "never_mention",
    "only_if_topic_raised",
    "answer_if_asked",
    "proactive",
  ] as const);
  const deniedAudienceBehavior = pick(random, ["omit", "render_boundary_when_relevant"] as const);
  const boundaryPrompt =
    deniedAudienceBehavior === "render_boundary_when_relevant" ? BOUNDARY_PROMPT : null;
  const base = {
    subject_may_know: pick(random, [null, true] as const),
    mention_policy: mentionPolicy,
    denied_audience_behavior: deniedAudienceBehavior,
    boundary_prompt: boundaryPrompt,
    topic_tags: [],
  };

  if (contentScope === "operator_only") {
    return disclosurePolicy({
      ...base,
      content_scope: contentScope,
      subject_may_know: pick(random, [null, true, false] as const),
    });
  }

  if (contentScope === "public") {
    return disclosurePolicy({
      ...base,
      content_scope: contentScope,
    });
  }

  if (contentScope === "allow_list") {
    const allowed = randomSubset(random, pool, { min: 1 });
    const excluded = randomSubset(random, pool, {
      min: 0,
      max: Math.max(0, pool.length - allowed.length),
      excluded: new Set(allowed),
    });

    return disclosurePolicy({
      ...base,
      content_scope: contentScope,
      allowed_entity_ids: allowed,
      excluded_entity_ids: excluded,
    });
  }

  if (contentScope === "subject_only") {
    return disclosurePolicy({
      ...base,
      content_scope: contentScope,
      subject_may_know: pick(random, [null, true] as const),
    });
  }

  const subjectMayKnow = pick(random, [null, true, false] as const);
  const excluded = new Set(randomSubset(random, pool, { min: 1 }));

  if (subjectMayKnow === false) {
    excluded.add(subject);
  }

  return disclosurePolicy({
    ...base,
    content_scope: contentScope,
    excluded_entity_ids: [...excluded],
    subject_may_know: subjectMayKnow,
  });
}

function randomActivationPolicy(random: () => number, pool: readonly EntityId[]): ActivationPolicy {
  const scope = pick(random, [
    "same_as_disclosure",
    "operator_only",
    "public",
    "allow_list",
    "subject_only",
    "all_except",
  ] as const);

  if (scope === "allow_list") {
    const allowed = randomSubset(random, pool, { min: 1 });

    return activationPolicy({
      scope,
      allowed_entity_ids: allowed,
      excluded_entity_ids: randomSubset(random, pool, {
        min: 0,
        max: Math.max(0, pool.length - allowed.length),
        excluded: new Set(allowed),
      }),
    });
  }

  if (scope === "all_except") {
    return activationPolicy({
      scope,
      excluded_entity_ids: randomSubset(random, pool, { min: 1 }),
    });
  }

  return activationPolicy({ scope });
}

function oracleDeniedMode(policy: DisclosurePolicy): OracleRenderMode {
  return policy.denied_audience_behavior === "render_boundary_when_relevant" ? "boundary" : "omit";
}

function oracleDisclosureMode(directive: CreatorDirective, audience: EntityId): OracleRenderMode {
  const policy = directive.disclosure_policy;
  const excluded = new Set(policy.excluded_entity_ids);

  if (excluded.has(audience)) {
    return oracleDeniedMode(policy);
  }

  if (
    directive.subject_entity_id !== null &&
    policy.subject_may_know === false &&
    audience === directive.subject_entity_id
  ) {
    return oracleDeniedMode(policy);
  }

  if (policy.content_scope === "public" || policy.content_scope === "all_except") {
    return "content";
  }

  if (policy.content_scope === "allow_list") {
    return new Set(policy.allowed_entity_ids).has(audience) ? "content" : "omit";
  }

  if (policy.content_scope === "subject_only") {
    return directive.subject_entity_id !== null && audience === directive.subject_entity_id
      ? "content"
      : "omit";
  }

  return audience === directive.created_by_entity_id ? "content" : "omit";
}

function oracleActivationActive(directive: CreatorDirective, audience: EntityId): boolean {
  const policy = directive.activation_policy;

  if (policy.scope === "same_as_disclosure") {
    return oracleDisclosureMode(directive, audience) !== "omit";
  }

  if (policy.scope === "operator_only") {
    return audience === directive.created_by_entity_id;
  }

  if (policy.scope === "public") {
    return true;
  }

  if (policy.scope === "allow_list") {
    const excluded = new Set(policy.excluded_entity_ids);

    if (excluded.has(audience)) {
      return false;
    }

    return new Set(policy.allowed_entity_ids).has(audience);
  }

  if (policy.scope === "subject_only") {
    return directive.subject_entity_id !== null && audience === directive.subject_entity_id;
  }

  return !new Set(policy.excluded_entity_ids).has(audience);
}

function oracleRealization(directive: CreatorDirective, audience: EntityId): OracleRealization {
  return {
    disclosure: ORACLE_RENDER_RANK[oracleDisclosureMode(directive, audience)],
    activation: oracleActivationActive(directive, audience) ? 1 : 0,
    mention: ORACLE_MENTION_RANK[directive.disclosure_policy.mention_policy],
    denied: ORACLE_DENIED_RANK[directive.disclosure_policy.denied_audience_behavior],
    subject: directive.disclosure_policy.subject_may_know === false ? 0 : 1,
  };
}

function oracleFamilyRealization(
  members: readonly CreatorDirective[],
  audience: EntityId,
): OracleRealization {
  const realizations = members.map((member) => oracleRealization(member, audience));

  return {
    disclosure: Math.min(...realizations.map((item) => item.disclosure)),
    activation: Math.min(...realizations.map((item) => item.activation)),
    mention: Math.min(...realizations.map((item) => item.mention)),
    denied: Math.min(...realizations.map((item) => item.denied)),
    subject: Math.min(...realizations.map((item) => item.subject)),
  };
}

function realizationStrictlyExpands(post: OracleRealization, pre: OracleRealization): boolean {
  const postValues = [post.disclosure, post.activation, post.mention, post.denied, post.subject];
  const preValues = [pre.disclosure, pre.activation, pre.mention, pre.denied, pre.subject];

  return (
    postValues.every((value, index) => value >= preValues[index]!) &&
    postValues.some((value, index) => value > preValues[index]!)
  );
}

function oracleAudienceUniverse(
  members: readonly CreatorDirective[],
  pool: readonly EntityId[],
): EntityId[] {
  const audienceIds = new Set(pool);

  for (const member of members) {
    audienceIds.add(member.created_by_entity_id);
    if (member.subject_entity_id !== null) {
      audienceIds.add(member.subject_entity_id);
    }
    for (const id of member.disclosure_policy.allowed_entity_ids) {
      audienceIds.add(id);
    }
    for (const id of member.disclosure_policy.excluded_entity_ids) {
      audienceIds.add(id);
    }
    for (const id of member.activation_policy.allowed_entity_ids) {
      audienceIds.add(id);
    }
    for (const id of member.activation_policy.excluded_entity_ids) {
      audienceIds.add(id);
    }
  }

  return [...audienceIds];
}

function oraclePostStrictlyExpandsFamily(
  survivor: CreatorDirective,
  losers: readonly CreatorDirective[],
  pool: readonly EntityId[],
): boolean {
  const beforeMembers = [survivor, ...losers];

  for (const audience of oracleAudienceUniverse(beforeMembers, pool)) {
    const before = oracleFamilyRealization(beforeMembers, audience);
    const after = oracleFamilyRealization([survivor], audience);

    if (realizationStrictlyExpands(after, before)) {
      return true;
    }
  }

  return false;
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
          resolution: "supersede_to_survivor",
          survivor_id: survivor.id,
          loser_ids: [older.id],
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

  it("revokes stale equal-or-more-permissive facts and reverses the revoke", async () => {
    const creator = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const stale = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Atlas rollback status may be discussed broadly.",
      disclosurePolicy: disclosurePolicy({
        content_scope: "public",
      }),
      priority: 1,
      createdAt: 1_000,
    });
    const survivor = queueDirective(harness, {
      createdByEntityId: creator,
      canonicalFact: "Atlas rollback status is operator-only.",
      disclosurePolicy: disclosurePolicy({
        content_scope: "operator_only",
      }),
      priority: 9,
      createdAt: 2_000,
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [stale.id, survivor.id],
          verdict: "same_intent",
          resolution: "revoke_stale",
          survivor_id: survivor.id,
          loser_ids: [stale.id],
          confidence: "high",
          rationale: "The older record is stale and the newer record replaces it.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(result.changes).toHaveLength(1);
    expect(result.changes[0]).toMatchObject({
      action: "creator_directive_revoke",
      targets: {
        survivor_id: survivor.id,
        loser_ids: [stale.id],
      },
    });
    expect(harness.creatorDirectiveRepository.get(survivor.id)).toMatchObject({
      status: "active",
    });
    expect(harness.creatorDirectiveRepository.get(stale.id)).toMatchObject({
      status: "revoked",
      record_version: 2,
    });

    const audit = harness.auditLog.list({ process: "creator-directive-reconciler" })[0];
    expect(audit).toMatchObject({
      action: "creator_directive_revoke",
      targets: {
        survivor_id: survivor.id,
        loser_ids: [stale.id],
      },
    });

    await harness.auditLog.revert(audit!.id, "test");

    expect(harness.creatorDirectiveRepository.get(stale.id)).toMatchObject({
      status: "active",
      revoked_reason: null,
      record_version: 3,
    });
  });

  it("routes disclosure-widening resolutions to review without mutation", async () => {
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
          resolution: "supersede_to_survivor",
          survivor_id: publicDirective.id,
          loser_ids: [operatorOnlyDirective.id],
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
        subkind: "disclosure_widening",
      },
    });
    expect(review).toMatchObject({
      kind: "creator_directive_reconciliation",
      refs: {
        subkind: "disclosure_widening",
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

  it("routes different-creator operator-only merges to disclosure widening review", async () => {
    const firstCreator = createEntityId();
    const secondCreator = createEntityId();
    const subject = createEntityId();
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient(),
    });
    cleanup.push(harness.cleanup);

    const first = queueDirective(harness, {
      createdByEntityId: firstCreator,
      subjectKind: "entity",
      subjectEntityId: subject,
      canonicalFact: "Keep the Atlas operator note private.",
      disclosurePolicy: disclosurePolicy({
        content_scope: "operator_only",
      }),
    });
    const survivor = queueDirective(harness, {
      createdByEntityId: secondCreator,
      subjectKind: "entity",
      subjectEntityId: subject,
      canonicalFact: "Keep the Atlas operator note private.",
      disclosurePolicy: disclosurePolicy({
        content_scope: "operator_only",
      }),
    });
    const llm = harness.llmClient as FakeLLMClient;
    llm.pushResponse(
      reconciliationResponse([
        {
          member_ids: [first.id, survivor.id],
          verdict: "same_intent",
          resolution: "supersede_to_survivor",
          survivor_id: survivor.id,
          loser_ids: [first.id],
          confidence: "high",
          rationale: "The records express the same directive.",
        },
      ]),
    );

    const result = await createProcess(harness).run(harness.createContext(), {});
    const review = harness.reviewQueueRepository.getOpen()[0];

    expect(result.errors).toEqual([]);
    expect(result.changes).toHaveLength(1);
    expect(result.changes[0]).toMatchObject({
      action: "enqueue_creator_directive_reconciliation_review",
      targets: {
        subkind: "disclosure_widening",
      },
    });
    expect(review).toMatchObject({
      refs: {
        subkind: "disclosure_widening",
        directive_ids: [first.id, survivor.id].sort(),
      },
    });
    expect(harness.creatorDirectiveRepository.get(first.id)?.status).toBe("active");
    expect(harness.creatorDirectiveRepository.get(survivor.id)?.status).toBe("active");
  });

  it("escalates deterministic disclosure-widening guard axes", () => {
    const creator = createEntityId();
    const otherCreator = createEntityId();
    const subject = createEntityId();
    const first = createEntityId();
    const second = createEntityId();
    const base = {
      createdByEntityId: creator,
      subjectEntityId: subject,
    };

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          ...base,
          disclosurePolicy: disclosurePolicy({ content_scope: "public" }),
          activationPolicy: activationPolicy({ scope: "public" }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({ content_scope: "public" }),
            activationPolicy: activationPolicy({ scope: "operator_only" }),
          }),
        ],
      ),
    ).toBe(true);

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          createdByEntityId: otherCreator,
          subjectEntityId: subject,
          disclosurePolicy: disclosurePolicy({ content_scope: "operator_only" }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({ content_scope: "operator_only" }),
          }),
        ],
      ),
    ).toBe(true);

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          ...base,
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [first],
          }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({
              content_scope: "all_except",
              excluded_entity_ids: [first, second],
            }),
          }),
        ],
      ),
    ).toBe(true);

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          ...base,
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
            subject_may_know: true,
          }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({
              content_scope: "operator_only",
              subject_may_know: false,
            }),
          }),
        ],
      ),
    ).toBe(true);

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          ...base,
          disclosurePolicy: disclosurePolicy({
            content_scope: "allow_list",
            allowed_entity_ids: [first, second],
          }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({
              content_scope: "allow_list",
              allowed_entity_ids: [first],
            }),
          }),
        ],
      ),
    ).toBe(true);

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          ...base,
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
            mention_policy: "proactive",
          }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({
              content_scope: "public",
              mention_policy: "never_mention",
            }),
          }),
        ],
      ),
    ).toBe(true);

    expect(
      mergeWidensDisclosure(
        makeDirectiveRecord({
          ...base,
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
        }),
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({
              content_scope: "public",
              denied_audience_behavior: "omit",
            }),
          }),
        ],
      ),
    ).toBe(true);

    const garbageScope = makeDirectiveRecord({
      ...base,
      disclosurePolicy: disclosurePolicy({ content_scope: "public" }),
    }) as CreatorDirective;

    expect(
      mergeWidensDisclosure(
        {
          ...garbageScope,
          disclosure_policy: {
            ...garbageScope.disclosure_policy,
            content_scope: "garbage",
          },
        } as unknown as CreatorDirective,
        [
          makeDirectiveRecord({
            ...base,
            disclosurePolicy: disclosurePolicy({ content_scope: "public" }),
          }),
        ],
      ),
    ).toBe(true);
  });

  it("fuzzes disclosure and activation widening against an independent oracle", () => {
    const random = seededRandom(87_001);
    const pool = Array.from({ length: 6 }, () => createEntityId());
    const subject = pool[0]!;
    const creatorPool = [pool[1]!, pool[2]!, pool[3]!];
    const cases: Array<{
      index: number;
      survivor: CreatorDirective;
      loser: CreatorDirective;
    }> = [
      {
        index: -1,
        survivor: makeDirectiveRecord({
          createdByEntityId: creatorPool[1]!,
          subjectEntityId: subject,
          disclosurePolicy: disclosurePolicy({ content_scope: "operator_only" }),
          activationPolicy: activationPolicy({ scope: "same_as_disclosure" }),
        }),
        loser: makeDirectiveRecord({
          createdByEntityId: creatorPool[0]!,
          subjectEntityId: subject,
          disclosurePolicy: disclosurePolicy({ content_scope: "operator_only" }),
          activationPolicy: activationPolicy({ scope: "same_as_disclosure" }),
        }),
      },
    ];

    for (let index = 0; index < 400; index += 1) {
      cases.push({
        index,
        survivor: makeDirectiveRecord({
          createdByEntityId: pick(random, creatorPool),
          subjectEntityId: subject,
          disclosurePolicy: randomDisclosurePolicy(random, pool, subject),
          activationPolicy: randomActivationPolicy(random, pool),
        }),
        loser: makeDirectiveRecord({
          createdByEntityId: pick(random, creatorPool),
          subjectEntityId: subject,
          disclosurePolicy: randomDisclosurePolicy(random, pool, subject),
          activationPolicy: randomActivationPolicy(random, pool),
        }),
      });
    }

    for (const { index, survivor, loser } of cases) {
      const oracleExpands = oraclePostStrictlyExpandsFamily(survivor, [loser], pool);

      if (oracleExpands) {
        const failureContext = {
          index,
          survivor: {
            created_by_entity_id: survivor.created_by_entity_id,
            disclosure_policy: survivor.disclosure_policy,
            activation_policy: survivor.activation_policy,
          },
          loser: {
            created_by_entity_id: loser.created_by_entity_id,
            disclosure_policy: loser.disclosure_policy,
            activation_policy: loser.activation_policy,
          },
        };
        expect(mergeWidensDisclosure(survivor, [loser]), JSON.stringify(failureContext)).toBe(true);
        expect(revokeWidensDisclosure(survivor, [loser]), JSON.stringify(failureContext)).toBe(
          true,
        );
      }
    }
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
          resolution: "escalate",
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
          resolution: "keep_independent",
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
          resolution: "supersede_to_survivor",
          survivor_id: spanish.id,
          loser_ids: [english.id],
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
          resolution: "supersede_to_survivor",
          survivor_id: survivor.id,
          loser_ids: [firstLoser.id, secondLoser.id],
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
          resolution: "supersede_to_survivor",
          survivor_id: survivor.id,
          loser_ids: [firstLoser.id, secondLoser.id],
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
          resolution: "supersede_to_survivor",
          survivor_id: second.id,
          loser_ids: [first.id],
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
          resolution: "escalate",
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
