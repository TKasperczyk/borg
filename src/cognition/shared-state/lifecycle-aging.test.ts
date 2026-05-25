import { describe, expect, it } from "vitest";

import {
  createActionId,
  createCommitmentId,
  createGoalId,
  createOpenQuestionId,
  createStreamEntryId,
  sharedStateEntryIdHelpers,
} from "../../util/ids.js";
import { makeSharedStateEntry } from "../../test-support/factories/shared-state.js";
import {
  applyLifecycleAging,
  blocksLiveToLowSalienceDemotion,
  blocksLowSalienceToDormantDemotion,
  entryProtectionState,
  materializeSharedStateEntriesAfterOperations,
  recordBlockerCounts,
  reactivatesDemoted,
  sharedStateLifecycleProtectionReasons,
  type LifecycleAgingBlockerCounts,
  type LifecycleProtectionReason,
} from "./lifecycle-aging.js";

const DEMOTED_KINDS = ["low_salience_live", "dormant_live"] as const;

function staleTurnInput(entryId: string) {
  return {
    currentTurnCounter: 30,
    lastUpdatedTurnByEntryId: { [entryId]: 1 },
    recentTurnThreshold: 5,
    dormantTurnThreshold: 15,
  };
}

function emptyBlockerCounts(): LifecycleAgingBlockerCounts {
  return {
    demotable_count: 0,
    unknown_age_count: 0,
    demoted_count: 0,
    blocked_by_current_turn_update: 0,
    blocked_by_patch_touch: 0,
    blocked_by_ledger_overlap: 0,
    blocked_by_recent_retrieval: 0,
    blocked_by_active_canonicalizer_critical: 0,
    blocked_by_active_canonicalizer_operational: 0,
    blocked_by_hard_total: 0,
    blocked_by_soft_total: 0,
    blocked_by_multiple_reasons: 0,
  };
}

describe("recordBlockerCounts", () => {
  it.each<{
    name: string;
    reasons: LifecycleProtectionReason[];
    expected: Partial<LifecycleAgingBlockerCounts>;
  }>([
    {
      name: "single hard reason",
      reasons: ["ledger_overlap"],
      expected: {
        blocked_by_ledger_overlap: 1,
        blocked_by_hard_total: 1,
        blocked_by_soft_total: 0,
        blocked_by_multiple_reasons: 0,
      },
    },
    {
      name: "single soft reason",
      reasons: ["recent_retrieval"],
      expected: {
        blocked_by_recent_retrieval: 1,
        blocked_by_hard_total: 0,
        blocked_by_soft_total: 1,
        blocked_by_multiple_reasons: 0,
      },
    },
    {
      name: "two hard reasons",
      reasons: ["current_turn_update", "ledger_overlap"],
      expected: {
        blocked_by_current_turn_update: 1,
        blocked_by_ledger_overlap: 1,
        blocked_by_hard_total: 1,
        blocked_by_soft_total: 0,
        blocked_by_multiple_reasons: 1,
      },
    },
    {
      name: "two soft reasons",
      reasons: ["active_canonicalizer_operational", "recent_retrieval"],
      expected: {
        blocked_by_active_canonicalizer_operational: 1,
        blocked_by_recent_retrieval: 1,
        blocked_by_hard_total: 0,
        blocked_by_soft_total: 1,
        blocked_by_multiple_reasons: 1,
      },
    },
    {
      name: "mixed hard and soft reasons",
      reasons: ["ledger_overlap", "recent_retrieval"],
      expected: {
        blocked_by_ledger_overlap: 1,
        blocked_by_recent_retrieval: 1,
        blocked_by_hard_total: 1,
        blocked_by_soft_total: 1,
        blocked_by_multiple_reasons: 1,
      },
    },
    {
      name: "three mixed reasons",
      reasons: ["current_turn_update", "ledger_overlap", "recent_retrieval"],
      expected: {
        blocked_by_current_turn_update: 1,
        blocked_by_ledger_overlap: 1,
        blocked_by_recent_retrieval: 1,
        blocked_by_hard_total: 1,
        blocked_by_soft_total: 1,
        blocked_by_multiple_reasons: 1,
      },
    },
  ])("records $name", ({ reasons, expected }) => {
    const counts = emptyBlockerCounts();

    recordBlockerCounts(counts, reasons);

    expect(counts).toMatchObject(expected);
  });
});

describe("applyLifecycleAging", () => {
  it("materializes global update turns for add update and supersede but not transitions", () => {
    const source = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_turn_global: 2,
      last_updated_stream_entry_ids: [source],
    });
    const audienceEntityId = entry.audience_entity_id;
    const transitionedOnly = materializeSharedStateEntriesAfterOperations({
      previousArtifact: {
        audience_entity_id: audienceEntityId,
        record_version: 1,
        created_at: 1_000,
        updated_at: 1_000,
        last_compiled_at: 1_000,
        last_compiled_stream_entry_id: source,
        entries: [entry],
      },
      audienceEntityId,
      nowMs: 2_000,
      lastUpdatedTurnGlobal: 10,
      operations: [
        {
          type: "transition_kind",
          id: entry.id,
          kind: "low_salience_live",
        },
      ],
    });

    const updated = materializeSharedStateEntriesAfterOperations({
      previousArtifact: {
        audience_entity_id: audienceEntityId,
        record_version: 1,
        created_at: 1_000,
        updated_at: 1_000,
        last_compiled_at: 1_000,
        last_compiled_stream_entry_id: source,
        entries: [entry],
      },
      audienceEntityId,
      nowMs: 2_000,
      lastUpdatedTurnGlobal: 10,
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.fixture",
          last_updated_stream_entry_ids: [secondSource],
        },
        {
          type: "transition_kind",
          id: entry.id,
          kind: "low_salience_live",
        },
        {
          type: "supersede",
          id: entry.id,
          replacement: {
            state_key: "decision.fixture",
            kind: "locked",
            text: "Replacement fixture",
            provenance_stream_entry_ids: [thirdSource],
          },
          last_updated_stream_entry_ids: [thirdSource],
        },
      ],
    });
    const oldEntry = updated.find((candidate) => candidate.id === entry.id);
    const replacement = updated.find((candidate) => candidate.id !== entry.id);

    expect(transitionedOnly[0]).toMatchObject({
      kind: "low_salience_live",
      last_updated_turn_global: 2,
    });
    expect(oldEntry).toMatchObject({
      kind: "low_salience_live",
      last_updated_turn_global: 10,
    });
    expect(replacement).toMatchObject({
      kind: "locked",
      last_updated_turn_global: 10,
    });
  });

  it("demotes old live entries without structural pull", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      currentTurnCounter: 12,
      lastUpdatedTurnByEntryId: { [entry.id]: 6 },
      recentTurnThreshold: 5,
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      },
    ]);
  });

  it("uses persisted global update turn before the stream-derived fallback", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_turn_global: 1,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      currentTurnCounter: 30,
      lastUpdatedTurnByEntryId: { [entry.id]: 30 },
      recentTurnThreshold: 5,
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      },
    ]);
  });

  it("falls back to stream-derived update turn for legacy entries", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_turn_global: null,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      currentTurnCounter: 12,
      lastUpdatedTurnByEntryId: { [entry.id]: 6 },
      recentTurnThreshold: 5,
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      },
    ]);
  });

  it("demotes old low-salience live entries to dormant", () => {
    const entry = makeSharedStateEntry({
      kind: "low_salience_live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      currentTurnCounter: 30,
      lastUpdatedTurnByEntryId: { [entry.id]: 14 },
      dormantTurnThreshold: 15,
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "low_salience_live",
        toKind: "dormant_live",
        reason: "old_low_salience_without_structural_pull",
        transition: "demoted",
      },
    ]);
  });

  it("reactivates demoted entries before demotion checks", () => {
    const currentSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind: "low_salience_live",
      last_updated_stream_entry_ids: [currentSource],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      currentUserStreamEntryId: currentSource,
      currentTurnCounter: 30,
      lastUpdatedTurnByEntryId: { [entry.id]: 1 },
      recentTurnThreshold: 5,
      dormantTurnThreshold: 15,
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "low_salience_live",
        toKind: "live",
        reason: "current_turn_update",
        transition: "reactivated",
      },
    ]);
  });

  it.each(DEMOTED_KINDS)("reactivates %s on a current-turn source", (kind) => {
    const currentSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind,
      last_updated_stream_entry_ids: [currentSource],
    });
    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      currentUserStreamEntryId: currentSource,
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: kind,
        toKind: "live",
        reason: "current_turn_update",
        transition: "reactivated",
      },
    ]);
  });

  it.each(DEMOTED_KINDS)("reactivates %s on ledger overlap", (kind) => {
    const ledgerSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind,
      provenance_stream_entry_ids: [ledgerSource],
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      ledgerStreamEntryIds: [ledgerSource],
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: kind,
        toKind: "live",
        reason: "ledger_overlap",
        transition: "reactivated",
      },
    ]);
  });

  it.each(DEMOTED_KINDS)("reactivates %s on active critical canonicalizer overlap", (kind) => {
    const activeCommitmentId = createCommitmentId();
    const entry = makeSharedStateEntry({
      kind,
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [activeCommitmentId],
        action_ids: [],
        open_question_ids: [],
      },
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      activeCriticalCommitmentIds: [activeCommitmentId],
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: kind,
        toKind: "live",
        reason: "active_canonicalizer_critical",
        transition: "reactivated",
      },
    ]);
  });

  it.each(DEMOTED_KINDS)("does not reactivate %s on recent retrieval citation", (kind) => {
    const entry = makeSharedStateEntry({
      kind,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      recentlyRetrievedEntryIds: [entry.id],
    });

    expect(result.transitions).toEqual([]);
  });

  it.each(DEMOTED_KINDS)("reactivates %s on a direct patch touch", (kind) => {
    const entry = makeSharedStateEntry({
      kind,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      touchedEntryIds: new Set([entry.id]),
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: kind,
        toKind: "live",
        reason: "touched_by_patch",
        transition: "reactivated",
      },
    ]);
  });

  it("keeps old live entries active when hard structural protection applies", () => {
    const currentSource = createStreamEntryId();
    const ledgerSource = createStreamEntryId();
    const activeCommitmentId = createCommitmentId();
    const cases = [
      makeSharedStateEntry({
        kind: "live",
        last_updated_stream_entry_ids: [currentSource],
      }),
      makeSharedStateEntry({
        kind: "live",
        provenance_stream_entry_ids: [ledgerSource],
        last_updated_stream_entry_ids: [createStreamEntryId()],
      }),
      makeSharedStateEntry({
        kind: "live",
        last_updated_stream_entry_ids: [createStreamEntryId()],
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [activeCommitmentId],
          action_ids: [],
          open_question_ids: [],
        },
      }),
    ];

    const result = applyLifecycleAging({
      entries: cases,
      currentUserStreamEntryId: currentSource,
      ledgerStreamEntryIds: [ledgerSource],
      activeCriticalCommitmentIds: [activeCommitmentId],
      currentTurnCounter: 30,
      lastUpdatedTurnByEntryId: Object.fromEntries(cases.map((entry) => [entry.id, 1])),
      recentTurnThreshold: 5,
    });

    expect(result.transitions).toEqual([]);
  });

  it("does not demote an old live entry with current-turn update protection", () => {
    const currentSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [currentSource],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      currentUserStreamEntryId: currentSource,
    });

    expect(result.transitions).toEqual([]);
  });

  it("does not demote an old live entry with ledger overlap protection", () => {
    const ledgerSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind: "live",
      provenance_stream_entry_ids: [ledgerSource],
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      ledgerStreamEntryIds: [ledgerSource],
    });

    expect(result.transitions).toEqual([]);
  });

  it("does not demote an old live entry with critical canonicalizer protection", () => {
    const activeCommitmentId = createCommitmentId();
    const entry = makeSharedStateEntry({
      kind: "live",
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [activeCommitmentId],
        action_ids: [],
        open_question_ids: [],
      },
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      activeCriticalCommitmentIds: [activeCommitmentId],
    });

    expect(result.transitions).toEqual([]);
  });

  it("demotes an old live entry with only recent retrieval protection", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      recentlyRetrievedEntryIds: [entry.id],
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      },
    ]);
  });

  it("demotes an old live entry with only operational canonicalizer protection", () => {
    const activeActionId = createActionId();
    const entry = makeSharedStateEntry({
      kind: "live",
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [activeActionId],
        open_question_ids: [],
      },
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      activeActionIds: [activeActionId],
    });

    expect(result.transitions).toEqual([
      {
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      },
    ]);
  });

  it("does not demote an old low-salience entry with recent retrieval protection", () => {
    const entry = makeSharedStateEntry({
      kind: "low_salience_live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      recentlyRetrievedEntryIds: [entry.id],
    });

    expect(result.transitions).toEqual([]);
    expect(result.blockerCountsLowSalienceToDormant).toMatchObject({
      demotable_count: 1,
      demoted_count: 0,
      blocked_by_recent_retrieval: 1,
      blocked_by_soft_total: 1,
    });
  });

  it("does not demote an old live entry with direct patch touch protection", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      touchedEntryIds: new Set([entry.id]),
    });

    expect(result.transitions).toEqual([]);
  });

  it("does not demote when turn age is unavailable", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      currentTurnCounter: 30,
      recentTurnThreshold: 5,
    });

    expect(result.transitions).toEqual([]);
  });

  it("reports blocker counts for old live entries without changing transitions", () => {
    const currentSource = createStreamEntryId();
    const ledgerSource = createStreamEntryId();
    const activeCommitmentId = createCommitmentId();
    const oldProtected = makeSharedStateEntry({
      kind: "live",
      state_key: "protected",
      provenance_stream_entry_ids: [ledgerSource],
      last_updated_stream_entry_ids: [currentSource],
    });
    const oldMultipleProtected = makeSharedStateEntry({
      kind: "live",
      state_key: "multiple",
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [activeCommitmentId],
        action_ids: [],
        open_question_ids: [],
      },
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const oldDemoted = makeSharedStateEntry({
      kind: "live",
      state_key: "demoted",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const unknownAge = makeSharedStateEntry({
      kind: "live",
      state_key: "unknown",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [oldProtected, oldMultipleProtected, oldDemoted, unknownAge],
      currentUserStreamEntryId: currentSource,
      ledgerStreamEntryIds: [ledgerSource],
      activeCriticalCommitmentIds: [activeCommitmentId],
      recentlyRetrievedEntryIds: [oldMultipleProtected.id],
      currentTurnCounter: 30,
      lastUpdatedTurnByEntryId: {
        [oldProtected.id]: 1,
        [oldMultipleProtected.id]: 2,
        [oldDemoted.id]: 3,
      },
      recentTurnThreshold: 5,
    });

    expect(result.transitions).toEqual([
      {
        entryId: oldDemoted.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      },
    ]);
    expect(result.blockerCountsLiveToLowSalience).toMatchObject({
      demotable_count: 3,
      unknown_age_count: 1,
      demoted_count: 1,
      blocked_by_current_turn_update: 1,
      blocked_by_patch_touch: 0,
      blocked_by_ledger_overlap: 1,
      blocked_by_recent_retrieval: 0,
      blocked_by_active_canonicalizer_critical: 1,
      blocked_by_active_canonicalizer_operational: 0,
      blocked_by_hard_total: 2,
      blocked_by_soft_total: 0,
      blocked_by_multiple_reasons: 1,
    });
    expect(result.blockedSample.map((entry) => entry.entry_id)).toEqual([
      oldProtected.id,
      oldMultipleProtected.id,
    ]);
    expect(result.unknownAgeSample).toEqual([
      {
        entry_id: unknownAge.id,
        state_key: "unknown",
        kind: "live",
        last_updated_stream_entry_ids_count: 1,
        last_updated_turn_global: null,
        rank: unknownAge.rank,
      },
    ]);
  });

  it("sorts unknown-age samples by rank then entry id", () => {
    const highRank = makeSharedStateEntry({
      id: sharedStateEntryIdHelpers.parse("dart_0000000000000004"),
      kind: "live",
      state_key: "unknown.high-rank",
      rank: 9,
    });
    const tiedLaterId = makeSharedStateEntry({
      id: sharedStateEntryIdHelpers.parse("dart_0000000000000003"),
      kind: "live",
      state_key: "unknown.tie-later",
      rank: 2,
    });
    const tiedEarlierId = makeSharedStateEntry({
      id: sharedStateEntryIdHelpers.parse("dart_0000000000000001"),
      kind: "live",
      state_key: "unknown.tie-earlier",
      rank: 2,
    });
    const lowRank = makeSharedStateEntry({
      id: sharedStateEntryIdHelpers.parse("dart_0000000000000002"),
      kind: "live",
      state_key: "unknown.low-rank",
      rank: 1,
    });

    const result = applyLifecycleAging({
      entries: [highRank, tiedLaterId, tiedEarlierId, lowRank],
      currentTurnCounter: 30,
      recentTurnThreshold: 5,
    });

    expect(result.unknownAgeSample.map((entry) => entry.entry_id)).toEqual([
      lowRank.id,
      tiedEarlierId.id,
      tiedLaterId.id,
      highRank.id,
    ]);
  });

  it("caps unknown-age samples at the first 10 entries by deterministic sort order", () => {
    const entries = Array.from({ length: 12 }, (_, index) =>
      makeSharedStateEntry({
        id: sharedStateEntryIdHelpers.parse(`dart_00000000000000${String(index).padStart(2, "0")}`),
        kind: "live",
        state_key: `unknown.${index}`,
        rank: 12 - index,
      }),
    );

    const result = applyLifecycleAging({
      entries,
      currentTurnCounter: 30,
      recentTurnThreshold: 5,
    });

    const expectedEntryIds = [...entries]
      .sort((left, right) => left.rank - right.rank || left.id.localeCompare(right.id))
      .slice(0, 10)
      .map((entry) => entry.id);

    expect(result.unknownAgeSample).toHaveLength(10);
    expect(result.unknownAgeSample.map((entry) => entry.entry_id)).toEqual(expectedEntryIds);
  });
});

describe("sharedStateLifecycleProtectionReasons", () => {
  it("returns all applicable reasons", () => {
    const currentSource = createStreamEntryId();
    const ledgerSource = createStreamEntryId();
    const activeOpenQuestionId = createOpenQuestionId();
    const activeActionId = createActionId();
    const activeGoalId = createGoalId();
    const activeCommitmentId = createCommitmentId();
    const entry = makeSharedStateEntry({
      kind: "live",
      provenance_stream_entry_ids: [ledgerSource],
      last_updated_stream_entry_ids: [currentSource],
      canonicalizes: {
        goal_ids: [activeGoalId],
        commitment_ids: [activeCommitmentId],
        action_ids: [activeActionId],
        open_question_ids: [activeOpenQuestionId],
      },
    });

    expect(
      sharedStateLifecycleProtectionReasons(entry, {
        entries: [entry],
        touchedEntryIds: new Set([entry.id]),
        currentUserStreamEntryId: currentSource,
        ledgerStreamEntryIds: [ledgerSource],
        activeOpenQuestionIds: [activeOpenQuestionId],
        activeActionIds: [activeActionId],
        activeGoalIds: [activeGoalId],
        activeCriticalCommitmentIds: [activeCommitmentId],
        recentlyRetrievedEntryIds: [entry.id],
      }),
    ).toEqual([
      { reason: "touched_by_patch", strength: "hard" },
      { reason: "current_turn_update", strength: "hard" },
      { reason: "ledger_overlap", strength: "hard" },
      { reason: "active_canonicalizer_critical", strength: "hard" },
      { reason: "active_canonicalizer_operational", strength: "soft" },
      { reason: "recent_retrieval", strength: "soft" },
    ]);
  });

  it("returns an empty array for unprotected entries", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    expect(sharedStateLifecycleProtectionReasons(entry, { entries: [entry] })).toEqual([]);
  });

  it("returns a single reason when one protection applies", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    expect(
      sharedStateLifecycleProtectionReasons(entry, {
        entries: [entry],
        recentlyRetrievedEntryIds: [entry.id],
      }),
    ).toEqual([{ reason: "recent_retrieval", strength: "soft" }]);
  });

  it("returns multiple reasons when multiple protections apply", () => {
    const currentSource = createStreamEntryId();
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [currentSource],
    });

    expect(
      sharedStateLifecycleProtectionReasons(entry, {
        entries: [entry],
        currentUserStreamEntryId: currentSource,
        recentlyRetrievedEntryIds: [entry.id],
      }),
    ).toEqual([
      { reason: "current_turn_update", strength: "hard" },
      { reason: "recent_retrieval", strength: "soft" },
    ]);
  });
});

describe("entry protection tiers", () => {
  it("splits hard and soft protections", () => {
    const currentSource = createStreamEntryId();
    const ledgerSource = createStreamEntryId();
    const activeCriticalCommitmentId = createCommitmentId();
    const activeOperationalCommitmentId = createCommitmentId();
    const activeActionId = createActionId();
    const entry = makeSharedStateEntry({
      kind: "live",
      provenance_stream_entry_ids: [ledgerSource],
      last_updated_stream_entry_ids: [currentSource],
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [activeCriticalCommitmentId, activeOperationalCommitmentId],
        action_ids: [activeActionId],
        open_question_ids: [],
      },
    });

    expect(
      entryProtectionState(entry, {
        entries: [entry],
        touchedEntryIds: new Set([entry.id]),
        currentUserStreamEntryId: currentSource,
        ledgerStreamEntryIds: [ledgerSource],
        activeActionIds: [activeActionId],
        activeCriticalCommitmentIds: [activeCriticalCommitmentId],
        activeOperationalCommitmentIds: [activeOperationalCommitmentId],
        recentlyRetrievedEntryIds: [entry.id],
      }),
    ).toEqual({
      hard: [
        "touched_by_patch",
        "current_turn_update",
        "ledger_overlap",
        "active_canonicalizer_critical",
      ],
      soft: ["active_canonicalizer_operational", "recent_retrieval"],
    });
  });

  it("uses only hard protections for live demotion and reactivation", () => {
    const activeActionId = createActionId();
    const softOnly = makeSharedStateEntry({
      kind: "low_salience_live",
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [activeActionId],
        open_question_ids: [],
      },
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const hardSource = createStreamEntryId();
    const hardEntry = makeSharedStateEntry({
      kind: "low_salience_live",
      last_updated_stream_entry_ids: [hardSource],
    });

    expect(
      blocksLiveToLowSalienceDemotion(softOnly, {
        entries: [softOnly],
        activeActionIds: [activeActionId],
        recentlyRetrievedEntryIds: [softOnly.id],
      }),
    ).toBeNull();
    expect(
      blocksLowSalienceToDormantDemotion(softOnly, {
        entries: [softOnly],
        activeActionIds: [activeActionId],
      }),
    ).toBe("active_canonicalizer_operational");
    expect(
      reactivatesDemoted(softOnly, {
        entries: [softOnly],
        activeActionIds: [activeActionId],
        recentlyRetrievedEntryIds: [softOnly.id],
      }),
    ).toBeNull();
    expect(
      reactivatesDemoted(hardEntry, {
        entries: [hardEntry],
        currentUserStreamEntryId: hardSource,
      }),
    ).toBe("current_turn_update");
  });

  it.each([
    {
      name: "open question canonicalizer",
      build: () => {
        const id = createOpenQuestionId();
        return {
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [id],
          },
          input: { activeOpenQuestionIds: [id] },
        };
      },
    },
    {
      name: "action canonicalizer",
      build: () => {
        const id = createActionId();
        return {
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [],
            action_ids: [id],
            open_question_ids: [],
          },
          input: { activeActionIds: [id] },
        };
      },
    },
    {
      name: "goal canonicalizer",
      build: () => {
        const id = createGoalId();
        return {
          canonicalizes: {
            goal_ids: [id],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
          input: { activeGoalIds: [id] },
        };
      },
    },
    {
      name: "operational commitment canonicalizer",
      build: () => {
        const id = createCommitmentId();
        return {
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [id],
            action_ids: [],
            open_question_ids: [],
          },
          input: { activeOperationalCommitmentIds: [id], activeCriticalCommitmentIds: [] },
        };
      },
    },
  ])("$name is soft-only lifecycle protection", ({ build }) => {
    const { canonicalizes, input } = build();
    const liveEntry = makeSharedStateEntry({
      kind: "live",
      canonicalizes,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });
    const lowSalienceEntry = makeSharedStateEntry({
      kind: "low_salience_live",
      canonicalizes,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    expect(
      blocksLiveToLowSalienceDemotion(liveEntry, {
        entries: [liveEntry],
        ...input,
      }),
    ).toBeNull();
    expect(
      blocksLowSalienceToDormantDemotion(lowSalienceEntry, {
        entries: [lowSalienceEntry],
        ...input,
      }),
    ).toBe("active_canonicalizer_operational");
    expect(
      reactivatesDemoted(lowSalienceEntry, {
        entries: [lowSalienceEntry],
        ...input,
      }),
    ).toBeNull();
  });
});
