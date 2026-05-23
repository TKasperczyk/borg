import { describe, expect, it } from "vitest";

import {
  createActionId,
  createCommitmentId,
  createGoalId,
  createOpenQuestionId,
  createSharedStateEntryId,
  createStreamEntryId,
} from "../../util/ids.js";
import { makeSharedStateEntry } from "../../test-support/factories/shared-state.js";
import { applyLifecycleAging, sharedStateLifecycleProtectionReasons } from "./lifecycle-aging.js";

const DEMOTED_KINDS = ["low_salience_live", "dormant_live"] as const;

function staleTurnInput(entryId: string) {
  return {
    currentTurnCounter: 30,
    lastUpdatedTurnByEntryId: { [entryId]: 1 },
    recentTurnThreshold: 5,
    dormantTurnThreshold: 15,
  };
}

describe("applyLifecycleAging", () => {
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

  it.each(DEMOTED_KINDS)("reactivates %s on active canonicalizer overlap", (kind) => {
    const activeActionId = createActionId();
    const entry = makeSharedStateEntry({
      kind,
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
        fromKind: kind,
        toKind: "live",
        reason: "active_canonicalizer_overlap",
        transition: "reactivated",
      },
    ]);
  });

  it.each(DEMOTED_KINDS)("reactivates %s on recent retrieval citation", (kind) => {
    const entry = makeSharedStateEntry({
      kind,
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
        fromKind: kind,
        toKind: "live",
        reason: "recent_retrieval",
        transition: "reactivated",
      },
    ]);
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

  it("keeps old live entries active when any structural protection applies", () => {
    const currentSource = createStreamEntryId();
    const ledgerSource = createStreamEntryId();
    const activeActionId = createActionId();
    const activeGoalId = createGoalId();
    const retrievedId = createSharedStateEntryId();
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
          commitment_ids: [],
          action_ids: [activeActionId],
          open_question_ids: [],
        },
      }),
      makeSharedStateEntry({
        kind: "live",
        id: retrievedId,
        last_updated_stream_entry_ids: [createStreamEntryId()],
      }),
      makeSharedStateEntry({
        kind: "live",
        last_updated_stream_entry_ids: [createStreamEntryId()],
        canonicalizes: {
          goal_ids: [activeGoalId],
          commitment_ids: [],
          action_ids: [],
          open_question_ids: [],
        },
      }),
    ];

    const result = applyLifecycleAging({
      entries: cases,
      currentUserStreamEntryId: currentSource,
      ledgerStreamEntryIds: [ledgerSource],
      activeActionIds: [activeActionId],
      activeGoalIds: [activeGoalId],
      recentlyRetrievedEntryIds: [retrievedId],
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

  it("does not demote an old live entry with canonicalizer protection", () => {
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

    expect(result.transitions).toEqual([]);
  });

  it("does not demote an old live entry with recent retrieval protection", () => {
    const entry = makeSharedStateEntry({
      kind: "live",
      last_updated_stream_entry_ids: [createStreamEntryId()],
    });

    const result = applyLifecycleAging({
      entries: [entry],
      ...staleTurnInput(entry.id),
      recentlyRetrievedEntryIds: [entry.id],
    });

    expect(result.transitions).toEqual([]);
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
    const activeActionId = createActionId();
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
        commitment_ids: [],
        action_ids: [activeActionId],
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
      activeActionIds: [activeActionId],
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
      blocked_by_current_turn_update: 0,
      blocked_by_patch_touch: 0,
      blocked_by_ledger_overlap: 0,
      blocked_by_recent_retrieval: 0,
      blocked_by_active_canonicalizer: 0,
      blocked_by_multiple_reasons: 2,
    });
    expect(result.blockedSample.map((entry) => entry.entry_id)).toEqual([
      oldProtected.id,
      oldMultipleProtected.id,
    ]);
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
      "touched_by_patch",
      "current_turn_update",
      "ledger_overlap",
      "active_canonicalizer_overlap",
      "recent_retrieval",
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
    ).toEqual(["recent_retrieval"]);
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
    ).toEqual(["current_turn_update", "recent_retrieval"]);
  });
});
