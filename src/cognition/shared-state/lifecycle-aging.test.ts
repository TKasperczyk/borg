import { describe, expect, it } from "vitest";

import {
  createActionId,
  createGoalId,
  createSharedStateEntryId,
  createStreamEntryId,
} from "../../util/ids.js";
import { makeSharedStateEntry } from "../../test-support/factories/shared-state.js";
import { applyLifecycleAging } from "./lifecycle-aging.js";

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
});
