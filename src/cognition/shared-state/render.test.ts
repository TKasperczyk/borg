import { describe, expect, it } from "vitest";

import type {
  SharedStateArtifact,
  SharedStateEntry,
  SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";
import {
  createActionId,
  createCommitmentId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createSharedStateEntryId,
  createStreamEntryId,
  type ActionId,
  type CommitmentId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { renderSharedStateArtifact, summarizeSharedStateArtifactRender } from "./render.js";

function entry(input: {
  audience: EntityId;
  kind: SharedStateEntryKind;
  rank: number;
  updatedAt: number;
  stateKey?: string | null;
  text?: string;
  provenanceStreamEntryIds?: StreamEntryId[];
  lastUpdatedStreamEntryIds?: StreamEntryId[];
  lastUpdatedTurnGlobal?: number | null;
  canonicalizes?: {
    goal_ids?: GoalId[];
    commitment_ids?: CommitmentId[];
    action_ids?: ActionId[];
    open_question_ids?: OpenQuestionId[];
  };
}): SharedStateEntry {
  const streamId = createStreamEntryId();
  const provenanceStreamEntryIds = input.provenanceStreamEntryIds ?? [streamId];
  const lastUpdatedStreamEntryIds = input.lastUpdatedStreamEntryIds ?? [streamId];

  return {
    id: createSharedStateEntryId(),
    audience_entity_id: input.audience,
    state_key: input.stateKey === undefined ? `${input.kind}.state` : input.stateKey,
    kind: input.kind,
    text: input.text ?? `${input.kind} state ${input.rank}`,
    owner_entity_id: null,
    provenance_stream_entry_ids: provenanceStreamEntryIds,
    last_updated_stream_entry_ids: lastUpdatedStreamEntryIds,
    created_at: input.updatedAt,
    last_updated_at: input.updatedAt,
    last_updated_turn_global: input.lastUpdatedTurnGlobal ?? null,
    superseded_by_id: null,
    rank: input.rank,
    canonicalizes: {
      goal_ids: input.canonicalizes?.goal_ids ?? [],
      commitment_ids: input.canonicalizes?.commitment_ids ?? [],
      action_ids: input.canonicalizes?.action_ids ?? [],
      open_question_ids: input.canonicalizes?.open_question_ids ?? [],
    },
  };
}

function artifact(entries: readonly SharedStateEntry[]): SharedStateArtifact {
  return {
    audience_entity_id: entries[0]?.audience_entity_id ?? createEntityId(),
    record_version: 1,
    created_at: 1,
    updated_at: 1,
    last_compiled_at: 1,
    last_compiled_stream_entry_id: null,
    entries: [...entries],
  };
}

describe("summarizeSharedStateArtifactRender", () => {
  it("reports newest reserved live entries in the render summary", () => {
    const audience = createEntityId();
    const entries = [
      ...Array.from({ length: 14 }, (_, index) =>
        entry({ audience, kind: "locked", rank: index, updatedAt: 1_000 + index }),
      ),
      ...Array.from({ length: 10 }, (_, index) =>
        entry({ audience, kind: "live", rank: 100 + index, updatedAt: 10_000 + index }),
      ),
      ...Array.from({ length: 5 }, (_, index) =>
        entry({ audience, kind: "pending", rank: 200 + index, updatedAt: 2_000 + index }),
      ),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 25,
      maxTokens: 50_000,
      reservedSlots: {
        live: 8,
        pending: 3,
        invalidated: 3,
      },
      lockedMaxEntries: 14,
      newestStateChangeReservedSlots: 3,
    });

    expect(summary.renderedEntryCount).toBe(25);
    expect(summary.renderedByKind.live).toBe(10);
    expect(summary.newestReservedEntryCount).toBe(3);
  });

  it("classifies omitted shared-state severity using turn and structural canonicalizer signals", () => {
    const audience = createEntityId();
    const currentUserStreamEntryId = createStreamEntryId();
    const operationalUpdateId = createStreamEntryId();
    const recentUpdateId = createStreamEntryId();
    const oldUpdateId = createStreamEntryId();
    const unknownUpdateId = createStreamEntryId();
    const openQuestionId = createOpenQuestionId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 100,
        text: "current turn live",
        lastUpdatedStreamEntryIds: [currentUserStreamEntryId],
      }),
      entry({
        audience,
        kind: "live",
        rank: 1,
        updatedAt: 95,
        text: "second current turn live",
        lastUpdatedStreamEntryIds: [currentUserStreamEntryId],
      }),
      entry({
        audience,
        kind: "live",
        rank: 2,
        updatedAt: 90,
        text: "operational live",
        lastUpdatedStreamEntryIds: [operationalUpdateId],
        canonicalizes: { open_question_ids: [openQuestionId] },
      }),
      entry({
        audience,
        kind: "live",
        rank: 3,
        updatedAt: 80,
        text: "recent low salience live",
        lastUpdatedStreamEntryIds: [recentUpdateId],
      }),
      entry({
        audience,
        kind: "live",
        rank: 4,
        updatedAt: 70,
        text: "old live",
        lastUpdatedStreamEntryIds: [oldUpdateId],
      }),
      entry({
        audience,
        kind: "live",
        rank: 5,
        updatedAt: 65,
        text: "unknown age live",
        lastUpdatedStreamEntryIds: [unknownUpdateId],
      }),
      entry({ audience, kind: "locked", rank: 6, updatedAt: 60 }),
      entry({ audience, kind: "pending", rank: 7, updatedAt: 50 }),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 0,
        pending: 0,
        invalidated: 0,
      },
      lockedMaxEntries: 0,
      newestStateChangeReservedSlots: 0,
      currentUserStreamEntryId,
      activeOpenQuestionIds: [openQuestionId],
      currentTurnCounter: 10,
      lastUpdatedTurnByStreamEntryId: {
        [currentUserStreamEntryId]: 10,
        [operationalUpdateId]: 7,
        [recentUpdateId]: 7,
        [oldUpdateId]: 0,
      },
    });

    expect(summary.omittedLiveRecentOperational).toBe(2);
    expect(summary.omittedLiveRecentLowSalience).toBe(1);
    expect(summary.omittedLiveOld).toBe(1);
    expect(summary.omittedLiveUnknownAge).toBe(1);
    expect(summary.omittedLocked).toBe(1);
    expect(summary.omittedPending).toBe(1);
    expect(summary.allActiveKeysIndexed).toBe(true);
  });

  it("classifies omitted locked entries by structural severity subtype", () => {
    const audience = createEntityId();
    const currentUserStreamEntryId = createStreamEntryId();
    const recentUpdateId = createStreamEntryId();
    const oldUpdateId = createStreamEntryId();
    const unknownUpdateId = createStreamEntryId();
    const commitmentId = createCommitmentId();
    const goalId = createGoalId();
    const entries = [
      entry({
        audience,
        kind: "locked",
        rank: 0,
        updatedAt: 100,
        stateKey: "locked.rendered",
        lastUpdatedStreamEntryIds: [currentUserStreamEntryId],
      }),
      entry({
        audience,
        kind: "locked",
        rank: 1,
        updatedAt: 90,
        stateKey: "locked.recent",
        lastUpdatedStreamEntryIds: [recentUpdateId],
      }),
      entry({
        audience,
        kind: "locked",
        rank: 2,
        updatedAt: 80,
        stateKey: "locked.old",
        lastUpdatedStreamEntryIds: [oldUpdateId],
      }),
      entry({
        audience,
        kind: "locked",
        rank: 3,
        updatedAt: 70,
        stateKey: "locked.unknown",
        lastUpdatedStreamEntryIds: [unknownUpdateId],
      }),
      entry({
        audience,
        kind: "locked",
        rank: 4,
        updatedAt: 60,
        stateKey: "locked.commitment",
        lastUpdatedStreamEntryIds: [oldUpdateId],
        canonicalizes: { commitment_ids: [commitmentId] },
      }),
      entry({
        audience,
        kind: "locked",
        rank: 5,
        updatedAt: 50,
        stateKey: "locked.goal",
        lastUpdatedStreamEntryIds: [recentUpdateId],
        canonicalizes: { goal_ids: [goalId] },
      }),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 0,
        pending: 0,
        invalidated: 0,
      },
      lockedMaxEntries: 6,
      newestStateChangeReservedSlots: 0,
      currentUserStreamEntryId,
      activeGoalIds: [goalId],
      activeCriticalCommitmentIds: [commitmentId],
      currentTurnCounter: 20,
      lastUpdatedTurnByStreamEntryId: {
        [currentUserStreamEntryId]: 20,
        [recentUpdateId]: 18,
        [oldUpdateId]: 10,
      },
    });

    expect(summary.omittedLocked).toBe(5);
    expect(summary.omittedLockedRecent).toBe(2);
    expect(summary.omittedLockedOld).toBe(2);
    expect(summary.omittedLockedUnknownAge).toBe(1);
    expect(summary.omittedLockedWithActiveCriticalCommitment).toBe(1);
    expect(summary.omittedLockedWithOperationalCanonicalizer).toBe(1);
    expect(summary.omittedLockedIndexedOnly).toBe(5);
  });

  it("classifies omitted live prior-session entries with old persisted turns as old", () => {
    const audience = createEntityId();
    const priorSessionId = createStreamEntryId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 100,
        text: "rendered live",
      }),
      entry({
        audience,
        kind: "live",
        rank: 1,
        updatedAt: 10,
        text: "old persisted live",
        lastUpdatedStreamEntryIds: [priorSessionId],
        lastUpdatedTurnGlobal: 1,
      }),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 0,
        pending: 0,
        invalidated: 0,
      },
      newestStateChangeReservedSlots: 0,
      currentTurnCounter: 10,
      lastUpdatedTurnByStreamEntryId: {},
    });

    expect(summary.omittedLiveOld).toBe(1);
    expect(summary.omittedLiveUnknownAge).toBe(0);
  });

  it("classifies omitted live prior-session entries with recent persisted turns as recent", () => {
    const audience = createEntityId();
    const priorSessionId = createStreamEntryId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 100,
        text: "rendered live",
      }),
      entry({
        audience,
        kind: "live",
        rank: 1,
        updatedAt: 10,
        text: "recent persisted live",
        lastUpdatedStreamEntryIds: [priorSessionId],
        lastUpdatedTurnGlobal: 8,
      }),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 0,
        pending: 0,
        invalidated: 0,
      },
      newestStateChangeReservedSlots: 0,
      currentTurnCounter: 10,
      lastUpdatedTurnByStreamEntryId: {},
    });

    expect(summary.omittedLiveRecentLowSalience).toBe(1);
    expect(summary.omittedLiveOld).toBe(0);
    expect(summary.omittedLiveUnknownAge).toBe(0);
  });

  it("keeps omitted live entries with null persisted turns and no stream lookup unknown age", () => {
    const audience = createEntityId();
    const priorSessionId = createStreamEntryId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 100,
        text: "rendered live",
      }),
      entry({
        audience,
        kind: "live",
        rank: 1,
        updatedAt: 10,
        text: "unknown persisted live",
        lastUpdatedStreamEntryIds: [priorSessionId],
        lastUpdatedTurnGlobal: null,
      }),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 0,
        pending: 0,
        invalidated: 0,
      },
      newestStateChangeReservedSlots: 0,
      currentTurnCounter: 10,
      lastUpdatedTurnByStreamEntryId: {},
    });

    expect(summary.omittedLiveOld).toBe(0);
    expect(summary.omittedLiveUnknownAge).toBe(1);
  });

  it("classifies omitted locked prior-session entries with old persisted turns as old", () => {
    const audience = createEntityId();
    const priorSessionId = createStreamEntryId();
    const entries = [
      entry({
        audience,
        kind: "locked",
        rank: 0,
        updatedAt: 100,
        text: "rendered locked",
      }),
      entry({
        audience,
        kind: "locked",
        rank: 1,
        updatedAt: 10,
        text: "old persisted locked",
        lastUpdatedStreamEntryIds: [priorSessionId],
        lastUpdatedTurnGlobal: 1,
      }),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 0,
        pending: 0,
        invalidated: 0,
      },
      lockedMaxEntries: 1,
      newestStateChangeReservedSlots: 0,
      currentTurnCounter: 10,
      lastUpdatedTurnByStreamEntryId: {},
    });

    expect(summary.omittedLockedOld).toBe(1);
    expect(summary.omittedLockedUnknownAge).toBe(0);
  });
});

describe("renderSharedStateArtifact", () => {
  it("renders a compact all-key index before detailed entries", () => {
    const audience = createEntityId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 20,
        stateKey: "project.alpha",
        text: "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho",
      }),
      entry({
        audience,
        kind: "locked",
        rank: 1,
        updatedAt: 10,
        stateKey: "project.alpha",
        text: "locked alpha detail",
      }),
      entry({
        audience,
        kind: "pending",
        rank: 2,
        updatedAt: 30,
        stateKey: null,
        text: "legacy pending detail",
      }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 1,
        maxTokens: 50_000,
        newestStateChangeReservedSlots: 1,
      }) ?? "";

    expect(rendered.indexOf("SharedStateArtifact compact active-key index:")).toBeLessThan(
      rendered.indexOf("state_key_bucket="),
    );
    expect(rendered).toContain("- legacy | kinds=pending |");
    expect(rendered).toContain("- project.alpha | kinds=locked,live |");
    expect(rendered).toContain("active_count=2");
    expect(rendered).toContain(
      'excerpt="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron..."',
    );
    expect(rendered).toContain("| expanded");
    expect(rendered).toContain("| omitted");

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      newestStateChangeReservedSlots: 1,
    });
    expect(summary.compactIndexLineCount).toBe(2);
    expect(summary.allActiveKeysIndexed).toBe(true);
  });

  it("keeps demoted live entries in the compact index without default detail expansion", () => {
    const audience = createEntityId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 50,
        stateKey: "state.current",
        text: "current live detail",
      }),
      entry({
        audience,
        kind: "low_salience_live",
        rank: 1,
        updatedAt: 40,
        stateKey: "state.low",
        text: "low salience detail",
      }),
      entry({
        audience,
        kind: "dormant_live",
        rank: 2,
        updatedAt: 30,
        stateKey: "state.dormant",
        text: "dormant detail",
      }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 1,
        maxTokens: 50_000,
        reservedSlots: {
          live: 1,
          pending: 0,
          invalidated: 0,
        },
        newestStateChangeReservedSlots: 0,
      }) ?? "";
    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 1,
        pending: 0,
        invalidated: 0,
      },
      newestStateChangeReservedSlots: 0,
    });

    expect(rendered).toContain("- state.low | kinds=low_salience_live |");
    expect(rendered).toContain("- state.dormant | kinds=dormant_live |");
    expect(rendered).toContain("text: current live detail");
    expect(rendered).not.toContain("text: low salience detail");
    expect(rendered).not.toContain("text: dormant detail");
    expect(summary.omittedLowSalienceLive).toBe(1);
    expect(summary.omittedDormantLive).toBe(1);
  });

  it("uses structural salience signals to choose detailed expansions", () => {
    const audience = createEntityId();
    const currentUserStreamEntryId = createStreamEntryId();
    const ledgerStreamEntryId = createStreamEntryId();
    const openQuestionId = createOpenQuestionId();
    const actionId = createActionId();
    const goalId = createGoalId();
    const commitmentId = createCommitmentId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 100,
        updatedAt: 1,
        text: "current detail",
        lastUpdatedStreamEntryIds: [currentUserStreamEntryId],
      }),
      entry({
        audience,
        kind: "live",
        rank: 101,
        updatedAt: 2,
        text: "ledger detail",
        provenanceStreamEntryIds: [ledgerStreamEntryId],
      }),
      entry({
        audience,
        kind: "live",
        rank: 102,
        updatedAt: 3,
        text: "open question detail",
        canonicalizes: { open_question_ids: [openQuestionId] },
      }),
      entry({
        audience,
        kind: "live",
        rank: 103,
        updatedAt: 4,
        text: "action detail",
        canonicalizes: { action_ids: [actionId] },
      }),
      entry({
        audience,
        kind: "live",
        rank: 104,
        updatedAt: 5,
        text: "goal detail",
        canonicalizes: { goal_ids: [goalId] },
      }),
      entry({
        audience,
        kind: "locked",
        rank: 105,
        updatedAt: 6,
        text: "critical commitment detail",
        canonicalizes: { commitment_ids: [commitmentId] },
      }),
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 10_000,
        text: "newer non salient detail",
      }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 6,
        maxTokens: 50_000,
        reservedSlots: {
          live: 0,
          pending: 0,
          invalidated: 0,
        },
        lockedMaxEntries: 6,
        newestStateChangeReservedSlots: 0,
        currentUserStreamEntryId,
        ledgerStreamEntryIds: [ledgerStreamEntryId],
        activeOpenQuestionIds: [openQuestionId],
        activeActionIds: [actionId],
        activeGoalIds: [goalId],
        activeCriticalCommitmentIds: [commitmentId],
      }) ?? "";

    expect(rendered).toContain("text: current detail");
    expect(rendered).toContain("text: ledger detail");
    expect(rendered).toContain("text: open question detail");
    expect(rendered).toContain("text: action detail");
    expect(rendered).toContain("text: goal detail");
    expect(rendered).toContain("text: critical commitment detail");
    expect(rendered).not.toContain("text: newer non salient detail");
  });

  it("drops lower-priority critical locked entries before newest reserved entries under token pressure", () => {
    const audience = createEntityId();
    const commitmentId = createCommitmentId();
    const pressureText = "token pressure ".repeat(260);
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 10_000,
        stateKey: "newest.live",
        text: `newest reserved detail ${pressureText}`,
      }),
      entry({
        audience,
        kind: "locked",
        rank: 1,
        updatedAt: 9_000,
        stateKey: "critical.locked",
        text: `critical commitment detail ${pressureText}`,
        canonicalizes: { commitment_ids: [commitmentId] },
      }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 2,
        maxTokens: 1_000,
        reservedSlots: {
          live: 0,
          pending: 0,
          invalidated: 0,
        },
        lockedMaxEntries: 2,
        newestStateChangeReservedSlots: 1,
        activeCriticalCommitmentIds: [commitmentId],
      }) ?? "";

    expect(rendered).toContain("state_key_bucket=newest.live");
    expect(rendered).toContain("text: newest reserved detail");
    expect(rendered).not.toContain("state_key_bucket=critical.locked");
    expect(rendered).not.toContain("text: critical commitment detail");
  });
});
