import { describe, expect, it } from "vitest";

import type {
  SharedStateArtifact,
  SharedStateEntry,
  SharedStateEntryKind,
} from "../../memory/shared-state/index.js";
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
  createdAt?: number;
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
    created_at: input.createdAt ?? input.updatedAt,
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
        entry({ audience, kind: "tentative", rank: 200 + index, updatedAt: 2_000 + index }),
      ),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 25,
      maxTokens: 50_000,
      reservedSlots: {
        live: 8,
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
  it("names the instant record_version was read, so a flat version is not read as a still store", () => {
    const audience = createEntityId();
    const entries = [
      entry({
        audience,
        kind: "live",
        rank: 0,
        updatedAt: 20,
        stateKey: "project.alpha",
        text: "alpha detail",
      }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 4,
        maxTokens: 50_000,
      }) ?? "";

    expect(rendered).toContain("record_version=1\nsnapshot_basis=turn_start");
    expect(rendered).toContain("read before this turn's shared-state compile");
  });

  it("names the snapshot basis on the omission-only render too", () => {
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
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 4,
        maxTokens: 40,
      }) ?? "";

    expect(rendered).toContain("SharedStateArtifact omitted:");
    expect(rendered).toContain("record_version=1\nsnapshot_basis=turn_start");
  });

  it("names omission as a render budget rather than a store cap", () => {
    const audience = createEntityId();
    const entries = Array.from({ length: 6 }, (_, index) =>
      entry({
        audience,
        kind: "live",
        rank: index,
        updatedAt: 20 + index,
        stateKey: `project.key${index}`,
        text: `body ${index}`,
      }),
    );

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 2,
        maxTokens: 50_000,
      }) ?? "";

    expect(rendered).toContain("SharedStateArtifact omitted:");
    expect(rendered).toContain("omission_basis=render_budget");
    expect(rendered).toContain("still active and unchanged in the store");
    expect(rendered).toContain("not the store's lifecycle cap");
  });

  it("names the omission basis on the omission-only render too", () => {
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
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 4,
        maxTokens: 40,
      }) ?? "";

    expect(rendered).toContain("SharedStateArtifact omitted:");
    expect(rendered).toContain("omission_basis=render_budget");
  });

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

  // The index is the only surface most entries ever get, so it invites being read as
  // an ordering over them -- in particular as the lifecycle-cap prune queue, whose
  // tiebreak within a shared `last_updated_at` is `rank ASC`. It is not that. Keys
  // here sort alphabetically in exactly the reverse of their rank, so an index that
  // leaked rank would fail; `rank` is not even a field on an index line.
  it("orders the compact index by state key, not by rank or prune position", () => {
    const audience = createEntityId();
    const entries = [
      entry({ audience, kind: "locked", rank: 40, updatedAt: 500, stateKey: "audit.zeta" }),
      entry({ audience, kind: "locked", rank: 41, updatedAt: 500, stateKey: "audit.mid" }),
      entry({ audience, kind: "locked", rank: 42, updatedAt: 500, stateKey: "audit.alpha" }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 1,
        maxTokens: 50_000,
        newestStateChangeReservedSlots: 1,
      }) ?? "";

    const indexedKeys = rendered
      .split("\n")
      .map((line) => /^- (?<key>[^ |]+) \| kinds=/u.exec(line)?.groups?.key)
      .filter((key): key is string => key !== undefined);

    expect(indexedKeys).toEqual(["audit.alpha", "audit.mid", "audit.zeta"]);
    expect(rendered).not.toContain("rank=");
  });

  // The disclosure fields are the largest term on an index line, and in a register whose rows all
  // came from one audience they are the same bytes on every line. They are hoisted to one line
  // above the index, and a row printing nothing is a row carrying the hoisted label -- so the
  // contract under test is that every row's label is still readable, not that it is still repeated.
  it("hoists the repeated index disclosure label and drops it from the rows that share it", () => {
    const audience = createEntityId();
    const entries = Array.from({ length: 6 }, (_, index) =>
      entry({
        audience,
        kind: "locked",
        rank: index,
        updatedAt: 500 + index,
        stateKey: `audit.key${index}`,
      }),
    );

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 1,
        maxTokens: 50_000,
        newestStateChangeReservedSlots: 1,
      }) ?? "";
    const indexLines = rendered
      .split("\n")
      .filter((line) => /^- audit\.key\d+ \| kinds=/u.test(line));

    expect(rendered).toContain(
      `  (disclosure label of every index row below that does not print its own: disclosure_class=relationship_private origin_audience=${audience} private-to=${audience})`,
    );
    expect(indexLines).toHaveLength(6);
    expect(indexLines.filter((line) => line.includes("disclosure_class="))).toEqual([]);
    // The expanded body keeps its own full label; only the index rows defer to the hoisted line.
    expect(rendered).toContain(`private-to=${audience}`);
  });

  // A register is not guaranteed to be label-uniform: an entry with an owner carries a wider
  // origin_audience than one without. The most repeated label is hoisted and the rows that differ
  // print their own, so a reader can always tell which rows the hoisted line does not cover.
  it("keeps per-row index disclosure fields on the rows whose label differs from the hoisted one", () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const entries = [
      ...Array.from({ length: 6 }, (_, index) =>
        entry({
          audience,
          kind: "locked",
          rank: index,
          updatedAt: 500 + index,
          stateKey: `audit.shared${index}`,
        }),
      ),
      ...Array.from({ length: 2 }, (_, index) => ({
        ...entry({
          audience,
          kind: "locked",
          rank: 10 + index,
          updatedAt: 600 + index,
          stateKey: `audit.owned${index}`,
        }),
        owner_entity_id: owner,
      })),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 1,
        maxTokens: 50_000,
        newestStateChangeReservedSlots: 1,
      }) ?? "";
    const indexLines = rendered.split("\n").filter((line) => /^- audit\.\w+ \| kinds=/u.test(line));
    const linesWithFields = indexLines.filter((line) => line.includes("disclosure_class="));

    expect(rendered).toContain(
      `  (disclosure label of every index row below that does not print its own: disclosure_class=relationship_private origin_audience=${audience} private-to=${audience})`,
    );
    expect(indexLines).toHaveLength(8);
    expect(linesWithFields).toHaveLength(2);
    for (const line of linesWithFields) {
      expect(line).toMatch(/^- audit\.owned\d+ \|/u);
      expect(line).toContain("disclosure_class=relationship_private");
      expect(line).toContain(owner);
      expect(line).toContain(audience);
    }
  });

  // Hoisting is gated on the arithmetic rather than on a tuned row count: stating the label once
  // costs a line, so a label carried by too few rows to pay for that line stays where it is.
  it("leaves a single-row index disclosure label in place rather than hoisting it", () => {
    const audience = createEntityId();
    const entries = [
      entry({ audience, kind: "locked", rank: 0, updatedAt: 500, stateKey: "audit.only" }),
    ];

    const rendered =
      renderSharedStateArtifact(artifact(entries), {
        maxEntries: 1,
        maxTokens: 50_000,
        newestStateChangeReservedSlots: 1,
      }) ?? "";
    const indexLine = rendered
      .split("\n")
      .find((line) => line.startsWith("- audit.only | kinds="));

    expect(rendered).not.toContain("disclosure label of every index row below");
    expect(indexLine).toContain(
      `disclosure_class=relationship_private origin_audience=${audience} private-to=${audience}`,
    );
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
          invalidated: 0,
        },
        newestStateChangeReservedSlots: 0,
      }) ?? "";
    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 1,
      maxTokens: 50_000,
      reservedSlots: {
        live: 1,
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

  it("renders created_at only on entries whose body was rewritten after it was first written", () => {
    const audience = createEntityId();
    const rewritten = entry({
      audience,
      kind: "locked",
      rank: 0,
      createdAt: 1_000,
      updatedAt: 9_000,
      stateKey: "audit.rewritten",
      text: "body carried forward by a later update",
    });
    const original = entry({
      audience,
      kind: "locked",
      rank: 1,
      updatedAt: 9_000,
      stateKey: "audit.original",
      text: "body still as first written",
    });

    const rendered =
      renderSharedStateArtifact(artifact([rewritten, original]), {
        maxEntries: 10,
        maxTokens: 50_000,
        lockedMaxEntries: 10,
      }) ?? "";

    const lineFor = (id: string): string =>
      rendered.split("\n").find((line) => line.includes(`id=${id}`)) ?? "";

    // The rewritten entry's stamp dates its newest sentence, not the whole body:
    // both stamps render so the span is visible.
    expect(lineFor(rewritten.id)).toContain("created_at=1000 last_updated_at=9000");
    // The untouched entry's stamp is the write instant of every sentence in it;
    // a redundant created_at would only cost prompt budget.
    expect(lineFor(original.id)).toContain("last_updated_at=9000");
    expect(lineFor(original.id)).not.toContain("created_at=");
  });

  it("carries created_at onto the compact index line so its absence reads the same there", () => {
    const audience = createEntityId();
    const rewritten = entry({
      audience,
      kind: "locked",
      rank: 0,
      createdAt: 1_000,
      updatedAt: 9_000,
      stateKey: "audit.rewritten",
      text: "body carried forward by a later update",
    });
    const original = entry({
      audience,
      kind: "locked",
      rank: 1,
      updatedAt: 9_000,
      stateKey: "audit.original",
      text: "body still as first written",
    });

    // Nothing expanded: the index line is the only thing said about either entry, which is
    // the position every omitted row is in.
    const rendered =
      renderSharedStateArtifact(artifact([rewritten, original]), {
        maxEntries: 0,
        maxTokens: 50_000,
      }) ?? "";

    const indexLineFor = (stateKey: string): string =>
      rendered.split("\n").find((line) => line.startsWith(`- ${stateKey} |`)) ?? "";

    expect(indexLineFor("audit.rewritten")).toContain("created_at=1000 | last_updated_at=9000");
    expect(indexLineFor("audit.original")).toContain("last_updated_at=9000");
    expect(indexLineFor("audit.original")).not.toContain("created_at=");
  });

  it("names what a successor retracted, so a supersede does not read like a prune", () => {
    const audience = createEntityId();
    const successor = entry({
      audience,
      kind: "locked",
      rank: 0,
      updatedAt: 9_000,
      stateKey: "audit.replaced",
      text: "corrected body",
    });
    const retracted: SharedStateEntry = {
      ...entry({
        audience,
        kind: "locked",
        rank: 1,
        updatedAt: 9_000,
        stateKey: "audit.replaced",
        text: "the wording that was withdrawn",
      }),
      superseded_by_id: successor.id,
    };
    const untouched = entry({
      audience,
      kind: "locked",
      rank: 2,
      updatedAt: 9_000,
      stateKey: "audit.untouched",
      text: "body that replaced nothing",
    });

    const rendered =
      renderSharedStateArtifact(artifact([successor, retracted, untouched]), {
        maxEntries: 10,
        maxTokens: 50_000,
        lockedMaxEntries: 10,
      }) ?? "";

    const lineFor = (id: string): string =>
      rendered.split("\n").find((line) => line.includes(`id=${id}`)) ?? "";

    // The retracted body stays off this surface -- that is what retraction means -- but the
    // fact that it was retracted does not.
    expect(rendered).not.toContain("text: the wording that was withdrawn");
    expect(lineFor(successor.id)).toContain(`supersedes=${retracted.id}`);
    expect(lineFor(untouched.id)).not.toContain("supersedes=");
  });

  it("carries the retraction onto the compact index line, where omitted rows live", () => {
    const audience = createEntityId();
    const successor = entry({
      audience,
      kind: "locked",
      rank: 0,
      updatedAt: 9_000,
      stateKey: "audit.replaced",
      text: "corrected body",
    });
    const retracted: SharedStateEntry = {
      ...entry({
        audience,
        kind: "locked",
        rank: 1,
        updatedAt: 9_000,
        stateKey: "audit.replaced",
        text: "the wording that was withdrawn",
      }),
      superseded_by_id: successor.id,
    };
    const untouched = entry({
      audience,
      kind: "locked",
      rank: 2,
      updatedAt: 9_000,
      stateKey: "audit.untouched",
      text: "body that replaced nothing",
    });

    // Nothing expanded: the index line is the only thing said about either key.
    const rendered =
      renderSharedStateArtifact(artifact([successor, retracted, untouched]), {
        maxEntries: 0,
        maxTokens: 50_000,
      }) ?? "";

    const indexLineFor = (stateKey: string): string =>
      rendered.split("\n").find((line) => line.startsWith(`- ${stateKey} |`)) ?? "";

    expect(indexLineFor("audit.replaced")).toContain("superseded_count=1");
    expect(indexLineFor("audit.untouched")).not.toContain("superseded_count=");
  });
});
