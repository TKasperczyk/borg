import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { commitmentMigrations } from "../commitments/migrations.js";
import { EntityRepository, type CommitmentRecord } from "../commitments/index.js";
import type { GoalRecord, GoalTreeNode } from "../self/index.js";
import { sessionMigrations, SessionsRepository } from "../../sessions/index.js";
import {
  composeMigrations,
  openDatabase,
  type SqliteDatabase,
} from "../../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  createCommitmentId,
  createEntityId,
  createGoalId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromMetadata,
  relationshipPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "./disclosure-label.js";
import {
  commitmentMemoryDisclosureLabel,
  goalMemoryDisclosureLabel,
} from "./disclosure-serializers.js";
import { SourceStreamAudienceDisclosureResolver } from "./source-stream-audience-disclosure.js";

function makeCommitment(input: {
  scope: EntityId;
  sourceStreamEntryIds?: readonly StreamEntryId[];
}): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "boundary",
    kind: "boundary",
    enforcement_class: "critical",
    critical_domain: "privacy",
    directive_family: "source_origin_test",
    closure_pressure_relevance: "neutral",
    directive: "Keep the source audience distinct from scope.",
    priority: 10,
    made_to_entity: null,
    restricted_audience: input.scope,
    about_entity: null,
    committed_by_entity_id: null,
    provenance: { kind: "manual" },
    ...(input.sourceStreamEntryIds === undefined
      ? {}
      : { source_stream_entry_ids: [...input.sourceStreamEntryIds] }),
    created_at: 1_000,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    last_reinforced_at: 1_000,
  };
}

function makeGoal(input: {
  scope: EntityId;
  sourceStreamEntryIds: readonly StreamEntryId[];
}): GoalRecord {
  return {
    id: createGoalId(),
    description: "Remember where this goal originated.",
    terminal_condition: null,
    priority: 5,
    parent_goal_id: null,
    status: "active",
    progress_notes: null,
    last_progress_ts: null,
    created_at: 1_000,
    target_at: null,
    audience_entity_id: input.scope,
    owner_entity_id: null,
    source_stream_entry_ids: [...input.sourceStreamEntryIds],
    provenance: { kind: "manual" },
  };
}

describe("SourceStreamAudienceDisclosureResolver", () => {
  let db: SqliteDatabase | undefined;
  const tempDirs: string[] = [];

  afterEach(() => {
    db?.close();
    db = undefined;
    for (const tempDir of tempDirs.splice(0)) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("resolves heterogeneous rows in one batch and preserves chronological audience order", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-source-audience-"));
    tempDirs.push(dataDir);
    db = openDatabase(":memory:", {
      migrations: composeMigrations(
        commitmentMigrations,
        sessionMigrations,
        streamEntryIndexMigrations,
      ),
    });
    const clock = new ManualClock(200);
    const entityRepository = new EntityRepository({ db, clock });
    const sessionsRepository = new SessionsRepository({ db, clock });
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
    const scope = entityRepository.add({
      id: createEntityId(),
      canonicalName: "Continuous room",
    }).id;
    const earlyAudience = entityRepository.add({
      id: createEntityId(),
      canonicalName: "Original operator thread",
    }).id;
    const lateAudience = entityRepository.add({
      id: createEntityId(),
      canonicalName: "Original arena thread",
      aliases: ["Arena alias"],
    }).id;
    const sessionFallbackAudience = entityRepository.add({
      id: createEntityId(),
      canonicalName: "Legacy session room",
    }).id;
    const earlySessionId = createSessionId();
    const lateSessionId = createSessionId();
    const fallbackSessionId = createSessionId();

    for (const [sessionId, audienceLabel, audienceEntityId] of [
      [earlySessionId, "wrong-current-early", scope],
      [lateSessionId, "wrong-current-late", scope],
      [fallbackSessionId, "Legacy session room", sessionFallbackAudience],
    ] as const) {
      sessionsRepository.ensure({
        session_id: sessionId,
        source_type: "demo",
        label: audienceLabel,
        audience_label: audienceLabel,
        audience_entity_id: audienceEntityId,
        conversation_kind: "demo",
      });
    }

    const earlyWriter = new StreamWriter({
      dataDir,
      sessionId: earlySessionId,
      clock,
      entryIndex,
    });
    const lateWriter = new StreamWriter({
      dataDir,
      sessionId: lateSessionId,
      clock,
      entryIndex,
    });
    const fallbackWriter = new StreamWriter({
      dataDir,
      sessionId: fallbackSessionId,
      clock,
      entryIndex,
    });

    clock.set(200);
    const late = await lateWriter.append({
      kind: "user_msg",
      content: "late",
      audience: "Arena alias",
    });
    clock.set(100);
    const early = await earlyWriter.append({
      kind: "user_msg",
      content: "early",
      audience: "Original operator thread",
    });
    clock.set(300);
    const duplicateEarlyAudience = await lateWriter.append({
      kind: "user_msg",
      content: "duplicate audience",
      audience: "Original operator thread",
    });
    clock.set(150);
    const sessionFallback = await fallbackWriter.append({
      kind: "internal_event",
      content: "legacy entry without an audience label",
    });
    earlyWriter.close();
    lateWriter.close();
    fallbackWriter.close();

    db.prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?").run(early.id);

    const lookupMany = vi.spyOn(entryIndex, "lookupMany");
    const findByNames = vi.spyOn(entityRepository, "findByNames");
    const getMany = vi.spyOn(sessionsRepository, "getMany");
    const resolver = new SourceStreamAudienceDisclosureResolver({
      dataDir,
      entryIndex,
      sessionsRepository,
      entityRepository,
    });
    const commitment = makeCommitment({
      scope,
      sourceStreamEntryIds: [late.id, duplicateEarlyAudience.id, early.id],
    });
    const goal = makeGoal({
      scope,
      sourceStreamEntryIds: [sessionFallback.id, early.id],
    });
    const goalTree: GoalTreeNode = { ...goal, children: [] };
    const resolved = resolver.resolve({ commitments: [commitment], goalTrees: [goalTree] });

    expect(lookupMany).toHaveBeenCalledTimes(1);
    expect(lookupMany).toHaveBeenCalledWith([
      late.id,
      duplicateEarlyAudience.id,
      early.id,
      sessionFallback.id,
    ]);
    expect(findByNames).toHaveBeenCalledTimes(1);
    expect(findByNames).toHaveBeenCalledWith(["Arena alias", "Original operator thread"]);
    expect(getMany).toHaveBeenCalledTimes(1);
    expect(getMany).toHaveBeenCalledWith([fallbackSessionId]);

    const commitmentLabel = commitmentMemoryDisclosureLabel(resolved.commitments[0]!);
    expect(commitmentLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [earlyAudience, lateAudience],
      privateToEntityIds: [scope],
      publicToEntityIds: [],
    });
    expect(goalMemoryDisclosureLabel(resolved.goalTrees[0]!)).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [earlyAudience, sessionFallbackAudience],
      privateToEntityIds: [scope],
      publicToEntityIds: [],
    });

    const duplicateIdLabels = resolver.resolveLabels({
      commitments: [
        commitment,
        { ...commitment, source_stream_entry_ids: [late.id] },
      ],
    }).commitmentLabels;
    expect(duplicateIdLabels.map((label) => label.originAudienceEntityIds)).toEqual([
      [earlyAudience, lateAudience],
      [lateAudience],
    ]);
  });

  it("falls back for the complete row when any source cannot be resolved", () => {
    const scope = createEntityId();
    const missingSource = createStreamEntryId();
    const commitment = makeCommitment({ scope, sourceStreamEntryIds: [missingSource] });
    const lookupMany = vi.fn(() => new Map());
    const findByNames = vi.fn(() => new Map());
    const getMany = vi.fn(() => []);
    const resolver = new SourceStreamAudienceDisclosureResolver({
      dataDir: "/does/not/matter",
      entryIndex: { lookupMany },
      sessionsRepository: { getMany },
      entityRepository: { findByNames },
    });

    const [resolved] = resolver.resolve({ commitments: [commitment] }).commitments;

    expect(commitmentMemoryDisclosureLabel(resolved!)).toEqual(
      relationshipPrivateMemoryDisclosureLabel([scope]),
    );
    expect(lookupMany).toHaveBeenCalledTimes(1);
    expect(findByNames).not.toHaveBeenCalled();
    expect(getMany).not.toHaveBeenCalled();
  });

  it("keeps attached metadata authoritative and combines labels in canonical order", () => {
    const canonicalFirst = createEntityId();
    const canonicalSecond = createEntityId();
    const additional = createEntityId();
    const scope = createEntityId();
    const commitment = makeCommitment({ scope });
    const attached = {
      disclosure_class: "relationship_private" as const,
      origin_audience_entity_ids: [canonicalFirst, canonicalSecond],
      private_to_entity_ids: [scope],
      public_to_entity_ids: [],
    };

    expect(commitmentMemoryDisclosureLabel({ ...commitment, disclosure_label: attached })).toEqual(
      memoryDisclosureLabelFromMetadata(attached),
    );
    expect(
      combineMemoryDisclosureLabels([
        commitmentMemoryDisclosureLabel({ ...commitment, disclosure_label: attached }),
        relationshipPrivateMemoryDisclosureLabel([additional, canonicalFirst]),
      ]),
    ).toMatchObject({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [canonicalFirst, canonicalSecond, additional],
    });
    expect(
      combineMemoryDisclosureLabels([
        relationshipPrivateMemoryDisclosureLabel([canonicalFirst]),
        unknownMemoryDisclosureLabel(),
      ]).disclosureClass,
    ).toBe("unknown");
  });
});
