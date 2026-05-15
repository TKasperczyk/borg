import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { BorgDependencies } from "../src/borg/types.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import { createEvalBorg } from "../eval/support/create-eval-borg.js";
import { ManualClock } from "../src/util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createDecisionArtifactEntryId,
  createEpisodeId,
  type SessionId,
} from "../src/util/ids.js";
import { BorgTransport, type AuditTranscriptEntry } from "../assessor/borg-transport.js";
import { estimatePromptTokens } from "../src/util/token-estimate.js";

import {
  buildMemorySnapshotMarkdown,
  MEMORY_SNAPSHOT_TARGET_TOKEN_BUDGET,
} from "./memory-snapshot.js";
import { MetricsCapture } from "./metrics.js";

type BorgInternal = {
  deps: BorgDependencies;
};

function oversizedText(index: number): string {
  return `oversized fixture row ${index} ${"grounded detail ".repeat(24)}`;
}

function fixedId(prefix: string, index: number): string {
  return `${prefix}_${String(index).padStart(16, "0")}`;
}

function fakeOversizedTransport(count: number): BorgTransport {
  const rows = Array.from({ length: count }, (_, index) => index + 1);
  const transcript = rows.map((index): AuditTranscriptEntry => {
    return {
      entry: {
        id: fixedId("strm", index) as never,
        timestamp: index,
        kind: index % 2 === 0 ? "agent_msg" : "user_msg",
        content: oversizedText(index),
        session_id: DEFAULT_SESSION_ID,
        compressed: false,
        sender_entity_id: null,
        reply_target_entity_id: null,
      },
      quarantined: false,
      quarantineReason: null,
    };
  });
  const records = rows.map((index) => ({
    id: fixedId("ep", index),
    start_time: index,
    end_time: index,
    updated_at: index,
    created_at: index,
    confidence: 0.9,
    significance: 0.9,
    source_stream_ids: [fixedId("strm", index)],
    source_episode_ids: [fixedId("ep", index)],
    evidence_episode_ids: [fixedId("ep", index)],
    evidence_stream_entry_ids: [fixedId("strm", index)],
    provenance_episode_ids: [fixedId("ep", index)],
    provenance_stream_entry_ids: [fixedId("strm", index)],
    resolved_episode_ids: [fixedId("ep", index)],
    key_episode_ids: [fixedId("ep", index)],
    related_episode_ids: [fixedId("ep", index)],
    related_semantic_node_ids: [fixedId("semn", index)],
    label: oversizedText(index),
    title: oversizedText(index),
    narrative: oversizedText(index),
    description: oversizedText(index),
    directive: oversizedText(index),
    question: oversizedText(index),
    what_changed: oversizedText(index),
    evidence_text: oversizedText(index),
    applies_when: oversizedText(index),
    approach: oversizedText(index),
    progress_notes: oversizedText(index),
    trigger_reason: oversizedText(index),
    reason: oversizedText(index),
    summary: oversizedText(index),
    kind: "fixture",
    relation: "related_to",
    from_node_id: fixedId("semn", index),
    to_node_id: fixedId("semn", index + 1),
    valid_from: index,
    valid_to: null,
    invalidated_at: null,
    invalidated_reason: null,
    state: "active",
    status: "active",
    priority: 1,
    urgency: 1,
    strength: 1,
    source: "fixture",
    provenance: { process: "fixture" },
    record_type: "value",
    record_id: fixedId("val", index),
    action: "upserted",
    review_item_id: null,
    type: "promise",
    directive_family: "fixture",
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    revoked_at: null,
    superseded_by: null,
    actor: "borg",
    completed_at: null,
    canonical_name: `Entity ${index}`,
    aliases: [`Alias ${index}`],
    name_provenance: "fixture",
    subject_entity_id: fixedId("ent", index),
    slot_key: "fixture.slot",
    value: oversizedText(index),
    entity_id: fixedId("ent", index),
    trust: 0.5,
    attachment: 0.5,
    interaction_count: index,
    commitment_count: index,
    sentiment_summary: oversizedText(index),
    ts: index,
    trust_delta: 0,
    attachment_delta: 0,
    valence: 0,
    arousal: 0,
    session_id: DEFAULT_SESSION_ID,
    recent_triggers: ["fixture"],
    attempts: index,
    successes: index,
    failures: 0,
    skill_id: fixedId("skl", index),
    context_key: "fixture",
    classification: "success",
    grounded: true,
    consumed_at: null,
    audience_entity_id: null,
    refs: [fixedId("ep", index)],
    applied_at: index,
    reverted_at: null,
  }));
  const borg = {
    episodic: { list: async () => ({ items: records }) },
    semantic: {
      nodes: { list: async () => records },
      edges: { list: async () => records },
    },
    self: {
      values: { list: () => records },
      goals: { list: () => records.map((record) => ({ ...record, children: [] })) },
      traits: { list: () => records },
      autobiographical: { listPeriods: () => records, currentPeriod: () => null },
      growthMarkers: { list: () => records },
      openQuestions: { list: () => records },
    },
    identity: { listEvents: () => records },
    commitments: { list: () => records },
    actions: { list: () => records },
    skills: { list: () => records },
    workmem: {
      load: () => ({
        session_id: DEFAULT_SESSION_ID,
        turn_counter: count,
        updated_at: count,
        mode: "fixture",
        hot_entities: [],
        pending_actions: [],
        suppressed: [],
        pending_procedural_attempts: [],
        discourse_state: null,
      }),
    },
    review: { list: () => records },
    audit: { list: () => records },
  } as Record<string, unknown>;

  return {
    getBorg: () => ({
      ...borg,
      deps: {
        entityRepository: { list: () => records },
        relationalSlotRepository: {
          list: () => records,
          countByState: () => ({ established: count, contested: 0, quarantined: 0, revoked: 0 }),
        },
        socialRepository: {
          list: () => records,
          listEvents: () => records,
        },
        moodRepository: {
          listStates: () => records,
          history: () => records,
        },
        skillRepository: {
          batchListContextStatsForSkills: () => new Map([["fixture", records]]),
        },
        proceduralEvidenceRepository: {
          list: () => records,
        },
      },
    }),
    async readAuditTranscript() {
      return transcript;
    },
  } as unknown as BorgTransport;
}

describe("simulator memory snapshot", () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    while (tempDirs.length > 0) {
      const dir = tempDirs.pop();

      if (dir !== undefined) {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  });

  it("renders all snapshot sections from a small Borg fixture", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-sim-snapshot-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const borg = await createEvalBorg({
      tempDir,
      llm: new FakeLLMClient({ responses: [] }),
      clock,
    });
    const internal = borg as unknown as BorgInternal;

    try {
      const userEntry = await borg.stream.append({
        kind: "user_msg",
        content: "Maya is my partner.",
      });
      const agentEntry = await borg.stream.append({
        kind: "agent_msg",
        content: "I will remember Maya as your partner.",
      });
      const episode = await internal.deps.episodicRepository.insert({
        id: createEpisodeId(),
        title: "Maya relationship note",
        narrative: "The user said Maya is their partner.",
        participants: ["user", "Maya"],
        location: null,
        start_time: clock.now(),
        end_time: clock.now(),
        source_stream_ids: [userEntry.id, agentEntry.id],
        significance: 0.8,
        tags: ["relationship"],
        confidence: 0.95,
        lineage: { derived_from: [], supersedes: [] },
        emotional_arc: null,
        embedding: Float32Array.from({ length: 64 }, () => 0.1),
        created_at: clock.now(),
        updated_at: clock.now(),
      });
      const mayaNode = await borg.semantic.nodes.add({
        kind: "entity",
        label: "Maya",
        description: "The user's partner.",
        sourceEpisodeIds: [episode.id],
      });
      const partnerNode = await borg.semantic.nodes.add({
        kind: "concept",
        label: "partner",
        description: "A close relational role.",
        sourceEpisodeIds: [episode.id],
      });
      borg.semantic.edges.add({
        from_node_id: mayaNode.id,
        to_node_id: partnerNode.id,
        relation: "related_to",
        confidence: 0.9,
        evidence_episode_ids: [episode.id],
        created_at: clock.now(),
        last_verified_at: clock.now(),
      });

      borg.self.values.add({
        label: "grounded recall",
        description: "Preserve user-stated details accurately.",
        priority: 10,
        provenance: { kind: "episodes", episode_ids: [episode.id] },
      });
      const goal = borg.self.goals.add({
        description: "Keep Maya grounded in memory.",
        priority: 8,
        provenance: { kind: "manual" },
      });
      borg.self.traits.add({
        label: "careful",
        delta: 0.4,
        timestamp: clock.now(),
        provenance: { kind: "manual" },
      });
      borg.self.autobiographical.upsertPeriod({
        label: "Maya context",
        start_ts: clock.now(),
        narrative: "A period focused on preserving relationship details.",
        key_episode_ids: [episode.id],
        themes: ["memory"],
        provenance: { kind: "episodes", episode_ids: [episode.id] },
      });
      borg.self.growthMarkers.add({
        ts: clock.now(),
        category: "understanding",
        what_changed: "Borg learned the Maya relationship anchor.",
        evidence_episode_ids: [episode.id],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [episode.id] },
      });
      borg.self.openQuestions.add({
        question: "What details about Maya matter most?",
        urgency: 0.5,
        related_episode_ids: [episode.id],
        related_semantic_node_ids: [mayaNode.id],
        source: "user",
        provenance: { kind: "manual" },
      });

      const mayaEntity = internal.deps.entityRepository.resolve("Maya", {
        kind: "person",
        provenance: "user_declared",
      });
      borg.commitments.add({
        type: "promise",
        directiveFamily: "maya recall",
        directive: "Keep Maya's relationship role grounded.",
        priority: 7,
        about: "Maya",
        provenance: { kind: "manual" },
      });
      borg.actions.add({
        id: createActionId(),
        description: "Remember Maya as the user's partner",
        actor: "borg",
        audience_entity_id: null,
        goal_id: goal.id,
        open_question_id: null,
        state: "completed",
        confidence: 0.9,
        provenance_episode_ids: [episode.id],
        provenance_stream_entry_ids: [userEntry.id],
        created_at: clock.now(),
        updated_at: clock.now(),
        considering_at: null,
        committed_at: null,
        scheduled_at: null,
        completed_at: clock.now(),
        not_done_at: null,
        unknown_at: null,
      });
      internal.deps.relationalSlotRepository.applyAssertion({
        subject_entity_id: mayaEntity,
        slot_key: "relationship.role",
        asserted_value: "partner",
        source_stream_entry_ids: [userEntry.id],
        confirmation: "direct",
      });
      internal.deps.socialRepository.recordInteraction(mayaEntity, {
        provenance: { kind: "manual" },
        valence: 0.2,
      });
      borg.mood.update(DEFAULT_SESSION_ID, {
        valence: 0.2,
        arousal: 0.3,
        reason: "fixture",
        provenance: { kind: "manual" },
      });
      const skill = await borg.skills.add({
        applies_when: "a user asks about a named partner",
        approach: "Check grounded memory before answering.",
        sourceEpisodes: [episode.id],
      });
      internal.deps.proceduralEvidenceRepository.insert({
        pendingAttemptSnapshot: {
          problem_text: "ground a partner claim",
          approach_summary: "check memory snapshot",
          selected_skill_id: skill.id,
          source_stream_ids: [userEntry.id],
          turn_counter: 1,
          audience_entity_id: null,
        },
        classification: "success",
        evidenceText: "The answer cited the Maya episode.",
        resolvedEpisodeIds: [episode.id],
      });

      const transcript: AuditTranscriptEntry[] = [userEntry, agentEntry].map((entry) => ({
        entry,
        quarantined: false,
        quarantineReason: null,
      }));
      const transport = {
        getBorg: () => borg,
        async readAuditTranscript() {
          return transcript;
        },
      } as unknown as BorgTransport;
      const snapshot = await buildMemorySnapshotMarkdown({
        transport,
        sessionIds: [DEFAULT_SESSION_ID as SessionId],
      });

      for (const heading of [
        "## Memory Snapshot",
        "### Scope And Counts",
        "### Stream Transcript",
        "### Episodic Memory",
        "### Semantic Nodes",
        "### Semantic Edges",
        "### Identity And Self",
        "### Goals And Open Questions",
        "### Commitments",
        "### Actions",
        "### Relational And Social",
        "### Affective State",
        "### Procedural Memory",
        "### Working Memory",
        "### Review And Audit Diagnostics",
      ]) {
        expect(snapshot).toContain(heading);
      }

      expect(snapshot).toContain("Maya is my partner.");
      expect(snapshot).toContain("Maya relationship note");
      expect(snapshot).toContain("Keep Maya grounded in memory.");
      expect(snapshot).toContain("Remember Maya as the user's partner");
      expect(snapshot).toContain("relationship.role");
      expect(snapshot).toContain("a user asks about a named partner");
    } finally {
      await borg.close();
    }
  });

  it("renders audience-scoped commitments in audit snapshots", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-sim-snapshot-"));
    tempDirs.push(tempDir);
    const borg = await createEvalBorg({
      tempDir,
      llm: new FakeLLMClient({ responses: [] }),
      clock: new ManualClock(1_000),
    });

    try {
      borg.commitments.add({
        type: "boundary",
        directiveFamily: "trip_group_private_details",
        directive: "Keep Alice and Ben trip details inside the trip group.",
        priority: 10,
        audience: "Trip Group",
        provenance: { kind: "manual" },
      });
      borg.commitments.add({
        type: "promise",
        directiveFamily: "trip_group_booking_followup",
        directive: "Follow up with the trip group about booking constraints.",
        priority: 8,
        madeTo: "Trip Group",
        provenance: { kind: "manual" },
      });

      const transport = {
        getBorg: () => borg,
        async readAuditTranscript() {
          return [];
        },
      } as unknown as BorgTransport;
      const snapshot = await buildMemorySnapshotMarkdown({
        transport,
        sessionIds: [DEFAULT_SESSION_ID as SessionId],
      });

      expect(snapshot).toContain("### Commitments");
      expect(snapshot).toContain("Keep Alice and Ben trip details inside the trip group.");
      expect(snapshot).toContain("Follow up with the trip group about booking constraints.");
      expect(snapshot).not.toContain("No commitments recorded.");
    } finally {
      await borg.close();
    }
  });

  it("renders commitment lifecycle breakdowns and active details", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-sim-snapshot-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const borg = await createEvalBorg({
      tempDir,
      llm: new FakeLLMClient({ responses: [] }),
      clock,
    });
    const internal = borg as unknown as BorgInternal;
    const commitments = internal.deps.commitmentRepository;

    try {
      const active = commitments.add({
        type: "promise",
        directiveFamily: "snapshot active",
        directive: "Keep the active snapshot fixture visible.",
        priority: 7,
        provenance: { kind: "manual" },
      });
      const revoked = commitments.add({
        type: "promise",
        directiveFamily: "snapshot revoked",
        directive: "Retire the revoked snapshot fixture.",
        priority: 6,
        provenance: { kind: "manual" },
      });
      const expired = commitments.add({
        type: "promise",
        directiveFamily: "snapshot expired",
        directive: "Expire the snapshot fixture.",
        priority: 5,
        provenance: { kind: "manual" },
        createdAt: 1_700_000_000_000,
        expiresAt: 1_750_000_000_000,
      });
      const canonicalized = commitments.add({
        type: "promise",
        directiveFamily: "snapshot canonicalized",
        directive: "Canonicalize the snapshot fixture.",
        priority: 5,
        provenance: { kind: "manual" },
      });
      const superseded = commitments.add({
        type: "promise",
        directiveFamily: "snapshot superseded",
        directive: "Replace the superseded snapshot fixture.",
        priority: 4,
        provenance: { kind: "manual" },
      });
      const replacement = commitments.add({
        type: "promise",
        directiveFamily: "snapshot replacement",
        directive: "Keep the replacement snapshot fixture visible.",
        priority: 8,
        provenance: { kind: "manual" },
      });
      commitments.revoke(revoked.id, "snapshot revoked", { kind: "manual" });
      commitments.revoke(
        canonicalized.id,
        "snapshot canonicalized",
        { kind: "manual" },
        undefined,
        {
          canonicalizedByArtifactEntryId: createDecisionArtifactEntryId(),
        },
      );
      commitments.supersede(superseded.id, replacement.id);

      const transport = {
        getBorg: () => borg,
        async readAuditTranscript() {
          return [];
        },
      } as unknown as BorgTransport;
      const snapshot = await buildMemorySnapshotMarkdown({
        transport,
        sessionIds: [DEFAULT_SESSION_ID as SessionId],
      });
      const metrics = await new MetricsCapture(join(tempDir, "metrics.jsonl")).capture(
        borg,
        "turn-snapshot-commitment-lifecycle",
        1,
        {
          sessionId: DEFAULT_SESSION_ID as SessionId,
          sessionIds: [DEFAULT_SESSION_ID as SessionId],
          transportChatAttempts: 1,
        },
      );

      expect(snapshot).toContain("### Commitments");
      expect(snapshot).toContain("- total_commitments=6");
      expect(snapshot).toContain(
        `- lifecycle_counts active=${metrics.commitment_count_active} revoked=${metrics.commitment_count_revoked} expired=${metrics.commitment_count_expired} canonicalized=${metrics.commitment_count_canonicalized} superseded=${metrics.commitment_count_superseded}`,
      );
      expect(metrics.commitment_count_expired).toBe(0);
      expect(snapshot).toContain("expired=0");
      expect(snapshot).toContain(active.id);
      expect(snapshot).toContain(replacement.id);
      expect(snapshot).toContain("Keep the active snapshot fixture visible.");
      expect(snapshot).toContain("Keep the replacement snapshot fixture visible.");
      expect(snapshot).not.toContain('directive="Retire the revoked snapshot fixture.');
      expect(snapshot).not.toContain('directive="Replace the superseded snapshot fixture.');
      expect(snapshot).toContain("Expire the snapshot fixture.");
    } finally {
      await borg.close();
    }
  });

  it("caps oversized snapshots and reports omitted rows explicitly", async () => {
    const snapshot = await buildMemorySnapshotMarkdown({
      transport: fakeOversizedTransport(500),
      sessionIds: [DEFAULT_SESSION_ID as SessionId],
    });

    expect(estimatePromptTokens(snapshot)).toBeLessThanOrEqual(MEMORY_SNAPSHOT_TARGET_TOKEN_BUDGET);
    expect(snapshot).toContain("omitted");
    expect(snapshot).toContain("per-section cap=");
    expect(snapshot).toContain("global token budget");
  });
});
