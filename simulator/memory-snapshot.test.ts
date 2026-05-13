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
  createEpisodeId,
  type SessionId,
} from "../src/util/ids.js";
import { BorgTransport, type AuditTranscriptEntry } from "../assessor/borg-transport.js";

import { buildMemorySnapshotMarkdown } from "./memory-snapshot.js";

type BorgInternal = {
  deps: BorgDependencies;
};

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
    const clock = new ManualClock(1_000);
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
});
