import { describe, expect, it, vi } from "vitest";

import { selfPrivateMemoryDisclosureLabel } from "../../memory/common/index.js";
import type { EpisodeSearchCandidate } from "../../memory/episodic/index.js";
import type { OpenQuestion } from "../../memory/self/index.js";
import {
  createEntityId,
  createEpisodeId,
  createOpenQuestionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
} from "../../util/ids.js";
import type { ToolInvocationContext } from "../dispatcher.js";
import { createEpisodicRecentTool } from "./episodic-recent.js";
import { createJournalAppendTool } from "./journal-append.js";
import { createOpenQuestionsResolveTool } from "./open-questions-resolve.js";

const context: ToolInvocationContext = {
  sessionId: DEFAULT_SESSION_ID,
  origin: "autonomous",
  turnOrigin: "autonomous",
  turnId: "turn-inner-life-tools",
};

function sampleOpenQuestion(overrides: Partial<OpenQuestion> = {}): OpenQuestion {
  return {
    id: createOpenQuestionId(),
    record_version: 1,
    question: "What should I settle?",
    urgency: 0.5,
    status: "open",
    goal_id: null,
    audience_entity_id: null,
    related_episode_ids: [],
    related_semantic_node_ids: [],
    provenance: { kind: "system" },
    source: "autonomy",
    created_at: 1_000,
    last_touched: 1_000,
    resolution_evidence_episode_ids: [],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: null,
    resolved_at: null,
    abandoned_reason: null,
    abandoned_at: null,
    resolved_by_artifact_entry_id: null,
    unresolved_rumination_ticks: 0,
    last_ruminated_at: null,
    ...overrides,
  };
}

describe("inner life internal tools", () => {
  it("appends a self-private journal entry with disclosure labels", async () => {
    const selfEntityId = createEntityId();
    const appendJournalEntry = vi.fn((input) => ({
      id: 1,
      self_entity_id: input.selfEntityId,
      text: input.text,
      disclosure_class: "self_private" as const,
      created_at: 1_000,
      updated_at: 1_000,
      source_turn_id: input.sourceTurnId ?? null,
      marker_stream_entry_id: null,
    }));
    const tool = createJournalAppendTool({
      resolveSelfEntityId: () => selfEntityId,
      appendJournalEntry,
    });

    const result = await tool.invoke({ text: "Keep this private note." }, context);

    expect(tool.allowedOrigins).toEqual(["autonomous"]);
    expect(tool.writeScope).toBe("write");
    expect(appendJournalEntry).toHaveBeenCalledWith({
      text: "Keep this private note.",
      selfEntityId,
      sourceTurnId: "turn-inner-life-tools",
    });
    expect(result.journalEntry).toMatchObject({
      text: "Keep this private note.",
      disclosure_label: {
        disclosure_class: "self_private",
      },
    });
  });

  // The store behind this tool is append-only: the repository has an INSERT and
  // no UPDATE or DELETE. Nothing on the surface said so, while three sibling
  // tools do mutate an existing row, so an intention to go back and mark an
  // earlier entry was readable as feasible. The surface has to carry it.
  it("names journal entries as immutable on both surfaces the model reads", () => {
    const tool = createJournalAppendTool({
      resolveSelfEntityId: () => createEntityId(),
      appendJournalEntry: () => {
        throw new Error("not invoked");
      },
    });

    expect(tool.description).toContain("immutable once written");
    expect(tool.description).toContain("no tool amends or deletes one");
    expect(tool.description).toContain("A correction is a new entry naming the one it corrects");
    expect(tool.menuSummary).toContain("immutable once written");
    expect(tool.menuSummary).toContain("never a change to it");
    // The store makes both entries permanent; it does not make them co-visible.
    // The reader is an origin-time range with no text or id query, so a claim
    // that a read-back sees both is true of the store and false of the only
    // instrument that reaches it. State the reachability, not the store.
    expect(tool.description).not.toContain("sees both");
    expect(tool.description).toContain("Nothing on either entry links them");
    expect(tool.description).toContain("no text or id query");
    expect(tool.description).toContain(
      "reading back to the corrected entry's own window can never contain it",
    );
  });

  // last_ruminated_at is nulled by the same UPDATE that sets the terminal status,
  // exactly as the ticks are zeroed. The surface named one and not the other, so a
  // reader who checked the unnamed field read the write's own doing as history.
  it("names both rumination fields this write resets, not only the counter", () => {
    const tool = createOpenQuestionsResolveTool({
      identityService: {
        resolveOpenQuestion: () => {
          throw new Error("not invoked");
        },
      },
      disclosureLabelForEvidence: async () => selfPrivateMemoryDisclosureLabel(),
    });

    expect(tool.description).toContain("unresolved_rumination_ticks reads 0");
    expect(tool.description).toContain("last_ruminated_at reads null");
    expect(tool.description).toContain("because this write set them there");
  });

  it("surfaces identity-governance review-required open question resolution as tool data", async () => {
    const current = sampleOpenQuestion();
    const resolveOpenQuestion = vi.fn(() => ({
      status: "requires_review" as const,
      current,
    }));
    const tool = createOpenQuestionsResolveTool({
      identityService: { resolveOpenQuestion },
      disclosureLabelForEvidence: async () => selfPrivateMemoryDisclosureLabel(),
    });
    const evidenceStreamEntryId = createStreamEntryId();

    const result = await tool.invoke(
      {
        open_question_id: current.id,
        resolution_note: "The evidence is not enough to bypass review.",
        resolution_evidence_stream_entry_ids: [evidenceStreamEntryId],
      },
      context,
    );

    expect(resolveOpenQuestion).toHaveBeenCalledWith(
      current.id,
      expect.objectContaining({
        resolution_note: "The evidence is not enough to bypass review.",
        resolution_evidence_stream_entry_ids: [evidenceStreamEntryId],
      }),
      {
        kind: "online_reflector",
        evidence_episode_ids: [],
        evidence_stream_entry_ids: [evidenceStreamEntryId],
      },
      undefined,
    );
    expect(result).toMatchObject({
      status: "requires_review",
      reason: "identity_governance_requires_review",
      openQuestion: {
        id: current.id,
        question: current.question,
        disclosure_label: {
          disclosure_class: "self_private",
        },
      },
    });
  });

  it("returns recent episodes with disclosure labels", async () => {
    const sourceStreamId = createStreamEntryId();
    const candidate: EpisodeSearchCandidate = {
      episode: {
        id: createEpisodeId(),
        title: "Recent reflection",
        narrative: "A recent episode with enough detail to recall.",
        participants: ["self"],
        location: null,
        start_time: 1_000,
        end_time: 1_100,
        source_stream_ids: [sourceStreamId],
        significance: 0.5,
        tags: ["reflection"],
        confidence: 0.8,
        lineage: {
          derived_from: [],
          supersedes: [],
        },
        emotional_arc: null,
        embedding: new Float32Array([0]),
        created_at: 1_000,
        updated_at: 1_100,
      },
      stats: {
        episode_id: createEpisodeId(),
        retrieval_count: 0,
        use_count: 0,
        last_retrieved: null,
        win_rate: 0,
        tier: "T3",
        promoted_at: 1_000,
        promoted_from: null,
        gist: null,
        gist_generated_at: null,
        last_decayed_at: null,
        heat_multiplier: 1,
        valence_mean: 0,
        archived: false,
      },
      similarity: 0,
    };
    const listRecentEpisodes = vi.fn(async () => [candidate]);
    const tool = createEpisodicRecentTool({ listRecentEpisodes });

    const result = await tool.invoke({ limit: 3 }, context);

    expect(tool.allowedOrigins).toEqual(["autonomous"]);
    expect(tool.writeScope).toBe("read");
    expect(listRecentEpisodes).toHaveBeenCalledWith(3, context);
    expect(result.episodes).toHaveLength(1);
    expect(result.episodes[0]).toMatchObject({
      id: candidate.episode.id,
      title: "Recent reflection",
      source_stream_ids: [sourceStreamId],
      disclosure_label: {
        disclosure_class: "public",
      },
    });
  });
});
