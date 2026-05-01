import { describe, expect, it } from "vitest";

import type { OpenQuestion } from "../memory/self/index.js";
import type { EpisodeSearchCandidate, EpisodeStats } from "../memory/episodic/types.js";
import type { SemanticEdge, SemanticNode } from "../memory/semantic/types.js";
import { createEpisodeFixture } from "../offline/test-support.js";
import { RetrievalError } from "../util/errors.js";
import type { EpisodeId, OpenQuestionId, SemanticEdgeId, SemanticNodeId } from "../util/ids.js";

import type { EvidenceItem, EvidencePool, RecallIntent } from "./recall-types.js";
import type { EpisodeScore } from "./scoring.js";
import {
  projectEpisodes,
  projectOpenQuestions,
  projectSemantic,
  type EpisodeProjectionSource,
} from "./evidence-projections.js";
import type { RetrievedSemantic } from "./semantic-retrieval.js";

const intent: RecallIntent = {
  id: "recall_raw_text_0",
  kind: "raw_text",
  query: "Atlas",
  terms: [],
  priority: 100,
  source: "raw-user-message",
};

describe("EvidencePool compatibility projections", () => {
  it("derive legacy fields only from evidence items in the pool", () => {
    const keptEpisode = createEpisodeFixture({
      id: "ep_aaaaaaaaaaaaaaaa" as EpisodeId,
      title: "Kept episode",
    });
    const droppedEpisode = createEpisodeFixture({
      id: "ep_bbbbbbbbbbbbbbbb" as EpisodeId,
      title: "Dropped episode",
    });
    const keptNode = semanticNode("semn_aaaaaaaaaaaaaaaa" as SemanticNodeId, "Kept node");
    const droppedNode = semanticNode("semn_bbbbbbbbbbbbbbbb" as SemanticNodeId, "Dropped node");
    const keptSupport = semanticNode("semn_cccccccccccccccc" as SemanticNodeId, "Kept support");
    const droppedSupport = semanticNode(
      "semn_dddddddddddddddd" as SemanticNodeId,
      "Dropped support",
    );
    const keptEdge = semanticEdge(
      "seme_aaaaaaaaaaaaaaaa" as SemanticEdgeId,
      keptNode.id,
      keptSupport.id,
    );
    const droppedEdge = semanticEdge(
      "seme_bbbbbbbbbbbbbbbb" as SemanticEdgeId,
      droppedNode.id,
      droppedSupport.id,
    );
    const keptQuestion = openQuestion("oq_aaaaaaaaaaaaaaaa" as OpenQuestionId, "Kept question?");
    const droppedQuestion = openQuestion(
      "oq_bbbbbbbbbbbbbbbb" as OpenQuestionId,
      "Dropped question?",
    );
    const keptEpisodeEvidence = evidence("evidence_episode_keep", "episode", {
      episodeId: keptEpisode.id,
    });
    const droppedEpisodeEvidence = evidence("evidence_episode_drop", "episode", {
      episodeId: droppedEpisode.id,
    });
    const keptQuestionEvidence = evidence("evidence_open_question_keep", "open_question", {
      openQuestionId: keptQuestion.id,
    });
    const droppedQuestionEvidence = evidence("evidence_open_question_drop", "open_question", {
      openQuestionId: droppedQuestion.id,
    });
    const pool: EvidencePool = {
      intents: [intent],
      items: [
        keptEpisodeEvidence,
        evidence("evidence_semantic_node_keep", "semantic_node", { nodeId: keptNode.id }),
        evidence("evidence_semantic_edge_keep", "semantic_edge", { edgeId: keptEdge.id }),
        keptQuestionEvidence,
      ],
    };

    const episodes = projectEpisodes(
      pool,
      new Map<string, EpisodeProjectionSource>([
        [keptEpisodeEvidence.id, episodeSource(keptEpisode, 0.9)],
        [droppedEpisodeEvidence.id, episodeSource(droppedEpisode, 1)],
      ]),
      {
        limit: 5,
        mmrLambda: 0.7,
      },
    );
    const semantic = projectSemantic(pool, {
      as_of: null,
      supports: [keptSupport, droppedSupport],
      contradicts: [],
      categories: [],
      matched_node_ids: [keptNode.id, droppedNode.id],
      matched_nodes: [keptNode, droppedNode],
      support_hits: [
        { root_node_id: keptNode.id, node: keptSupport, edgePath: [keptEdge] },
        { root_node_id: droppedNode.id, node: droppedSupport, edgePath: [droppedEdge] },
      ],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    } satisfies RetrievedSemantic);
    const questions = projectOpenQuestions(
      pool,
      new Map([
        [keptQuestionEvidence.id, keptQuestion],
        [droppedQuestionEvidence.id, droppedQuestion],
      ]),
    );

    expect(episodes.episodes.map((item) => item.episode.id)).toEqual([keptEpisode.id]);
    expect(episodes.selectedEvidence.map((item) => item.id)).toEqual([keptEpisodeEvidence.id]);
    expect(semantic.matched_node_ids).toEqual([keptNode.id]);
    expect(semantic.support_hits.map((hit) => hit.edgePath[0]?.id)).toEqual([keptEdge.id]);
    expect(semantic.supports.map((node) => node.id)).toEqual([keptSupport.id]);
    expect(questions.map((question) => question.id)).toEqual([keptQuestion.id]);
  });

  it("throws when episode evidence is missing its hydrated projection source", () => {
    const episode = createEpisodeFixture({
      id: "ep_aaaaaaaaaaaaaaaa" as EpisodeId,
      title: "Missing hydration",
    });
    const pool: EvidencePool = {
      intents: [intent],
      items: [
        evidence("evidence_episode_missing", "episode", {
          episodeId: episode.id,
        }),
      ],
    };

    expect(() =>
      projectEpisodes(pool, new Map(), {
        limit: 5,
        mmrLambda: 0.7,
      }),
    ).toThrow(RetrievalError);
  });

  it("throws when semantic evidence is missing hydrated nodes or hits", () => {
    const node = semanticNode("semn_aaaaaaaaaaaaaaaa" as SemanticNodeId, "Missing node");
    const support = semanticNode("semn_bbbbbbbbbbbbbbbb" as SemanticNodeId, "Missing support");
    const edge = semanticEdge("seme_aaaaaaaaaaaaaaaa" as SemanticEdgeId, node.id, support.id);

    expect(() =>
      projectSemantic(
        {
          intents: [intent],
          items: [
            evidence("evidence_semantic_node_missing", "semantic_node", { nodeId: node.id }),
          ],
        },
        emptySemantic(),
      ),
    ).toThrow(RetrievalError);

    expect(() =>
      projectSemantic(
        {
          intents: [intent],
          items: [
            evidence("evidence_semantic_edge_missing", "semantic_edge", { edgeId: edge.id }),
          ],
        },
        {
          ...emptySemantic(),
          matched_node_ids: [node.id],
          matched_nodes: [node],
        },
      ),
    ).toThrow(RetrievalError);
  });

  it("throws when open-question evidence is missing its hydrated question", () => {
    const question = openQuestion(
      "oq_aaaaaaaaaaaaaaaa" as OpenQuestionId,
      "Missing question?",
    );
    const pool: EvidencePool = {
      intents: [intent],
      items: [
        evidence("evidence_open_question_missing", "open_question", {
          openQuestionId: question.id,
        }),
      ],
    };

    expect(() => projectOpenQuestions(pool, new Map())).toThrow(RetrievalError);
  });
});

function emptySemantic(): RetrievedSemantic {
  return {
    as_of: null,
    supports: [],
    contradicts: [],
    categories: [],
    matched_node_ids: [],
    matched_nodes: [],
    support_hits: [],
    causal_hits: [],
    contradiction_hits: [],
    category_hits: [],
  };
}

function episodeSource(episode: ReturnType<typeof createEpisodeFixture>, score: number) {
  return {
    candidate: {
      episode,
      stats: episodeStats(episode.id),
      similarity: score,
    } satisfies EpisodeSearchCandidate,
    score: episodeScore(score),
    citationChain: () => [],
  };
}

function episodeStats(episodeId: EpisodeId): EpisodeStats {
  return {
    episode_id: episodeId,
    retrieval_count: 0,
    use_count: 0,
    last_retrieved: null,
    win_rate: 0,
    tier: "T2",
    promoted_at: 0,
    promoted_from: null,
    gist: null,
    gist_generated_at: null,
    last_decayed_at: null,
    heat_multiplier: 1,
    valence_mean: 0,
    archived: false,
  };
}

function episodeScore(score: number): EpisodeScore {
  return {
    decayedSalience: score,
    heat: 0,
    goalRelevance: 0,
    valueAlignment: 0,
    timeRelevance: 0,
    moodBoost: 0,
    socialRelevance: 0,
    entityRelevance: 0,
    suppressionPenalty: 0,
    score,
  };
}

function semanticNode(id: SemanticNodeId, label: string): SemanticNode {
  return {
    id,
    kind: "entity",
    label,
    description: `${label} description`,
    domain: null,
    aliases: [],
    confidence: 0.9,
    source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
    created_at: 1,
    updated_at: 1,
    last_verified_at: 1,
    embedding: Float32Array.from([1, 0, 0, 0]),
    archived: false,
    superseded_by: null,
  };
}

function semanticEdge(
  id: SemanticEdgeId,
  fromNodeId: SemanticNodeId,
  toNodeId: SemanticNodeId,
): SemanticEdge {
  return {
    id,
    from_node_id: fromNodeId,
    to_node_id: toNodeId,
    relation: "supports",
    confidence: 0.8,
    evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
    created_at: 1,
    last_verified_at: 1,
    valid_from: 1,
    valid_to: null,
    invalidated_at: null,
    invalidated_by_edge_id: null,
    invalidated_by_review_id: null,
    invalidated_by_process: null,
    invalidated_reason: null,
  };
}

function openQuestion(id: OpenQuestionId, question: string): OpenQuestion {
  return {
    id,
    question,
    urgency: 0.8,
    status: "open",
    audience_entity_id: null,
    related_episode_ids: [],
    related_semantic_node_ids: ["semn_aaaaaaaaaaaaaaaa" as SemanticNodeId],
    provenance: { kind: "manual" },
    source: "reflection",
    created_at: 1,
    last_touched: 1,
    resolution_episode_id: null,
    resolution_note: null,
    resolved_at: null,
    abandoned_reason: null,
    abandoned_at: null,
  };
}

function evidence(
  id: string,
  source: EvidenceItem["source"],
  provenance: NonNullable<EvidenceItem["provenance"]>,
): EvidenceItem {
  return {
    id,
    source,
    text: id,
    provenance,
    recallIntentId: intent.id,
    matchedTerms: [],
    score: 0.9,
    scoreBreakdown: {},
  };
}
