import {
  createEpisodeFixture,
  createRetrievalScoreFixture,
  createSemanticNodeFixture,
  createWorkingMemoryFixture,
} from "../offline/test-support.js";
import type { RetrievedEpisode, RetrievedSemantic, EvidenceItem } from "../retrieval/index.js";
import { publicMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import { DEFAULT_SESSION_ID, createSemanticEdgeId, createSemanticNodeId } from "../util/ids.js";
import type { AnnotationTarget } from "../memory/identity/annotations.js";
import type { EvidenceLedgerBuildInput } from "./evidence-ledger/builder-types.js";
import { EvidenceLedgerBuilder } from "./evidence-ledger/builder.js";
import type { RecordEvidenceReader } from "../memory/common/record-evidence.js";
import type { StreamReader } from "../stream/index.js";

export function createRecordEvidenceFixture() {
  const episodes: RetrievedEpisode[] = ["First", "Second"].map((label) => ({
    episode: createEpisodeFixture({
      title: `Synthetic ${label}`,
      narrative: `${label} original text.\n原文。`,
      shared: true,
    }),
    score: 0.7,
    rawScore: 0.7,
    scoreBreakdown: createRetrievalScoreFixture(),
    citationChain: [],
  }));
  const node = createSemanticNodeFixture({
    label: "Synthetic node",
    description: "Original node description. 原文。",
    source_episode_ids: [episodes[1]!.episode.id],
  });
  const semantic: RetrievedSemantic = {
    supports: [],
    contradicts: [],
    categories: [],
    matched_node_ids: [node.id],
    matched_nodes: [node],
    support_hits: [],
    causal_hits: [],
    contradiction_hits: [],
    category_hits: [],
  };
  const evidence: EvidenceItem[] = [
    {
      id: "synthetic-episode-evidence",
      source: "episode",
      text: episodes[0]!.episode.narrative,
      provenance: { episodeId: episodes[0]!.episode.id },
      recallIntentId: "synthetic",
      matchedTerms: [],
      score: 0.7,
      scoreBreakdown: {},
      disclosureLabel: publicMemoryDisclosureLabel(),
    },
    {
      id: "synthetic-node-evidence",
      source: "semantic_node",
      text: node.description,
      provenance: { nodeId: node.id },
      recallIntentId: "synthetic",
      matchedTerms: [],
      score: 0.7,
      scoreBreakdown: {},
      disclosureLabel: publicMemoryDisclosureLabel(),
    },
  ];
  const input: EvidenceLedgerBuildInput = {
    sessionId: DEFAULT_SESSION_ID,
    audienceEntityId: null,
    currentUserMessage: "Synthetic current turn",
    nowMs: 2_000_000,
    workingMemory: createWorkingMemoryFixture(),
    applicableCommitments: [],
    openQuestions: [],
    pendingCorrections: [],
    retrievedEpisodes: episodes,
    retrievedSemantic: semantic,
    retrievedEvidence: evidence,
    autobiographicalRecall: {
      window: { startMs: 0, endMs: 2_000_000, label: "Synthetic window", source: "recent_default" },
      evidence: [
        {
          id: "synthetic-autobiographical-episode",
          kind: "episode",
          groupId: "episodes",
          groupLabel: "Episodes",
          occurredAt: 1_000_000,
          relativeAge: "1m ago",
          score: 0.7,
          text: episodes[0]!.episode.narrative,
          disclosureLabel: publicMemoryDisclosureLabel(),
          sourceStreamEntryIds: [],
          sourceEpisodeIds: [episodes[0]!.episode.id],
          metadata: { episode_id: episodes[0]!.episode.id },
        },
      ],
    },
  };
  return { episodes, node, semantic, evidence, input };
}

export function createWarmRecordEvidenceFixture() {
  const fixture = createRecordEvidenceFixture();
  const episode = fixture.episodes[0]!.episode;
  const edgeId = createSemanticEdgeId();
  const records: Array<{ target: AnnotationTarget; evidence: EvidenceItem; correction: string }> = [
    {
      target: { record_type: "episode", record_id: episode.id },
      evidence: {
        ...fixture.evidence[0]!,
        id: "synthetic-warm-episode",
        source: "warm_recall",
        provenance: { episodeId: episode.id, streamIds: episode.source_stream_ids },
      },
      correction: "Warm episode correction. 更正。",
    },
    {
      target: { record_type: "semantic_node", record_id: fixture.node.id },
      evidence: {
        ...fixture.evidence[1]!,
        id: "synthetic-warm-node",
        source: "warm_recall",
        source_episode_ids: [episode.id],
      },
      correction: "Warm node correction. تصحيح.",
    },
    {
      target: { record_type: "semantic_edge", record_id: edgeId },
      evidence: {
        ...fixture.evidence[1]!,
        id: "synthetic-warm-edge",
        source: "warm_recall",
        provenance: { edgeId, nodeId: fixture.node.id },
        source_episode_ids: [episode.id],
      },
      correction: "Warm edge correction. Zażółć. 🧭",
    },
  ];
  const unrelatedEvidence: EvidenceItem[] = [
    {
      ...fixture.evidence[0]!,
      id: "synthetic-warm-raw-stream",
      source: "warm_recall",
      provenance: { streamIds: episode.source_stream_ids, parentEpisodeId: episode.id },
    },
    {
      ...fixture.evidence[1]!,
      id: "synthetic-warm-derived-node",
      source: "warm_recall",
      provenance: { nodeId: createSemanticNodeId() },
      source_episode_ids: [episode.id],
    },
  ];
  const evidence = [...records.map((record) => record.evidence), ...unrelatedEvidence];
  return {
    records,
    unrelatedEvidence,
    evidence,
    input: {
      ...fixture.input,
      retrievedEpisodes: [],
      retrievedSemantic: { ...fixture.semantic, matched_nodes: [], matched_node_ids: [] },
      autobiographicalRecall: undefined,
      retrievedEvidence: evidence,
    },
  };
}

export function buildRecordEvidenceLedger(
  input: EvidenceLedgerBuildInput,
  recordEvidenceReader?: RecordEvidenceReader,
) {
  return new EvidenceLedgerBuilder({
    recordEvidenceReader,
    createStreamReader: () =>
      ({
        scanReverse: () => ({ entries: [], scannedBytes: 0, capReached: null }),
      }) as unknown as StreamReader,
    relationalSlotRepository: { list: () => [] },
    actionRepository: { list: () => [] },
    currentSessionTranscriptTokenBudget: 2_000,
  }).build(input);
}
