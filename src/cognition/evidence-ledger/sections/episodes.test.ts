import { describe, expect, it } from "vitest";

import {
  createEpisodeFixture,
  createRetrievalScoreFixture,
} from "../../../offline/test-support.js";
import type { EvidenceItem, RetrievedEpisode } from "../../../retrieval/index.js";
import {
  DEFAULT_SESSION_ID,
  createEntityId,
  createEpisodeId,
  createStreamEntryId,
} from "../../../util/ids.js";
import type { BuilderSectionContext } from "../builder-context.js";
import { createSectionBuckets } from "../section-buckets.js";
import { addEpisodesSection } from "./episodes.js";
import {
  addRetrievedRawStreamEvidenceSection,
  addRetrievedStructuredEvidenceSection,
} from "./retrieved-evidence.js";

const NOW_MS = 1_800_000_000_000;

describe("evidence-ledger episode section", () => {
  it("renders disclosure labels for private recalled episodes", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const streamId = createStreamEntryId();
    const occurredAt = NOW_MS - 5 * 60_000;
    const buckets = createSectionBuckets();
    const retrievedEpisode: RetrievedEpisode = {
      episode: createEpisodeFixture({
        id: createEpisodeId(),
        title: "Alice private planning",
        narrative: "Alice private planning should be usable internally but not disclosed.",
        end_time: occurredAt,
        source_stream_ids: [streamId],
        audience_entity_id: alice,
        shared: false,
      }),
      score: 0.81,
      rawScore: 0.81,
      scoreBreakdown: createRetrievalScoreFixture(),
      citationChain: [],
    };

    addEpisodesSection({
      input: {
        sessionId: DEFAULT_SESSION_ID,
        nowMs: NOW_MS,
        audienceEntityId: bob,
        retrievedEpisodes: [retrievedEpisode],
      },
      nowMs: NOW_MS,
      resolver: {
        currentSessionId: DEFAULT_SESSION_ID,
        streamEntriesById: new Map(),
        streamOrderById: new Map(),
        episodeScopesById: new Map(),
        episodeSourceStreamIdsById: new Map(),
      },
      buckets,
    } as unknown as BuilderSectionContext);

    const entry = buckets.get("episodes")?.entries[0];
    expect(entry?.state).toContain("disclosure_class=relationship_private");
    expect(entry?.state).toContain(`private-to=${alice}`);
    expect(entry?.state).toContain(
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
    expect(entry?.state_metadata).toMatchObject({
      occurred_at: new Date(occurredAt).toISOString(),
      relative_age: "5m ago",
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
      },
      disclosure_note:
        "I can use this internally; I do not disclose it to the current audience unless authorized",
      current_audience_entity_id: bob,
    });
  });

  it("renders disclosure labels for private raw-stream evidence", () => {
    const alice = createEntityId();
    const streamId = createStreamEntryId();
    const buckets = createSectionBuckets();
    const evidence: EvidenceItem = {
      id: "evidence_raw_stream_private_intent",
      source: "raw_stream",
      text: "Alice private raw-stream detail.",
      provenance: {
        streamIds: [streamId],
        parentEpisodeId: createEpisodeId(),
      },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 1,
      scoreBreakdown: {
        provenance: 1,
      },
      disclosureLabel: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: [alice],
        privateToEntityIds: [alice],
        publicToEntityIds: [],
      },
    };

    addRetrievedRawStreamEvidenceSection({
      input: {
        retrievedEvidence: [evidence],
      },
      resolver: {
        currentSessionId: DEFAULT_SESSION_ID,
        streamEntriesById: new Map(),
        streamOrderById: new Map(),
        episodeScopesById: new Map(),
        episodeSourceStreamIdsById: new Map(),
      },
      transcript: {
        rawStreamIds: new Set(),
      },
      buckets,
    } as unknown as BuilderSectionContext);

    const entry = buckets.get("retrieved_raw_stream_evidence")?.entries[0];
    expect(entry?.state).toContain("disclosure_class=relationship_private");
    expect(entry?.state).toContain(`private-to=${alice}`);
    expect(entry?.state).toContain(
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
    expect(entry?.state_metadata).toMatchObject({
      stream_ids: [streamId],
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
      },
      disclosure_note:
        "I can use this internally; I do not disclose it to the current audience unless authorized",
    });
  });

  it("fails closed for parentless warm raw-stream evidence without a source label", () => {
    const streamId = createStreamEntryId();
    const buckets = createSectionBuckets();
    const evidence: EvidenceItem = {
      id: "warm_recall_raw_stream_parentless",
      source: "warm_recall",
      text: "Parentless persisted raw stream detail.",
      provenance: {
        streamIds: [streamId],
      },
      recallIntentId: "warm_recall",
      matchedTerms: [],
      score: 0.5,
      scoreBreakdown: {
        provenance: 1,
      },
    };

    addRetrievedStructuredEvidenceSection({
      input: {
        retrievedEvidence: [evidence],
      },
      resolver: {
        currentSessionId: DEFAULT_SESSION_ID,
        streamEntriesById: new Map(),
        streamOrderById: new Map(),
        episodeScopesById: new Map(),
        episodeSourceStreamIdsById: new Map(),
      },
      buckets,
    } as unknown as BuilderSectionContext);

    const entry = buckets.get("retrieved_memory_evidence")?.entries[0];
    expect(entry?.state).toContain("disclosure_class=unknown");
    expect(entry?.state_metadata).toMatchObject({
      stream_ids: [streamId],
      disclosure_label: {
        disclosure_class: "unknown",
        origin_audience_entity_ids: [],
        private_to_entity_ids: [],
        public_to_entity_ids: [],
      },
      disclosure_note:
        "I can use this internally; I do not disclose it to the current audience unless authorized",
    });
  });
});
