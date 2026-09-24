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
  createSessionId,
  createStreamEntryId,
} from "../../../util/ids.js";
import type { BuilderSectionContext } from "../builder-context.js";
import { RAW_STREAM_TRUST_RANK, createSectionBuckets } from "../section-buckets.js";
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

  describe("raw-stream evidence the builder's entry map does not hold", () => {
    const otherSession = createSessionId();

    const rawStreamEvidence = (overrides: Partial<EvidenceItem>): EvidenceItem => ({
      id: "evidence_raw_stream_unresolved",
      source: "raw_stream",
      text: "Recalled raw-stream detail.",
      provenance: { streamIds: [createStreamEntryId()] },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 1,
      scoreBreakdown: { provenance: 1 },
      ...overrides,
    });

    const renderRow = (evidence: EvidenceItem, streamEntriesById = new Map()) => {
      const buckets = createSectionBuckets();
      addRetrievedRawStreamEvidenceSection({
        input: { retrievedEvidence: [evidence] },
        resolver: {
          currentSessionId: DEFAULT_SESSION_ID,
          streamEntriesById,
          streamOrderById: new Map(),
          episodeScopesById: new Map(),
          episodeSourceStreamIdsById: new Map(),
        },
        transcript: { rawStreamIds: new Set() },
        buckets,
      } as unknown as BuilderSectionContext);

      return buckets.get("retrieved_raw_stream_evidence")?.entries[0];
    };

    const sectionOf = (evidence: EvidenceItem) => {
      const buckets = createSectionBuckets();
      addRetrievedRawStreamEvidenceSection({
        input: { retrievedEvidence: [evidence] },
        resolver: {
          currentSessionId: DEFAULT_SESSION_ID,
          streamEntriesById: new Map(),
          streamOrderById: new Map(),
          episodeScopesById: new Map(),
          episodeSourceStreamIdsById: new Map(),
        },
        transcript: { rawStreamIds: new Set() },
        buckets,
      } as unknown as BuilderSectionContext);

      return [...buckets.entries()]
        .filter(([, bucket]) => bucket.entries.length > 0)
        .map(([sectionId]) => sectionId);
    };

    it("reads the carried session as current when it matches, rather than reporting global", () => {
      const entry = renderRow(
        rawStreamEvidence({
          streamEntrySessionId: DEFAULT_SESSION_ID,
          streamEntryKind: "user_msg",
        }),
      );

      expect(entry?.session_scope).toBe("current_session");
      expect(entry?.source_type).toBe("current_session_stream");
      expect(entry?.actor).toBe("user");
      expect(entry?.trust_rank).toBe(RAW_STREAM_TRUST_RANK);
    });

    it("reads a carried session other than the current one as prior", () => {
      const entry = renderRow(
        rawStreamEvidence({
          streamEntrySessionId: otherSession,
          streamEntryKind: "agent_msg",
        }),
      );

      expect(entry?.session_scope).toBe("prior_session");
      expect(entry?.source_type).toBe("prior_session_stream");
      expect(entry?.actor).toBe("assistant");
    });

    it("keeps corrected prior-session raw evidence in its source section with capped trust", () => {
      const evidence = rawStreamEvidence({
        streamEntrySessionId: otherSession,
        streamEntryKind: "agent_msg",
      });

      expect(sectionOf(evidence)).toEqual(["retrieved_raw_stream_evidence"]);
      expect(renderRow(evidence)?.trust_rank).toBeLessThan(RAW_STREAM_TRUST_RANK);
      expect(sectionOf(rawStreamEvidence({ streamEntryKind: "agent_msg" }))).toEqual([
        "retrieved_raw_stream_evidence",
      ]);
    });

    it("keeps global and memory when neither the map nor the item carries the fact", () => {
      const entry = renderRow(rawStreamEvidence({}));

      expect(entry?.session_scope).toBe("global");
      expect(entry?.source_type).toBe("system_metadata");
      expect(entry?.actor).toBe("memory");
    });

    it("prefers the entry map over the carried copy when the map has the row", () => {
      const streamId = createStreamEntryId();
      const entry = renderRow(
        rawStreamEvidence({
          provenance: { streamIds: [streamId] },
          // Disagrees with the map on both fields, so a row built from these instead of
          // from the map would be visible rather than coincidentally equal.
          streamEntrySessionId: otherSession,
          streamEntryKind: "agent_msg",
        }),
        new Map([
          [
            streamId,
            {
              id: streamId,
              kind: "user_msg",
              session_id: DEFAULT_SESSION_ID,
              timestamp: NOW_MS,
              content: "Live entry.",
            },
          ],
        ]),
      );

      expect(entry?.session_scope).toBe("current_session");
      expect(entry?.actor).toBe("user");
    });
  });

  describe("warm recall of stream entries", () => {
    const warmEvidence = (overrides: Partial<EvidenceItem>): EvidenceItem => ({
      id: "warm_recall_raw_stream_unresolved",
      source: "warm_recall",
      text: "Warm-recalled stream detail.",
      provenance: { streamIds: [createStreamEntryId()] },
      recallIntentId: "warm_recall",
      matchedTerms: [],
      score: 0.15,
      scoreBreakdown: { provenance: 1 },
      ...overrides,
    });

    const renderRow = (evidence: EvidenceItem, streamEntriesById = new Map()) => {
      const buckets = createSectionBuckets();
      addRetrievedStructuredEvidenceSection({
        input: { retrievedEvidence: [evidence] },
        resolver: {
          currentSessionId: DEFAULT_SESSION_ID,
          streamEntriesById,
          streamOrderById: new Map(),
          episodeScopesById: new Map(),
          episodeSourceStreamIdsById: new Map(),
        },
        buckets,
      } as unknown as BuilderSectionContext);

      return buckets.get("retrieved_memory_evidence")?.entries[0];
    };

    it("takes one entry's actor from its carried kind and leaves its scope to the resolver", () => {
      const entry = renderRow(warmEvidence({ streamEntryKind: "agent_msg" }));

      expect(entry?.actor).toBe("assistant");
      expect(entry?.session_scope).toBe("global");
      expect(entry?.source_type).toBe("system_metadata");
      expect(entry?.value).toBe("warm_recall");
    });

    it("prefers the entry map over the carried kind when the map has the row", () => {
      const streamId = createStreamEntryId();
      const entry = renderRow(
        warmEvidence({
          provenance: { streamIds: [streamId] },
          streamEntryKind: "agent_msg",
        }),
        new Map([
          [
            streamId,
            {
              id: streamId,
              kind: "user_msg",
              session_id: DEFAULT_SESSION_ID,
              timestamp: NOW_MS,
              content: "Live entry.",
            },
          ],
        ]),
      );

      expect(entry?.session_scope).toBe("current_session");
      expect(entry?.source_type).toBe("current_session_stream");
      expect(entry?.actor).toBe("user");
    });

    it("keeps global and memory for a join that carries no kind", () => {
      const entry = renderRow(
        warmEvidence({
          provenance: { streamIds: [createStreamEntryId(), createStreamEntryId()] },
        }),
      );

      expect(entry?.session_scope).toBe("global");
      expect(entry?.source_type).toBe("system_metadata");
      expect(entry?.actor).toBe("memory");
    });
  });
});
