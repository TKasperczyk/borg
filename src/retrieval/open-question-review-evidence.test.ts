import { describe, expect, it } from "vitest";
import {
  renderPlanRequestedVerificationRetrieval,
  summarizeRetrievedEvidence,
} from "../cognition/deliberation/prompt/retrieval.js";
import { compactEvidenceLedger, renderEvidenceLedger } from "../cognition/evidence-ledger/index.js";
import {
  capFullLedgerSection,
  sectionAccountingEntry,
} from "../cognition/evidence-ledger/compaction.js";
import { renderSection } from "../cognition/evidence-ledger/section-rendering.js";
import { buildRecordEvidenceLedger } from "../cognition/record-evidence-test-support.js";
import {
  createOfflineTestHarness,
  createWorkingMemoryFixture,
  TestEmbeddingClient,
} from "../offline/test-support.js";
import type { OpenQuestionReview } from "../memory/self/open-question-review.js";
import { createOpenQuestionId, createSemanticNodeId, DEFAULT_SESSION_ID } from "../util/ids.js";
import { estimatePromptTokens } from "../util/token-estimate.js";
import { RetrievalPipeline, type RetrievedContext } from "./pipeline.js";
import { SELF_RECALL_SCOPE } from "./recall-context.js";
import type { EvidenceItem } from "./recall-types.js";

const at = Date.UTC(2026, 8, 2);
const review: OpenQuestionReview = {
  id: 101,
  kind: "contradiction",
  status: "resolved",
  resolution: "dismiss",
  resolved_at: new Date(at).toISOString(),
};
const emptySemantic: RetrievedContext["semantic"] = {
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

function unifiedContext(evidence: EvidenceItem[]) {
  return { evidence, episodes: [], semantic: emptySemantic, open_questions: [] };
}

async function ledgerFor(evidence: EvidenceItem[]) {
  return buildRecordEvidenceLedger({
    sessionId: DEFAULT_SESSION_ID,
    audienceEntityId: null,
    currentUserMessage: "Synthetic turn",
    nowMs: at,
    workingMemory: createWorkingMemoryFixture(),
    applicableCommitments: [],
    openQuestions: [],
    pendingCorrections: [],
    retrievedEpisodes: [],
    retrievedSemantic: emptySemantic,
    retrievedEvidence: evidence,
  });
}

function syntheticEvidence(index = 0): EvidenceItem {
  return {
    id: `synthetic-review-evidence-${index}`,
    source: "open_question",
    text: "x".repeat(5_000),
    provenance: { openQuestionId: createOpenQuestionId() },
    review,
    recallIntentId: "synthetic",
    matchedTerms: [],
    score: 0.5,
    scoreBreakdown: {},
  };
}

describe("review metadata in unified question evidence", () => {
  for (const length of [300, 500, 5_000]) {
    it.each([null, "dismiss", "accept"] as const)(
      `${length}-character questions keep complete %s labels in fresh/warm retrieval and compacted ledgers`,
      async (resolution) => {
        const text = "q".repeat(length);
        const query = "Synthetic question query";
        const embeddingClient = new TestEmbeddingClient(
          new Map([
            [text, [1, 0, 0, 0]],
            [query, [1, 0, 0, 0]],
          ]),
        );
        const harness = await createOfflineTestHarness({ embeddingClient });
        try {
          const item = harness.reviewQueueRepository.enqueue({
            kind: "contradiction",
            refs: { node_ids: [createSemanticNodeId(), createSemanticNodeId()] },
            reason: "Synthetic review",
          });
          const question = harness.openQuestionsRepository.add({
            question: text,
            source: "contradiction",
            urgency: 0.5,
            provenance: { kind: "manual" },
            review_id: item.id,
          });
          if (resolution !== null)
            harness.db
              .prepare("UPDATE review_queue SET resolution = ?, resolved_at = ? WHERE id = ?")
              .run(resolution, at, item.id);
          await harness.openQuestionsRepository.waitForPendingEmbeddings();
          const expected = harness.openQuestionsRepository.get(question.id)!.review!;
          const pipeline = new RetrievalPipeline({
            embeddingClient,
            episodicRepository: harness.episodicRepository,
            openQuestionsRepository: harness.openQuestionsRepository,
            recallStateRepository: harness.recallStateRepository,
            dataDir: harness.tempDir,
            clock: harness.clock,
          });
          const fresh = await pipeline.searchWithContextForDisclosure(query, {
            includeOpenQuestions: true,
            recordRetrieval: false,
          });
          harness.recallStateRepository.save({
            scopeKey: SELF_RECALL_SCOPE,
            activeHandles: [
              {
                handle: { source: "open_question", openQuestionId: question.id },
                firstSeenTurn: 0,
                lastSeenTurn: 0,
                lastRenderedTurn: null,
                expiresAfterTurn: 10,
                reinforcementCount: 1,
              },
            ],
            suppressedHandles: {},
            lastRefreshTurn: 0,
            updatedAt: harness.clock.now(),
            ttlTurns: 6,
          });
          const warm = await pipeline.recallEpisodesForCognition(query, {
            includeOpenQuestions: false,
            turnCounter: 1,
            recallContext: {
              reader: SELF_RECALL_SCOPE,
              currentSessionId: DEFAULT_SESSION_ID,
              currentAudienceEntityId: null,
              currentParticipantEntityIds: [],
            },
          });
          for (const [context, source] of [
            [fresh, "open_question"],
            [warm, "warm_recall"],
          ] as const) {
            const evidence = context.evidence.find(
              (row) => row.source === source && row.provenance?.openQuestionId === question.id,
            )!;
            expect(evidence).toBeDefined();
            expect(evidence.text).toBe(text);
            expect(evidence.review).toEqual(expected);
            const summary = summarizeRetrievedEvidence(
              "Evidence",
              unifiedContext([evidence]),
              1_000,
            )!;
            expect(summary).toContain(`[review=${JSON.stringify(expected)}]`);
            expect(estimatePromptTokens(summary)).toBeLessThanOrEqual(1_000);
            const verification = renderPlanRequestedVerificationRetrieval(
              unifiedContext([evidence]),
              1,
              1_000,
            );
            expect(verification.replaceAll("&quot;", '"')).toContain(
              `review="${JSON.stringify(expected)}"`,
            );
            expect(verification).toContain('payload_status="check_not_completed_budget"');
            const ledger = await ledgerFor([evidence]);
            const compacted = compactEvidenceLedger(ledger, {
              targetTokens: 8_000,
              hardCapTokens: 8_000,
              maxEntryTextTokens: 40,
            }).ledger;
            const entry = compacted.sections
              .flatMap((section) => section.entries)
              .find((row) => row.state_metadata?.open_question_id === question.id)!;
            expect(entry.state_metadata?.review).toEqual(expected);
            expect(entry.text!.length).toBeLessThan(text.length);
            expect(renderEvidenceLedger(compacted)).toContain(JSON.stringify(expected));
          }
        } finally {
          await harness.cleanup();
        }
      },
    );
  }

  it("charges complete summary labels to the budget and counts omitted labels", () => {
    const evidence = [syntheticEvidence(0), syntheticEvidence(1), syntheticEvidence(2)];
    const unlinked = evidence.map(({ review: _, ...row }) => row);
    const unlinkedPage = summarizeRetrievedEvidence(
      "Evidence",
      unifiedContext([unlinked[0]!]),
      1_000,
    )!;
    const tooSmall = estimatePromptTokens(unlinkedPage) + 1;
    const page = summarizeRetrievedEvidence("Evidence", unifiedContext(evidence), tooSmall)!;
    expect(page).not.toContain("[review=");
    expect(page).toContain("review_labels_omitted_count=3");
    expect(estimatePromptTokens(page)).toBeLessThanOrEqual(tooSmall);
    const onePage = summarizeRetrievedEvidence("Evidence", unifiedContext([evidence[0]!]), 1_000)!;
    const limit = estimatePromptTokens(onePage) + 30;
    const partial = summarizeRetrievedEvidence("Evidence", unifiedContext(evidence), limit)!;
    expect(partial.match(/\[review=/g)).toHaveLength(1);
    expect(partial).toContain(`[review=${JSON.stringify(review)}]`);
    expect(partial).toContain("review_labels_omitted_count=2");
    expect(estimatePromptTokens(partial)).toBeLessThanOrEqual(limit);
  });

  it("counts review labels excluded by the verification membership budget", () => {
    const page = renderPlanRequestedVerificationRetrieval(
      unifiedContext([syntheticEvidence()]),
      1,
      1,
    );
    expect(page).not.toContain(' review="');
    expect(page).toContain('review_labels_omitted_count="1"');
    expect(page).toContain("<omitted_count>1</omitted_count>");
  });

  it.each([400, 20])(
    "counts omitted ledger labels at a %i-token section cap, including a second compaction",
    async (maxTokens) => {
      const ledger = await ledgerFor([syntheticEvidence(0), syntheticEvidence(1)]);
      const options = {
        targetTokens: 8_000,
        hardCapTokens: 8_000,
        sectionOptions: { open_questions: { maxEntries: 0, maxTokens } },
      };
      const first = compactEvidenceLedger(ledger, options).ledger;
      const second = compactEvidenceLedger(first, options).ledger;
      for (const result of [first, second]) {
        const page = renderEvidenceLedger(result)!;
        expect(page).not.toContain(JSON.stringify(review));
        expect(page).toContain('"review_labels_omitted_count":2');
        expect(page).toContain('"review_label_count":2');
        expect(estimatePromptTokens(page)).toBeLessThanOrEqual(8_000);
      }
    },
  );

  it("charges the complete review label against the ledger section budget", async () => {
    const evidence = { ...syntheticEvidence(), text: "Synthetic short question?" };
    const ledger = await ledgerFor([evidence]);
    const section = ledger.sections.find((candidate) => candidate.id === "open_questions")!;
    const unlinked = {
      ...section,
      entries: section.entries.map((entry) => {
        const { review: _, ...metadata } = entry.state_metadata!;
        return { ...entry, state_metadata: metadata };
      }),
    };
    const maxTokens = estimatePromptTokens(renderSection(unlinked));
    const options = { maxEntries: 10, maxTokens };
    const ordinary = capFullLedgerSection({
      section: unlinked,
      options,
      maxEntryTextTokens: 1_000,
    });
    expect(ordinary.section.entries).toHaveLength(1);
    const linked = capFullLedgerSection({ section, options, maxEntryTextTokens: 1_000 });
    expect(linked.section.entries).toHaveLength(0);
    expect(sectionAccountingEntry(linked)?.omission_accounting?.counts).toMatchObject({
      review_label_count: 1,
      review_labels_omitted_count: 1,
    });
  });
});
