import type { OpenQuestion } from "./types.js";
import { createWorkingMemoryFixture } from "../../offline/test-support.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

export const REVIEW_TEST_NOW = Date.parse("2026-09-02T12:00:00.000Z");

// Load the same public renderers from either this tree or the frozen dev source.
export async function renderQuestionSurfaces(root: string, questions: readonly OpenQuestion[]) {
  const load = async (path: string) => import(/* @vite-ignore */ `${root}/src/${path}.ts`);
  const workingMemory = createWorkingMemoryFixture({ updated_at: REVIEW_TEST_NOW });
  const { EvidenceLedgerBuilder } = await load("cognition/evidence-ledger/builder");
  const ledger = await new EvidenceLedgerBuilder({
    createStreamReader: () => ({
      scanReverse: () => ({ entries: [], scannedBytes: 0, capReached: null }),
    }),
    relationalSlotRepository: { list: () => [] },
    actionRepository: { list: () => [] },
    currentSessionTranscriptTokenBudget: 2_000,
  }).build({
    sessionId: DEFAULT_SESSION_ID,
    audienceEntityId: null,
    currentUserMessage: "Synthetic current turn",
    nowMs: REVIEW_TEST_NOW,
    workingMemory,
    applicableCommitments: [],
    openQuestions: questions,
    pendingCorrections: [],
    retrievedEpisodes: [],
    retrievedSemantic: null,
    retrievedEvidence: [],
  });
  const { renderEvidenceLedger } = await load("cognition/evidence-ledger/finalizer-ledger");
  const { compactEvidenceLedger } = await load("cognition/evidence-ledger/compaction");
  const { renderCompactPlannerLedger } = await load("cognition/evidence-ledger/compact-planner");
  const context = {
    sessionId: DEFAULT_SESSION_ID,
    nowMs: REVIEW_TEST_NOW,
    userMessage: "Synthetic current turn",
    turnOrigin: "autonomous",
    audience: "self",
    isSelfAudience: true,
    perception: {
      entities: [],
      mode: "reflective",
      affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
      temporalCue: null,
    },
    retrievalResult: [],
    workingMemory,
    selfSnapshot: { values: [], goals: [], traits: [] },
    openQuestionsContext: questions,
    evidenceLedger: ledger,
  };
  const { buildCompactPlannerSystemPrompt } = await load(
    "cognition/deliberation/prompt/planner-context",
  );
  const { buildBaseSystemPrompt } = await load("cognition/deliberation/prompt/system-prompt");
  const { summarizeRetrievedEvidence, renderPlanRequestedVerificationRetrieval } = await load(
    "cognition/deliberation/prompt/retrieval",
  );
  const { createOpenQuestionDormantTrigger } = await load(
    "autonomy/triggers/open-question-dormant",
  );
  const trigger = createOpenQuestionDormantTrigger({
    clock: { now: () => REVIEW_TEST_NOW },
    dormantMs: 1,
    openQuestionsRepository: { list: () => questions },
    watermarkRepository: { get: () => null },
  });
  const events = await trigger.scan();
  const { formatAutonomyTriggerContext } = await load("cognition/autonomy-trigger");
  const wake =
    events.length === 0
      ? ""
      : formatAutonomyTriggerContext(trigger.buildTurn(events[0]).autonomyTrigger);
  const { openQuestionDuplicatePresentationRow } = await load(
    "memory/self/open-question-duplicates",
  );
  const { createOpenQuestionsRuminationsTool } = await load(
    "tools/internal/open-questions-ruminations",
  );
  const tool = createOpenQuestionsRuminationsTool({
    clock: { now: () => REVIEW_TEST_NOW },
    listRuminations: () =>
      questions.map((q, index) => ({
        id: index + 1,
        open_question_id: q.id,
        note: "Synthetic rumination.",
        tensions: [],
        connected_open_question_ids: [],
        evidence_episode_ids: [],
        evidence_stream_entry_ids: [],
        source_process: "synthetic",
        source_run_id: null,
        source_turn_id: null,
        provenance: { kind: "manual" },
        created_at: REVIEW_TEST_NOW - 1,
      })),
    getOpenQuestion: (id: string) => questions.find((q) => q.id === id) ?? null,
    countRuminations: () => 1,
    mergeTombstonePrimaryId: () => null,
  });
  return {
    ledger: renderEvidenceLedger(ledger),
    compacted_ledger: renderEvidenceLedger(compactEvidenceLedger(ledger).ledger),
    planner_ledger: renderCompactPlannerLedger(ledger),
    planner: buildCompactPlannerSystemPrompt({
      context,
      staticPrefix: "SYNTHETIC",
      compactPlannerLedger: null,
    })
      .system.map((b: { text: string }) => b.text)
      .join("\n"),
    legacy_planner: buildBaseSystemPrompt(
      { ...context, evidenceLedger: null },
      { retrievalContextBudget: 3000, semanticContextBudget: 3000 },
    ),
    retrieval: summarizeRetrievedEvidence("Recall", { openQuestions: questions }),
    verification: renderPlanRequestedVerificationRetrieval(
      {
        evidence: [],
        episodes: [],
        semantic: {
          supports: [],
          contradicts: [],
          categories: [],
          matched_node_ids: [],
          matched_nodes: [],
          support_hits: [],
          contradiction_hits: [],
          category_hits: [],
          causal_hits: [],
        },
        open_questions: questions,
      },
      3_000,
      3_000,
      { rowsTotalReadAtMs: REVIEW_TEST_NOW, currentTimeMs: REVIEW_TEST_NOW },
    ),
    wake,
    duplicates: JSON.stringify(questions.map(openQuestionDuplicatePresentationRow)),
    tool: JSON.stringify(
      tool.outputSchema.parse(
        await tool.invoke(
          tool.inputSchema.parse({
            since: new Date(0).toISOString(),
            until: new Date(REVIEW_TEST_NOW).toISOString(),
          }),
          { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" },
        ),
      ),
    ),
  } as Record<string, string>;
}
