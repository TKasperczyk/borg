import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { AnnotationTarget, AuditedAnnotation } from "../memory/identity/annotations.js";
import type { EpisodeId } from "../util/ids.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import type { createRecordEvidenceFixture } from "./record-evidence-test-support.js";

/** Executes the public renderers and captures the online reflector's model input. */
export async function renderOrdinaryRecordSurfaces(
  root: string,
  fixture: ReturnType<typeof createRecordEvidenceFixture>,
  annotationsForTarget?: (target: AnnotationTarget) => AuditedAnnotation[],
) {
  const load = async (path: string) => import(/* @vite-ignore */ `${root}/src/${path}.ts`);
  const reader = {
    identityEventRepository: {
      listAnnotationsForTargets: (targets: readonly AnnotationTarget[]) =>
        targets.map((target) => ({ ...target, annotations: annotationsForTarget?.(target) ?? [] })),
    },
    episodicRepository: {
      getLifecycleMany: (ids: readonly EpisodeId[]) =>
        new Map(ids.map((id) => [id, "active" as const])),
    },
  };
  const { EvidenceLedgerBuilder } = await load("cognition/evidence-ledger/builder");
  const built = await new EvidenceLedgerBuilder({
    recordEvidenceReader: reader,
    createStreamReader: () => ({
      scanReverse: () => ({ entries: [], scannedBytes: 0, capReached: null }),
    }),
    relationalSlotRepository: { list: () => [] },
    actionRepository: { list: () => [] },
    currentSessionTranscriptTokenBudget: 2_000,
  }).build(fixture.input);
  const { renderEvidenceLedger } = await load("cognition/evidence-ledger/finalizer-ledger");
  const { compactEvidenceLedger } = await load("cognition/evidence-ledger/compaction");
  const { renderCompactPlannerLedger } = await load("cognition/evidence-ledger/compact-planner");
  const { renderSection } = await load("cognition/evidence-ledger/section-rendering");
  const surfaces: Record<string, string> = {
    ledger: renderEvidenceLedger(built),
    compacted_ledger: renderEvidenceLedger(compactEvidenceLedger(built).ledger),
    planner_ledger: renderCompactPlannerLedger(built),
  };
  for (const section of built.sections)
    if (
      [
        "episodes",
        "semantic_graph",
        "retrieved_memory_evidence",
        "autobiographical_recall",
      ].includes(section.id)
    )
      surfaces[section.id] = renderSection(section);
  const retrieval = await load("cognition/deliberation/prompt/retrieval");
  const records =
    annotationsForTarget === undefined
      ? fixture
      : await (
          await load("cognition/record-evidence")
        ).withRetrievedRecordEvidence(reader, fixture);
  surfaces.legacy_episodes = retrieval.summarizeRetrievedEpisodes("Recall", records.episodes);
  surfaces.legacy_semantic = retrieval.summarizeSemanticContext(records.semantic, 3_000, 2_000_000);
  surfaces.unified_retrieval = retrieval.summarizeRetrievedEvidence("Recall", records);
  surfaces.additional_fallback = retrieval.summarizeRetrievedEvidence("Additional", {
    ...records,
    evidence: [],
  });
  surfaces.verification = retrieval.renderPlanRequestedVerificationRetrieval(
    { ...records, open_questions: [] },
    3_000,
    3_000,
    { rowsTotalReadAtMs: 2_000_000, currentTimeMs: 2_000_000 },
  );
  const invocation = { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" };
  const { createEpisodicGetTool } = await load("tools/internal/episodic-get");
  const get = createEpisodicGetTool({
    recordEvidenceReader: reader,
    episodicRepository: {
      get: async () => fixture.episodes[0]!.episode,
      getStats: () => null,
      isEpisodeEffectivelyVisible: () => true,
      getLifecycleMany: reader.episodicRepository.getLifecycleMany,
    },
  });
  surfaces.tool_get = JSON.stringify(
    get.outputSchema.parse(
      await get.invoke({ episode_id: fixture.episodes[0]!.episode.id }, invocation),
    ),
  );
  const { createEpisodicSearchTool } = await load("tools/internal/episodic-search");
  const search = createEpisodicSearchTool({
    recordEvidenceReader: reader,
    searchEpisodes: async () => fixture.episodes,
  });
  const searchOutput = search.outputSchema.parse(
    await search.invoke({ query: "synthetic" }, invocation),
  );
  surfaces.tool_search = JSON.stringify(searchOutput);
  const { createEpisodicRecentTool } = await load("tools/internal/episodic-recent");
  const recent = createEpisodicRecentTool({
    recordEvidenceReader: reader,
    listRecentEpisodes: async () => fixture.episodes,
  });
  surfaces.tool_recent = JSON.stringify(
    recent.outputSchema.parse(await recent.invoke({}, invocation)),
  );
  const { createSemanticWalkTool } = await load("tools/internal/semantic-walk");
  const walk = createSemanticWalkTool({
    recordEvidenceReader: reader,
    walkGraph: async () => [{ node: fixture.node, edgePath: [] }],
  });
  surfaces.tool_walk = JSON.stringify(
    walk.outputSchema.parse(
      await walk.invoke({ node_id: fixture.node.id, relation: "supports" }, invocation),
    ),
  );
  const { formatAutonomyTriggerContext } = await load("cognition/autonomy-trigger");
  const wake = {
    source_name: "open_question_dormant",
    source_type: "trigger",
    event_id: "synthetic-wake",
    sort_ts: 2_000_000,
    payload: {
      open_question_id: "oq_synthetic",
      question: "Synthetic question?",
      last_touched: 1_000_000,
      related_episodes: searchOutput.episodes,
    },
  };
  surfaces.wake = formatAutonomyTriggerContext(wake);
  surfaces.bounded_wake = formatAutonomyTriggerContext(wake, 2_000);
  const { Reflector } = await load("cognition/reflection/reflector");
  const llm = new FakeLLMClient({
    responses: [
      {
        text: "",
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "synthetic-reflection",
            name: "EmitTurnReflection",
            input: {
              advanced_goals: [],
              procedural_outcomes: [],
              trait_demonstrations: [],
              intent_updates: [],
              step_outcomes: [],
              proposed_steps: [],
              open_questions: [],
              resolved_open_questions: [],
              retired_goals: [],
            },
          },
        ],
      },
    ],
  });
  const reflector = new Reflector({
    recordEvidenceReader: reader,
    llmClient: llm,
    model: "synthetic-model",
    episodicRepository: {},
    goalsRepository: {},
    traitsRepository: {},
  });
  await reflector.runReflectionJudgment(
    {
      userMessage: "Synthetic question",
      actionResult: { response: "Synthetic response", tool_calls: [] },
      workingMemory: fixture.input.workingMemory,
      selfSnapshot: { values: [], goals: [], traits: [] },
      retrievedEpisodes: fixture.episodes,
    },
    { shownResolutions: new Map(), fields: {} },
  );
  surfaces.online_reflector = String(llm.requests[0]!.messages[0]!.content);
  return surfaces;
}
