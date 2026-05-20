import { afterEach, describe, expect, it, vi } from "vitest";

import type { LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../../cognition/tracing/tracer.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";
import type { Episode } from "../../memory/episodic/index.js";
import { createStreamEntryId, type StreamEntryId } from "../../util/ids.js";
import { createEpisodeFixture, createOfflineTestHarness, createSemanticNodeFixture } from "../test-support.js";
import { ReviewResolverProcess } from "./index.js";

const REVIEW_RESOLVER_TOOL_NAME = "EmitReviewResolverDecision";

type OfflineHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;
type TraceEvent = { event: TurnTraceEventName } & TurnTraceData;

class ArrayTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;
  readonly events: TraceEvent[] = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({
      event,
      ...data,
    });
  }
}

function resolverResponse(input: Record<string, unknown>): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 7,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_review_resolver",
        name: REVIEW_RESOLVER_TOOL_NAME,
        input,
      },
    ],
  };
}

async function runResolver(harness: OfflineHarness, maxItemsPerPass = 3) {
  const process = new ReviewResolverProcess({
    db: harness.db,
    maxItemsPerPass,
  });

  return process.run(harness.createContext(), {});
}

async function insertSource(harness: OfflineHarness, content: string) {
  const entry = await harness.streamWriter.append({
    kind: "user_msg",
    content,
  });
  const episode = await harness.episodicRepository.insert(
    createEpisodeFixture({
      narrative: content,
      source_stream_ids: [entry.id],
    }),
  );

  return { entry, episode };
}

async function insertAssistantSource(harness: OfflineHarness, content: string) {
  const entry = await harness.streamWriter.append({
    kind: "agent_msg",
    content,
  });
  const episode = await harness.episodicRepository.insert(
    createEpisodeFixture({
      narrative: content,
      source_stream_ids: [entry.id],
    }),
  );

  return { entry, episode };
}

function overseerFlag(input: {
  kind: "misattribution" | "identity_inconsistency" | "temporal_drift";
  reason: string;
  sourceEntryId: StreamEntryId;
  sourceEpisodeId: Episode["id"];
  sourceEntryIds?: StreamEntryId[];
  sourceEpisodeIds?: Episode["id"][];
  patch?: Record<string, unknown>;
}) {
  const sourceEntryIds = input.sourceEntryIds ?? [input.sourceEntryId];
  const sourceEpisodeIds = input.sourceEpisodeIds ?? [input.sourceEpisodeId];

  return {
    kind: input.kind,
    flag_kind: input.kind,
    reason: input.reason,
    confidence: 0.9,
    ...(input.patch === undefined ? {} : { patch: input.patch }),
    source_assessment: "supports_flag",
    cited_stream_ids: sourceEntryIds,
    quoted_span: "deployment script",
    audience_entities: [],
    source_episode_ids: sourceEpisodeIds,
    source_stream_ids: sourceEntryIds,
  };
}

async function enqueueSemanticMisattribution(
  harness: OfflineHarness,
  options: {
    descriptionPatch?: string;
    aliasesPatch?: string[];
    sourceEpisodeIdsPatch?: Episode["id"][];
  } = {},
) {
  const source = await insertSource(
    harness,
    "Ben wrote the deployment script and Alice reviewed it.",
  );
  const node = await harness.semanticNodeRepository.insert(
    createSemanticNodeFixture({
      label: "Deployment script authorship",
      description: "Alice wrote the deployment script.",
      source_episode_ids: [source.episode.id],
    }),
  );
  const patch = {
    ...(options.descriptionPatch === undefined
      ? {}
      : { description: options.descriptionPatch }),
    ...(options.aliasesPatch === undefined ? {} : { aliases: options.aliasesPatch }),
    ...(options.sourceEpisodeIdsPatch === undefined
      ? {}
      : { source_episode_ids: options.sourceEpisodeIdsPatch }),
  };
  const item = harness.reviewQueueRepository.enqueue({
    kind: "misattribution",
    reason: "The node attributes the deployment script to the wrong actor.",
    refs: {
      target_type: "semantic_node",
      target_id: node.id,
      patch,
      evidence_stream_ids: [source.entry.id],
      overseer_flag: overseerFlag({
        kind: "misattribution",
        reason: "The node attributes the deployment script to the wrong actor.",
        sourceEntryId: source.entry.id,
        sourceEpisodeId: source.episode.id,
        patch,
      }),
    },
  });

  return {
    item,
    nodeId: node.id,
    sourceEntryId: source.entry.id,
  };
}

async function enqueueEpisodeMisattribution(harness: OfflineHarness) {
  const source = await insertSource(harness, "Ben wrote the deployment helper.");
  const episode = await harness.episodicRepository.insert(
    createEpisodeFixture({
      title: "Deployment helper authorship",
      narrative: "Alice wrote the deployment helper.",
      participants: ["Alice"],
      source_stream_ids: [source.entry.id],
    }),
  );
  const patch = {
    participants: ["Ben"],
  };
  const item = harness.reviewQueueRepository.enqueue({
    kind: "misattribution",
    reason: "The episode attributes the deployment helper to the wrong actor.",
    refs: {
      target_type: "episode",
      target_id: episode.id,
      patch,
      evidence_stream_ids: [source.entry.id],
      overseer_flag: overseerFlag({
        kind: "misattribution",
        reason: "The episode attributes the deployment helper to the wrong actor.",
        sourceEntryId: source.entry.id,
        sourceEpisodeId: source.episode.id,
        patch,
      }),
    },
  });

  return {
    item,
    episodeId: episode.id,
    sourceEntryId: source.entry.id,
  };
}

async function enqueueTaintedSemanticMisattribution(
  harness: OfflineHarness,
  legitimateSource?: Awaited<ReturnType<typeof insertSource>>,
) {
  const tainted = await insertAssistantSource(
    harness,
    "Borg called Priya one of the three siblings.",
  );
  const evidenceSources = legitimateSource === undefined ? [tainted] : [tainted, legitimateSource];
  const patch = {
    description: "Nora and Julian are siblings; Priya is Nora's partner.",
  };
  const node = await harness.semanticNodeRepository.insert(
    createSemanticNodeFixture({
      label: "Family sibling group",
      description: "Nora, Julian, and Priya are the three siblings.",
      source_episode_ids: evidenceSources.map((source) => source.episode.id),
    }),
  );
  const item = harness.reviewQueueRepository.enqueue({
    kind: "misattribution",
    reason: "The node may have laundered Borg's own sibling label.",
    refs: {
      target_type: "semantic_node",
      target_id: node.id,
      patch,
      evidence_stream_ids: evidenceSources.map((source) => source.entry.id),
      reviewed_assistant_stream_entry_id: tainted.entry.id,
      overseer_flag: overseerFlag({
        kind: "misattribution",
        reason: "The node may have laundered Borg's own sibling label.",
        sourceEntryId: tainted.entry.id,
        sourceEpisodeId: tainted.episode.id,
        sourceEntryIds: evidenceSources.map((source) => source.entry.id),
        sourceEpisodeIds: evidenceSources.map((source) => source.episode.id),
        patch,
      }),
    },
  });

  return {
    item,
    node,
    taintedSourceEntryId: tainted.entry.id,
    legitimateSourceEntryId: legitimateSource?.entry.id ?? null,
  };
}

describe("review resolver process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("resolves accepted semantic-node misattribution repairs by superseding instead of inline-patching embedded text", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited source says Ben wrote the script.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.errors).toEqual([]);
    expect(resolved).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "accept",
    });
    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
      corrected_by: sourceEntryId,
    });
    expect(node).toMatchObject({
      status: "superseded",
      corrected_by: sourceEntryId,
      description: "Alice wrote the deployment script.",
    });
    expect(
      tracer.events.find((event) => event.event === "semantic_node.status.transitioned"),
    ).toMatchObject({
      nodeId,
      toStatus: "superseded",
      correctedBy: sourceEntryId,
      source: "review_resolver",
    });
    expect(
      tracer.events.find((event) => event.event === "review_resolver.decision.completed"),
    ).toMatchObject({
      review_id: item.id,
      kind: "misattribution",
      verdict: "accept_repair",
      applied_resolution: "repair_via_supersede",
    });
  });

  it("keeps needs_manual reviews open with a resolver diagnostic", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "The source needs a human canonicalization decision.",
        cited_stream_ids: [],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.changes).toHaveLength(1);
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "The source needs a human canonicalization decision.",
      process: "review-resolver",
    });
    expect(node?.status).toBe("active");
  });

  it("does not accept a repair supported only by the assistant output under review", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, node, taintedSourceEntryId } =
      await enqueueTaintedSemanticMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited utterance says Priya was treated as a sibling.",
        cited_stream_ids: [taintedSourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const storedNode = await harness.semanticNodeRepository.get(node.id);
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(result.candidate_stats).toMatchObject({
      proposed: 1,
      accepted: 0,
    });
    expect(open).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "tainted_assistant_output_under_review_cannot_independently_support_claim",
    });
    expect(storedNode?.status).toBe("active");
    expect(prompt).toContain("evidence_hierarchy");
    expect(prompt).toContain("assistant_output_under_review");
    expect(prompt).toContain("cannot independently support the claim");
  });

  it("does not accept a zero-citation repair", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The repair is asserted without a citation.",
        cited_stream_ids: [],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.candidate_stats).toMatchObject({
      accepted: 0,
    });
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "accept_repair_requires_loaded_non_tainted_citation",
    });
    expect(node?.status).toBe("active");
  });

  it("does not accept a repair cited by tainted and unloaded sources only", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, node, taintedSourceEntryId } =
      await enqueueTaintedSemanticMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The repair cites one tainted source and one unloaded source.",
        cited_stream_ids: [taintedSourceEntryId, createStreamEntryId()],
      }),
    );

    await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const storedNode = await harness.semanticNodeRepository.get(node.id);

    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "tainted_assistant_output_under_review_cannot_independently_support_claim",
    });
    expect(storedNode?.status).toBe("active");
  });

  it("accepts a tainted citation only when a loaded non-tainted citation also supports it", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const legitimate = await insertSource(
      harness,
      "Nora and Julian are siblings; Priya is Nora's partner.",
    );
    const { item, node, taintedSourceEntryId, legitimateSourceEntryId } =
      await enqueueTaintedSemanticMisattribution(harness, legitimate);

    if (legitimateSourceEntryId === null) {
      throw new Error("expected legitimate source entry id");
    }

    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The user source supports the repair independently of the tainted utterance.",
        cited_stream_ids: [taintedSourceEntryId, legitimateSourceEntryId],
        support_basis: "mixed",
      }),
    );

    const result = await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const storedNode = await harness.semanticNodeRepository.get(node.id);

    expect(result.candidate_stats).toMatchObject({
      accepted: 1,
    });
    expect(resolved?.resolution).toBe("accept");
    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
      corrected_by: legitimateSourceEntryId,
    });
    expect(storedNode?.status).toBe("superseded");
    expect(storedNode?.corrected_by).toBe(legitimateSourceEntryId);
  });

  it("allows manual resolution after a needs_manual diagnostic is present", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, episodeId } = await enqueueEpisodeMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "needs_manual",
        reason: "The episode needs a human attribution check.",
        cited_stream_ids: [],
      }),
    );

    await runResolver(harness);
    await expect(harness.reviewQueueRepository.resolve(item.id, "accept")).resolves.toMatchObject({
      resolution: "accept",
    });
    const episode = await harness.episodicRepository.get(episodeId);

    expect(episode?.participants).toEqual(["Ben"]);
  });

  it("routes aliases-only semantic-node patches through supersede", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      aliasesPatch: ["deployment script author"],
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The alias update changes embedded semantic text.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(resolved?.resolution).toBe("accept");
    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
    });
    expect(node?.status).toBe("superseded");
    expect(node?.aliases).toEqual([]);
  });

  it("routes source_episode_ids-only semantic-node patches through supersede", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const replacementSource = await insertSource(
      harness,
      "Ben wrote the deployment script during the release cleanup.",
    );
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      sourceEpisodeIdsPatch: [replacementSource.episode.id],
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "Changing source episodes changes the semantic basis of the node.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(resolved?.refs.__borg_review_resolver_repair).toMatchObject({
      mode: "repair_via_supersede",
    });
    expect(node?.status).toBe("superseded");
    expect(node?.source_episode_ids).not.toEqual([replacementSource.episode.id]);
  });

  it("still applies non-semantic episode patches through the existing handler", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, episodeId, sourceEntryId } = await enqueueEpisodeMisattribution(harness);
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The source clearly identifies Ben as the helper author.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    await runResolver(harness);
    const resolved = harness.reviewQueueRepository.get(item.id);
    const episode = await harness.episodicRepository.get(episodeId);

    expect(resolved?.resolution).toBe("accept");
    expect(episode?.participants).toEqual(["Ben"]);
  });

  it("fails open and emits degraded when the LLM call fails", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(result.errors).toHaveLength(1);
    expect(
      tracer.events.find((event) => event.event === "review_resolver.degraded"),
    ).toMatchObject({
      review_id: item.id,
    });
  });

  it("fails open when the repair handler rejects the accepted repair", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, sourceEntryId } = await enqueueEpisodeMisattribution(harness);
    vi.spyOn(harness.episodicRepository, "update").mockRejectedValue(
      new Error("repair handler failed"),
    );
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The alias patch is supported by the source.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(result.errors).toHaveLength(1);
    expect(
      tracer.events.find((event) => event.event === "review_resolver.degraded"),
    ).toMatchObject({
      review_id: item.id,
      reason: "repair handler failed",
    });
  });

  it("rolls back semantic supersede when review queue finalization fails", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const { item, nodeId, sourceEntryId } = await enqueueSemanticMisattribution(harness, {
      descriptionPatch: "Ben wrote the deployment script.",
    });
    const originalPrepare = harness.db.prepare.bind(harness.db);
    vi.spyOn(harness.db, "prepare").mockImplementation((sql: string) => {
      if (
        sql.indexOf("UPDATE review_queue") >= 0 &&
        sql.indexOf("resolved_at = ?") >= 0
      ) {
        return {
          run() {
            throw new Error("queue update failed");
          },
        } as never;
      }

      return originalPrepare(sql);
    });
    llm.pushResponse(
      resolverResponse({
        verdict: "accept_repair",
        reason: "The cited source says Ben wrote the script.",
        cited_stream_ids: [sourceEntryId],
      }),
    );

    const result = await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);
    const node = await harness.semanticNodeRepository.get(nodeId);

    expect(result.errors).toHaveLength(1);
    expect(open?.resolved_at).toBeNull();
    expect(node?.status).toBe("active");
  });

  it("processes at most the configured cap per pass", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const items: ReviewQueueItem[] = [];

    for (let index = 0; index < 5; index += 1) {
      const source = await insertSource(
        harness,
        `Ben wrote deployment helper ${index}; Alice reviewed it.`,
      );
      const episode = await harness.episodicRepository.insert(
        createEpisodeFixture({
          title: `Deployment helper ${index}`,
          narrative: `Alice wrote deployment helper ${index}.`,
          participants: ["Alice"],
          source_stream_ids: [source.entry.id],
        }),
      );
      items.push(
        harness.reviewQueueRepository.enqueue({
          kind: "misattribution",
          reason: "Episode participant attribution needs review.",
          refs: {
            target_type: "episode",
            target_id: episode.id,
            patch: {
              participants: ["Ben"],
            },
            evidence_stream_ids: [source.entry.id],
            overseer_flag: overseerFlag({
              kind: "misattribution",
              reason: "Episode participant attribution needs review.",
              sourceEntryId: source.entry.id,
              sourceEpisodeId: source.episode.id,
              patch: {
                participants: ["Ben"],
              },
            }),
          },
        }),
      );
      llm.pushResponse(
        resolverResponse({
          verdict: "dismiss_false_positive",
          reason: "The cited source does not require a repair.",
          cited_stream_ids: [],
        }),
      );
    }

    const result = await runResolver(harness, 3);
    const resolved = items.filter(
      (item) => harness.reviewQueueRepository.get(item.id)?.resolved_at !== null,
    );
    const open = items.filter(
      (item) => harness.reviewQueueRepository.get(item.id)?.resolved_at === null,
    );

    expect(result.changes).toHaveLength(3);
    expect(resolved).toHaveLength(3);
    expect(open).toHaveLength(2);
    expect(llm.requests).toHaveLength(3);
  });

  it("stubs valid identity inconsistency refs as needs_manual without judging", async () => {
    const llm = new FakeLLMClient();
    const tracer = new ArrayTracer();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      tracer,
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const evidenceEpisode = createEpisodeFixture().id;
    const value = harness.valuesRepository.add({
      label: "groundedness",
      description: "Stay grounded in cited evidence.",
      priority: 5,
      provenance: {
        kind: "manual",
      },
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      reason: "The value has new supporting evidence.",
      refs: {
        target_type: "value",
        target_id: value.id,
        repair_op: "reinforce",
        evidence_episode_ids: [evidenceEpisode],
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
    });

    await runResolver(harness);
    const open = harness.reviewQueueRepository.get(item.id);

    expect(open?.resolved_at).toBeNull();
    expect(open?.refs.__borg_review_resolver_diagnostic).toMatchObject({
      verdict: "needs_manual",
      reason: "identity_inconsistency_auto_resolution_not_yet_supported",
    });
    expect(
      tracer.events.find((event) => event.event === "review_resolver.decision.completed"),
    ).toMatchObject({
      verdict: "needs_manual",
      reason: "identity_kind_not_yet_supported",
    });
    expect(llm.requests).toHaveLength(0);
  });
});
