import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { FakeLLMClient } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import { EpisodicExtractor } from "../episodic/index.js";

import { AffectiveExtractor } from "./extractor.js";

describe("AffectiveExtractor", () => {
  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("captures joy, sadness, and arousal cues heuristically", async () => {
    const extractor = new AffectiveExtractor();

    await expect(extractor.analyze("I am thrilled this works!")).resolves.toMatchObject({
      dominant_emotion: "joy",
    });
    await expect(extractor.analyze("I feel sad and stuck...")).resolves.toMatchObject({
      dominant_emotion: "sadness",
    });

    const calm = await extractor.analyze("okay");
    const intense = await extractor.analyze("THIS IS BROKEN!!!");

    expect(calm.arousal).toBeLessThan(intense.arousal);
    expect(intense.valence).toBeLessThan(0);
  });

  it("falls back to the llm for long ambiguous text when enabled", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: JSON.stringify({
            valence: -0.2,
            arousal: 0.4,
            dominant_emotion: "curiosity",
          }),
          input_tokens: 10,
          output_tokens: 10,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const extractor = new AffectiveExtractor({
      llmClient: llm,
      model: "haiku",
      useLlmFallback: true,
    });

    const signal = await extractor.analyze(
      "This extended neutral paragraph discusses connectors adapters fixtures migrations selectors repositories pipelines orchestrators prompts contracts registries contexts boundaries schemas interfaces daemons budgets audits reviews cadences provenance vectors aliases wrappers snapshots cursors sectors batches windows clusters layers hooks policies reversers scoring heuristics embeddings identities semantics maintenance telemetry and serialization without any obvious emotional vocabulary.",
    );

    expect(signal).toMatchObject({
      dominant_emotion: "curiosity",
      valence: -0.2,
      arousal: 0.4,
    });
  });

  it("populates emotional arcs during episodic extraction", async () => {
    const llm = new FakeLLMClient();
    harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default",
      clock: harness.clock,
    });
    let firstId = "";
    let secondId = "";

    try {
      firstId = (
        await writer.append({
          kind: "user_msg",
          content: "I am frustrated and stuck with Rust lifetimes.",
        })
      ).id;
      secondId = (
        await writer.append({
          kind: "agent_msg",
          content: "Great, that fix works now!",
        })
      ).id;
    } finally {
      writer.close();
    }

    llm.pushResponse({
      text: JSON.stringify({
        episodes: [
          {
            title: "Rust lifetime debugging spiral",
            narrative: "The user struggled with Rust lifetimes and eventually found a fix.",
            source_stream_ids: [firstId, secondId],
            participants: ["user"],
            tags: ["rust", "debugging"],
            confidence: 0.8,
            significance: 0.7,
          },
        ],
      }),
      input_tokens: 20,
      output_tokens: 20,
      stop_reason: "end_turn",
      tool_calls: [],
    });

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: harness.llmClient,
      model: "haiku",
      clock: harness.clock,
    });

    await extractor.extractFromStream();
    const [episode] = await harness.episodicRepository.listAll();

    expect(episode?.emotional_arc).not.toBeNull();
    expect(episode?.emotional_arc?.start.valence ?? 0).toBeLessThan(0);
  });
});
