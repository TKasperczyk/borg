import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { FakeLLMClient } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import { EpisodicExtractor } from "../episodic/index.js";

import { AffectiveExtractor } from "./extractor.js";

const AFFECTIVE_TOOL_NAME = "EmitAffectiveSignal";
const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

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

  it("treats short gratitude follow-ups as positive enough for lagged attribution", async () => {
    const extractor = new AffectiveExtractor();

    expect((await extractor.analyze("thanks")).valence).toBeGreaterThan(0.2);
    expect((await extractor.analyze("thanks!")).valence).toBeGreaterThan(0.2);
    expect((await extractor.analyze("okay, thanks")).valence).toBeGreaterThan(0.2);
    expect((await extractor.analyze("okay")).valence).toBeLessThan(0.2);
  });

  it("uses the llm as the primary affect classifier when configured", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 8,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: AFFECTIVE_TOOL_NAME,
              input: {
                valence: -0.45,
                arousal: 0.35,
                dominant_emotion: "anger",
              },
            },
          ],
        },
      ],
    });
    const extractor = new AffectiveExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const signal = await extractor.analyze("Yeah great, exactly what I wanted");

    expect(signal).toMatchObject({
      dominant_emotion: "anger",
      valence: -0.45,
      arousal: 0.35,
    });
    expect(llm.requests).toHaveLength(1);
    expect(llm.requests[0]?.budget).toBe("perception-affective");
  });

  it("caps llm affective calls for one extractor instance", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 8,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: AFFECTIVE_TOOL_NAME,
              input: {
                valence: -0.45,
                arousal: 0.35,
                dominant_emotion: "anger",
              },
            },
          ],
        },
      ],
    });
    const extractor = new AffectiveExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await extractor.analyze("Yeah great, exactly what I wanted");
    const capped = await extractor.analyze("Sure, perfect.");

    expect(llm.requests).toHaveLength(1);
    expect(capped).toMatchObject({
      valence: 0,
      dominant_emotion: "neutral",
    });
  });

  it("does not let coding-error lexicon words override the llm signal", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 10,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: AFFECTIVE_TOOL_NAME,
              input: {
                valence: 0,
                arousal: 0.2,
                dominant_emotion: "neutral",
              },
            },
          ],
        },
      ],
    });
    const extractor = new AffectiveExtractor({
      llmClient: llm,
      model: "haiku",
      useLlmFallback: true,
    });

    const signal = await extractor.analyze("The build is blocked by a broken test error.");

    expect(signal).toMatchObject({
      dominant_emotion: "neutral",
      valence: 0,
      arousal: 0.2,
    });
    expect(llm.requests).toHaveLength(1);
  });

  it("uses the llm for long ambiguous text when enabled", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 10,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: AFFECTIVE_TOOL_NAME,
              input: {
                valence: -0.2,
                arousal: 0.4,
                dominant_emotion: "curiosity",
              },
            },
          ],
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
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: AFFECTIVE_TOOL_NAME,
    });
  });

  it("stores LLM-emitted emotional arcs during episodic extraction", async () => {
    const llm = new FakeLLMClient();
    harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    const writer = new StreamWriter({
      dataDir: harness.tempDir,
      sessionId: "default" as never,
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
      text: "",
      input_tokens: 20,
      output_tokens: 20,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_1",
          name: EPISODE_TOOL_NAME,
          input: {
            episodes: [
              {
                title: "Rust lifetime debugging spiral",
                narrative: "The user struggled with Rust lifetimes and eventually found a fix.",
                source_stream_ids: [firstId, secondId],
                participants: ["user"],
                location: null,
                tags: ["rust", "debugging"],
                emotional_arc: {
                  start: {
                    valence: -0.6,
                    arousal: 0.5,
                  },
                  peak: {
                    valence: -0.7,
                    arousal: 0.6,
                  },
                  end: {
                    valence: -0.2,
                    arousal: 0.25,
                  },
                  dominant_emotion: "anger",
                },
                confidence: 0.8,
                significance: 0.7,
              },
            ],
          },
        },
      ],
    });

    const extractor = new EpisodicExtractor({
      dataDir: harness.tempDir,
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      llmClient: harness.llmClient,
      model: "haiku",
      entityRepository: harness.entityRepository,
      clock: harness.clock,
    });

    await extractor.extractFromStream();
    const [episode] = await harness.episodicRepository.listAll();

    expect(episode?.emotional_arc).not.toBeNull();
    expect(episode?.emotional_arc?.start.valence).toBe(-0.6);
  });
});
