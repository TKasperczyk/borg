import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { FixedClock } from "../../util/clock.js";
import { EntityExtractor } from "./entity-extractor.js";
import { ModeDetector } from "./mode-detector.js";
import { Perceiver, runPerceptionClassifierSafely } from "./perceive.js";
import { detectTemporalCue } from "./temporal-cue.js";

const ENTITY_TOOL_NAME = "EmitEntityExtraction";
const MODE_TOOL_NAME = "EmitModeDetection";

function invalidEntityResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: ENTITY_TOOL_NAME,
        input: { entities: [1] },
      },
    ],
  };
}

function invalidModeResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: MODE_TOOL_NAME,
        input: { mode: "unknown" },
      },
    ],
  };
}

function entityResponse(entities: readonly unknown[], userIdentityNames: readonly string[] = []) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: ENTITY_TOOL_NAME,
        input: { entities, user_identity_names: userIdentityNames },
      },
    ],
  };
}

function modeResponse(mode: string, isOperational = false) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: MODE_TOOL_NAME,
        input: { mode, is_operational: isOperational },
      },
    ],
  };
}

function temporalCueResponse(input: { since?: string; until?: string; label?: string }) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_temporal",
        name: "EmitTemporalCue",
        input: { has_cue: true, ...input },
      },
    ],
  };
}

describe("perception", () => {
  it("returns classifier results without notifying on the safe-wrapper success path", async () => {
    const onFailure = vi.fn();

    const result = await runPerceptionClassifierSafely({
      classifier: "mode_detector",
      run: async () => "reflective",
      fallback: () => "idle",
      onFailure,
    });

    expect(result).toBe("reflective");
    expect(onFailure).not.toHaveBeenCalled();
  });

  it("returns the fallback and notifies on the safe-wrapper failure path", async () => {
    const error = new Error("classifier exploded");
    const onFailure = vi.fn();

    const result = await runPerceptionClassifierSafely({
      classifier: "entity_extractor",
      run: async () => {
        throw error;
      },
      fallback: () => ["Atlas"],
      onFailure,
    });

    expect(result).toEqual(["Atlas"]);
    expect(onFailure).toHaveBeenCalledWith({
      classifier: "entity_extractor",
      error,
    });
  });

  it("returns empty entities when no LLM client is configured", async () => {
    // Perception entity extraction is LLM-only: the previous regex
    // heuristic produced false-positive entities at high rates
    // ('Good', 'If', '[End.]') that poisoned downstream retrieval.
    // Without an LLM the honest signal is empty.
    const extractor = new EntityExtractor();

    expect(await extractor.extractEntities("Jane Doe said yesterday was rough")).toEqual([]);
  });

  it("keeps language-neutral entity sanitizer checks only", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_entity",
              name: ENTITY_TOOL_NAME,
              input: {
                entities: ["李", "the", "Human:", "[end]", "!!!"],
              },
            },
          ],
        },
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });

    expect(await extractor.extractEntities("irrelevant")).toEqual(["李", "the", "Human:", "[end]"]);
  });

  it("defaults to idle when no LLM client is configured", async () => {
    // The heuristic tier was removed; without an LLM, the safe neutral
    // default is "idle" (skips S2 planning, uses default retrieval weights).
    const detector = new ModeDetector({ llmEnabled: false });
    await expect(detector.detectMode("pnpm build throws an error trace")).resolves.toEqual({
      mode: "idle",
      isOperational: false,
    });
    await expect(detector.detectMode("Why do I keep avoiding this?")).resolves.toEqual({
      mode: "idle",
      isOperational: false,
    });
    await expect(detector.detectMode("ok")).resolves.toEqual({
      mode: "idle",
      isOperational: false,
    });
  });

  it("classifies every message via the LLM when configured", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: ENTITY_TOOL_NAME,
              input: { entities: ["Atlas"] },
            },
          ],
        },
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_2",
              name: MODE_TOOL_NAME,
              input: { mode: "reflective", is_operational: true },
            },
          ],
        },
      ],
    });
    const entityExtractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });
    const modeDetector = new ModeDetector({
      llmClient: llm,
      model: "haiku",
      llmEnabled: true,
    });

    expect(await entityExtractor.extractEntities("something vague")).toEqual(["Atlas"]);
    expect(await modeDetector.detectMode("maybe this", [])).toEqual({
      mode: "reflective",
      isOperational: true,
    });
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: ENTITY_TOOL_NAME,
    });
    expect(llm.requests[1]?.tool_choice).toEqual({
      type: "tool",
      name: MODE_TOOL_NAME,
    });
  });

  it("surfaces the LLM operational-turn signal on perception results", async () => {
    const llm = new FakeLLMClient({
      responses: [entityResponse(["Atlas"]), modeResponse("problem_solving", true)],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveLlmEnabled: false,
      temporalCueLlmEnabled: false,
    });

    const perceived = await perceiver.perceive("Recap the locked Atlas deployment state.");

    expect(perceived.mode).toBe("problem_solving");
    expect(perceived.isOperational).toBe(true);
  });

  it("passes the full message text to the entity LLM without truncation", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: ENTITY_TOOL_NAME,
              input: { entities: ["pgvector", "qdrant"] },
            },
          ],
        },
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });
    const longLowercaseText = `${"pgvector qdrant ".repeat(180)}borg memory index drift`;

    const entities = await extractor.extractEntities(longLowercaseText);

    expect(entities).toEqual(["pgvector", "qdrant"]);
    expect(llm.requests[0]?.messages[0]?.content).toBe(longLowercaseText);
  });

  it("returns the LLM entity payload after output sanitization", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: ENTITY_TOOL_NAME,
              input: { entities: ["Sam", "bicycles"] },
            },
          ],
        },
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });

    expect(await extractor.extractEntities("yesterday Sam mentioned bicycles")).toEqual([
      "Sam",
      "bicycles",
    ]);
    expect(llm.requests).toHaveLength(1);
  });

  it("surfaces LLM-declared user identity names separately from entity mentions", async () => {
    const llm = new FakeLLMClient({
      responses: [entityResponse(["Tom"], ["Tom"])],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });

    expect(await extractor.extract("I'm Tom")).toEqual({
      entities: ["Tom"],
      entityMentions: [{ name: "Tom", kind: "person" }],
      userIdentityNames: ["Tom"],
    });
  });

  it("accepts typed entity kinds and defaults omitted kinds to person", async () => {
    const llm = new FakeLLMClient({
      responses: [
        entityResponse([
          { name: "planning-room", kind: "group" },
          { name: "Atlas", kind: "abstract" },
          { name: "Maya" },
        ]),
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });

    expect(await extractor.extract("planning-room discussed Atlas with Maya")).toMatchObject({
      entities: ["planning-room", "Atlas", "Maya"],
      entityMentions: [
        { name: "planning-room", kind: "group" },
        { name: "Atlas", kind: "abstract" },
        { name: "Maya", kind: "person" },
      ],
    });
  });

  it("falls back to idle when mode detection throws", async () => {
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [entityResponse(["Atlas"]), invalidModeResponse()],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveLlmEnabled: false,
      temporalCueLlmEnabled: false,
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive("plain lower text");

    expect(perceived.mode).toBe("idle");
    expect(perceived.isOperational).toBe(false);
    expect(perceived.entities).toEqual(["Atlas"]);
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "mode_detector",
        error: expect.any(Error),
      }),
    );
  });

  it("falls back to empty entities when entity extraction throws", async () => {
    // The previous regex-heuristic fallback was removed -- it
    // produced false-positive entities at high rates that poisoned
    // downstream retrieval. Empty entities is the honest signal on
    // failure. The turn proceeds; mode and other classifiers run
    // independently per Promise.all.
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [invalidEntityResponse(), modeResponse("problem_solving")],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveLlmEnabled: false,
      temporalCueLlmEnabled: false,
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive(
      'Talk to @alice about "Project Atlas" with Jane Doe.',
    );

    expect(perceived.entities).toEqual([]);
    expect(perceived.mode).toBe("problem_solving");
    expect(perceived.isOperational).toBe(false);
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "entity_extractor",
        error: expect.any(Error),
      }),
    );
  });

  it("degrades mode and entities independently when both classifiers throw", async () => {
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [invalidEntityResponse(), invalidModeResponse()],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveLlmEnabled: false,
      temporalCueLlmEnabled: false,
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive('Meet @alice about "Project Atlas".');

    expect(perceived.entities).toEqual([]);
    expect(perceived.mode).toBe("idle");
    expect(perceived.isOperational).toBe(false);
    expect(onClassifierFailure).toHaveBeenCalledTimes(2);
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "entity_extractor",
        error: expect.any(Error),
      }),
    );
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "mode_detector",
        error: expect.any(Error),
      }),
    );
  });

  it("keeps degraded perception structurally identical to successful perception", async () => {
    const successful = await new Perceiver({
      llmClient: new FakeLLMClient({
        responses: [entityResponse(["Atlas"]), modeResponse("reflective")],
      }),
      model: "haiku",
      affectiveLlmEnabled: false,
      temporalCueLlmEnabled: false,
    }).perceive("plain lower text");
    const degraded = await new Perceiver({
      llmClient: new FakeLLMClient({
        responses: [invalidEntityResponse(), invalidModeResponse()],
      }),
      model: "haiku",
      affectiveLlmEnabled: false,
      temporalCueLlmEnabled: false,
    }).perceive("plain lower text");

    expect(Object.keys(degraded).sort()).toEqual(Object.keys(successful).sort());
    expect({
      ...degraded,
      entities: successful.entities,
      entityMentions: successful.entityMentions,
      mode: successful.mode,
      isOperational: successful.isOperational,
    }).toEqual(successful);
  });

  it("produces a perception result with empty entities and null temporal cue when no LLM is configured", async () => {
    // Without an LLM, perception's entity extraction returns []
    // (heuristic was removed -- it produced too many false positives
    // like 'Good', 'If', '[End.]'). Mode degrades to 'idle' and the
    // temporal cue degrades to null. The turn proceeds with an
    // empty perception payload rather than confidently-wrong tags.
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const perceiver = new Perceiver({
      llmEnabled: false,
      clock: new FixedClock(nowMs),
    });
    const perceived = await perceiver.perceive("Jane Doe said yesterday was rough");

    expect(perceived.entities).toEqual([]);
    expect(perceived.temporalCue).toBeNull();
    expect(perceived.mode).toBe("idle");
    expect(perceived.isOperational).toBe(false);
    expect(perceived.affectiveSignal).toEqual({
      valence: 0,
      arousal: 0,
      dominant_emotion: null,
    });
    expect(perceived.affectiveSignalDegraded).toBe(true);
  });

  it("notifies when temporal cue extraction degrades without an LLM", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const onDegraded = vi.fn();

    const cue = await detectTemporalCue("Jane said yesterday was rough", nowMs, {
      onDegraded,
    });

    expect(cue).toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_unavailable");
  });

  it("emits the perception degraded trace when temporal cue extraction fails", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        entityResponse(["Jane"]),
        modeResponse("reflective"),
        () => {
          throw new Error("temporal unavailable");
        },
      ],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveLlmEnabled: false,
      clock: new FixedClock(nowMs),
      tracer,
      turnId: "turn-1",
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive("Jane said yesterday was rough");

    expect(perceived.temporalCue).toBeNull();
    expect(tracer.emit).toHaveBeenCalledWith("perception.classifier.degraded", {
      turnId: "turn-1",
      classifier: "temporal_cue",
      reason: "llm_failed",
    });
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "temporal_cue",
        error: expect.any(Error),
      }),
    );
  });

  it("routes entity_extractor and temporal_cue to fastModel while mode_detector stays on model", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const llm = new FakeLLMClient({
      responses: [
        entityResponse(["Atlas"]),
        modeResponse("problem_solving"),
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_temporal",
              name: "EmitTemporalCue",
              input: { has_cue: false },
            },
          ],
        },
      ],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "background-opus",
      fastModel: "haiku-fast",
      affectiveLlmEnabled: false,
      clock: new FixedClock(nowMs),
    });

    await perceiver.perceive("Plain lower text about Atlas yesterday.");

    const byBudget = new Map(llm.requests.map((request) => [request.budget, request.model]));
    expect(byBudget.get("perception-entity-fallback")).toBe("haiku-fast");
    expect(byBudget.get("perception-temporal-cue")).toBe("haiku-fast");
    expect(byBudget.get("perception-mode-fallback")).toBe("background-opus");
  });

  it("extracts a temporal cue via the LLM when one is configured", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const llm = new FakeLLMClient({
      responses: [
        temporalCueResponse({
          since: "2026-04-20T12:00:00Z",
          until: "2026-04-21T12:00:00Z",
          label: "yesterday",
        }),
      ],
    });

    const cue = await detectTemporalCue("Jane said yesterday was rough", nowMs, {
      llmClient: llm,
      model: "haiku",
    });

    expect(cue).toEqual({
      sinceTs: nowMs - 24 * 60 * 60 * 1_000,
      untilTs: nowMs,
      label: "yesterday",
    });
  });

  it("hands the classifier an ISO 'now' rather than raw epoch milliseconds", async () => {
    // The window arithmetic is the harness's job, not the model's: asking for
    // epoch numbers produced windows two years off their own label.
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const llm = new FakeLLMClient({
      responses: [temporalCueResponse({ since: "2026-04-20T12:00:00Z", label: "yesterday" })],
    });

    await detectTemporalCue("Jane said yesterday was rough", nowMs, {
      llmClient: llm,
      model: "haiku",
    });

    const payload = llm.requests[0]?.messages[0]?.content ?? "";
    expect(payload).toContain("2026-04-21T12:00:00.000Z");
    expect(payload).not.toContain(String(nowMs));
  });

  it("drops a cue bound that does not parse as an instant, keeping the one that does", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const llm = new FakeLLMClient({
      responses: [
        temporalCueResponse({
          since: "2026-04-20T12:00:00Z",
          until: "sometime",
          label: "yesterday",
        }),
      ],
    });

    const cue = await detectTemporalCue("Jane said yesterday was rough", nowMs, {
      llmClient: llm,
      model: "haiku",
    });

    expect(cue).toEqual({ sinceTs: nowMs - 24 * 60 * 60 * 1_000, label: "yesterday" });
  });

  it("treats an unparseable window as no cue at all", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const llm = new FakeLLMClient({
      responses: [
        temporalCueResponse({ since: "last-ish", until: "not a date", label: "yesterday" }),
      ],
    });

    const cue = await detectTemporalCue("Jane said yesterday was rough", nowMs, {
      llmClient: llm,
      model: "haiku",
    });

    expect(cue).toBeNull();
  });
});
