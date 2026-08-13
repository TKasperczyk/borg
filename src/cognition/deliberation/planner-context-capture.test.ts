import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LLMCompleteResult } from "../../llm/index.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { DeliberationContext } from "./types.js";
import {
  buildBaseSystemPrompt,
  buildCacheableBaseSystemPromptParts,
} from "./prompt/system-prompt.js";
import {
  buildPlannerContextCaptureRecord,
  anchorPlannerRequest,
  captureCompactPlannerContext,
  createPlannerCaptureRenderInput,
  parsePlannerContextCaptureRecord,
  PlannerContextCapture,
  plannerContextCapturePath,
  renderCapturedPlannerSurfacePair,
  type PlannerContextCaptureRecord,
} from "./planner-context-capture.js";
import { replayPlannerContextCapture } from "./planner-ab-replay.js";
import { createS2PlannerRequestSnapshot } from "./s2-planner.js";

const NOW_MS = Date.UTC(2026, 7, 13, 20, 0, 0);
const tempDirs: string[] = [];

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function context(overrides: Partial<DeliberationContext> = {}): DeliberationContext {
  return {
    sessionId: DEFAULT_SESSION_ID,
    nowMs: NOW_MS,
    turnId: "turn_capture_test",
    userMessage: "Please plan this carefully.",
    perception: {
      entities: [],
      mode: "reflective",
      affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
      temporalCue: null,
    },
    retrievalResult: [],
    workingMemory: {
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 8,
      hot_entities: [],
      pending_actions: [],
      pending_social_attribution: null,
      pending_trait_attribution: null,
      suppressed: [],
      mood: null,
      pending_procedural_attempts: [],
      discourse_state: { stop_until_substantive_content: null },
      mode: "reflective",
      updated_at: NOW_MS,
    },
    selfSnapshot: { values: [], goals: [], traits: [] },
    ...overrides,
  };
}

function renderInput(sourceContext = context()) {
  const options = {
    retrievalContextBudget: 8_000,
    semanticContextBudget: 16_000,
    nowMs: NOW_MS,
  };
  const baseSystemPrompt = buildBaseSystemPrompt(sourceContext, options);
  const cacheable = buildCacheableBaseSystemPromptParts(sourceContext, options);

  return createPlannerCaptureRenderInput({
    context: sourceContext,
    legacyBaseSystemPrompt: baseSystemPrompt,
    compactStaticPrefix: cacheable.staticPrefix,
    compactPlannerLedger: null,
    additionalPromptSections: [
      { blockId: "borg_session_reentry_continuity", text: "re-entry test block" },
    ],
    dialogueMessages: [{ role: "user", content: sourceContext.userMessage }],
    model: "claude-opus-test",
    maxTokens: 4_096,
  });
}

function record(input = renderInput()): PlannerContextCaptureRecord {
  const liveSurface = renderCapturedPlannerSurfacePair(input).compact.rendered.system;
  const liveOutput = {
    plan: {
      uncertainty: "",
      verification_steps: [],
      tensions: [],
      voice_note: "",
      emission_recommendation: "emit" as const,
      intents: [],
    },
    reasoning: "reference reasoning",
    usage: { input_tokens: 100, output_tokens: 20, stop_reason: "tool_use" },
  };
  return buildPlannerContextCaptureRecord({
    captureId: "capture_test",
    capturedAt: NOW_MS,
    liveSurfaceVariant: "compact",
    renderInput: input,
    liveOutcome: {
      status: "completed",
      attempts: 1,
      structuralReason: "emit_turn_plan",
    },
    liveOutput,
    liveRequest: anchorPlannerRequest(
      createS2PlannerRequestSnapshot({
        attempt: 1,
        system: liveSurface,
        messages: input.dialogueMessages,
        model: input.model,
        maxTokens: input.maxTokens,
        ...(input.thinking === undefined ? {} : { thinking: input.thinking }),
        ...(input.effort === undefined ? {} : { effort: input.effort }),
        ...(input.compactContext.turnOrigin === undefined
          ? {}
          : { turnOrigin: input.compactContext.turnOrigin }),
      }),
    ),
  });
}

describe("planner context capture", () => {
  it("keeps capture disabled at sample rate zero without touching the data dir", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-planner-capture-off-"));
    tempDirs.push(dataDir);
    const capture = new PlannerContextCapture({
      dataDir,
      sampleRate: 0,
      random: () => 0,
    });

    expect(capture.shouldCapture()).toBe(false);
    expect(existsSync(plannerContextCapturePath(dataDir))).toBe(false);
    expect(existsSync(join(dataDir, "captures"))).toBe(false);
  });

  it("round-trips the rendering closure through JSON with byte-identical paired surfaces", () => {
    const repositoryGet = vi.fn();
    const reRetrieve = vi.fn();
    const input = renderInput(
      context({
        entityRepository: {
          get: repositoryGet,
        } as unknown as DeliberationContext["entityRepository"],
        reRetrieve,
      }),
    );
    const livePair = renderCapturedPlannerSurfacePair(input);
    const serialized = JSON.stringify(record(input));
    const parsed = parsePlannerContextCaptureRecord(JSON.parse(serialized) as unknown);
    const replayPair = renderCapturedPlannerSurfacePair(parsed.render_input);

    expect(replayPair.compact.rendered.system).toEqual(livePair.compact.rendered.system);
    expect(replayPair.legacy.rendered.system).toEqual(livePair.legacy.rendered.system);
    expect(replayPair.compact.fingerprint).toEqual(livePair.compact.fingerprint);
    expect(replayPair.legacy.fingerprint).toEqual(livePair.legacy.fingerprint);
    expect(parsed.expected_surfaces).toEqual({
      compact: livePair.compact.fingerprint,
      legacy: livePair.legacy.fingerprint,
    });
    expect(parsed.render_input.compactContext).not.toHaveProperty("entityRepository");
    expect(parsed.render_input.compactContext).not.toHaveProperty("reRetrieve");
    expect(repositoryGet).not.toHaveBeenCalled();
    expect(reRetrieve).not.toHaveBeenCalled();
  });

  it("skips oversized records and counts the skip without creating a capture file", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-planner-capture-size-"));
    tempDirs.push(dataDir);
    const capture = new PlannerContextCapture({
      dataDir,
      sampleRate: 1,
      maxRecordBytes: 64,
    });

    const result = await capture.write(record());

    expect(result).toMatchObject({ status: "skipped", reason: "record_oversized" });
    expect(capture.snapshotStats()).toEqual({
      captured: 0,
      oversizedSkipped: 1,
      fileFullSkipped: 0,
      failed: 0,
    });
    expect(existsSync(plannerContextCapturePath(dataDir))).toBe(false);
  });

  it("caps append-only file growth and counts records skipped at the cap", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-planner-capture-cap-"));
    tempDirs.push(dataDir);
    const captureRecord = record();
    const lineBytes = Buffer.byteLength(`${JSON.stringify(captureRecord)}\n`);
    const capture = new PlannerContextCapture({
      dataDir,
      sampleRate: 1,
      maxRecordBytes: lineBytes,
      maxFileBytes: lineBytes,
    });

    await expect(capture.write(captureRecord)).resolves.toMatchObject({ status: "captured" });
    await expect(capture.write(captureRecord)).resolves.toMatchObject({
      status: "skipped",
      reason: "file_full",
    });

    const path = plannerContextCapturePath(dataDir);
    expect(readFileSync(path, "utf8").trim().split("\n")).toHaveLength(1);
    expect(capture.snapshotStats()).toEqual({
      captured: 1,
      oversizedSkipped: 0,
      fileFullSkipped: 1,
      failed: 0,
    });
  });

  it("repairs pre-existing capture directory and file permissions under umask 0022", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-planner-capture-mode-"));
    tempDirs.push(dataDir);
    const captureDirectory = join(dataDir, "captures");
    const path = plannerContextCapturePath(dataDir);
    mkdirSync(captureDirectory, { mode: 0o755 });
    writeFileSync(path, "", { mode: 0o644 });
    chmodSync(captureDirectory, 0o755);
    chmodSync(path, 0o644);
    const capture = new PlannerContextCapture({ dataDir, sampleRate: 1 });
    const previousUmask = process.umask(0o022);

    try {
      await expect(capture.write(record())).resolves.toMatchObject({ status: "captured" });
    } finally {
      process.umask(previousUmask);
    }

    expect(statSync(captureDirectory).mode & 0o777).toBe(0o700);
    expect(statSync(path).mode & 0o777).toBe(0o600);
  });

  it("rejects a captures-directory symlink that escapes dataDir", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-planner-capture-contained-"));
    const outside = mkdtempSync(join(tmpdir(), "borg-planner-capture-outside-"));
    tempDirs.push(dataDir, outside);
    symlinkSync(outside, join(dataDir, "captures"));
    const capture = new PlannerContextCapture({ dataDir, sampleRate: 1 });

    await expect(capture.write(record())).resolves.toMatchObject({
      status: "failed",
      reason: expect.stringContaining("must resolve below"),
    });
    expect(existsSync(join(outside, "planner-contexts.jsonl"))).toBe(false);
  });

  it("captures only the compact renderer's assembled context closure", () => {
    const captured = captureCompactPlannerContext(context());

    expect(captured).toMatchObject({
      sessionId: DEFAULT_SESSION_ID,
      nowMs: NOW_MS,
      selfSnapshot: { values: [], goals: [], traits: [] },
    });
    expect(captured).not.toHaveProperty("userMessage");
    expect(captured).not.toHaveProperty("retrievalResult");
    expect(captured).not.toHaveProperty("recencyMessages");
    expect(captured).not.toHaveProperty("currentUserContent");
  });

  it("omits unused nested sensitive payloads while preserving disclosure labels", () => {
    const disclosure = {
      disclosureClass: "relationship_private" as const,
      originAudienceEntityIds: ["ent_origin"],
      privateToEntityIds: ["ent_private"],
      publicToEntityIds: [],
    };
    const source = context({
      perception: {
        entities: ["UNUSED_PERCEPTION_ENTITY"],
        entityMentions: [{ name: "UNUSED_ENTITY_MENTION", kind: "person" }],
        mode: "reflective",
        affectiveSignal: { valence: 0.2, arousal: 0.4, dominant_emotion: "curiosity" },
        temporalCue: { label: "UNUSED_TEMPORAL_PAYLOAD" },
      },
      workingMemory: {
        ...context().workingMemory,
        pending_actions: [
          { description: "UNUSED_PENDING_ACTION", next_action: null, created_at: NOW_MS },
        ],
        suppressed: [{ id: "unused", reason: "UNUSED_SUPPRESSION", until_turn: 99 }],
        pending_procedural_attempts: [
          {
            problem_text: "UNUSED_PROCEDURAL_ATTEMPT",
            approach_summary: "unused",
            selected_skill_id: null,
            source_stream_ids: ["strm_unused"],
            turn_counter: 1,
            audience_entity_id: null,
          },
        ],
      } as DeliberationContext["workingMemory"],
      selectedSkill: {
        skill: {
          id: "skill_selected",
          applies_when: "selected applies",
          approach: "selected approach",
          disclosure_label: disclosure,
        },
        evaluatedCandidates: [{ skill: { approach: "UNUSED_EVALUATED_CANDIDATE" } }],
      } as unknown as DeliberationContext["selectedSkill"],
      evidenceLedger: {
        sections: [],
        audienceStanding: {
          recentLivedExperienceEntries: [
            {
              id: "lived_1",
              source_type: "system_metadata",
              session_scope: "global",
              actor: "memory",
              trust_rank: 1,
              text: "visible lived text",
              state_metadata: {
                disclosure_label: disclosure,
                unused_sensitive_payload: "UNUSED_LEDGER_METADATA",
              },
            },
          ],
          renderRecentLivedExperience: true,
          observedEventIntrospectionEntries: [],
          commitmentEntries: [],
          relationalEntries: [],
        },
        sharedState: {
          entries: [{ superseded_by_id: null, value: "UNUSED_SHARED_STATE_PAYLOAD" }],
        },
        imageAttachments: [{ label: "UNUSED_IMAGE_PAYLOAD" }],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      } as unknown as DeliberationContext["evidenceLedger"],
    });

    const captured = captureCompactPlannerContext(source);
    const serialized = JSON.stringify(captured);

    expect(captured.perception).not.toHaveProperty("entities");
    expect(captured.perception).not.toHaveProperty("temporalCue");
    expect(captured.workingMemory).toMatchObject({
      pendingActionCount: 1,
      pendingProceduralAttemptCount: 1,
    });
    expect(captured.workingMemory).not.toHaveProperty("suppressed");
    expect(captured.selectedSkill).not.toHaveProperty("evaluatedCandidates");
    expect(serialized).not.toContain("UNUSED_PERCEPTION_ENTITY");
    expect(serialized).not.toContain("UNUSED_ENTITY_MENTION");
    expect(serialized).not.toContain("UNUSED_TEMPORAL_PAYLOAD");
    expect(serialized).not.toContain("UNUSED_PENDING_ACTION");
    expect(serialized).not.toContain("UNUSED_PROCEDURAL_ATTEMPT");
    expect(serialized).not.toContain("UNUSED_SUPPRESSION");
    expect(serialized).not.toContain("UNUSED_EVALUATED_CANDIDATE");
    expect(serialized).not.toContain("UNUSED_LEDGER_METADATA");
    expect(serialized).not.toContain("UNUSED_SHARED_STATE_PAYLOAD");
    expect(serialized).not.toContain("UNUSED_IMAGE_PAYLOAD");
    expect(serialized).toContain('"disclosureClass":"relationship_private"');
    expect(serialized).toContain('"privateToEntityIds":["ent_private"]');
  });

  it("dry replay renders both variants without invoking the LLM", async () => {
    const llm = new FakeLLMClient();
    const captureRecord = record();

    const result = await replayPlannerContextCapture(captureRecord, {
      mode: "dry",
      pairIndex: 0,
      now: () => NOW_MS + 1,
    });

    expect(llm.requests).toHaveLength(0);
    expect(result.mode).toBe("dry");
    expect(result.surfaces.compact.byteFaithfulToCapture).toBe(true);
    expect(result.surfaces.legacy.byteFaithfulToCapture).toBe(true);
    expect(result.surfaces.compact.fingerprint.systemChars).toBeGreaterThan(0);
    expect(result.surfaces.legacy.fingerprint.systemChars).toBeGreaterThan(0);
    expect(result).not.toHaveProperty("live");
  });

  it("live paired replay is write-isolated from retrieval and substrate repositories", async () => {
    const reRetrieve = vi.fn();
    const repositoryUpdate = vi.fn();
    const entityRepository = {
      get: vi.fn(),
      upsert: repositoryUpdate,
    } as unknown as DeliberationContext["entityRepository"];
    const captureRecord = record(renderInput(context({ reRetrieve, entityRepository })));
    const response = {
      text: "planner reasoning",
      input_tokens: 123,
      output_tokens: 17,
      cache_creation_input_tokens: 7,
      cache_read_input_tokens: 11,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_plan",
          name: "EmitTurnPlan",
          input: {
            uncertainty: "",
            verification_steps: [],
            tensions: [],
            voice_note: "",
            emission_recommendation: "emit",
            intents: [],
          },
        },
      ],
    } satisfies LLMCompleteResult;
    const llm = new FakeLLMClient({ responses: [response, response] });

    const result = await replayPlannerContextCapture(captureRecord, {
      mode: "live",
      llmClient: llm,
      pairIndex: 1,
      now: () => NOW_MS + 2,
    });

    expect(llm.requests).toHaveLength(2);
    expect(llm.requests.map((request) => request.budget)).toEqual([
      "cognition-plan",
      "cognition-plan",
    ]);
    expect(result.execution_order).toEqual(["legacy", "compact"]);
    expect(result.live?.legacy).toMatchObject({
      status: "completed",
      usage: {
        input_tokens: 123,
        cache_creation_input_tokens: 7,
        cache_read_input_tokens: 11,
      },
    });
    expect(result.live?.compact).toMatchObject({ status: "completed" });
    expect(reRetrieve).not.toHaveBeenCalled();
    expect(repositoryUpdate).not.toHaveBeenCalled();
  });

  it("excludes degraded source captures by default and labels included replay degradation", async () => {
    const completed = record();
    const degraded = parsePlannerContextCaptureRecord({
      ...completed,
      live_outcome: {
        status: "degraded",
        attempts: 2,
        structuralReason: "missing_emit_turn_plan_tool_use",
        plan: null,
        reasoning: "no tool",
        usage: { input_tokens: 2, output_tokens: 1, stop_reason: "end_turn" },
      },
    });
    const excludedLlm = new FakeLLMClient();

    const excluded = await replayPlannerContextCapture(degraded, {
      mode: "live",
      llmClient: excludedLlm,
    });
    expect(excluded.pairing_status).toBe("excluded_source_outcome");
    expect(excludedLlm.requests).toHaveLength(0);

    const miss = {
      text: "no plan tool",
      input_tokens: 2,
      output_tokens: 1,
      stop_reason: "end_turn",
      tool_calls: [],
    } satisfies LLMCompleteResult;
    const includedLlm = new FakeLLMClient({ responses: [miss, miss, miss, miss] });
    const included = await replayPlannerContextCapture(degraded, {
      mode: "live",
      llmClient: includedLlm,
      includeNonCompleted: true,
    });
    expect(included.pairing_status).toBe("paired");
    expect(included.live?.compact).toMatchObject({
      status: "degraded",
      attempts: 2,
      structuralReason: "missing_emit_turn_plan_tool_use",
    });
    expect(included.live?.legacy).toMatchObject({ status: "degraded", attempts: 2 });
  });

  it("skips live replay when the stored exact-surface fidelity is false", async () => {
    const captureRecord = parsePlannerContextCaptureRecord({
      ...record(),
      fidelity: {
        ...record().fidelity,
        verified: false,
        exactLiveSurfaceMatchesProjection: false,
      },
    });
    const llm = new FakeLLMClient();

    const result = await replayPlannerContextCapture(captureRecord, {
      mode: "live",
      llmClient: llm,
    });

    expect(result.pairing_status).toBe("skipped_fidelity");
    expect(llm.requests).toHaveLength(0);
  });

  it("skips before live calls when the current canonical request drifts from capture", async () => {
    const source = record();
    const captureRecord = parsePlannerContextCaptureRecord({
      ...source,
      fidelity: {
        ...source.fidelity,
        liveRequest: {
          ...source.fidelity.liveRequest,
          canonicalSha256: "0".repeat(64),
        },
      },
    });
    const llm = new FakeLLMClient();

    const result = await replayPlannerContextCapture(captureRecord, {
      mode: "live",
      llmClient: llm,
    });

    expect(result.pairing_status).toBe("skipped_fidelity");
    expect(result.fidelity).toEqual({
      storedVerified: true,
      currentSourceRequestMatchesCapture: false,
    });
    expect(llm.requests).toHaveLength(0);
  });
});
