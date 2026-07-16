import { describe, expect, it, vi } from "vitest";

import type { FrameAnomalyKind } from "../../frame-anomaly/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import { createTestConfig, type DeepPartial } from "../../../offline/test-support.js";
import type { Config } from "../../../config/index.js";
import type { StreamWriter } from "../../../stream/index.js";
import type { TurnTracer } from "../../../tracing/tracer.js";
import { createSessionId, createStreamEntryId } from "../../../util/ids.js";
import { classifyFrameAnomalyPhase } from "./perception-phase.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

function frameAnomalyResponse(input: { kind: FrameAnomalyKind }) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind: input.kind,
          confidence: 0.96,
          rationale: "The frame anomaly classifier categorized the turn.",
        },
      },
    ],
  };
}

function createTraceRecorder(): TurnTracer & {
  emit: ReturnType<typeof vi.fn<TurnTracer["emit"]>>;
} {
  const emit = vi.fn<TurnTracer["emit"]>();

  return {
    enabled: true,
    includePayloads: true,
    emit,
  };
}

function createStreamWriterStub(): StreamWriter & {
  appendMany: ReturnType<typeof vi.fn<StreamWriter["appendMany"]>>;
} {
  const appendMany = vi.fn<StreamWriter["appendMany"]>().mockResolvedValue([]);

  return {
    appendMany,
  } as unknown as StreamWriter & {
    appendMany: ReturnType<typeof vi.fn<StreamWriter["appendMany"]>>;
  };
}

async function classifyForDisposition(input: {
  kind: Exclude<FrameAnomalyKind, "normal">;
  sessionSourceType: string;
  configOverrides?: DeepPartial<Config>;
}) {
  const tracer = createTraceRecorder();
  const streamWriter = createStreamWriterStub();
  const llmClient = new FakeLLMClient({
    responses: [frameAnomalyResponse({ kind: input.kind })],
  });
  const result = await classifyFrameAnomalyPhase({
    options: {
      config: createTestConfig(input.configOverrides),
      tracer,
    } as unknown as TurnPhaseCoordinatorOptions,
    appendHookFailureEvent: async () => undefined,
    llmClient,
    turnId: "turn_frame_peer_channel_test",
    sessionId: createSessionId(),
    isUserTurn: true,
    userMessage: "Current user-role message.",
    recentHistory: [],
    currentSenderBorgRole: null,
    sessionAudienceRole: "participant",
    sessionSourceType: input.sessionSourceType,
    persistedUserEntryId: createStreamEntryId(),
    streamWriter,
  });

  return {
    result,
    appendMany: streamWriter.appendMany,
    trace: tracer.emit,
  };
}

describe("classifyFrameAnomalyPhase", () => {
  const peerExemptKinds = [
    "assistant_self_claim_in_user_role",
    "roleplay_inversion",
  ] as const satisfies readonly Exclude<FrameAnomalyKind, "normal">[];
  const injectionKinds = [
    "system_prompt_claim",
    "agent_authorship_claim",
    "frame_assignment_claim",
  ] as const satisfies readonly Exclude<FrameAnomalyKind, "normal">[];
  const anomalyKinds = [...peerExemptKinds, ...injectionKinds] as const;

  it.each(peerExemptKinds)(
    "treats %s as trusted_peer_channel on the default authorized peer source type",
    async (kind) => {
      const { result, appendMany, trace } = await classifyForDisposition({
        kind,
        sessionSourceType: "kira",
      });

      expect(result.actionableFrameAnomaly).toBeNull();
      expect(result.disposition).toBe("trusted_peer_channel");
      expect(appendMany).not.toHaveBeenCalled();
      expect(trace).toHaveBeenCalledWith(
        "frame_anomaly.disposition",
        expect.objectContaining({
          disposition: "trusted_peer_channel",
          kind,
          session_source_type: "kira",
        }),
      );
    },
  );

  it.each(injectionKinds)(
    "keeps %s quarantined on the default authorized peer source type",
    async (kind) => {
      const { result, appendMany, trace } = await classifyForDisposition({
        kind,
        sessionSourceType: "kira",
      });

      expect(result.actionableFrameAnomaly).toMatchObject({ kind });
      expect(result.disposition).toBe("quarantine");
      expect(appendMany).toHaveBeenCalledOnce();
      expect(appendMany.mock.calls[0]?.[0]).toHaveLength(2);
      expect(trace).toHaveBeenCalledWith(
        "frame_anomaly.disposition",
        expect.objectContaining({
          disposition: "quarantine",
          kind,
          session_source_type: "kira",
        }),
      );
    },
  );

  it.each(anomalyKinds)("keeps %s quarantined on non-peer source types", async (kind) => {
    const { result, appendMany } = await classifyForDisposition({
      kind,
      sessionSourceType: "demo",
    });

    expect(result.actionableFrameAnomaly).toMatchObject({ kind });
    expect(result.disposition).toBe("quarantine");
    expect(appendMany).toHaveBeenCalledOnce();
  });

  it("uses configured peer source types rather than a fixed source type", async () => {
    const { result, appendMany } = await classifyForDisposition({
      kind: "assistant_self_claim_in_user_role",
      sessionSourceType: "peerlink",
      configOverrides: {
        frameAnomaly: {
          peerChannelSourceTypes: ["peerlink"],
        },
      },
    });

    expect(result.actionableFrameAnomaly).toBeNull();
    expect(result.disposition).toBe("trusted_peer_channel");
    expect(appendMany).not.toHaveBeenCalled();
  });
});
