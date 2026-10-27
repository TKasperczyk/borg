import { describe, expect, it, vi } from "vitest";

import type { FrameAnomalyKind } from "../../frame-anomaly/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import { createTestConfig, type DeepPartial } from "../../../offline/test-support.js";
import type { Config } from "../../../config/index.js";
import { streamEntrySchema, type StreamEntry, type StreamWriter } from "../../../stream/index.js";
import {
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
} from "../../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../../memory/common/disclosure-serializers.js";
import type { TurnTracer } from "../../../tracing/tracer.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../../util/ids.js";
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
  entries?: StreamEntry[];
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
    sessionId: input.entries?.[0]?.session_id ?? createSessionId(),
    isUserTurn: true,
    userMessage: "Current user-role message.",
    recentHistory: [],
    currentSenderBorgRole: null,
    sessionAudienceRole: "participant",
    sessionSourceType: input.sessionSourceType,
    persistedUserEntryId: input.entries?.[0]?.id ?? createStreamEntryId(),
    sourceUserEntryIds: input.entries?.map((entry) => entry.id),
    sourceUserEntries: input.entries,
    streamWriter,
  });

  return {
    result,
    appendMany: streamWriter.appendMany,
    trace: tracer.emit,
  };
}

describe("classifyFrameAnomalyPhase", () => {
  it("labels a shared rationale with every cited source while preserving each original disclosure label", async () => {
    const audienceId = createEntityId();
    const sessionId = createSessionId();
    const entries = [
      publicMemoryDisclosureLabel(),
      relationshipPrivateMemoryDisclosureLabel([audienceId]),
    ].map((label) =>
      streamEntrySchema.parse({
        id: createStreamEntryId(),
        session_id: sessionId,
        timestamp: 1_000,
        kind: "user_msg",
        content: "報告。",
        metadata: { ...memoryDisclosurePayloadFields(label) },
      }),
    );
    const { appendMany } = await classifyForDisposition({
      kind: "system_prompt_claim",
      sessionSourceType: "kira",
      entries,
    });
    const events = appendMany.mock.calls[0]![0];
    expect(events).toHaveLength(2);
    expect(events[0]?.content).toMatchObject({
      source_stream_entry_id: entries[0]!.id,
      source_disclosure_label: { disclosure_class: "public" },
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [audienceId],
      },
    });
    expect(events[1]?.content).toMatchObject({
      source_stream_entry_id: entries[1]!.id,
      source_disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [audienceId],
      },
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [audienceId],
      },
    });
  });
  const peerExemptKinds = [
    "assistant_self_claim_in_user_role",
    "roleplay_inversion",
    "agent_authorship_claim",
  ] as const satisfies readonly Exclude<FrameAnomalyKind, "normal">[];
  const injectionKinds = [
    "system_prompt_claim",
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
    "annotates %s without striking it on the default authorized peer source type",
    async (kind) => {
      const { result, appendMany, trace } = await classifyForDisposition({
        kind,
        sessionSourceType: "kira",
      });

      expect(result.actionableFrameAnomaly).toMatchObject({ kind });
      expect(result.disposition).toBe("annotate");
      expect(appendMany).toHaveBeenCalledOnce();
      expect(appendMany.mock.calls[0]?.[0]).toHaveLength(1);
      expect(appendMany.mock.calls[0]?.[0][0]?.content).toMatchObject({
        event: "frame_anomaly_annotation",
        kind,
        restores_quarantine_marker_ids: [],
      });
      expect(trace).toHaveBeenCalledWith(
        "frame_anomaly.disposition",
        expect.objectContaining({
          disposition: "annotate",
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
      kind: "agent_authorship_claim",
      sessionSourceType: "claude_code",
      configOverrides: {
        frameAnomaly: {
          peerChannelSourceTypes: ["claude_code"],
        },
      },
    });

    expect(result.actionableFrameAnomaly).toBeNull();
    expect(result.disposition).toBe("trusted_peer_channel");
    expect(appendMany).not.toHaveBeenCalled();
  });
});
