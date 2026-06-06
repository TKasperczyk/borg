import { afterEach, describe, expect, it } from "vitest";

import { DemoMessageConnector } from "../../outbound/index.js";
import type { EpisodicRepository } from "../../memory/episodic/index.js";
import { createEpisodeFixture } from "../../offline/test-support.js";
import { StreamReader, type StreamEntry, type StreamWriter } from "../../stream/index.js";
import type { SessionId } from "../../util/ids.js";
import {
  Borg,
  FakeLLMClient,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createEmitAnswerResponse,
  createSessionId,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

function createOutboundToolResponse(input: { targetSessionId: SessionId; instruction: string }) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_outbound",
        name: "tool.outbound.post",
        input: {
          target_session_id: input.targetSessionId,
          instruction: input.instruction,
        },
      },
    ],
  };
}

function createNoCreatorDirectiveResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_creator_directive",
        name: "EmitCreatorDirectives",
        input: {
          decision: "none",
          reason: "No durable creator directive detected.",
          candidates: [],
        },
      },
    ],
  };
}

function requestText(request: { system?: unknown; messages?: unknown } | undefined): string {
  return JSON.stringify({
    system: request?.system ?? null,
    messages: request?.messages ?? null,
  });
}

describe("proactive outbound human-mind invariants", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("plans directed outbound with durable non-target memory while emitting target-safe text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-invariant-"));
    tempDirs.push(tempDir);
    const operatorSessionId = createSessionId();
    const targetSessionId = createSessionId();
    const durableCrossAudienceMemory = "DURABLE_CROSS_AUDIENCE_MEMORY";
    const targetVisibleContext = "TARGET_AUDIENCE_CONTEXT";
    const targetInstruction = "Tell the Launch Room the launch checklist is ready.";
    const liveAppends: StreamEntry[] = [];
    const llm = new FakeLLMClient({
      responses: [
        createNoCreatorDirectiveResponse(),
        createOutboundToolResponse({
          targetSessionId,
          instruction: targetInstruction,
        }),
        createEmitAnswerResponse("Launch Room, the launch checklist is ready."),
        createEmitAnswerResponse("I sent the Launch Room a message in its session."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      liveExtraction: false,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      onStreamAppend: (entries) => liveAppends.push(...entries),
      outboundConnectors: [new DemoMessageConnector()],
    });

    try {
      const tomId = borg.entities.resolve("Tom");
      borg.entities.setBorgRole(tomId, "creator");
      const groupId = borg.entities.resolve("Launch Room", {
        kind: "group",
      });

      borg.sessions.ensure({
        session_id: operatorSessionId,
        source_type: "demo",
        label: "operator",
        audience_label: "Tom",
        audience_entity_id: tomId,
        conversation_kind: "demo",
        audience_role: "operator",
      });
      borg.sessions.ensure({
        session_id: targetSessionId,
        source_type: "demo",
        label: "launch-room",
        audience_label: "Launch Room",
        audience_entity_id: groupId,
        conversation_kind: "demo",
      });

      const internal = borgInternals<{
        deps: {
          episodicRepository: Pick<EpisodicRepository, "createEpisode">;
          createStreamWriter: (sessionId: SessionId) => StreamWriter;
        };
      }>(borg);
      await internal.deps.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "Durable operator-only launch memory",
            narrative: `${durableCrossAudienceMemory} belongs to the operator audience and is relevant to launch checklist planning.`,
            participants: ["Tom"],
            tags: ["launch", "checklist"],
            audience_entity_id: tomId,
            shared: false,
          },
          [0, 1, 0, 0],
        ),
      );
      const targetWriter = internal.deps.createStreamWriter(targetSessionId);
      try {
        await targetWriter.append({
          kind: "user_msg",
          content: `${targetVisibleContext} is visible in the target session.`,
          audience: "Launch Room",
        });
      } finally {
        targetWriter.close();
      }

      await borg.turn({
        sessionId: operatorSessionId,
        audience: "Tom",
        userMessage: "Send the Launch Room a note about the checklist.",
      });

      const targetRequests = llm.requests
        .map(requestText)
        .filter((text) => text.includes(targetInstruction));
      expect(targetRequests.length).toBeGreaterThan(0);
      for (const text of targetRequests) {
        expect(text).toContain(durableCrossAudienceMemory);
        expect(text).toContain(targetVisibleContext);
        expect(text).toContain("disclosure_class=relationship_private");
        expect(text).toContain(
          "usable internally; do not disclose to current audience unless authorized",
        );
        expect(text).toContain("Treat disclosure labels as target-audience constraints");
        expect(text).toContain("Memory disclosure labels are input-side guidance");
      }

      const targetEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: targetSessionId,
      }).tail(20);
      const targetAgentMessage = targetEntries.find(
        (entry) => entry.kind === "agent_msg" && entry.audience === "Launch Room",
      );
      expect(targetAgentMessage?.content).toBe("Launch Room, the launch checklist is ready.");
      expect(String(targetAgentMessage?.content ?? "")).not.toContain(durableCrossAudienceMemory);
    } finally {
      await borg.close();
    }
  });
});
