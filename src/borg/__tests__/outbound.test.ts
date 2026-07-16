import { afterEach, describe, expect, it } from "vitest";

import { StreamReader, type StreamEntry, type StreamWriter } from "../../stream/index.js";
import { DemoMessageConnector } from "../../outbound/index.js";
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
  ManualClock,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

function createOutboundToolResponse(input: { targetSessionId: SessionId; instruction: string }) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
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

function requestText(request: { system?: unknown; messages?: unknown } | undefined): string {
  return JSON.stringify({
    system: request?.system ?? null,
    messages: request?.messages ?? null,
  });
}

function createEmitNoOutputResponse(reason = "No current-session message is needed.") {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_emit_no_output",
        name: "EmitNoOutput",
        input: {
          reason,
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
    stop_reason: "tool_use",
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

function createCommitmentVerdictResponse(
  violations: Array<{
    commitment_id: string;
    reason: string;
    confidence: number;
    violating_span_or_topic?: string;
  }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_commitment",
        name: "EmitCommitmentViolations",
        input: {
          violations,
        },
      },
    ],
  };
}

describe("proactive outbound", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("runs outbound composition as a target-scoped turn and delivers to a demo group session", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-"));
    tempDirs.push(tempDir);
    const operatorSessionId = createSessionId();
    const targetSessionId = createSessionId();
    const directingOnlySecret = "DIRECTING_ONLY_SECRET";
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
          config: { host_capabilities: string };
          createStreamWriter: (sessionId: SessionId) => StreamWriter;
        };
      }>(borg);
      const writer = internal.deps.createStreamWriter(operatorSessionId);
      try {
        await writer.append({
          kind: "user_msg",
          content: `${directingOnlySecret} stays only in the operator session.`,
          audience: "Tom",
        });
      } finally {
        writer.close();
      }
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

      const result = await borg.turn({
        sessionId: operatorSessionId,
        audience: "Tom",
        userMessage: "Send the Launch Room a note about the checklist.",
      });

      expect(result.response).toBe("I sent the Launch Room a message in its session.");
      expect(internal.deps.config.host_capabilities).toContain(
        "Proactive outbound messaging via wired source_type connector(s): demo",
      );

      const targetAgentAppends = liveAppends.filter(
        (entry) =>
          entry.session_id === targetSessionId &&
          entry.kind === "agent_msg" &&
          entry.content === "Launch Room, the launch checklist is ready.",
      );
      expect(targetAgentAppends).toHaveLength(1);

      const targetEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: targetSessionId,
      }).tail(20);
      expect(targetEntries.some((entry) => entry.kind === "perception")).toBe(false);
      expect(
        targetEntries.find(
          (entry) =>
            entry.kind === "agent_msg" &&
            entry.content === "Launch Room, the launch checklist is ready.",
        )?.audience,
      ).toBe("Launch Room");

      const operatorEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: operatorSessionId,
      }).tail(20);
      const outboundToolResult = operatorEntries.find((entry) => {
        const content = entry.content as { output?: { outbound?: unknown } };

        return entry.kind === "tool_result" && content.output?.outbound !== undefined;
      });
      expect(outboundToolResult?.content).toMatchObject({
        ok: true,
        output: {
          outbound: {
            target_session_id: targetSessionId,
            status: "completed",
            emitted: true,
            delivery: {
              status: "transported",
              source_type: "demo",
            },
            delivery_outcome: {
              state: "delivered",
              agent_message_id: targetAgentAppends[0]?.id,
              delivery_status: "transported",
              source_type: "demo",
            },
          },
        },
      });
      const outboundCompletedActions = borg.actions
        .list({ state: "completed", limit: 10 })
        .filter((action) => action.description.includes("Outbound post"));
      expect(outboundCompletedActions).toHaveLength(1);
      expect(outboundCompletedActions[0]).toMatchObject({
        actor: "borg",
        audience_entity_id: groupId,
        state: "completed",
        completed_at: expect.any(Number),
        not_done_at: null,
        session_scope: "current_session",
        session_anchor_id: targetSessionId,
      });
      expect(outboundCompletedActions[0]?.description).toContain("delivered");
      expect(outboundCompletedActions[0]?.provenance_stream_entry_ids).toEqual(
        expect.arrayContaining([targetAgentAppends[0]?.id]),
      );
      expect(borg.actions.getCreationCountsBySource()).toMatchObject({
        api: 0,
        tool: 1,
      });

      const targetScopedRequests = llm.requests
        .map(requestText)
        .filter((text) => text.includes(targetInstruction));
      expect(targetScopedRequests.length).toBeGreaterThan(0);
      for (const text of targetScopedRequests) {
        expect(text).not.toContain(directingOnlySecret);
        expect(text).toContain(targetVisibleContext);
      }
    } finally {
      await borg.close();
    }
  });

  it("records suppressed directed outbound as a not_done ActionRecord", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-suppressed-"));
    tempDirs.push(tempDir);
    const operatorSessionId = createSessionId();
    const targetSessionId = createSessionId();
    const targetInstruction = "Tell Alice the maintenance window moved if useful.";
    const llm = new FakeLLMClient({
      responses: [
        createNoCreatorDirectiveResponse(),
        createOutboundToolResponse({
          targetSessionId,
          instruction: targetInstruction,
        }),
        createEmitNoOutputResponse("The target-scoped finalizer chose no visible message."),
        createEmitAnswerResponse("I handed the note to Alice's session; no message was emitted."),
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
      outboundConnectors: [new DemoMessageConnector()],
    });

    try {
      const tomId = borg.entities.resolve("Tom");
      borg.entities.setBorgRole(tomId, "creator");
      const aliceId = borg.entities.resolve("Alice");

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
        label: "alice",
        audience_label: "Alice",
        audience_entity_id: aliceId,
        conversation_kind: "demo",
      });

      const result = await borg.turn({
        sessionId: operatorSessionId,
        audience: "Tom",
        userMessage: "Send Alice a maintenance-window note.",
      });

      expect(result.response).toBe("I handed the note to Alice's session; no message was emitted.");

      const targetEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: targetSessionId,
      }).tail(20);
      expect(targetEntries.filter((entry) => entry.kind === "agent_msg")).toHaveLength(0);
      const suppressionMarker = targetEntries.find((entry) => entry.kind === "agent_suppressed");
      expect(suppressionMarker?.content).toMatchObject({
        reason: "finalizer_no_output",
      });

      const operatorEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: operatorSessionId,
      }).tail(20);
      const outboundToolResult = operatorEntries.find((entry) => {
        const content = entry.content as { output?: { outbound?: unknown } };

        return entry.kind === "tool_result" && content.output?.outbound !== undefined;
      });
      expect(outboundToolResult?.content).toMatchObject({
        ok: true,
        output: {
          outbound: {
            target_session_id: targetSessionId,
            status: "completed",
            emitted: false,
            delivery_outcome: {
              state: "suppressed",
              reason: "finalizer_no_output",
              marker_entry_id: suppressionMarker?.id,
            },
          },
        },
      });

      const outboundNotDoneActions = borg.actions
        .list({ state: "not_done", limit: 10 })
        .filter((action) => action.description.includes("Outbound post"));
      expect(outboundNotDoneActions).toHaveLength(1);
      expect(outboundNotDoneActions[0]).toMatchObject({
        actor: "borg",
        audience_entity_id: aliceId,
        state: "not_done",
        completed_at: null,
        not_done_at: expect.any(Number),
        session_scope: "current_session",
        session_anchor_id: targetSessionId,
      });
      expect(outboundNotDoneActions[0]?.description).toContain("finalizer_no_output");
      expect(outboundNotDoneActions[0]?.provenance_stream_entry_ids).toEqual(
        expect.arrayContaining([suppressionMarker?.id]),
      );
      expect(borg.actions.getCreationCountsBySource()).toMatchObject({
        api: 0,
        tool: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("does not advertise demo outbound from a stream append observer alone", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-no-bridge-"));
    tempDirs.push(tempDir);
    const liveAppends: StreamEntry[] = [];
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
      llmClient: new FakeLLMClient({ responses: [] }),
      onStreamAppend: (entries) => liveAppends.push(...entries),
    });

    try {
      const internal = borgInternals<{
        deps: {
          config: { host_capabilities: string };
        };
      }>(borg);

      expect(internal.deps.config.host_capabilities).toContain(
        "Proactive outbound messaging (I cannot reach out to participants later on my own initiative)",
      );
      expect(internal.deps.config.host_capabilities).not.toContain(
        "Proactive outbound messaging via wired source_type connector(s): demo",
      );
      expect(liveAppends).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("does not double-deliver outbound when commitment regeneration reruns the finalizer", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-regeneration-"));
    tempDirs.push(tempDir);
    const operatorSessionId = createSessionId();
    const targetSessionId = createSessionId();
    const targetInstruction = "Tell Alice the maintenance checklist is ready.";
    let commitmentIdForGuard = "";
    const llm = new FakeLLMClient({
      responses: [
        createNoCreatorDirectiveResponse(),
        createOutboundToolResponse({
          targetSessionId,
          instruction: targetInstruction,
        }),
        createEmitAnswerResponse("Alice, the maintenance checklist is ready."),
        createEmitAnswerResponse("I sent it. ORCHID-17"),
        () =>
          createCommitmentVerdictResponse([
            {
              commitment_id: commitmentIdForGuard,
              reason: "The visible answer disclosed ORCHID-17.",
              confidence: 0.99,
              violating_span_or_topic: "ORCHID-17",
            },
          ]),
        createOutboundToolResponse({
          targetSessionId,
          instruction: targetInstruction,
        }),
        createEmitAnswerResponse("I already posted the Alice update."),
        createCommitmentVerdictResponse([]),
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
      outboundConnectors: [new DemoMessageConnector()],
    });

    try {
      const tomId = borg.entities.resolve("Tom");
      borg.entities.setBorgRole(tomId, "creator");
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
        label: "alice",
        audience_label: "Alice",
        conversation_kind: "demo",
      });
      const commitment = borg.commitments.add({
        type: "boundary",
        kind: "boundary",
        enforcementClass: "critical",
        criticalDomain: "audience_scope",
        directiveFamily: "operator_orchid_boundary",
        directive: "Do not disclose ORCHID-17 to Tom.",
        priority: 10,
        audience: "Tom",
        provenance: { kind: "manual" },
      });
      commitmentIdForGuard = commitment.id;
      const result = await borg.turn({
        sessionId: operatorSessionId,
        audience: "Tom",
        userMessage: "Send Alice a maintenance checklist update.",
      });

      expect(result.response).toBe("I already posted the Alice update.");

      const targetEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: targetSessionId,
      }).tail(20);
      expect(
        targetEntries.filter(
          (entry) =>
            entry.kind === "agent_msg" &&
            entry.content === "Alice, the maintenance checklist is ready.",
        ),
      ).toHaveLength(1);

      const operatorEntries = new StreamReader({
        dataDir: tempDir,
        sessionId: operatorSessionId,
      }).tail(50);
      const outboundToolCallIds = operatorEntries
        .filter((entry) => {
          const content = entry.content as { call_id?: string; tool_name?: string };

          return entry.kind === "tool_call" && content.tool_name === "tool.outbound.post";
        })
        .map((entry) => (entry.content as { call_id: string }).call_id);
      const outboundToolCallIdSet = new Set(outboundToolCallIds);
      const outboundToolResults = operatorEntries.filter((entry) => {
        const content = entry.content as { call_id?: string };

        return (
          entry.kind === "tool_result" &&
          content.call_id !== undefined &&
          outboundToolCallIdSet.has(content.call_id)
        );
      });
      expect(outboundToolCallIds).toHaveLength(2);
      expect(outboundToolResults).toHaveLength(2);
      expect(
        outboundToolResults.filter((entry) => {
          const content = entry.content as { output?: { outbound?: unknown } };

          return content.output?.outbound !== undefined;
        }),
      ).toHaveLength(1);
      expect(
        outboundToolResults.filter((entry) => {
          const content = entry.content as { error?: string };

          return content.error?.includes("may be used only once per turn") === true;
        }),
      ).toHaveLength(1);
    } finally {
      await borg.close();
    }
  });

  it("lets a scheduled autonomous wake use config-gated outbound and still respects wake budget", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autonomous-outbound-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(10_000);
    const targetSessionId = createSessionId();
    const liveAppends: StreamEntry[] = [];
    const llm = new FakeLLMClient({
      responses: [
        createOutboundToolResponse({
          targetSessionId,
          instruction: "Tell Alice the maintenance window moved.",
        }),
        createEmitAnswerResponse("Alice, the maintenance window moved."),
        createEmitNoOutputResponse(),
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
        autonomy: {
          enabled: true,
          maxWakesPerWindow: 1,
          budgetWindowMs: 3_600_000,
          proactiveOutbound: {
            enabled: true,
            maxPostsPerWindow: 1,
            windowMs: 3_600_000,
            allowByConfig: {
              sessionIds: [targetSessionId],
              sourceTypes: [],
            },
          },
          executiveFocus: {
            enabled: false,
          },
          triggers: {
            commitmentExpiring: {
              enabled: false,
            },
            openQuestionDormant: {
              enabled: false,
            },
            scheduledReflection: {
              enabled: false,
            },
            scheduledWake: {
              enabled: true,
            },
            goalFollowupDue: {
              enabled: false,
            },
          },
          conditions: {
            commitmentRevoked: {
              enabled: false,
            },
            moodValenceDrop: {
              enabled: false,
            },
            openQuestionUrgencyBump: {
              enabled: false,
            },
          },
        },
      }),
      liveExtraction: false,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      onStreamAppend: (entries) => liveAppends.push(...entries),
      outboundConnectors: [new DemoMessageConnector()],
    });

    try {
      borg.sessions.ensure({
        session_id: targetSessionId,
        source_type: "demo",
        label: "alice",
        audience_label: "Alice",
        conversation_kind: "demo",
      });

      const internal = borgInternals<{
        deps: {
          scheduledWakesRepository: {
            schedule(input: { delaySeconds: number; note: string }): unknown;
          };
          autonomyScheduler: {
            tick(): Promise<{ firedEvents: number; budgetSkipped: number }>;
          };
        };
      }>(borg);
      internal.deps.scheduledWakesRepository.schedule({
        delaySeconds: 1,
        note: "Proactively update Alice.",
      });
      clock.advance(1_000);

      const firstTick = await internal.deps.autonomyScheduler.tick();
      expect(firstTick.firedEvents).toBe(1);

      const targetMessages = liveAppends.filter(
        (entry) =>
          entry.session_id === targetSessionId &&
          entry.kind === "agent_msg" &&
          entry.content === "Alice, the maintenance window moved.",
      );
      expect(targetMessages).toHaveLength(1);
      expect(
        llm.requests.some((request) => requestText(request).includes("borg_autonomous_reflection")),
      ).toBe(true);

      internal.deps.scheduledWakesRepository.schedule({
        delaySeconds: 1,
        note: "A second wake should stay inside the wake budget.",
      });
      clock.advance(1_000);

      const secondTick = await internal.deps.autonomyScheduler.tick();
      expect(secondTick.budgetSkipped).toBe(1);
      expect(
        liveAppends.filter(
          (entry) => entry.session_id === targetSessionId && entry.kind === "agent_msg",
        ),
      ).toHaveLength(1);
    } finally {
      await borg.close();
    }
  });
});
