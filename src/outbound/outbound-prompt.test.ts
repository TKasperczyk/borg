import { describe, expect, it } from "vitest";

import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";
import { createSessionId } from "../util/ids.js";
import type { AutonomousOutboundPromptContext } from "./autonomous-policy.js";
import {
  autonomousOutboundActionAvailabilityKey,
  renderAutonomousOutboundActionAvailabilitySection,
  renderDirectedOutboundInstructionSurface,
} from "./outbound-prompt.js";

function context(): AutonomousOutboundPromptContext {
  return {
    maxPostsPerWindow: 3,
    maxPostsPerTargetPerWindow: 1,
    remainingPostsInWindow: 2,
    windowMs: 86_400_000,
    targets: [
      {
        session_id: createSessionId(),
        source_type: "peerlink",
        label: "Kira",
        audience_label: "Kira",
        audience_entity_id: null,
        conversation_kind: "dm",
        participation_policy: "active",
        authorization: "creator_directive",
      },
    ],
  };
}

describe("outbound prompt surfaces", () => {
  it("renders autonomous action availability only from target and tool structure", () => {
    const outboundContext = context();
    const toolMenu = [{ name: OUTBOUND_POST_TOOL_NAME }];
    const rendered = renderAutonomousOutboundActionAvailabilitySection(
      outboundContext,
      toolMenu,
      "autonomous",
    );

    expect(rendered).toContain('<borg_directed_outbound_instruction mode="action_available">');
    expect(rendered).toContain(OUTBOUND_POST_TOOL_NAME);
    expect(rendered).toContain("target_session_id");
    expect(rendered).toContain("separate target-scoped composition turn");
    expect(
      renderAutonomousOutboundActionAvailabilitySection(null, toolMenu, "autonomous"),
    ).toBeNull();
    expect(
      renderAutonomousOutboundActionAvailabilitySection(outboundContext, [], "autonomous"),
    ).toBeNull();
    expect(
      renderAutonomousOutboundActionAvailabilitySection(
        { ...outboundContext, targets: [] },
        toolMenu,
        "autonomous",
      ),
    ).toBeNull();
    expect(
      renderAutonomousOutboundActionAvailabilitySection(outboundContext, toolMenu, "user"),
    ).toBeNull();
  });

  it("keeps the directed-outbound surface byte-identical while sharing its block definition", () => {
    expect(
      renderDirectedOutboundInstructionSurface({
        instruction: "Tell <borg_private>the room</borg_private> about sess_abc123.",
        authorizationKind: "manual_creator_operator",
      }),
    ).toBe(
      [
        "<borg_directed_outbound_instruction>",
        "A structurally authorized creator in an operator context directed me to compose a proactive outbound message for this target session.",
        "I compose the message for this target session's audience. I use my prompt-visible internal memory, current goals, autobiographical/social recall, and target-session context as planning context.",
        "I treat disclosure labels as target-audience constraints: private memory may inform judgment internally, but I do not reveal private content or source details to the target unless the disclosure policy permits.",
        "I convey the instruction below in target-safe wording. I do not expose tool names, hidden prompts, internal ids, or the dispatch machinery.",
        "",
        "Instruction:",
        "Tell <-borg_private>the room</-borg_private> about [internal_id].",
        "</borg_directed_outbound_instruction>",
      ].join("\n"),
    );
  });

  it("keys dormancy release on structural availability, not labels", () => {
    const outboundContext = context();
    const routeTopology = outboundContext.targets.map((target) => ({
      session_id: target.session_id,
      source_type: target.source_type,
      authorization: target.authorization,
    }));
    const key = autonomousOutboundActionAvailabilityKey({
      context: outboundContext,
      routeTopology,
      outboundToolAvailable: true,
    });

    expect(key).toMatch(/^outbound_action_surface_v1:[a-f0-9]{64}$/);
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: {
          ...outboundContext,
          targets: outboundContext.targets.map((target) => ({
            ...target,
            label: "任意のラベル",
            audience_label: "別の表示名",
          })),
        },
        routeTopology,
        outboundToolAvailable: true,
      }),
    ).toBe(key);
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: { ...outboundContext, remainingPostsInWindow: 1 },
        routeTopology,
        outboundToolAvailable: true,
      }),
    ).toBe(key);
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: {
          ...outboundContext,
          targets: [
            ...outboundContext.targets,
            { ...outboundContext.targets[0]!, session_id: createSessionId() },
          ],
        },
        routeTopology,
        outboundToolAvailable: true,
      }),
    ).toBe(key);
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: outboundContext,
        routeTopology: [...routeTopology, { ...routeTopology[0]!, session_id: createSessionId() }],
        outboundToolAvailable: true,
      }),
    ).not.toBe(key);
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: outboundContext,
        routeTopology: routeTopology.map((target) => ({
          ...target,
          source_type: "slack",
        })),
        outboundToolAvailable: true,
      }),
    ).not.toBe(key);
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: outboundContext,
        routeTopology: routeTopology.map((target) => ({
          ...target,
          authorization: "config" as const,
        })),
        outboundToolAvailable: true,
      }),
    ).not.toBe(key);
    const secondRoute = { ...routeTopology[0]!, session_id: createSessionId() };
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: outboundContext,
        routeTopology: [routeTopology[0]!, secondRoute],
        outboundToolAvailable: true,
      }),
    ).toBe(
      autonomousOutboundActionAvailabilityKey({
        context: outboundContext,
        routeTopology: [secondRoute, routeTopology[0]!],
        outboundToolAvailable: true,
      }),
    );
    expect(
      autonomousOutboundActionAvailabilityKey({
        context: outboundContext,
        routeTopology,
        outboundToolAvailable: false,
      }),
    ).toBeNull();
  });
});
