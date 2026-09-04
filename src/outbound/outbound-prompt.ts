import { createHash } from "node:crypto";

import {
  PROMPT_SURFACES,
  renderPromptSurface,
  type PromptSurfaceRenderContext,
} from "../cognition/prompts/prompt-surface-registry.js";
import type { TurnOrigin } from "../contracts/cognitive-contracts.js";
import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";
import { escapeReservedBorgTags, scrubCreatorDirectiveInternalIds } from "../util/prompt-tags.js";
import type {
  AutonomousOutboundPromptContext,
  AutonomousOutboundRouteTopologyTarget,
} from "./autonomous-policy.js";

type AutonomousToolMenuItem = {
  name: string;
};

type DirectedOutboundAuthorizationKind = "manual_creator_operator" | "autonomous_policy";

const OUTBOUND_ACTION_TOPOLOGY_KEY_VERSION = 1;

const TARGET_COMPOSITION_CONTEXT =
  "I compose the message for this target session's audience. I use my prompt-visible internal memory, current goals, autobiographical/social recall, and target-session context as planning context.";
const TARGET_DISCLOSURE_CONTRACT =
  "I treat disclosure labels as target-audience constraints: private memory may inform judgment internally, but I do not reveal private content or source details to the target unless the disclosure policy permits.";
const TARGET_DELIVERY_CONTRACT =
  "I convey the instruction below in target-safe wording. I do not expose tool names, hidden prompts, internal ids, or the dispatch machinery.";

function directedOutboundProvenanceLine(
  authorizationKind: DirectedOutboundAuthorizationKind,
): string {
  return authorizationKind === "autonomous_policy"
    ? "An autonomous wake, structurally authorized by proactive outbound policy, directed me to compose a proactive outbound message for this target session."
    : "A structurally authorized creator in an operator context directed me to compose a proactive outbound message for this target session.";
}

function renderRegisteredOutboundBlock(
  surface:
    | typeof PROMPT_SURFACES.directedOutboundFraming
    | typeof PROMPT_SURFACES.autonomousOutboundActionFraming,
  promptSection: string,
): string {
  const renderContext: PromptSurfaceRenderContext = {
    renderBlock: (id) => (id === "borg_directed_outbound_instruction" ? promptSection : null),
  };

  return renderPromptSurface(surface, renderContext) ?? "";
}

export function renderDirectedOutboundInstructionSurface(input: {
  instruction: string;
  authorizationKind: DirectedOutboundAuthorizationKind;
}): string {
  const promptSection = [
    "<borg_directed_outbound_instruction>",
    directedOutboundProvenanceLine(input.authorizationKind),
    TARGET_COMPOSITION_CONTEXT,
    TARGET_DISCLOSURE_CONTRACT,
    TARGET_DELIVERY_CONTRACT,
    "",
    "Instruction:",
    escapeReservedBorgTags(scrubCreatorDirectiveInternalIds(input.instruction)),
    "</borg_directed_outbound_instruction>",
  ].join("\n");

  return renderRegisteredOutboundBlock(PROMPT_SURFACES.directedOutboundFraming, promptSection);
}

function autonomousOutboundActionIsAvailable(
  context: AutonomousOutboundPromptContext | null | undefined,
  toolMenu: readonly AutonomousToolMenuItem[] | null | undefined,
): context is AutonomousOutboundPromptContext {
  return (
    context !== null &&
    context !== undefined &&
    context.targets.length > 0 &&
    (toolMenu?.some((item) => item.name === OUTBOUND_POST_TOOL_NAME) ?? false)
  );
}

export function renderAutonomousOutboundActionAvailabilitySection(
  context: AutonomousOutboundPromptContext | null | undefined,
  toolMenu: readonly AutonomousToolMenuItem[] | null | undefined,
  turnOrigin: TurnOrigin | undefined,
): string | null {
  if (turnOrigin !== "autonomous" || !autonomousOutboundActionIsAvailable(context, toolMenu)) {
    return null;
  }

  const promptSection = [
    '<borg_directed_outbound_instruction mode="action_available">',
    "A structurally authorized proactive outbound action is available on this autonomous turn for the targets listed in <reachable_threads>.",
    "Each listed route has a target session whose stored status is active, structural caps that still leave this action available, and a connector registered for its source type -- that last check is one bit per source type, read once at startup, so every route sharing a source type carries the same verdict and none of them can lose it while I am running. None of those checks reads a delivery outcome or contacts the far side: a listed route can still fail at transport, a failed send never removes a route from this list however often it repeats, and a route missing from this list can be missing because its session status is no longer active rather than because an authorization changed.",
    "A reach can also come back target_busy, which is neither a transport failure nor a refusal by the far side: that target was holding its own turn lock when I called, so no composition turn opened and nothing was sent. It is a fact about when I reached, not about the route or about how the far side received me, and a later reach at the same route can land.",
    `If I choose to reach one, I call ${OUTBOUND_POST_TOOL_NAME} with target_session_id set to that target's session_id and an instruction describing what the target-scoped turn should convey. The tool opens a separate target-scoped composition turn; it does not send my current draft directly.`,
    TARGET_COMPOSITION_CONTEXT,
    TARGET_DISCLOSURE_CONTRACT,
    "The target-scoped turn conveys my instruction in target-safe wording and does not expose tool names, hidden prompts, internal ids, or dispatch machinery.",
    "I wait for the outbound tool result before choosing the terminal result for this autonomous interval.",
    "</borg_directed_outbound_instruction>",
  ].join("\n");

  return renderRegisteredOutboundBlock(
    PROMPT_SURFACES.autonomousOutboundActionFraming,
    promptSection,
  );
}

function structuralTarget(target: AutonomousOutboundRouteTopologyTarget) {
  return {
    authorization: target.authorization,
    session_id: target.session_id,
    source_type: target.source_type,
  };
}

// This key releases an otherwise-infinite empty-wake dormancy only when an
// executable outbound path is structurally present. Pre-fix dormant rows have
// no key and therefore become eligible once; prompt text and goal language never
// participate. The version namespaces the durable format, and the bounded
// digest keeps watermark metadata small.
export function autonomousOutboundActionAvailabilityKey(input: {
  context: AutonomousOutboundPromptContext | null | undefined;
  routeTopology: readonly AutonomousOutboundRouteTopologyTarget[];
  outboundToolAvailable: boolean;
}): string | null {
  if (
    input.outboundToolAvailable !== true ||
    input.context === null ||
    input.context === undefined ||
    input.context.targets.length === 0 ||
    input.routeTopology.length === 0
  ) {
    return null;
  }

  const structuralState = JSON.stringify({
    topology_key_version: OUTBOUND_ACTION_TOPOLOGY_KEY_VERSION,
    outbound_tool: OUTBOUND_POST_TOOL_NAME,
    targets: input.routeTopology
      .map(structuralTarget)
      .sort((left, right) => left.session_id.localeCompare(right.session_id)),
  });
  const fingerprint = createHash("sha256").update(structuralState).digest("hex");

  return `outbound_action_surface_v${OUTBOUND_ACTION_TOPOLOGY_KEY_VERSION}:${fingerprint}`;
}
