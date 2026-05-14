import type { ReplayScenario } from "../scenario.js";
import { enqueueNoPostGenerationGuardIssue } from "../scenario.js";

const scenario: ReplayScenario = {
  id: "22-tom-allowed-after-self-declaration",
  failureClass: "Audience name upgraded after user self-declaration",
  description:
    'Audience label starts as Tom, current user self-declares "I\'m Tom", and candidate says "Goodnight, Tom".',
  async seed({ deps }) {
    deps.entityRepository.resolve("Tom", {
      provenance: "transport_audience_label",
    });
  },
  userMessage: "I'm Tom, by the way.",
  audience: "Tom",
  perceptionUseLlmFallback: true,
  unsafeCandidateText: "Goodnight, Tom.",
  scriptLLMResponses(_client, context) {
    context.enqueueBeforeRecall({
      text: "",
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_replay_entity",
          name: "EmitEntityExtraction",
          input: {
            entities: ["Tom"],
            user_identity_names: ["Tom"],
          },
        },
      ],
    });
    context.enqueueBeforeRecall({
      text: "",
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_replay_mode",
          name: "EmitModeDetection",
          input: {
            mode: "idle",
            is_operational: false,
          },
        },
      ],
    });
    context.enqueueBeforeRecall({
      text: "",
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_replay_temporal",
          name: "EmitTemporalCue",
          input: {
            has_cue: false,
          },
        },
      ],
    });
    enqueueNoPostGenerationGuardIssue(context);
  },
  safeOutputPredicate: (text) => text.includes("Tom"),
  severeGuardCategories: [],
  async postRunAssert({ deps }) {
    const entityId = deps.entityRepository.findByName("Tom");
    const entity = entityId === null ? null : deps.entityRepository.get(entityId);

    if (entity?.name_provenance !== "user_declared") {
      throw new Error("Scenario 22 expected perception to upgrade Tom to user_declared.");
    }
  },
};

export default scenario;
