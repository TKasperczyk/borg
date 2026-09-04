import { z } from "zod";

import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromMetadata,
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../memory/common/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import { episodeIdSchema } from "../../memory/episodic/index.js";
import type { ResolveOpenQuestionIdentityService } from "../../memory/lifecycle-ops/index.js";
import { resolveOpenQuestionThroughIdentityService } from "../../memory/lifecycle-ops/index.js";
import {
  openQuestionIdSchema,
  openQuestionSchema,
  type OpenQuestion,
} from "../../memory/self/index.js";
import { streamEntryIdSchema } from "../../util/id-schemas.js";
import type { ToolDefinition } from "../dispatcher.js";

const openQuestionsResolveInputSchema = z
  .object({
    open_question_id: openQuestionIdSchema,
    resolution_note: z.string().min(1),
    resolution_evidence_episode_ids: z.array(episodeIdSchema).optional(),
    resolution_evidence_stream_entry_ids: z.array(streamEntryIdSchema).optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (
      (value.resolution_evidence_episode_ids?.length ?? 0) === 0 &&
      (value.resolution_evidence_stream_entry_ids?.length ?? 0) === 0
    ) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Open question resolution requires episode or stream evidence",
        path: ["resolution_evidence_episode_ids"],
      });
    }
  });

const openQuestionWithDisclosureSchema = openQuestionSchema.extend({
  disclosure: z.string().min(1),
  disclosure_label: memoryDisclosureLabelMetadataSchema,
});

const openQuestionsResolveOutputSchema = z.discriminatedUnion("status", [
  z.object({
    status: z.literal("applied"),
    openQuestion: openQuestionWithDisclosureSchema,
  }),
  z.object({
    status: z.literal("requires_review"),
    reason: z.literal("identity_governance_requires_review"),
    openQuestion: openQuestionWithDisclosureSchema,
  }),
]);

export type OpenQuestionsResolveToolOptions = {
  identityService: ResolveOpenQuestionIdentityService;
  disclosureLabelForEvidence: (
    episodeIds: readonly z.infer<typeof episodeIdSchema>[],
    streamEntryIds: readonly z.infer<typeof streamEntryIdSchema>[],
  ) => Promise<MemoryDisclosureLabel>;
};

function disclosureLabelForQuestion(question: OpenQuestion): MemoryDisclosureLabel {
  const resolutionDisclosureLabel =
    question.resolution_disclosure_label === undefined
      ? null
      : memoryDisclosureLabelFromMetadata(question.resolution_disclosure_label);

  return combineMemoryDisclosureLabels([
    openQuestionMemoryDisclosureLabel(question),
    ...(resolutionDisclosureLabel === null ? [] : [resolutionDisclosureLabel]),
  ]);
}

function openQuestionOutput(question: OpenQuestion): z.infer<typeof openQuestionWithDisclosureSchema> {
  return {
    ...question,
    ...memoryDisclosurePayloadFields(disclosureLabelForQuestion(question)),
  };
}

export function createOpenQuestionsResolveTool(
  options: OpenQuestionsResolveToolOptions,
): ToolDefinition<
  z.infer<typeof openQuestionsResolveInputSchema>,
  z.infer<typeof openQuestionsResolveOutputSchema>
> {
  return {
    name: "tool.openQuestions.resolve",
    description:
      "Resolve an open question from prompt-visible evidence during autonomous reflection. If identity governance requires review, I report that status honestly instead of treating it as a tool failure. The openQuestion I get back is the record as this write left it, not as it stood before: an applied resolution clears the whole active-open rumination lifecycle on the record, so unresolved_rumination_ticks reads 0 and last_ruminated_at reads null because this write set them there. Neither reports how much the question was turned over first, and a null there is never evidence that nothing ever ruminated on it. The rumination notes are that record and they outlive the resolution.",
    menuSummary: "Resolve an open question with evidence, or surface identity review.",
    allowedOrigins: ["autonomous"],
    writeScope: "write",
    inputSchema: openQuestionsResolveInputSchema,
    outputSchema: openQuestionsResolveOutputSchema,
    async invoke(input, context) {
      const evidenceEpisodeIds = input.resolution_evidence_episode_ids ?? [];
      const evidenceStreamEntryIds = input.resolution_evidence_stream_entry_ids ?? [];
      const resolutionDisclosureLabel = await options.disclosureLabelForEvidence(
        evidenceEpisodeIds,
        evidenceStreamEntryIds,
      );
      const result = resolveOpenQuestionThroughIdentityService({
        openQuestionId: input.open_question_id,
        identityService: options.identityService,
        resolution: {
          resolution_evidence_episode_ids: evidenceEpisodeIds,
          resolution_evidence_stream_entry_ids: evidenceStreamEntryIds,
          resolution_disclosure_label: resolutionDisclosureLabel,
          resolution_note: input.resolution_note,
        },
        provenance: {
          kind: "online_reflector",
          evidence_episode_ids: evidenceEpisodeIds,
          evidence_stream_entry_ids: evidenceStreamEntryIds,
        },
        turnId: context.turnId,
        traceSourcePath: "tool.openQuestions.resolve",
        traceDecisionReason: "autonomous_tool",
      });

      if (result.status === "no_op" && result.reason === "requires_review") {
        const identityResult = result.value?.result;

        if (identityResult?.status !== "requires_review") {
          throw new Error("Open question resolution review result was missing");
        }

        return {
          status: "requires_review",
          reason: "identity_governance_requires_review",
          openQuestion: openQuestionOutput(identityResult.current),
        };
      }

      if (result.status === "conflict") {
        throw result.error;
      }

      if (result.status !== "success") {
        throw new Error(`Open question resolution did not apply: ${result.reason}`);
      }

      const identityResult = result.value.result;

      return identityResult.status === "applied"
        ? {
            status: "applied",
            openQuestion: openQuestionOutput(identityResult.record),
          }
        : {
            status: "requires_review",
            reason: "identity_governance_requires_review",
            openQuestion: openQuestionOutput(identityResult.current),
          };
    },
  };
}
