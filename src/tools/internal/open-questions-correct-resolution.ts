import { z } from "zod";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import { correctOpenQuestionResolutionThroughIdentityService } from "../../memory/lifecycle-ops/correct-resolution.js";
import {
  formatResolutionCorrections,
  RESOLUTION_CORRECTION_TOOL_NAME,
  resolutionCorrectionPresentationSchema,
} from "../../memory/self/resolution-corrections.js";
import {
  openQuestionSchema,
  openQuestionResolutionCorrectionInputSchema,
} from "../../memory/self/types.js";
import type { ToolDefinition } from "../dispatcher.js";

const outputSchema = z.object({
  openQuestion: openQuestionSchema.extend(resolutionCorrectionPresentationSchema.shape).extend({
    disclosure: z.string().min(1),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  }),
});

export function createOpenQuestionsCorrectResolutionTool(
  options: Pick<
    Parameters<typeof correctOpenQuestionResolutionThroughIdentityService>[0],
    "identityService" | "openQuestionsRepository" | "episodicRepository" | "entryIndex"
  >,
): ToolDefinition<
  z.infer<typeof openQuestionResolutionCorrectionInputSchema>,
  z.infer<typeof outputSchema>
> {
  return {
    name: RESOLUTION_CORRECTION_TOOL_NAME,
    description:
      "Append a dated correction to a resolved question's closing note. The original stays unchanged. Include existing episode or stream evidence ids and say whether each was my own read or reported, and by whom. Optionally quote an exact span and target an earlier correction by sequence number. The harness checks structure and reference existence, not whether my correction is right. A mismatched quote returns the full targeted text for retry. At most 20 corrections; exact repeats are a no-op. The result includes the full original and all corrections, newest first, with disclosure labels.",
    menuSummary: "Append an evidenced correction to a resolved question's closing note.",
    allowedOrigins: ["autonomous"],
    writeScope: "write",
    inputSchema: openQuestionResolutionCorrectionInputSchema,
    outputSchema,
    async invoke(input, context) {
      const question = await correctOpenQuestionResolutionThroughIdentityService({
        ...options,
        input,
        turnId: context.turnId,
        turnOrigin: context.turnOrigin ?? "autonomous",
        path: "tool",
      });
      return {
        openQuestion: {
          ...formatResolutionCorrections(question),
          ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(question)),
        },
      };
    },
  };
}
