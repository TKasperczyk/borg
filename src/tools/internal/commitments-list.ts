import { z } from "zod";

import { commitmentSchema, type CommitmentRecord } from "../../memory/commitments/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import type { MemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import {
  commitmentMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const commitmentsListInputSchema = z.object({}).strict();
const commitmentForCognitionSchema = commitmentSchema.extend({
  disclosure: z.string().min(1),
  disclosure_label: memoryDisclosureLabelMetadataSchema,
});

const commitmentsListOutputSchema = z.object({
  commitments: z.array(commitmentForCognitionSchema),
});

export type CommitmentsListToolOptions = {
  listCommitments: (
    context: ToolInvocationContext,
  ) => CommitmentRecord[] | Promise<CommitmentRecord[]>;
  disclosureLabelsForCommitments?: (
    commitments: readonly CommitmentRecord[],
    context: ToolInvocationContext,
  ) =>
    | ReadonlyMap<CommitmentRecord["id"], MemoryDisclosureLabel>
    | Promise<ReadonlyMap<CommitmentRecord["id"], MemoryDisclosureLabel>>;
};

export function createCommitmentsListTool(
  options: CommitmentsListToolOptions,
): ToolDefinition<
  z.infer<typeof commitmentsListInputSchema>,
  z.infer<typeof commitmentsListOutputSchema>
> {
  return {
    name: "tool.commitments.list",
    description: "List active commitments from the being's global memory with disclosure labels.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: commitmentsListInputSchema,
    outputSchema: commitmentsListOutputSchema,
    async invoke(_input, context) {
      const commitments = await options.listCommitments(context);
      const disclosureLabels = await options.disclosureLabelsForCommitments?.(commitments, context);

      return {
        commitments: commitments.map((commitment) => ({
          ...commitment,
          ...memoryDisclosurePayloadFields(
            disclosureLabels?.get(commitment.id) ?? commitmentMemoryDisclosureLabel(commitment),
          ),
        })),
      };
    },
  };
}
