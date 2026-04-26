import { z } from "zod";

import { commitmentSchema, type CommitmentRecord } from "../../memory/commitments/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const commitmentsListInputSchema = z.object({}).strict();

const commitmentsListOutputSchema = z.object({
  commitments: z.array(commitmentSchema),
});

export type CommitmentsListToolOptions = {
  listCommitments: (context: ToolInvocationContext) => CommitmentRecord[];
};

export function createCommitmentsListTool(
  options: CommitmentsListToolOptions,
): ToolDefinition<
  z.infer<typeof commitmentsListInputSchema>,
  z.infer<typeof commitmentsListOutputSchema>
> {
  return {
    name: "tool.commitments.list",
    description: "List active commitments.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: commitmentsListInputSchema,
    outputSchema: commitmentsListOutputSchema,
    async invoke(_input, context) {
      return {
        commitments: options.listCommitments(context),
      };
    },
  };
}
