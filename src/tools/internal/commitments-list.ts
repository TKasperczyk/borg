import { z } from "zod";

import { commitmentSchema, type CommitmentRecord } from "../../memory/commitments/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const commitmentsListInputSchema = z.object({}).strict();

const commitmentsListOutputSchema = z.object({
  commitments: z.array(commitmentSchema),
});

export type CommitmentsListToolOptions = {
  listCommitments: () => CommitmentRecord[];
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
    inputSchema: commitmentsListInputSchema,
    outputSchema: commitmentsListOutputSchema,
    async invoke() {
      return {
        commitments: options.listCommitments(),
      };
    },
  };
}
