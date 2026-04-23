import { z } from "zod";

import {
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  semanticRelationSchema,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "../../memory/semantic/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const semanticWalkInputSchema = z.object({
  node_id: semanticNodeIdSchema,
  relation: semanticRelationSchema,
  depth: z.number().int().positive().max(4).optional(),
  maxNodes: z.number().int().positive().max(32).optional(),
});

const semanticWalkOutputSchema = z.object({
  steps: z.array(
    z.object({
      node: semanticNodeSchema,
      edgePath: z.array(semanticEdgeSchema),
    }),
  ),
});

export type SemanticWalkToolOptions = {
  walkGraph: (
    fromId: z.infer<typeof semanticWalkInputSchema>["node_id"],
    options?: SemanticWalkOptions,
  ) => Promise<SemanticWalkStep[]>;
};

export function createSemanticWalkTool(
  options: SemanticWalkToolOptions,
): ToolDefinition<
  z.infer<typeof semanticWalkInputSchema>,
  z.infer<typeof semanticWalkOutputSchema>
> {
  return {
    name: "tool.semantic.walk",
    description: "Walk the semantic graph from a node across one relation family.",
    inputSchema: semanticWalkInputSchema,
    outputSchema: semanticWalkOutputSchema,
    async invoke(input) {
      return {
        steps: await options.walkGraph(input.node_id, {
          relations: [input.relation],
          depth: input.depth ?? 2,
          maxNodes: input.maxNodes ?? 16,
        }),
      };
    },
  };
}
