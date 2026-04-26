import { z } from "zod";

import {
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  semanticRelationSchema,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "../../memory/semantic/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const semanticWalkInputSchema = z.object({
  node_id: semanticNodeIdSchema,
  relation: semanticRelationSchema,
  depth: z.number().int().positive().max(4).optional(),
  maxNodes: z.number().int().positive().max(32).optional(),
  asOf: z.number().finite().optional(),
});

const semanticWalkNodeOutputSchema = semanticNodeSchema.omit({
  embedding: true,
});

const semanticWalkOutputSchema = z.object({
  steps: z.array(
    z.object({
      node: semanticWalkNodeOutputSchema,
      edgePath: z.array(semanticEdgeSchema),
    }),
  ),
});

function toSemanticWalkNodeOutput(
  node: SemanticWalkStep["node"],
): z.infer<typeof semanticWalkNodeOutputSchema> {
  return {
    id: node.id,
    kind: node.kind,
    label: node.label,
    description: node.description,
    domain: node.domain,
    aliases: node.aliases,
    confidence: node.confidence,
    source_episode_ids: node.source_episode_ids,
    created_at: node.created_at,
    updated_at: node.updated_at,
    last_verified_at: node.last_verified_at,
    archived: node.archived,
    superseded_by: node.superseded_by,
  };
}

export type SemanticWalkToolOptions = {
  walkGraph: (
    fromId: z.infer<typeof semanticWalkInputSchema>["node_id"],
    options?: SemanticWalkOptions,
    context?: ToolInvocationContext,
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
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: semanticWalkInputSchema,
    outputSchema: semanticWalkOutputSchema,
    async invoke(input, context) {
      const steps = await options.walkGraph(
        input.node_id,
        {
          relations: [input.relation],
          depth: input.depth ?? 2,
          maxNodes: input.maxNodes ?? 16,
          asOf: input.asOf,
        },
        context,
      );

      return {
        steps: steps.map((step) => {
          return {
            node: toSemanticWalkNodeOutput(step.node),
            edgePath: step.edgePath,
          };
        }),
      };
    },
  };
}
