import { z } from "zod";

import {
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  semanticRelationSchema,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "../../memory/semantic/index.js";
import { MEMORY_DISCLOSURE_CLASSES, type MemoryDisclosureLabel } from "../../retrieval/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

// TODO(post-inversion tool output cleanup): migrate semantic walk disclosure output to the
// shared snake_case disclosure_label serializer used by episodic search. This is an observable
// tool contract change, so keep the current camelCase shape during pure refactor sessions.
const semanticWalkDisclosureLabelSchema = z.object({
  disclosureClass: z.enum(MEMORY_DISCLOSURE_CLASSES),
  originAudienceEntityIds: z.array(z.string()),
  privateToEntityIds: z.array(z.string()),
  publicToEntityIds: z.array(z.string()),
});

const semanticWalkInputSchema = z.object({
  node_id: semanticNodeIdSchema,
  relation: semanticRelationSchema,
  depth: z.number().int().positive().max(4).optional(),
  maxNodes: z.number().int().positive().max(32).optional(),
  asOf: z.number().finite().optional(),
});

const semanticWalkNodeOutputSchema = semanticNodeSchema
  .omit({
    corrected_by: true,
    embedding: true,
  })
  .extend({
    partial_source_visibility: z.boolean().optional(),
    source_visibility_fraction: z.number().min(0).max(1).optional(),
    disclosureLabel: semanticWalkDisclosureLabelSchema.optional(),
  });
const semanticWalkEdgeOutputSchema = semanticEdgeSchema.extend({
  disclosureLabel: semanticWalkDisclosureLabelSchema.optional(),
});

const semanticWalkOutputSchema = z.object({
  steps: z.array(
    z.object({
      node: semanticWalkNodeOutputSchema,
      edgePath: z.array(semanticWalkEdgeOutputSchema),
    }),
  ),
});

type SemanticWalkNodeWithDisclosure = SemanticWalkStep["node"] & {
  partial_source_visibility?: boolean;
  source_visibility_fraction?: number;
  disclosureLabel?: MemoryDisclosureLabel;
};

type SemanticWalkEdgeWithDisclosure = SemanticWalkStep["edgePath"][number] & {
  disclosureLabel?: MemoryDisclosureLabel;
};

type SemanticWalkStepWithDisclosure = Omit<SemanticWalkStep, "node" | "edgePath"> & {
  node: SemanticWalkNodeWithDisclosure;
  edgePath: SemanticWalkEdgeWithDisclosure[];
};

function toSemanticWalkDisclosureLabelOutput(
  label: MemoryDisclosureLabel,
): z.infer<typeof semanticWalkDisclosureLabelSchema> {
  return {
    disclosureClass: label.disclosureClass,
    originAudienceEntityIds: [...label.originAudienceEntityIds],
    privateToEntityIds: [...label.privateToEntityIds],
    publicToEntityIds: [...label.publicToEntityIds],
  };
}

function toSemanticWalkNodeOutput(
  node: SemanticWalkNodeWithDisclosure,
): z.infer<typeof semanticWalkNodeOutputSchema> {
  return {
    id: node.id,
    kind: node.kind,
    label: node.label,
    description: node.description,
    domain: node.domain,
    aliases: node.aliases,
    observation_metadata: node.observation_metadata,
    confidence: node.confidence,
    source_episode_ids: node.source_episode_ids,
    created_at: node.created_at,
    updated_at: node.updated_at,
    last_verified_at: node.last_verified_at,
    archived: node.archived,
    superseded_by: node.superseded_by,
    status: node.status,
    superseded_at: node.superseded_at,
    ...(node.partial_source_visibility === undefined
      ? {}
      : { partial_source_visibility: node.partial_source_visibility }),
    ...(node.source_visibility_fraction === undefined
      ? {}
      : { source_visibility_fraction: node.source_visibility_fraction }),
    ...(node.disclosureLabel === undefined
      ? {}
      : { disclosureLabel: toSemanticWalkDisclosureLabelOutput(node.disclosureLabel) }),
  };
}

function toSemanticWalkEdgeOutput(
  edge: SemanticWalkEdgeWithDisclosure,
): z.infer<typeof semanticWalkEdgeOutputSchema> {
  const { disclosureLabel, ...edgeFields } = edge;

  return {
    ...edgeFields,
    ...(disclosureLabel === undefined
      ? {}
      : { disclosureLabel: toSemanticWalkDisclosureLabelOutput(disclosureLabel) }),
  };
}

export type SemanticWalkToolOptions = {
  walkGraph: (
    fromId: z.infer<typeof semanticWalkInputSchema>["node_id"],
    options?: SemanticWalkOptions,
    context?: ToolInvocationContext,
  ) => Promise<SemanticWalkStepWithDisclosure[]>;
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
            edgePath: step.edgePath.map(toSemanticWalkEdgeOutput),
          };
        }),
      };
    },
  };
}
