import { z } from "zod";

import {
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  semanticRelationSchema,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "../../memory/semantic/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  memoryDisclosureLabelMetadata,
  type MemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../../retrieval/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

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
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  });
const semanticWalkEdgeOutputSchema = semanticEdgeSchema.extend({
  disclosure_label: memoryDisclosureLabelMetadataSchema,
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
    disclosure_label: memoryDisclosureLabelMetadata(
      node.disclosureLabel ?? unknownMemoryDisclosureLabel(),
    ),
  };
}

function toSemanticWalkEdgeOutput(
  edge: SemanticWalkEdgeWithDisclosure,
): z.infer<typeof semanticWalkEdgeOutputSchema> {
  const { disclosureLabel, ...edgeFields } = edge;

  return {
    ...edgeFields,
    disclosure_label: memoryDisclosureLabelMetadata(
      disclosureLabel ?? unknownMemoryDisclosureLabel(),
    ),
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
    menuSummary: "Walk semantic memory from a known node.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    // Vector search over the Lance index; between compactions fragment counts
    // climb into the hundreds and honest queries exceed the 5s dispatcher
    // default (observed live: seven consecutive ~5s timeouts across two days).
    timeoutMs: 15_000,
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
