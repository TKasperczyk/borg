import { SemanticError } from "../../util/errors.js";
import type { SemanticNodeId } from "../../util/ids.js";
import { SemanticEdgeRepository, SemanticNodeRepository } from "./repository.js";
import type {
  SemanticEdge,
  SemanticNode,
  SemanticRelation,
  SemanticWalkOptions,
  SemanticWalkStep,
} from "./types.js";

export type SemanticGraphOptions = {
  nodeRepository: SemanticNodeRepository;
  edgeRepository: SemanticEdgeRepository;
};

const SYMMETRIC_WALK_RELATIONS = new Set<SemanticRelation>([
  "contradicts",
  "related_to",
]);

function defaultDirectionForRelation(relation: SemanticRelation): "out" | "both" {
  return SYMMETRIC_WALK_RELATIONS.has(relation) ? "both" : "out";
}

function resolveWalkDirection(
  relations: readonly SemanticRelation[] | undefined,
  override: "out" | "in" | "both" | undefined,
): "out" | "in" | "both" {
  if (override !== undefined) {
    return override;
  }

  if (relations === undefined || relations.length === 0) {
    return "both";
  }

  const directions = new Set(relations.map((relation) => defaultDirectionForRelation(relation)));
  return directions.size === 1 ? [...directions][0]! : "both";
}

export class SemanticGraph {
  constructor(private readonly options: SemanticGraphOptions) {}

  async neighbors(
    id: SemanticNodeId,
    options: {
      relations?: readonly SemanticRelation[];
      direction: "out" | "in" | "both";
      asOf?: number;
      includeInvalid?: boolean;
    },
  ): Promise<Array<{ node: SemanticNode; edge: SemanticEdge }>> {
    const validityOptions = {
      asOf: options.asOf,
      includeInvalid: options.includeInvalid,
    };
    const edges =
      options.direction === "both"
        ? [
            ...this.options.edgeRepository.listEdges({
              fromId: id,
              ...validityOptions,
            }),
            ...this.options.edgeRepository.listEdges({
              toId: id,
              ...validityOptions,
            }),
          ]
        : this.options.edgeRepository.listEdges(
            options.direction === "out"
              ? {
                  fromId: id,
                  ...validityOptions,
                }
              : {
                  toId: id,
                  ...validityOptions,
                },
          );
    const filtered =
      options.relations === undefined || options.relations.length === 0
        ? edges
        : edges.filter((edge) => options.relations?.includes(edge.relation));
    const nodeIds = filtered.map((edge) =>
      edge.from_node_id === id ? edge.to_node_id : edge.from_node_id,
    );
    const nodes = await this.options.nodeRepository.getMany(nodeIds, {
      includeArchived: false,
    });

    const results: Array<{ node: SemanticNode; edge: SemanticEdge }> = [];

    for (const [index, edge] of filtered.entries()) {
      const node = nodes[index];

      if (node === null || node === undefined) {
        continue;
      }

      results.push({
        node,
        edge,
      });
    }

    return results;
  }

  async walk(
    fromId: SemanticNodeId,
    options: SemanticWalkOptions = {},
  ): Promise<SemanticWalkStep[]> {
    const depth = options.depth ?? 2;
    const maxNodes = options.maxNodes ?? 32;

    if (depth <= 0 || maxNodes <= 0) {
      throw new SemanticError("Graph walk bounds must be positive", {
        code: "SEMANTIC_WALK_INVALID",
      });
    }

    const visited = new Set<string>([fromId]);
    const queue: Array<{ id: SemanticNodeId; depth: number; path: SemanticEdge[] }> = [
      { id: fromId, depth: 0, path: [] },
    ];
    const steps: SemanticWalkStep[] = [];
    const direction = resolveWalkDirection(options.relations, options.direction);

    while (queue.length > 0 && steps.length < maxNodes) {
      const next = queue.shift();

      if (next === undefined || next.depth >= depth) {
        continue;
      }

      const neighbors = await this.neighbors(next.id, {
        relations: options.relations,
        direction,
        asOf: options.asOf,
        includeInvalid: options.includeInvalid,
      });

      for (const neighbor of neighbors) {
        if (visited.has(neighbor.node.id)) {
          continue;
        }

        visited.add(neighbor.node.id);
        const edgePath = [...next.path, neighbor.edge];
        steps.push({
          node: neighbor.node,
          edgePath,
        });

        if (steps.length >= maxNodes) {
          break;
        }

        queue.push({
          id: neighbor.node.id,
          depth: next.depth + 1,
          path: edgePath,
        });
      }
    }

    return steps;
  }

  async contradictionsOf(id: SemanticNodeId): Promise<SemanticNode[]> {
    const neighbors = await this.neighbors(id, {
      relations: ["contradicts"],
      direction: "both",
    });

    return neighbors.map((item) => item.node);
  }

  async supportsFor(id: SemanticNodeId): Promise<SemanticNode[]> {
    const neighbors = await this.neighbors(id, {
      relations: ["supports"],
      direction: "out",
    });

    return neighbors.map((item) => item.node);
  }
}
