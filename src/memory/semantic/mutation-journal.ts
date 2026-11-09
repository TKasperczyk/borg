import type { SemanticEdge, SemanticNode } from "./types.js";

export type SemanticMutationRecorder = {
  nodeBefore(id: SemanticNode["id"], before: SemanticNode | null): void;
  edgeBefore(id: SemanticEdge["id"], before: SemanticEdge | null): void;
};

/** Owned by one extraction run, never installed globally on a repository. */
export class SemanticMutationJournal implements SemanticMutationRecorder {
  readonly nodes = new Map<SemanticNode["id"], SemanticNode | null>();
  readonly edges = new Map<SemanticEdge["id"], SemanticEdge | null>();

  nodeBefore(id: SemanticNode["id"], before: SemanticNode | null): void {
    if (!this.nodes.has(id)) this.nodes.set(id, structuredClone(before));
  }

  edgeBefore(id: SemanticEdge["id"], before: SemanticEdge | null): void {
    if (!this.edges.has(id)) this.edges.set(id, structuredClone(before));
  }
}
