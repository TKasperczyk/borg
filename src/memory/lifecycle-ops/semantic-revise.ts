import type { SemanticNodeRepository } from "../semantic/repository.js";
import type { SemanticNodeCorrectionRef, SemanticNodeIdValue } from "../semantic/types.js";
import type { SemanticNodeStatusTransition } from "../semantic/repository.js";
import type { LifecycleOperationResult, LifecycleTracer } from "./types.js";

export type SemanticStatusRepository = Pick<
  SemanticNodeRepository,
  "markSuperseded" | "markContradicted"
>;

export type SemanticLifecycleTraceSource =
  | "belief_reviser"
  | "decision_artifact_semantic_revision"
  | "review_handler"
  | "review_resolver";

function traceSemanticStatusTransition(input: {
  tracer?: LifecycleTracer;
  turnId?: string;
  transition: SemanticNodeStatusTransition | null;
  source: SemanticLifecycleTraceSource;
}): void {
  if (input.transition === null || input.tracer?.enabled !== true || input.turnId === undefined) {
    return;
  }

  input.tracer.emit("semantic_node.status.transitioned", {
    turnId: input.turnId,
    nodeId: input.transition.id,
    fromStatus: input.transition.fromStatus,
    toStatus: input.transition.toStatus,
    correctedBy: input.transition.correctedBy,
    source: input.source,
  });
}

export async function markSemanticSuperseded(input: {
  nodeId: SemanticNodeIdValue;
  correctedBy: SemanticNodeCorrectionRef;
  supersededAt: number;
  repository: Pick<SemanticNodeRepository, "markSuperseded">;
  tracer?: LifecycleTracer;
  turnId?: string;
  traceSource: SemanticLifecycleTraceSource;
}): Promise<LifecycleOperationResult<{ transition: SemanticNodeStatusTransition | null }>> {
  const transition = await input.repository.markSuperseded(
    input.nodeId,
    input.correctedBy,
    input.supersededAt,
  );

  traceSemanticStatusTransition({
    tracer: input.tracer,
    turnId: input.turnId,
    transition,
    source: input.traceSource,
  });

  if (transition === null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        transition: null,
      },
    };
  }

  return {
    status: "success",
    value: {
      transition,
    },
  };
}

export async function markSemanticContradicted(input: {
  nodeId: SemanticNodeIdValue;
  correctedBy: SemanticNodeCorrectionRef;
  supersededAt: number;
  repository: Pick<SemanticNodeRepository, "markContradicted">;
  tracer?: LifecycleTracer;
  turnId?: string;
  traceSource: SemanticLifecycleTraceSource;
}): Promise<LifecycleOperationResult<{ transition: SemanticNodeStatusTransition | null }>> {
  const transition = await input.repository.markContradicted(
    input.nodeId,
    input.correctedBy,
    input.supersededAt,
  );

  traceSemanticStatusTransition({
    tracer: input.tracer,
    turnId: input.turnId,
    transition,
    source: input.traceSource,
  });

  if (transition === null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        transition: null,
      },
    };
  }

  return {
    status: "success",
    value: {
      transition,
    },
  };
}
