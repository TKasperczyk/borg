import type { JsonValue } from "../../util/json-value.js";

export type LifecycleOperationResult<T = unknown> =
  | {
      status: "success";
      value: T;
    }
  | {
      status: "no_op";
      reason: string;
      value?: T;
    }
  | {
      status: "conflict";
      error: Error;
      value?: T;
    };

export type LifecycleTraceEventName =
  | "action_state.transitioned"
  | "action_state.borg_self_performance.completed"
  | "action_state.archived"
  | "action_session_scope.expired"
  | "action_session_scope.rolled_over"
  | "extraction.commitments.transitioned"
  | "extraction.goals.transitioned"
  | "open_question_resolution.transitioned"
  | "semantic_node.status.transitioned";

export type LifecycleTraceData = {
  turnId: string;
  [key: string]: JsonValue | undefined;
};

export type LifecycleTracer = {
  readonly enabled: boolean;
  emit(event: LifecycleTraceEventName, data: LifecycleTraceData): void;
};
