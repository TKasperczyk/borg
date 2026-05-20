import type {
  TurnTraceData,
  TurnTraceEventName,
  TurnTracer,
} from "../../cognition/tracing/tracer.js";

export type TestTurnTraceRecorder = TurnTracer & {
  events: Array<{ event: TurnTraceEventName; data: TurnTraceData }>;
};

export function makeTestTurnTraceRecorder(): TestTurnTraceRecorder {
  const events: TestTurnTraceRecorder["events"] = [];

  return {
    enabled: true,
    includePayloads: true,
    events,
    emit: (event, data) => {
      events.push({ event, data });
    },
  };
}
