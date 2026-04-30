export type AutonomyTriggerContext = {
  source_name: string;
  source_type: "trigger" | "condition";
  event_id: string;
  sort_ts: number;
  payload: Record<string, unknown>;
};

export function formatAutonomyTriggerContext(context: AutonomyTriggerContext): string {
  const payload = JSON.stringify(context.payload, null, 2) ?? "{}";
  const sortTs = Number.isFinite(context.sort_ts)
    ? new Date(context.sort_ts).toISOString()
    : String(context.sort_ts);

  return [
    "Autonomous wake context:",
    `source_name: ${context.source_name}`,
    `source_type: ${context.source_type}`,
    `event_id: ${context.event_id}`,
    `sort_ts: ${sortTs}`,
    "payload:",
    payload,
  ].join("\n");
}
