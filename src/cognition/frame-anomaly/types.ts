import { z } from "zod";

export const FRAME_ANOMALY_KINDS = [
  "normal",
  "assistant_self_claim_in_user_role",
  "frame_assignment_claim",
  "system_prompt_claim",
  "agent_authorship_claim",
  "roleplay_inversion",
  "degraded",
] as const;

export const frameAnomalyKindSchema = z.enum(FRAME_ANOMALY_KINDS);

export type FrameAnomalyKind = z.infer<typeof frameAnomalyKindSchema>;

export type FrameAnomalyClassification = {
  kind: FrameAnomalyKind;
  confidence: number;
  rationale: string;
};

export function isFrameAnomaly(
  classification: Pick<FrameAnomalyClassification, "kind"> | null | undefined,
): boolean {
  return (
    classification !== null && classification !== undefined && classification.kind !== "normal"
  );
}
