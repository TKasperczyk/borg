import { z } from "zod";

export const FRAME_ANOMALY_KINDS = [
  "normal",
  "assistant_self_claim_in_user_role",
  "frame_assignment_claim",
  "system_prompt_claim",
  "agent_authorship_claim",
  "roleplay_inversion",
] as const;

export const frameAnomalyKindSchema = z.enum(FRAME_ANOMALY_KINDS);

export type FrameAnomalyKind = z.infer<typeof frameAnomalyKindSchema>;

export type FrameAnomalyClassification =
  | {
      status: "ok";
      kind: FrameAnomalyKind;
      confidence: number;
      rationale: string;
    }
  | {
      status: "degraded";
      reason: string;
    };

export type ActualFrameAnomalyClassification = Extract<
  FrameAnomalyClassification,
  { status: "ok" }
> & {
  kind: Exclude<FrameAnomalyKind, "normal">;
};

export function isFrameAnomaly(
  classification: FrameAnomalyClassification | null | undefined,
): classification is ActualFrameAnomalyClassification {
  return (
    classification !== null &&
    classification !== undefined &&
    classification.status === "ok" &&
    classification.kind !== "normal"
  );
}
