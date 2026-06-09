import type { TurnOrigin } from "../contracts/cognitive-contracts.js";

export {
  COGNITIVE_MODES,
  TURN_ORIGINS,
  affectiveSignalSchema,
  attentionWeightsSchema,
  cognitiveModeSchema,
  intentRecordSchema,
  perceivedEntitySchema,
  perceptionResultSchema,
  temporalCueSchema,
  turnOriginSchema,
  type AffectiveSignal,
  type AttentionWeights,
  type CognitiveMode,
  type IntentRecord,
  type PerceptionResult,
  type TemporalCue,
  type TurnOrigin,
} from "../contracts/cognitive-contracts.js";

export function isUserTurnOrigin(origin: TurnOrigin | undefined): boolean {
  return origin === undefined || origin === "user";
}

export function isAutonomousLikeTurnOrigin(origin: TurnOrigin | undefined): boolean {
  return origin === "autonomous" || origin === "directed_outbound";
}

export function isDirectedOutboundTurnOrigin(origin: TurnOrigin | undefined): boolean {
  return origin === "directed_outbound";
}

export function runsExtraction(origin: TurnOrigin | undefined): boolean {
  return !isDirectedOutboundTurnOrigin(origin);
}

export function persistsPerception(origin: TurnOrigin | undefined): boolean {
  return !isDirectedOutboundTurnOrigin(origin);
}

export function runsReflectionPersistence(origin: TurnOrigin | undefined): boolean {
  return !isDirectedOutboundTurnOrigin(origin);
}

export function exposesOutboundTool(origin: TurnOrigin | undefined): boolean {
  return !isDirectedOutboundTurnOrigin(origin);
}

export function hasAutonomousTriggerUntrustedContext(origin: TurnOrigin | undefined): boolean {
  return origin === "autonomous";
}
