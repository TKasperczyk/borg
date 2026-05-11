import sarahPartnerName from "./01-sarah-partner-name.js";
import martaTutorInvention from "./02-marta-tutor-invention.js";
import barcelonaItinerary from "./03-barcelona-itinerary.js";
import threeHundredSoups from "./04-three-hundred-soups.js";
import falseActionCompletion from "./05-false-action-completion.js";
import falsePriorCallback from "./06-false-prior-callback.js";
import frameInversion from "./07-frame-inversion.js";
import agentSelfHistory from "./08-agent-self-history.js";
import phenomenology from "./09-phenomenology.js";
import closureLoop from "./10-closure-loop.js";
import crossSessionItalki from "./11-cross-session-italki.js";
import selfReportNotProof from "./17-self-report-not-proof.js";
import tomLeakConfigName from "./21-tom-leak-config-name.js";
import tomAllowedAfterSelfDeclaration from "./22-tom-allowed-after-self-declaration.js";
import suppressionNetworkWeather from "./27-suppression-network-weather.js";
import closureLoopPersistence from "./28-closure-loop-persistence.js";
import crossSessionPartnerNameConflict from "./29-cross-session-partner-name-conflict.js";
import tomLeakUnflaggedVocative from "./32-tom-leak-unflagged-vocative.js";
import tripVsRelocation from "./33-trip-vs-relocation.js";
import secondaryThreadCoverage from "./44-secondary-thread-coverage.js";
import type { ReplayScenario } from "../scenario.js";

export const REPLAY_SCENARIOS: readonly ReplayScenario[] = [
  sarahPartnerName,
  martaTutorInvention,
  barcelonaItinerary,
  threeHundredSoups,
  falseActionCompletion,
  falsePriorCallback,
  frameInversion,
  agentSelfHistory,
  phenomenology,
  closureLoop,
  crossSessionItalki,
  selfReportNotProof,
  tomLeakConfigName,
  tomAllowedAfterSelfDeclaration,
  suppressionNetworkWeather,
  closureLoopPersistence,
  crossSessionPartnerNameConflict,
  tomLeakUnflaggedVocative,
  tripVsRelocation,
  secondaryThreadCoverage,
];
