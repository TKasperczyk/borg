import type {
  ReviewKind,
  ReviewQueueHandler,
  ReviewQueueHandlerRegistry,
  ReviewQueueRepository,
} from "../review-queue.js";
import { createBeliefRevisionReviewQueueHandler } from "./belief-revision.js";
import { createIdentityInconsistencyReviewQueueHandler } from "./identity-inconsistency.js";
import { createMisattributionReviewQueueHandler } from "./misattribution.js";
import { createNewInsightReviewQueueHandler } from "./new-insight.js";
import { createSemanticPairReviewQueueHandler } from "./semantic-pair.js";
import { createTemporalDriftReviewQueueHandler } from "./temporal-drift.js";

type ReviewQueueHandlerRegistrar =
  | Pick<ReviewQueueHandlerRegistry, "register">
  | Pick<ReviewQueueRepository, "registerHandler">;

function registerHandler<K extends ReviewKind, TRefs, TState>(
  registrar: ReviewQueueHandlerRegistrar,
  handler: ReviewQueueHandler<K, TRefs, TState>,
): void {
  if ("register" in registrar) {
    registrar.register(handler);
    return;
  }

  registrar.registerHandler(handler);
}

export function registerBuiltinReviewQueueHandlers(registrar: ReviewQueueHandlerRegistrar): void {
  registerHandler(registrar, createSemanticPairReviewQueueHandler("contradiction"));
  registerHandler(registrar, createSemanticPairReviewQueueHandler("duplicate"));
  registerHandler(registrar, createNewInsightReviewQueueHandler());
  registerHandler(registrar, createMisattributionReviewQueueHandler());
  registerHandler(registrar, createTemporalDriftReviewQueueHandler());
  registerHandler(registrar, createIdentityInconsistencyReviewQueueHandler());
  registerHandler(registrar, createBeliefRevisionReviewQueueHandler());
}
