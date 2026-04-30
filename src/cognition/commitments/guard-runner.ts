import type { LLMClient } from "../../llm/index.js";
import {
  CommitmentChecker,
  type CommitmentCheckResult,
  type CommitmentRecord,
  type EntityRepository,
} from "../../memory/commitments/index.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { TurnTracer } from "../tracing/tracer.js";

export type CommitmentGuardRunnerOptions = {
  detectionModel: string;
  rewriteModel: string;
  entityRepository: EntityRepository;
  tracer: TurnTracer;
};

export type CommitmentGuardRunnerInput = {
  turnId: string;
  llmClient: LLMClient;
  response: string;
  userMessage: string;
  cognitionInput: string;
  origin?: "user" | "autonomous";
  autonomyTrigger?: AutonomyTriggerContext | null;
  commitments: readonly CommitmentRecord[];
  relevantEntities: readonly string[];
};

export class CommitmentGuardRunner {
  constructor(private readonly options: CommitmentGuardRunnerOptions) {}

  async run(input: CommitmentGuardRunnerInput): Promise<CommitmentCheckResult> {
    const commitmentChecker = new CommitmentChecker({
      llmClient: input.llmClient,
      detectionModel: this.options.detectionModel,
      rewriteModel: this.options.rewriteModel,
      entityRepository: this.options.entityRepository,
    });
    const commitmentCheckerUserMessage =
      input.origin === "autonomous" ? input.userMessage : input.cognitionInput;
    const commitmentCheck = await commitmentChecker.check({
      response: input.response,
      userMessage: commitmentCheckerUserMessage,
      untrustedContext:
        input.origin === "autonomous" &&
        input.autonomyTrigger !== null &&
        input.autonomyTrigger !== undefined
          ? input.cognitionInput
          : null,
      commitments: input.commitments,
      relevantEntities: input.relevantEntities,
    });
    if (this.options.tracer.enabled) {
      this.options.tracer.emit("commitment_check", {
        turnId: input.turnId,
        verdict:
          commitmentCheck.emission.kind === "suppressed"
            ? "suppressed"
            : commitmentCheck.revised
              ? "rewritten"
              : "passed",
        rewriteTriggered: commitmentCheck.revised,
        violationCount: commitmentCheck.violations.length,
      });
    }

    return commitmentCheck;
  }
}
