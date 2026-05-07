import type { LLMClient } from "../../llm/index.js";
import type { PostGenerationGuardMode } from "../../config/index.js";
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
  mode?: PostGenerationGuardMode;
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

  private mode(): PostGenerationGuardMode {
    return this.options.mode ?? "enforce";
  }

  async run(input: CommitmentGuardRunnerInput): Promise<CommitmentCheckResult> {
    const mode = this.mode();
    const commitmentChecker = new CommitmentChecker({
      llmClient: input.llmClient,
      detectionModel: this.options.detectionModel,
      rewriteModel: this.options.rewriteModel,
      entityRepository: this.options.entityRepository,
    });
    const commitmentCheckerUserMessage =
      input.origin === "autonomous" ? input.userMessage : input.cognitionInput;
    let commitmentCheck: CommitmentCheckResult;

    try {
      commitmentCheck = await commitmentChecker.check({
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
    } catch (error) {
      if (mode !== "shadow") {
        throw error;
      }

      if (this.options.tracer.enabled) {
        this.options.tracer.emit("commitment_check", {
          turnId: input.turnId,
          mode,
          verdict: "passed",
          shadowError: error instanceof Error ? error.message : String(error),
          rewriteTriggered: false,
          violationCount: 0,
        });
      }

      return {
        passed: true,
        violations: [],
        revised: false,
        emission: {
          kind: "message",
          content: input.response,
        },
      };
    }
    const wouldHaveVerdict =
      commitmentCheck.emission.kind === "suppressed"
        ? "suppressed"
        : commitmentCheck.revised
          ? "rewritten"
          : "passed";
    const wouldHaveSuppressionReason =
      commitmentCheck.emission.kind === "suppressed" ? commitmentCheck.emission.reason : undefined;
    const actualCommitmentCheck: CommitmentCheckResult =
      mode === "shadow" && wouldHaveVerdict !== "passed"
        ? {
            ...commitmentCheck,
            revised: false,
            emission: {
              kind: "message",
              content: input.response,
            },
          }
        : commitmentCheck;
    const actualVerdict =
      actualCommitmentCheck.emission.kind === "suppressed"
        ? "suppressed"
        : actualCommitmentCheck.revised
          ? "rewritten"
          : "passed";

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("commitment_check", {
        turnId: input.turnId,
        mode,
        verdict: actualVerdict,
        wouldHaveVerdict,
        ...(wouldHaveSuppressionReason === undefined ? {} : { wouldHaveSuppressionReason }),
        rewriteTriggered: commitmentCheck.revised,
        violationCount: commitmentCheck.violations.length,
      });
    }

    return actualCommitmentCheck;
  }
}
