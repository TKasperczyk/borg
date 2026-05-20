import type { LLMClient } from "../../llm/index.js";
import type { PostGenerationGuardMode } from "../../config/index.js";
import {
  CommitmentChecker,
  type CommitmentCheckResult,
  type CommitmentKind,
  type CommitmentRecord,
  type EntityRepository,
} from "../../memory/commitments/index.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { TurnTracer } from "../tracing/tracer.js";

export type CommitmentGuardRunnerOptions = {
  detectionModel: string;
  rewriteModel: string;
  mode?: PostGenerationGuardMode;
  criticalKinds?: readonly CommitmentKind[];
  rewriteOnViolation?: boolean;
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

const DEFAULT_CRITICAL_COMMITMENT_KINDS = [
  "boundary",
  "audience_rule",
] as const satisfies readonly CommitmentKind[];

function passedResponse(
  response: string,
  violations: CommitmentCheckResult["violations"] = [],
): CommitmentCheckResult {
  return {
    passed: true,
    violations,
    revised: false,
    emission: {
      kind: "message",
      content: response,
    },
  };
}

function commitmentCheckVerdict(result: CommitmentCheckResult): "passed" | "rewritten" | "suppressed" {
  return result.emission.kind === "suppressed" ? "suppressed" : result.revised ? "rewritten" : "passed";
}

function uniqueCommitmentKinds(commitments: readonly CommitmentRecord[]): CommitmentKind[] {
  return [...new Set(commitments.map((commitment) => commitment.kind))];
}

export class CommitmentGuardRunner {
  constructor(private readonly options: CommitmentGuardRunnerOptions) {}

  private mode(): PostGenerationGuardMode {
    return this.options.mode ?? "enforce";
  }

  private criticalKinds(): readonly CommitmentKind[] {
    return this.options.criticalKinds ?? DEFAULT_CRITICAL_COMMITMENT_KINDS;
  }

  private rewriteOnViolation(): boolean {
    return this.options.rewriteOnViolation === true;
  }

  async run(input: CommitmentGuardRunnerInput): Promise<CommitmentCheckResult> {
    const mode = this.mode();
    const criticalKinds = new Set<CommitmentKind>(this.criticalKinds());
    const enforceCommitments =
      mode === "enforce"
        ? input.commitments.filter((commitment) => criticalKinds.has(commitment.kind))
        : [];
    const shadowCommitments =
      mode === "enforce"
        ? input.commitments.filter((commitment) => !criticalKinds.has(commitment.kind))
        : input.commitments;
    const commitmentChecker = new CommitmentChecker({
      llmClient: input.llmClient,
      detectionModel: this.options.detectionModel,
      rewriteModel: this.options.rewriteModel,
      entityRepository: this.options.entityRepository,
    });
    const commitmentCheckerUserMessage =
      input.origin === "autonomous" ? input.userMessage : input.cognitionInput;
    let shadowCheck: CommitmentCheckResult | null = null;
    let shadowError: string | undefined;

    if (shadowCommitments.length > 0) {
      try {
        shadowCheck = await commitmentChecker.check({
          response: input.response,
          userMessage: commitmentCheckerUserMessage,
          untrustedContext:
            input.origin === "autonomous" &&
            input.autonomyTrigger !== null &&
            input.autonomyTrigger !== undefined
              ? input.cognitionInput
              : null,
          commitments: shadowCommitments,
          relevantEntities: input.relevantEntities,
          rewriteOnViolation: false,
        });
      } catch (error) {
        shadowError = error instanceof Error ? error.message : String(error);
      }

      if (this.options.tracer.enabled && shadowCheck !== null && shadowCheck.violations.length > 0) {
        const wouldHaveVerdict = commitmentCheckVerdict(shadowCheck);

        this.options.tracer.emit("commitment_guard.shadow_observation", {
          turnId: input.turnId,
          mode: "shadow",
          verdict: "passed",
          wouldHaveVerdict,
          ...(shadowCheck.emission.kind === "suppressed"
            ? { wouldHaveSuppressionReason: shadowCheck.emission.reason }
            : {}),
          rewriteTriggered: false,
          violationCount: shadowCheck.violations.length,
          commitmentIds: shadowCheck.violations.map((violation) => violation.commitment_id),
          commitmentKinds: uniqueCommitmentKinds(shadowCommitments),
        });
      }
    }

    if (mode === "shadow") {
      const wouldHaveVerdict = shadowCheck === null ? "passed" : commitmentCheckVerdict(shadowCheck);
      const actualCommitmentCheck =
        shadowCheck === null
          ? passedResponse(input.response)
          : passedResponse(input.response, shadowCheck.violations);

      if (this.options.tracer.enabled) {
        this.options.tracer.emit("commitment_check.completed", {
          turnId: input.turnId,
          mode,
          verdict: "passed",
          wouldHaveVerdict,
          ...(shadowCheck?.emission.kind === "suppressed"
            ? { wouldHaveSuppressionReason: shadowCheck.emission.reason }
            : {}),
          ...(shadowError === undefined ? {} : { shadowError }),
          rewriteTriggered: false,
          violationCount: shadowCheck?.violations.length ?? 0,
        });
      }

      return actualCommitmentCheck;
    }

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
        commitments: enforceCommitments,
        relevantEntities: input.relevantEntities,
        rewriteOnViolation: this.rewriteOnViolation(),
      });
    } catch (error) {
      throw error;
    }
    const wouldHaveVerdict = commitmentCheckVerdict(commitmentCheck);
    const wouldHaveSuppressionReason =
      commitmentCheck.emission.kind === "suppressed" ? commitmentCheck.emission.reason : undefined;
    const actualVerdict = commitmentCheckVerdict(commitmentCheck);

    if (this.options.tracer.enabled && commitmentCheck.emission.kind === "suppressed") {
      this.options.tracer.emit("commitment_guard.enforce_suppression", {
        turnId: input.turnId,
        mode: "enforce",
        verdict: "suppressed",
        reason: commitmentCheck.emission.reason,
        rewriteTriggered: commitmentCheck.revised,
        violationCount: commitmentCheck.violations.length,
        commitmentIds: commitmentCheck.violations.map((violation) => violation.commitment_id),
        commitmentKinds: uniqueCommitmentKinds(enforceCommitments),
      });
    }

    if (
      this.options.tracer.enabled &&
      commitmentCheck.revised &&
      commitmentCheck.emission.kind === "message"
    ) {
      this.options.tracer.emit("commitment_guard.enforce_rewrite", {
        turnId: input.turnId,
        mode: "enforce",
        verdict: "rewritten",
        rewriteTriggered: true,
        violationCount: commitmentCheck.violations.length,
        commitmentIds: commitmentCheck.violations.map((violation) => violation.commitment_id),
        commitmentKinds: uniqueCommitmentKinds(enforceCommitments),
      });
    }

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("commitment_check.completed", {
        turnId: input.turnId,
        mode,
        verdict: actualVerdict,
        wouldHaveVerdict,
        ...(wouldHaveSuppressionReason === undefined ? {} : { wouldHaveSuppressionReason }),
        ...(shadowError === undefined ? {} : { shadowError }),
        rewriteTriggered: commitmentCheck.revised,
        violationCount: commitmentCheck.violations.length,
        ...(shadowCheck === null || shadowCheck.violations.length === 0
          ? {}
          : { shadowViolationCount: shadowCheck.violations.length }),
      });
    }

    if (enforceCommitments.length === 0 && shadowCheck !== null) {
      return passedResponse(input.response, shadowCheck.violations);
    }

    return commitmentCheck;
  }
}
