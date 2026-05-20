import type { OpenQuestion, OpenQuestionsRepository } from "../self/open-questions.js";
import type {
  IdentityService,
  IdentityUpdateOptions,
  IdentityUpdateResult,
} from "../identity/service.js";
import type { Provenance } from "../common/provenance.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import type {
  EpisodeId,
  OpenQuestionId,
  SharedStateEntryId,
  StreamEntryId,
} from "../../util/ids.js";
import type { LifecycleOperationResult, LifecycleTracer } from "./types.js";

export type ResolveOpenQuestionRepository = Pick<OpenQuestionsRepository, "resolve"> &
  Partial<Pick<OpenQuestionsRepository, "get">>;
export type ResolveOpenQuestionIdentityService = Pick<IdentityService, "resolveOpenQuestion">;

export function resolveOpenQuestionWithEvidence(input: {
  openQuestionId: OpenQuestionId;
  repository: ResolveOpenQuestionRepository;
  resolutionEvidenceEpisodeIds?: readonly EpisodeId[];
  resolutionEvidenceStreamEntryIds?: readonly StreamEntryId[];
  resolutionNote: string;
  resolvedByArtifactEntryId?: SharedStateEntryId | null;
  tracer?: LifecycleTracer;
  turnId?: string;
  traceSourcePath?: string;
  traceDecisionReason?: string;
}): LifecycleOperationResult<{ question: OpenQuestion | null }> {
  const previous = input.repository.get?.(input.openQuestionId);

  if (input.repository.get !== undefined && previous == null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        question: null,
      },
    };
  }

  if (previous !== undefined && previous !== null && previous.status !== "open") {
    return {
      status: "no_op",
      reason: "terminal",
      value: {
        question: previous,
      },
    };
  }

  let resolved: OpenQuestion;

  try {
    resolved = input.repository.resolve(
      input.openQuestionId,
      {
        resolution_evidence_episode_ids: input.resolutionEvidenceEpisodeIds,
        resolution_evidence_stream_entry_ids: input.resolutionEvidenceStreamEntryIds,
        resolution_note: input.resolutionNote,
      },
      {
        resolvedByArtifactEntryId: input.resolvedByArtifactEntryId,
      },
    );
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) {
      return {
        status: "conflict",
        error,
      };
    }

    throw error;
  }

  if (input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("open_question_resolution.transitioned", {
      turnId: input.turnId,
      oq_id: input.openQuestionId,
      source_path: input.traceSourcePath,
      decision: "resolved",
      decision_reason: input.traceDecisionReason,
      evidence_episode_count: input.resolutionEvidenceEpisodeIds?.length ?? 0,
      evidence_stream_entry_count: input.resolutionEvidenceStreamEntryIds?.length ?? 0,
    });
  }

  return {
    status: "success",
    value: {
      question: resolved,
    },
  };
}

export function resolveOpenQuestionThroughIdentityService(input: {
  openQuestionId: OpenQuestionId;
  identityService: ResolveOpenQuestionIdentityService;
  resolution: Parameters<OpenQuestionsRepository["resolve"]>[1];
  provenance: Provenance;
  options?: IdentityUpdateOptions;
  tracer?: LifecycleTracer;
  turnId?: string;
  traceSourcePath?: string;
  traceDecisionReason?: string;
}): LifecycleOperationResult<{
  result: IdentityUpdateResult<OpenQuestion>;
}> {
  let result: IdentityUpdateResult<OpenQuestion>;

  try {
    result = input.identityService.resolveOpenQuestion(
      input.openQuestionId,
      input.resolution,
      input.provenance,
      input.options,
    );
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) {
      return {
        status: "conflict",
        error,
      };
    }

    throw error;
  }

  if (result.status === "requires_review") {
    return {
      status: "no_op",
      reason: "requires_review",
      value: {
        result,
      },
    };
  }

  if (input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("open_question_resolution.transitioned", {
      turnId: input.turnId,
      oq_id: input.openQuestionId,
      source_path: input.traceSourcePath,
      decision: "resolved",
      decision_reason: input.traceDecisionReason,
      evidence_episode_count:
        input.resolution.resolution_evidence_episode_ids?.length ?? 0,
      evidence_stream_entry_count:
        input.resolution.resolution_evidence_stream_entry_ids?.length ?? 0,
    });
  }

  return {
    status: "success",
    value: {
      result,
    },
  };
}
