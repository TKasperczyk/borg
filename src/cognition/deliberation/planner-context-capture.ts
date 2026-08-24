import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { z } from "zod";

import type { LLMCompleteOptions, LLMMessage, LLMSystemBlock } from "../../llm/index.js";
import type {
  MemoryDisclosureLabel,
  MemoryDisclosureLabelMetadata,
} from "../../memory/common/disclosure-label.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { NOOP_TRACER } from "../../tracing/tracer.js";
import type { Clock } from "../../util/clock.js";
import { SystemClock } from "../../util/clock.js";
import type { SessionId } from "../../util/ids.js";
import type { CompactPlannerLedgerPrompt } from "../evidence-ledger/index.js";
import type { PromptSurfaceAdditionalSection } from "../prompts/prompt-surface-registry.js";
import { buildCompactPlannerSystemPrompt } from "./prompt/planner-context.js";
import {
  createS2PlannerRequestSnapshot,
  renderS2PlannerSurface,
  type RenderedS2PlannerSurface,
  type S2PlannerOutcome,
  type S2PlannerRequestSnapshot,
  type S2PlannerResult,
} from "./s2-planner.js";
import type { DeliberationContext, SelfSnapshot } from "./types.js";
import {
  fingerprintCanonicalRequest,
  fingerprintSystemSurface,
  llmSystemText,
} from "./request-fingerprint.js";
import {
  appendBoundedContextCapture,
  resolveContextCaptureStoragePath,
} from "./context-capture-storage.js";

export const PLANNER_CONTEXT_CAPTURE_SCHEMA_VERSION = 2 as const;
export const PLANNER_CONTEXT_CAPTURE_RELATIVE_PATH = join("captures", "planner-contexts.jsonl");
export const DEFAULT_PLANNER_CONTEXT_CAPTURE_MAX_RECORD_BYTES = 16 * 1024 * 1024;
export const DEFAULT_PLANNER_CONTEXT_CAPTURE_MAX_FILE_BYTES = 512 * 1024 * 1024;

type JsonObject = Record<string, unknown>;

function isJsonObject(value: unknown): value is JsonObject {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function projectDisclosureLabel(
  value: MemoryDisclosureLabel | MemoryDisclosureLabelMetadata | null | undefined,
): MemoryDisclosureLabel | MemoryDisclosureLabelMetadata | undefined {
  if (value === null || value === undefined) {
    return undefined;
  }

  if ("disclosureClass" in value) {
    return {
      disclosureClass: value.disclosureClass,
      originAudienceEntityIds: [...value.originAudienceEntityIds],
      privateToEntityIds: [...value.privateToEntityIds],
      publicToEntityIds: [...value.publicToEntityIds],
    };
  }

  return {
    disclosure_class: value.disclosure_class,
    origin_audience_entity_ids: [...value.origin_audience_entity_ids],
    private_to_entity_ids: [...value.private_to_entity_ids],
    public_to_entity_ids: [...value.public_to_entity_ids],
  };
}

function projectCreatorDirective(
  directive: NonNullable<DeliberationContext["creatorDirectiveBriefing"]>["directives"][number],
) {
  const common = {
    priority: directive.priority,
    createdAt: directive.createdAt,
    ...(directive.scope === undefined
      ? {}
      : {
          scope: {
            ...directive.scope,
            allowedEntityIds: [...directive.scope.allowedEntityIds],
            excludedEntityIds: [...directive.scope.excludedEntityIds],
            activationAllowedEntityIds: [...directive.scope.activationAllowedEntityIds],
            activationExcludedEntityIds: [...directive.scope.activationExcludedEntityIds],
          },
        }),
  };
  if (directive.renderMode === "boundary") {
    return { renderMode: directive.renderMode, ...common };
  }
  if (directive.renderMode === "private" && directive.privateKind === "operation") {
    return {
      renderMode: directive.renderMode,
      privateKind: directive.privateKind,
      kind: directive.kind,
      operationalDirective: directive.operationalDirective,
      ...common,
    };
  }

  const shared = {
    renderMode: directive.renderMode,
    ...(directive.renderMode === "private" ? { privateKind: directive.privateKind } : {}),
    kind: directive.kind,
    subjectKind: directive.subjectKind,
    subjectLabel: directive.subjectLabel,
    semanticSlot: directive.semanticSlot,
    mentionPolicy: directive.mentionPolicy,
    ...common,
  };
  if (
    directive.renderMode === "content" &&
    (directive.kind === "response_policy" || directive.kind === "routing_instruction")
  ) {
    return { ...shared, operationalDirective: directive.operationalDirective };
  }
  if (directive.semanticSlot !== null) {
    return { ...shared, semanticValue: directive.semanticValue };
  }
  return { ...shared, canonicalFact: directive.canonicalFact };
}

function projectSocialProfileFields(profile: NonNullable<DeliberationContext["audienceProfile"]>) {
  return {
    trust: profile.trust,
    attachment: profile.attachment,
    communication_style: profile.communication_style,
    shared_history_summary: profile.shared_history_summary,
    last_interaction_at: profile.last_interaction_at,
    interaction_count: profile.interaction_count,
    commitment_count: profile.commitment_count,
    notes: profile.notes,
  };
}

function projectEvidenceMetadata(metadata: Record<string, unknown> | undefined) {
  if (metadata === undefined) {
    return undefined;
  }

  const projected: Record<string, unknown> = {};
  for (const key of [
    "occurred_at",
    "lived_experience_kind",
    "source_kind",
    "status",
    "state",
    "outcome",
    "action_id",
    "open_question_id",
    "stance",
    "belief_effect",
    "recurrence_count",
    "relative_age",
    "disclosure_label",
  ] as const) {
    if (Object.hasOwn(metadata, key)) {
      projected[key] =
        key === "disclosure_label" && isJsonObject(metadata[key])
          ? projectDisclosureLabel(metadata[key] as unknown as MemoryDisclosureLabel)
          : metadata[key];
    }
  }

  return Object.keys(projected).length === 0 ? undefined : projected;
}

function projectEvidenceEntry(entry: EvidenceLedgerEntry) {
  return {
    id: entry.id,
    source_type: entry.source_type,
    session_scope: entry.session_scope,
    actor: entry.actor,
    trust_rank: entry.trust_rank,
    ...(entry.text === undefined ? {} : { text: entry.text }),
    ...(entry.value === undefined ? {} : { value: entry.value }),
    ...(entry.state === undefined ? {} : { state: entry.state }),
    ...(entry.stream_index === undefined ? {} : { stream_index: entry.stream_index }),
    ...(entry.taint === undefined ? {} : { taint: entry.taint }),
    ...(projectEvidenceMetadata(entry.state_metadata) === undefined
      ? {}
      : { state_metadata: projectEvidenceMetadata(entry.state_metadata) }),
    ...(entry.planner_metadata === undefined
      ? {}
      : {
          planner_metadata: {
            ...(entry.planner_metadata.decision_outcome_ref === undefined
              ? {}
              : { decision_outcome_ref: entry.planner_metadata.decision_outcome_ref }),
            ...(entry.planner_metadata.decision_summary === undefined
              ? {}
              : { decision_summary: entry.planner_metadata.decision_summary }),
            ...(entry.planner_metadata.decision_rationale === undefined
              ? {}
              : { decision_rationale: entry.planner_metadata.decision_rationale }),
          },
        }),
  };
}

function projectEvidenceLedger(ledger: EvidenceLedger | null | undefined) {
  if (ledger === null || ledger === undefined) {
    return null;
  }

  const standing = ledger.audienceStanding;
  return {
    sections: ledger.sections
      .filter((section) => section.id === "autobiographical_recall")
      .map((section) => ({
        id: section.id,
        label: section.label,
        entries: section.entries.map(projectEvidenceEntry),
      })),
    sectionEntryCounts: ledger.sections.map((section) => ({
      id: section.id,
      count: section.entries.length,
    })),
    activeSharedStateEntryCount:
      ledger.sharedState?.entries.filter((entry) => entry.superseded_by_id === null).length ?? 0,
    ...(standing === undefined
      ? {}
      : {
          audienceStanding: {
            recentLivedExperienceEntries:
              standing.recentLivedExperienceEntries.map(projectEvidenceEntry),
            renderRecentLivedExperience: standing.renderRecentLivedExperience,
            observedEventIntrospectionEntries:
              standing.observedEventIntrospectionEntries.map(projectEvidenceEntry),
          },
        }),
  };
}

function projectSelfSnapshot(snapshot: SelfSnapshot) {
  return {
    values: snapshot.values.map((value) => ({
      id: value.id,
      label: value.label,
      description: value.description,
      priority: value.priority,
      created_at: value.created_at,
      state: value.state,
      confidence: value.confidence,
    })),
    goals: snapshot.goals.map((goal) => ({
      id: goal.id,
      description: goal.description,
      terminal_condition: goal.terminal_condition,
      priority: goal.priority,
      status: goal.status,
      progress_notes: goal.progress_notes,
      last_progress_ts: goal.last_progress_ts,
      created_at: goal.created_at,
      target_at: goal.target_at,
      audience_entity_id: goal.audience_entity_id,
      owner_entity_id: goal.owner_entity_id,
      ...(projectDisclosureLabel(goal.disclosure_label) === undefined
        ? {}
        : { disclosure_label: projectDisclosureLabel(goal.disclosure_label) }),
    })),
    traits: snapshot.traits.map((trait) => ({
      id: trait.id,
      label: trait.label,
      strength: trait.strength,
      state: trait.state,
      confidence: trait.confidence,
    })),
    ...(snapshot.currentPeriod === null || snapshot.currentPeriod === undefined
      ? { currentPeriod: snapshot.currentPeriod }
      : {
          currentPeriod: {
            id: snapshot.currentPeriod.id,
            label: snapshot.currentPeriod.label,
            narrative: snapshot.currentPeriod.narrative,
            last_updated: snapshot.currentPeriod.last_updated,
            ...(projectDisclosureLabel(snapshot.currentPeriod.disclosure_label) === undefined
              ? {}
              : {
                  disclosure_label: projectDisclosureLabel(snapshot.currentPeriod.disclosure_label),
                }),
          },
        }),
    ...(snapshot.recentGrowthMarkers === undefined
      ? {}
      : {
          recentGrowthMarkers: snapshot.recentGrowthMarkers.map((marker) => ({
            id: marker.id,
            ts: marker.ts,
            category: marker.category,
            what_changed: marker.what_changed,
            confidence: marker.confidence,
            ...(projectDisclosureLabel(marker.disclosure_label) === undefined
              ? {}
              : { disclosure_label: projectDisclosureLabel(marker.disclosure_label) }),
          })),
        }),
  };
}

function projectNullable<T, U>(
  value: T | null | undefined,
  project: (present: T) => U,
): U | null | undefined {
  if (value === null || value === undefined) {
    return value as null | undefined;
  }
  return project(value);
}

function projectCurrentTimeContext(value: DeliberationContext["currentTimeContext"]) {
  return projectNullable(value, (context) => ({
    previousUserMessageAt: context.previousUserMessageAt,
    recentLifeElsewhere: {
      windowMs: context.recentLifeElsewhere.windowMs,
      autonomousReflectionCount: context.recentLifeElsewhere.autonomousReflectionCount,
      crossSessionConversationTurnCount:
        context.recentLifeElsewhere.crossSessionConversationTurnCount,
    },
  }));
}

function projectCreatorContext(value: DeliberationContext["creatorContext"]) {
  return projectNullable(value, (context) => ({
    currentSenderEntityId: context.currentSenderEntityId,
    currentSenderDisplayName: context.currentSenderDisplayName,
    currentSenderBorgRole: context.currentSenderBorgRole,
    sessionAudienceRole: context.sessionAudienceRole,
  }));
}

function projectDirectiveBriefing(value: DeliberationContext["creatorDirectiveBriefing"]) {
  return projectNullable(value, (briefing) => ({
    directives: briefing.directives.map(projectCreatorDirective),
  }));
}

function projectAutonomousOutbound(value: DeliberationContext["autonomousOutbound"]) {
  return projectNullable(value, (outbound) => ({
    maxPostsPerWindow: outbound.maxPostsPerWindow,
    maxPostsPerTargetPerWindow: outbound.maxPostsPerTargetPerWindow,
    remainingPostsInWindow: outbound.remainingPostsInWindow,
    windowMs: outbound.windowMs,
    targets: outbound.targets.map((target) => ({
      session_id: target.session_id,
      source_type: target.source_type,
      label: target.label,
      audience_label: target.audience_label,
      audience_entity_id: target.audience_entity_id,
      conversation_kind: target.conversation_kind,
      participation_policy: target.participation_policy,
      authorization: target.authorization,
    })),
  }));
}

function projectAutonomyTrigger(value: DeliberationContext["autonomyTrigger"]) {
  return projectNullable(value, (trigger) => ({
    source_name: trigger.source_name,
    source_type: trigger.source_type,
    event_id: trigger.event_id,
    sort_ts: trigger.sort_ts,
    payload: trigger.payload,
  }));
}

function projectCommitment(
  commitment: NonNullable<DeliberationContext["applicableCommitments"]>[number],
) {
  return {
    id: commitment.id,
    type: commitment.type,
    kind: commitment.kind,
    enforcement_class: commitment.enforcement_class,
    critical_domain: commitment.critical_domain,
    directive_family: commitment.directive_family,
    closure_pressure_relevance: commitment.closure_pressure_relevance,
    directive: commitment.directive,
    priority: commitment.priority,
    made_to_entity: commitment.made_to_entity,
    restricted_audience: commitment.restricted_audience,
    about_entity: commitment.about_entity,
    committed_by_entity_id: commitment.committed_by_entity_id,
    created_at: commitment.created_at,
    expires_at: commitment.expires_at,
    expired_at: commitment.expired_at,
    revoked_at: commitment.revoked_at,
    last_reinforced_at: commitment.last_reinforced_at,
  };
}

function projectRelationalSlot(slot: NonNullable<DeliberationContext["relationalSlots"]>[number]) {
  return {
    id: slot.id,
    subject_entity_id: slot.subject_entity_id,
    slot_key: slot.slot_key,
    value: slot.value,
    state: slot.state,
    alternate_values: slot.alternate_values.map((alternate) => ({ value: alternate.value })),
    updated_at: slot.updated_at,
  };
}

function projectActiveParticipant(
  participant: NonNullable<DeliberationContext["activeParticipants"]>[number],
) {
  return {
    entityId: participant.entityId,
    displayName: participant.displayName,
    role: participant.role,
  };
}

function projectParticipantProfile(
  participant: NonNullable<DeliberationContext["participantProfiles"]>[number],
) {
  return {
    ...projectActiveParticipant(participant),
    profile: participant.profile === null ? null : projectSocialProfileFields(participant.profile),
  };
}

function projectSelectedSkill(value: DeliberationContext["selectedSkill"]) {
  return projectNullable(value, (selection) => {
    const disclosureLabel = projectDisclosureLabel(selection.skill.disclosure_label);
    return {
      skill: {
        id: selection.skill.id,
        applies_when: selection.skill.applies_when,
        approach: selection.skill.approach,
        ...(disclosureLabel === undefined ? {} : { disclosure_label: disclosureLabel }),
      },
    };
  });
}

function projectTurnMechanismEvidence(value: DeliberationContext["turnMechanismEvidence"]) {
  if (value === undefined) {
    return undefined;
  }
  return {
    recentSuppressions: value.recentSuppressions.map((entry) => ({
      turnId: entry.turnId,
      reason: entry.reason,
      ts: entry.ts,
    })),
    recentRegenerations: value.recentRegenerations.map((entry) => ({
      turnId: entry.turnId,
      mechanism: entry.mechanism,
      ts: entry.ts,
      ...(entry.commitments === undefined || entry.commitments.length === 0
        ? {}
        : { commitments: entry.commitments.map((commitment) => ({ ...commitment })) }),
    })),
    ...(value.autonomySchedulerState === undefined
      ? {}
      : {
          autonomySchedulerState: {
            observedAt: value.autonomySchedulerState.observedAt,
            enabled: value.autonomySchedulerState.enabled,
            nextTickAt: value.autonomySchedulerState.nextTickAt,
            scheduledTickAt: value.autonomySchedulerState.scheduledTickAt,
            fleetBrake: {
              ...value.autonomySchedulerState.fleetBrake,
              window_outcomes: { ...value.autonomySchedulerState.fleetBrake.window_outcomes },
            },
            budget: {
              ...value.autonomySchedulerState.budget,
              wakes_in_current_window_by_trigger:
                value.autonomySchedulerState.budget.wakes_in_current_window_by_trigger.map(
                  (group) => ({
                    ...group,
                    outcome_counts: { ...group.outcome_counts },
                  }),
                ),
            },
          },
        }),
  };
}

function projectWorkingMemory(context: DeliberationContext) {
  const working = context.workingMemory;
  const discourse = working.discourse_state;
  const mechanismEvidence = context.turnMechanismEvidence;
  return {
    focus: working.hot_entities[0] ?? null,
    pendingActionCount: working.pending_actions.length,
    pending_actions: working.pending_actions.map((action) => ({
      description: action.description,
      next_action: action.next_action,
      ...(action.created_at === undefined ? {} : { created_at: action.created_at }),
    })),
    updated_at: working.updated_at,
    pendingProceduralAttemptCount: working.pending_procedural_attempts.length,
    mood:
      working.mood === null
        ? null
        : {
            valence: working.mood.valence,
            arousal: working.mood.arousal,
            dominant_emotion: working.mood.dominant_emotion,
          },
    discourseState: {
      stop_until_substantive_content:
        discourse.stop_until_substantive_content === null
          ? null
          : {
              provenance: discourse.stop_until_substantive_content.provenance,
              reason: discourse.stop_until_substantive_content.reason,
              since_turn: discourse.stop_until_substantive_content.since_turn,
            },
      ...(discourse.closure_loop === undefined
        ? {}
        : {
            closure_loop:
              discourse.closure_loop === null
                ? null
                : {
                    status: discourse.closure_loop.status,
                    reason: discourse.closure_loop.reason,
                    since_turn: discourse.closure_loop.since_turn,
                  },
          }),
      ...(discourse.closure_pressure_history === undefined
        ? {}
        : {
            closure_pressure_history: discourse.closure_pressure_history.map((entry) => ({
              turn_id: entry.turn_id,
              reason: entry.reason,
              ts: entry.ts,
            })),
          }),
      ...(mechanismEvidence !== undefined || discourse.recent_suppressions === undefined
        ? {}
        : {
            recent_suppressions: discourse.recent_suppressions.map((entry) => ({
              turn_id: entry.turn_id,
              reason: entry.reason,
              ts: entry.ts,
            })),
          }),
      ...(mechanismEvidence !== undefined || discourse.recent_regenerations === undefined
        ? {}
        : {
            recent_regenerations: discourse.recent_regenerations.map((entry) => ({
              turn_id: entry.turn_id,
              mechanism: entry.mechanism,
              ts: entry.ts,
            })),
          }),
    },
  };
}

function projectExecutiveFocus(value: DeliberationContext["executiveFocus"]) {
  return projectNullable(value, (focus) => {
    const nextStep = projectNullable(focus.next_step, (step) => {
      const disclosureLabel = projectDisclosureLabel(step.disclosure_label);
      return {
        id: step.id,
        goal_id: step.goal_id,
        description: step.description,
        status: step.status,
        kind: step.kind,
        due_at: step.due_at,
        last_attempt_ts: step.last_attempt_ts,
        ...(disclosureLabel === undefined ? {} : { disclosure_label: disclosureLabel }),
      };
    });
    return {
      selectedGoalId: focus.selected_goal?.id ?? null,
      nextStep,
      candidates: focus.candidates.map((candidate) => ({
        goal_id: candidate.goal_id,
        score: candidate.score,
        components: {
          priority: candidate.components.priority,
          deadline_pressure: candidate.components.deadline_pressure,
          context_fit: candidate.components.context_fit,
          progress_debt: candidate.components.progress_debt,
        },
        reason: candidate.reason,
      })),
    };
  });
}

export function captureCompactPlannerContext(context: DeliberationContext) {
  return {
    sessionId: context.sessionId,
    nowMs: context.nowMs,
    currentTimeContext: projectCurrentTimeContext(context.currentTimeContext),
    participationPolicy: context.participationPolicy,
    creatorIdentity: projectNullable(context.creatorIdentity, (identity) => ({
      displayName: identity.displayName,
    })),
    creatorContext: projectCreatorContext(context.creatorContext),
    creatorDirectiveBriefing: projectDirectiveBriefing(context.creatorDirectiveBriefing),
    autonomousOutbound: projectAutonomousOutbound(context.autonomousOutbound),
    autonomousFinalizerToolMenu: context.autonomousFinalizerToolMenu?.map((item) => ({
      name: item.name,
      menuSummary: item.menuSummary,
    })),
    turnId: context.turnId,
    turnOrigin: context.turnOrigin,
    audience: context.audience,
    isSelfAudience: context.isSelfAudience,
    audienceEntityId: context.audienceEntityId,
    autonomyTrigger: projectAutonomyTrigger(context.autonomyTrigger),
    perception: {
      mode: context.perception.mode,
      affectiveSignal: {
        valence: context.perception.affectiveSignal.valence,
        arousal: context.perception.affectiveSignal.arousal,
      },
    },
    applicableCommitments: context.applicableCommitments?.map(projectCommitment),
    openQuestionsContext: context.openQuestionsContext?.map((question) => ({
      id: question.id,
      question: question.question,
      status: question.status,
      urgency: question.urgency,
      audience_entity_id: question.audience_entity_id,
      last_touched: question.last_touched,
      ...(projectDisclosureLabel(question.disclosure_label) === undefined
        ? {}
        : { disclosure_label: projectDisclosureLabel(question.disclosure_label) }),
    })),
    relationalSlots: context.relationalSlots?.map(projectRelationalSlot),
    activeParticipants: context.activeParticipants?.map(projectActiveParticipant),
    participantProfiles: context.participantProfiles?.map(projectParticipantProfile),
    selectedSkill: projectSelectedSkill(context.selectedSkill),
    workingMemory: projectWorkingMemory(context),
    turnMechanismEvidence: projectTurnMechanismEvidence(context.turnMechanismEvidence),
    affectiveTrajectory: context.affectiveTrajectory?.map((entry) => ({
      ts: entry.ts,
      valence: entry.valence,
      arousal: entry.arousal,
    })),
    selfSnapshot: projectSelfSnapshot(context.selfSnapshot),
    executiveFocus: projectExecutiveFocus(context.executiveFocus),
    audienceProfile: projectNullable(context.audienceProfile, (profile) => ({
      entity_id: profile.entity_id,
      ...projectSocialProfileFields(profile),
    })),
    frameAnomaly: projectNullable(context.frameAnomaly, (anomaly) => ({
      kind: anomaly.kind,
      confidence: anomaly.confidence,
      rationale: anomaly.rationale,
    })),
    evidenceLedger: projectEvidenceLedger(context.evidenceLedger),
  };
}

export type CompactPlannerContextCapture = ReturnType<typeof captureCompactPlannerContext>;

export type PlannerCaptureRenderInput = {
  compactContext: CompactPlannerContextCapture;
  legacyBaseSystemPrompt: string;
  compactStaticPrefix: string;
  compactPlannerLedgerTrace: CompactPlannerLedgerPrompt["traceSummary"] | null;
  additionalPromptSections: readonly PromptSurfaceAdditionalSection[];
  dialogueMessages: readonly LLMMessage[];
  model: string;
  maxTokens: number;
  thinking?: LLMCompleteOptions["thinking"];
  effort?: LLMCompleteOptions["effort"];
};

export type PlannerSurfaceFingerprint = {
  systemChars: number;
  systemSha256: string;
  transportSha256: string;
  systemBlockCount: number;
  cacheBreakpointCount: number;
};

export type PlannerRequestFingerprint = {
  canonicalChars: number;
  canonicalSha256: string;
};

export type PlannerRequestAnchor = {
  attempt: number;
  surface: PlannerSurfaceFingerprint;
  request: PlannerRequestFingerprint;
};

export type CapturedPlannerSurface = {
  rendered: RenderedS2PlannerSurface;
  fingerprint: PlannerSurfaceFingerprint;
};

export type CapturedPlannerSurfacePair = {
  compact: CapturedPlannerSurface;
  legacy: CapturedPlannerSurface;
};

export type CapturedPlannerOutcome =
  | (Extract<S2PlannerOutcome, { status: "completed" | "degraded" }> & {
      plan: S2PlannerResult["plan"];
      reasoning: string;
      usage: S2PlannerResult["usage"];
    })
  | Extract<S2PlannerOutcome, { status: "threw" }>;

export type PlannerContextCaptureRecord = {
  schema_version: typeof PLANNER_CONTEXT_CAPTURE_SCHEMA_VERSION;
  capture_id: string;
  captured_at: number;
  turn_id: string | null;
  session_id: SessionId;
  live_surface_variant: "compact" | "legacy";
  render_input: PlannerCaptureRenderInput;
  expected_surfaces: {
    compact: PlannerSurfaceFingerprint;
    legacy: PlannerSurfaceFingerprint;
  };
  fidelity: {
    verified: boolean;
    exactLiveSurfaceMatchesProjection: boolean;
    exactLiveRequestMatchesProjection: boolean;
    liveSurface: PlannerSurfaceFingerprint | null;
    liveRequest: PlannerRequestFingerprint | null;
  };
  live_outcome: CapturedPlannerOutcome;
};

export type BuildPlannerContextCaptureRecordInput = {
  capturedAt: number;
  liveSurfaceVariant: "compact" | "legacy";
  renderInput: PlannerCaptureRenderInput;
  liveOutcome: S2PlannerOutcome;
  liveOutput?: S2PlannerResult;
  liveRequest?: PlannerRequestAnchor;
  captureId?: string;
};

export type PlannerContextCaptureWriteResult =
  | { status: "captured"; path: string; bytes: number; record: PlannerContextCaptureRecord }
  | { status: "skipped"; reason: "record_oversized" | "file_full"; bytes: number }
  | { status: "failed"; reason: string };

export type PlannerContextCaptureStats = {
  captured: number;
  oversizedSkipped: number;
  fileFullSkipped: number;
  failed: number;
};

export type PlannerContextCaptureOptions = {
  dataDir: string;
  sampleRate: number;
  clock?: Clock;
  tracer?: TurnTracer;
  random?: () => number;
  maxRecordBytes?: number;
  maxFileBytes?: number;
};

const plannerSurfaceFingerprintSchema = z
  .object({
    systemChars: z.number().int().nonnegative(),
    systemSha256: z.string().length(64),
    transportSha256: z.string().length(64),
    systemBlockCount: z.number().int().positive(),
    cacheBreakpointCount: z.number().int().nonnegative(),
  })
  .strict();

const plannerRequestFingerprintSchema = z
  .object({
    canonicalChars: z.number().int().nonnegative(),
    canonicalSha256: z.string().length(64),
  })
  .strict();

const plannerOutcomeSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("completed"),
      attempts: z.number().int().positive(),
      structuralReason: z.literal("emit_turn_plan"),
      plan: z.unknown().nullable(),
      reasoning: z.string(),
      usage: z.record(z.string(), z.unknown()),
    })
    .strict(),
  z
    .object({
      status: z.literal("degraded"),
      attempts: z.number().int().positive(),
      structuralReason: z.enum([
        "missing_emit_turn_plan_tool_use",
        "invalid_emit_turn_plan_input",
        "retryable_transport_error",
      ]),
      plan: z.unknown().nullable(),
      reasoning: z.string(),
      usage: z.record(z.string(), z.unknown()),
    })
    .strict(),
  z
    .object({
      status: z.literal("threw"),
      attempts: z.number().int().nonnegative(),
      structuralReason: z.literal("non_retryable_planner_error"),
      error: z
        .object({ name: z.string(), message: z.string(), code: z.string().optional() })
        .strict(),
    })
    .strict(),
]);

const plannerContextCaptureRecordEnvelopeSchema = z
  .object({
    schema_version: z.literal(PLANNER_CONTEXT_CAPTURE_SCHEMA_VERSION),
    capture_id: z.string().min(1),
    captured_at: z.number().finite(),
    turn_id: z.string().min(1).nullable(),
    session_id: z.string().min(1),
    live_surface_variant: z.enum(["compact", "legacy"]),
    render_input: z
      .object({
        compactContext: z.record(z.string(), z.unknown()),
        legacyBaseSystemPrompt: z.string(),
        compactStaticPrefix: z.string(),
        compactPlannerLedgerTrace: z.unknown().nullable(),
        additionalPromptSections: z.array(
          z.object({ blockId: z.string(), text: z.string() }).strict(),
        ),
        dialogueMessages: z.array(
          z.object({ role: z.enum(["user", "assistant"]), content: z.string() }).strict(),
        ),
        model: z.string().min(1),
        maxTokens: z.number().int().positive(),
        thinking: z.unknown().optional(),
        effort: z.unknown().optional(),
      })
      .strict(),
    expected_surfaces: z
      .object({ compact: plannerSurfaceFingerprintSchema, legacy: plannerSurfaceFingerprintSchema })
      .strict(),
    fidelity: z
      .object({
        verified: z.boolean(),
        exactLiveSurfaceMatchesProjection: z.boolean(),
        exactLiveRequestMatchesProjection: z.boolean(),
        liveSurface: plannerSurfaceFingerprintSchema.nullable(),
        liveRequest: plannerRequestFingerprintSchema.nullable(),
      })
      .strict(),
    live_outcome: plannerOutcomeSchema,
  })
  .strict();

function jsonRoundTrip<T>(value: T): T {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new TypeError("Planner context capture value is not JSON serializable");
  }
  return JSON.parse(serialized) as T;
}

export function createPlannerCaptureRenderInput(input: {
  context: DeliberationContext;
  legacyBaseSystemPrompt: string;
  compactStaticPrefix: string;
  compactPlannerLedger: CompactPlannerLedgerPrompt | null;
  additionalPromptSections: readonly PromptSurfaceAdditionalSection[];
  dialogueMessages: readonly LLMMessage[];
  model: string;
  maxTokens: number;
  thinking?: LLMCompleteOptions["thinking"];
  effort?: LLMCompleteOptions["effort"];
}): PlannerCaptureRenderInput {
  return {
    compactContext: captureCompactPlannerContext(input.context),
    legacyBaseSystemPrompt: input.legacyBaseSystemPrompt,
    compactStaticPrefix: input.compactStaticPrefix,
    compactPlannerLedgerTrace: input.compactPlannerLedger?.traceSummary ?? null,
    additionalPromptSections: input.additionalPromptSections,
    dialogueMessages: input.dialogueMessages,
    model: input.model,
    maxTokens: input.maxTokens,
    ...(input.thinking === undefined ? {} : { thinking: input.thinking }),
    ...(input.effort === undefined ? {} : { effort: input.effort }),
  };
}

function restoreEvidenceLedger(
  captured: CompactPlannerContextCapture["evidenceLedger"],
): EvidenceLedger | null {
  if (captured === null) {
    return null;
  }
  const capturedAutobiographical = new Map(
    (captured.sections ?? []).map((section) => [section.id, section]),
  );
  return {
    sections: captured.sectionEntryCounts.map((section) => {
      const exact = capturedAutobiographical.get(section.id);
      return exact === undefined
        ? {
            id: section.id,
            label: "",
            entries: Array.from({ length: section.count }, () => ({}) as EvidenceLedgerEntry),
          }
        : {
            id: exact.id,
            label: exact.label,
            entries: exact.entries as EvidenceLedgerEntry[],
          };
    }),
    ...(captured.audienceStanding === undefined
      ? {}
      : {
          audienceStanding: {
            recentLivedExperienceEntries: captured.audienceStanding
              .recentLivedExperienceEntries as EvidenceLedgerEntry[],
            renderRecentLivedExperience: captured.audienceStanding.renderRecentLivedExperience,
            observedEventIntrospectionEntries: captured.audienceStanding
              .observedEventIntrospectionEntries as EvidenceLedgerEntry[],
            commitmentEntries: [],
            relationalEntries: [],
          },
        }),
    sharedState: {
      entries: Array.from({ length: captured.activeSharedStateEntryCount }, () => ({
        superseded_by_id: null,
      })),
    } as EvidenceLedger["sharedState"],
    transcriptIncluded: false,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
  };
}

function restoreCompactPlannerContext(context: CompactPlannerContextCapture): DeliberationContext {
  const working = context.workingMemory;
  const executive = context.executiveFocus;
  return {
    userMessage: "",
    retrievalResult: [],
    ...context,
    perception: {
      entities: [],
      mode: context.perception.mode,
      affectiveSignal: { ...context.perception.affectiveSignal, dominant_emotion: null },
      temporalCue: null,
    },
    workingMemory: {
      session_id: context.sessionId,
      turn_counter: 0,
      hot_entities: working.focus === null ? [] : [working.focus],
      pending_actions:
        working.pending_actions === undefined
          ? (Array.from({ length: working.pendingActionCount }, () => ({})) as never[])
          : (working.pending_actions as DeliberationContext["workingMemory"]["pending_actions"]),
      pending_social_attribution: null,
      pending_trait_attribution: null,
      suppressed: [],
      mood: working.mood,
      pending_procedural_attempts: Array.from(
        { length: working.pendingProceduralAttemptCount },
        () => ({}),
      ) as never[],
      discourse_state:
        working.discourseState as unknown as DeliberationContext["workingMemory"]["discourse_state"],
      mode: null,
      updated_at: working.updated_at ?? 0,
    },
    selfSnapshot: context.selfSnapshot as unknown as SelfSnapshot,
    selectedSkill: context.selectedSkill as DeliberationContext["selectedSkill"],
    applicableCommitments:
      context.applicableCommitments as DeliberationContext["applicableCommitments"],
    relationalSlots: context.relationalSlots as DeliberationContext["relationalSlots"],
    participantProfiles: context.participantProfiles as DeliberationContext["participantProfiles"],
    turnMechanismEvidence:
      context.turnMechanismEvidence as DeliberationContext["turnMechanismEvidence"],
    affectiveTrajectory: context.affectiveTrajectory as DeliberationContext["affectiveTrajectory"],
    executiveFocus:
      executive === null || executive === undefined
        ? executive
        : ({
            selected_goal:
              executive.selectedGoalId === null ? null : { id: executive.selectedGoalId },
            selected_score: null,
            next_step: executive.nextStep,
            candidates: executive.candidates,
            threshold: 0,
            score_basis: {
              score_context: "turn_selection",
              deadline_lookahead_ms: 0,
              progress_debt_stale_ms: 0,
            },
          } as DeliberationContext["executiveFocus"]),
    audienceProfile: context.audienceProfile as DeliberationContext["audienceProfile"],
    frameAnomaly: context.frameAnomaly as DeliberationContext["frameAnomaly"],
    evidenceLedger: restoreEvidenceLedger(context.evidenceLedger),
  } as unknown as DeliberationContext;
}

export function plannerSurfaceText(system: string | readonly LLMSystemBlock[]): string {
  return llmSystemText(system);
}

export function fingerprintPlannerSurface(
  surface: Pick<RenderedS2PlannerSurface, "system">,
): PlannerSurfaceFingerprint {
  return fingerprintSystemSurface(surface.system);
}

export function fingerprintPlannerRequest(
  request: S2PlannerRequestSnapshot,
): PlannerRequestFingerprint {
  return fingerprintCanonicalRequest({
    system: request.system,
    messages: request.messages,
    tools: request.tools,
    ...request.callOptions,
  });
}

/** Compute immutable fidelity evidence synchronously before transport starts. */
export function anchorPlannerRequest(request: S2PlannerRequestSnapshot): PlannerRequestAnchor {
  return {
    attempt: request.attempt,
    surface: fingerprintPlannerSurface({ system: request.system }),
    request: fingerprintPlannerRequest(request),
  };
}

function capturedSurface(rendered: RenderedS2PlannerSurface): CapturedPlannerSurface {
  return { rendered, fingerprint: fingerprintPlannerSurface(rendered) };
}

function capturedCompactPlannerLedger(
  input: PlannerCaptureRenderInput,
): CompactPlannerLedgerPrompt | null {
  if (input.compactPlannerLedgerTrace === null) {
    return null;
  }
  const promptSection =
    input.additionalPromptSections.find(
      (section) => section.blockId === "borg_compact_planner_ledger",
    )?.text ?? null;
  return { promptSection, traceSummary: input.compactPlannerLedgerTrace };
}

export function renderCapturedPlannerSurfacePair(
  input: PlannerCaptureRenderInput,
): CapturedPlannerSurfacePair {
  const compactContext = restoreCompactPlannerContext(input.compactContext);
  const compactPlannerSurface = {
    variant: "compact" as const,
    ...buildCompactPlannerSystemPrompt({
      context: compactContext,
      staticPrefix: input.compactStaticPrefix,
      compactPlannerLedger: capturedCompactPlannerLedger(input),
      additionalPromptSections: input.additionalPromptSections,
    }),
  };
  const common = {
    baseSystemPrompt: input.legacyBaseSystemPrompt,
    selfSnapshot: compactContext.selfSnapshot,
    additionalPromptSections: input.additionalPromptSections,
    turnOrigin: compactContext.turnOrigin,
  };
  return {
    compact: capturedSurface(
      renderS2PlannerSurface({ ...common, plannerSurface: compactPlannerSurface }),
    ),
    legacy: capturedSurface(
      renderS2PlannerSurface({ ...common, plannerSurface: { variant: "legacy" } }),
    ),
  };
}

export function parsePlannerContextCaptureRecord(value: unknown): PlannerContextCaptureRecord {
  return plannerContextCaptureRecordEnvelopeSchema.parse(
    value,
  ) as unknown as PlannerContextCaptureRecord;
}

function capturedOutcome(
  outcome: S2PlannerOutcome,
  output: S2PlannerResult | undefined,
): CapturedPlannerOutcome {
  if (outcome.status === "threw") {
    return outcome;
  }
  if (output === undefined) {
    throw new TypeError(`Planner ${outcome.status} outcome requires its result payload`);
  }
  return { ...outcome, plan: output.plan, reasoning: output.reasoning, usage: output.usage };
}

export function buildPlannerContextCaptureRecord(
  input: BuildPlannerContextCaptureRecordInput,
): PlannerContextCaptureRecord {
  const renderInput = jsonRoundTrip(input.renderInput);
  const paired = renderCapturedPlannerSurfacePair(renderInput);
  const expected = paired[input.liveSurfaceVariant].fingerprint;
  const expectedRequest = createS2PlannerRequestSnapshot({
    attempt: 1,
    system: paired[input.liveSurfaceVariant].rendered.system,
    messages: renderInput.dialogueMessages,
    model: renderInput.model,
    maxTokens: renderInput.maxTokens,
    ...(renderInput.thinking === undefined ? {} : { thinking: renderInput.thinking }),
    ...(renderInput.effort === undefined ? {} : { effort: renderInput.effort }),
    ...(renderInput.compactContext.turnOrigin === undefined
      ? {}
      : { turnOrigin: renderInput.compactContext.turnOrigin }),
  });
  const expectedRequestFingerprint = fingerprintPlannerRequest(expectedRequest);
  const liveSurface = input.liveRequest?.surface ?? null;
  const exactLiveSurfaceMatchesProjection =
    liveSurface !== null && JSON.stringify(liveSurface) === JSON.stringify(expected);
  const liveRequest = input.liveRequest?.request ?? null;
  const exactLiveRequestMatchesProjection =
    liveRequest !== null &&
    JSON.stringify(liveRequest) === JSON.stringify(expectedRequestFingerprint);
  const record = {
    schema_version: PLANNER_CONTEXT_CAPTURE_SCHEMA_VERSION,
    capture_id: input.captureId ?? randomUUID(),
    captured_at: input.capturedAt,
    turn_id: renderInput.compactContext.turnId ?? null,
    session_id: renderInput.compactContext.sessionId,
    live_surface_variant: input.liveSurfaceVariant,
    render_input: renderInput,
    expected_surfaces: {
      compact: paired.compact.fingerprint,
      legacy: paired.legacy.fingerprint,
    },
    fidelity: {
      verified: exactLiveSurfaceMatchesProjection && exactLiveRequestMatchesProjection,
      exactLiveSurfaceMatchesProjection,
      exactLiveRequestMatchesProjection,
      liveSurface,
      liveRequest,
    },
    live_outcome: capturedOutcome(input.liveOutcome, input.liveOutput),
  } satisfies PlannerContextCaptureRecord;
  return parsePlannerContextCaptureRecord(jsonRoundTrip(record));
}

export function plannerContextCapturePath(dataDir: string): string {
  return join(dataDir, PLANNER_CONTEXT_CAPTURE_RELATIVE_PATH);
}

export class PlannerContextCapture {
  private readonly clock: Clock;
  private readonly tracer: TurnTracer;
  private readonly random: () => number;
  private readonly maxRecordBytes: number;
  private readonly maxFileBytes: number;
  private readonly stats: PlannerContextCaptureStats = {
    captured: 0,
    oversizedSkipped: 0,
    fileFullSkipped: 0,
    failed: 0,
  };

  constructor(private readonly options: PlannerContextCaptureOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.random = options.random ?? Math.random;
    this.maxRecordBytes =
      options.maxRecordBytes ?? DEFAULT_PLANNER_CONTEXT_CAPTURE_MAX_RECORD_BYTES;
    this.maxFileBytes = options.maxFileBytes ?? DEFAULT_PLANNER_CONTEXT_CAPTURE_MAX_FILE_BYTES;
  }

  private resolvedStoragePath(): { path: string; captureDirectory: string } {
    return resolveContextCaptureStoragePath(this.options.dataDir, "planner-contexts.jsonl");
  }

  shouldCapture(): boolean {
    return this.options.sampleRate > 0 && this.random() < this.options.sampleRate;
  }

  capturedAt(): number {
    return this.clock.now();
  }

  snapshotStats(): PlannerContextCaptureStats {
    return { ...this.stats };
  }

  private emit(
    record: Pick<PlannerContextCaptureRecord, "turn_id" | "session_id">,
    status: "captured" | "skipped" | "failed",
    details: Record<string, string | number>,
  ): void {
    if (!this.tracer.enabled) return;
    this.tracer.emit(`deliberation.planner_context_capture.${status}`, {
      turnId: record.turn_id ?? "unknown",
      session_id: record.session_id,
      ...details,
      captured_count: this.stats.captured,
      oversized_skip_count: this.stats.oversizedSkipped,
      file_full_skip_count: this.stats.fileFullSkipped,
      failed_count: this.stats.failed,
    });
  }

  async write(record: PlannerContextCaptureRecord): Promise<PlannerContextCaptureWriteResult> {
    const bytes = Buffer.byteLength(`${JSON.stringify(record)}\n`);
    if (bytes > this.maxRecordBytes) {
      this.stats.oversizedSkipped += 1;
      this.emit(record, "skipped", { reason: "record_oversized", record_bytes: bytes });
      return { status: "skipped", reason: "record_oversized", bytes };
    }

    try {
      const { path } = this.resolvedStoragePath();
      const result = await appendBoundedContextCapture({
        dataDir: this.options.dataDir,
        fileName: "planner-contexts.jsonl",
        record,
        maxFileBytes: this.maxFileBytes,
      });
      if (result.status === "file_full") {
        this.stats.fileFullSkipped += 1;
        this.emit(record, "skipped", { reason: "file_full", record_bytes: bytes });
        return { status: "skipped", reason: "file_full", bytes };
      }
      this.stats.captured += 1;
      this.emit(record, "captured", { record_bytes: bytes });
      return { status: "captured", path, bytes, record };
    } catch (error) {
      this.stats.failed += 1;
      const reason = error instanceof Error ? error.message : String(error);
      this.emit(record, "failed", { reason });
      return { status: "failed", reason };
    }
  }

  async capture(
    input: BuildPlannerContextCaptureRecordInput,
  ): Promise<PlannerContextCaptureWriteResult> {
    try {
      return await this.write(buildPlannerContextCaptureRecord(input));
    } catch (error) {
      this.stats.failed += 1;
      const reason = error instanceof Error ? error.message : String(error);
      this.emit(
        {
          turn_id: input.renderInput.compactContext.turnId ?? null,
          session_id: input.renderInput.compactContext.sessionId,
        },
        "failed",
        { reason },
      );
      return { status: "failed", reason };
    }
  }
}
