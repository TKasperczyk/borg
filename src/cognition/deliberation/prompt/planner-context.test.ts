import { describe, expect, it } from "vitest";

import type { CommitmentRecord } from "../../../memory/commitments/index.js";
import {
  buildCompactPlannerLedgerPrompt,
  type EvidenceLedger,
  type EvidenceLedgerEntry,
} from "../../evidence-ledger/index.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createCreatorDirectiveId,
  createEntityId,
  createExecutiveStepId,
  createGoalId,
  createRelationalSlotId,
  createSessionId,
  createStreamEntryId,
  createTraitId,
  createValueId,
  entityIdHelpers,
  type EntityId,
} from "../../../util/ids.js";
import type { DeliberationContext, SelfSnapshotGoal } from "../types.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../../tools/internal/outbound-post-name.js";
import { buildBaseSystemPrompt } from "./system-prompt.js";
import {
  buildCompactPlannerSystemPrompt,
  COMPACT_PLANNER_TARGET_TOKENS,
  headTailPlannerExcerpt,
  PLANNER_GOAL_TARGET_TOKENS,
} from "./planner-context.js";

const NOW_MS = Date.UTC(2026, 7, 13, 0, 0, 0);

function goal(description: string, overrides: Partial<SelfSnapshotGoal> = {}): SelfSnapshotGoal {
  return {
    id: createGoalId(),
    description,
    terminal_condition: `Complete ${description}`,
    priority: 5,
    parent_goal_id: null,
    status: "active",
    progress_notes: "Initial progress",
    last_progress_ts: NOW_MS - 2 * 60 * 60_000,
    created_at: NOW_MS - 2 * 24 * 60 * 60_000,
    target_at: NOW_MS + 3 * 24 * 60 * 60_000,
    audience_entity_id: null,
    owner_entity_id: null,
    provenance: { kind: "manual" },
    ...overrides,
  };
}

function commitment(
  directive: string,
  overrides: Partial<CommitmentRecord> = {},
): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "promise",
    kind: "assistant_commitment",
    enforcement_class: "advisory",
    critical_domain: null,
    directive_family: "planner_context_test",
    closure_pressure_relevance: "neutral",
    directive,
    priority: 5,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    committed_by_entity_id: null,
    provenance: { kind: "manual" },
    source_stream_entry_ids: [createStreamEntryId()],
    created_at: NOW_MS - 24 * 60 * 60_000,
    updated_at: NOW_MS - 24 * 60 * 60_000,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    canonicalized_by_artifact_entry_id: null,
    last_reinforced_at: NOW_MS - 60 * 60_000,
    ...overrides,
  };
}

function evidenceLedger(recentLivedExperienceEntries: EvidenceLedgerEntry[] = []): EvidenceLedger {
  return {
    sections: [],
    audienceStanding: {
      recentLivedExperienceEntries,
      renderRecentLivedExperience: false,
      observedEventIntrospectionEntries: [],
      commitmentEntries: [],
      relationalEntries: [],
    },
    transcriptIncluded: false,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
  };
}

function context(overrides: Partial<DeliberationContext> = {}): DeliberationContext {
  return {
    sessionId: DEFAULT_SESSION_ID,
    nowMs: NOW_MS,
    userMessage: "Please think this through.",
    perception: {
      entities: [],
      mode: "reflective",
      affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
      temporalCue: null,
    },
    retrievalResult: [],
    workingMemory: {
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 4,
      hot_entities: [],
      pending_actions: [],
      pending_social_attribution: null,
      pending_trait_attribution: null,
      suppressed: [],
      mood: null,
      pending_procedural_attempts: [],
      discourse_state: { stop_until_substantive_content: null },
      mode: "reflective",
      updated_at: NOW_MS,
    },
    selfSnapshot: { values: [], goals: [], traits: [] },
    evidenceLedger: evidenceLedger(),
    ...overrides,
  };
}

function build(inputContext: DeliberationContext) {
  return buildCompactPlannerSystemPrompt({
    context: inputContext,
    staticPrefix: "STATIC_HEAD_SENTINEL",
    compactPlannerLedger: null,
  });
}

function allSystemText(result: ReturnType<typeof build>): string {
  return result.system.map((block) => block.text).join("\n\n");
}

function taggedBlock(text: string, tag: string): string {
  return text.match(new RegExp(`<${tag}(?:\\s[^>]*)?>[\\s\\S]*?</${tag}>`))?.[0] ?? "";
}

function entityMembership(text: string, entityIds: readonly string[]): string[] {
  return entityIds.filter((entityId) => text.includes(entityId)).sort();
}

function profileContext(
  entityId: ReturnType<typeof createEntityId>,
  index: number,
  freeText = `profile-${index}`,
): NonNullable<DeliberationContext["participantProfiles"]>[number] {
  return {
    entityId,
    displayName: freeText,
    role: index === 0 ? "speaker" : "participant",
    profile: {
      entity_id: entityId,
      trust: 0.5,
      attachment: 0.4,
      communication_style: freeText,
      shared_history_summary: freeText,
      last_interaction_at: NOW_MS - index * 60_000,
      interaction_count: index + 1,
      commitment_count: index,
      sentiment_history: [],
      notes: freeText,
      created_at: NOW_MS - 100_000,
      updated_at: NOW_MS,
    },
  };
}

function relationalSlot(
  entityId: ReturnType<typeof createEntityId>,
  freeText: string,
): NonNullable<DeliberationContext["relationalSlots"]>[number] {
  return {
    id: createRelationalSlotId(),
    subject_entity_id: entityId,
    slot_key: freeText,
    value: freeText,
    state: "contested",
    evidence_stream_entry_ids: [createStreamEntryId()],
    contradicted_by_stream_entry_ids: [createStreamEntryId()],
    alternate_values: [{ value: freeText, evidence_stream_entry_ids: [createStreamEntryId()] }],
    created_at: NOW_MS - 100_000,
    updated_at: NOW_MS,
  };
}

function creatorDirective(
  index: number,
  canonicalFact: string,
): NonNullable<DeliberationContext["creatorDirectiveBriefing"]>["directives"][number] {
  return {
    renderMode: "content",
    kind: "subject_fact",
    subjectKind: "entity",
    subjectLabel: `subject-${index}`,
    semanticSlot: null,
    semanticValue: null,
    canonicalFact,
    operationalDirective: null,
    mentionPolicy: "answer_if_asked",
    priority: 1_000 - index,
    createdAt: NOW_MS + index,
  };
}

function rowIds(text: string, tag: "goal" | "c"): string[] {
  return [...text.matchAll(new RegExp(`<${tag} i="([^"]+)"`, "g"))]
    .map((match) => match[1]!)
    .sort();
}

function selfClosingRows(text: string, tag: string): string[] {
  return [...text.matchAll(new RegExp(`<${tag} [^\n]*?/>`, "g"))].map((match) => match[0]);
}

function livedEntry(input: {
  id: string;
  kind: string;
  occurredAt: number;
  text: string;
  outcomeReference?: string;
  disclosureClass?: "public" | "self_private" | "sensitive";
  originAudienceEntityIds?: readonly EntityId[];
  stance?: string;
  beliefEffect?: string;
}): EvidenceLedgerEntry {
  const disclosureClass = input.disclosureClass ?? "self_private";
  return {
    id: input.id,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 70,
    text: input.text,
    value: input.kind,
    state: "active",
    state_metadata: {
      lived_experience_kind: input.kind,
      occurred_at: input.occurredAt,
      ...(input.stance === undefined ? {} : { stance: input.stance }),
      ...(input.beliefEffect === undefined ? {} : { belief_effect: input.beliefEffect }),
      disclosure_label: {
        disclosure_class: disclosureClass,
        origin_audience_entity_ids: [...(input.originAudienceEntityIds ?? [])],
        private_to_entity_ids: [],
        public_to_entity_ids: [],
      },
    },
    ...(input.outcomeReference === undefined
      ? {}
      : {
          planner_metadata: {
            decision_outcome_ref: input.outcomeReference,
            decision_summary: input.text,
            decision_rationale: null,
          },
        }),
    taint: "none",
  };
}

describe("compact planner context", () => {
  it("renders the structurally available outbound action in the turn-local autonomous surface", () => {
    const outboundContext = {
      maxPostsPerWindow: 3,
      maxPostsPerTargetPerWindow: 1,
      remainingPostsInWindow: 2,
      windowMs: 86_400_000,
      targets: [
        {
          session_id: createSessionId(),
          source_type: "peerlink",
          label: "Kira",
          audience_label: "Kira",
          audience_entity_id: null,
          conversation_kind: "dm" as const,
          participation_policy: "active" as const,
          authorization: "config" as const,
        },
      ],
    };
    const withAction = build(
      context({
        turnOrigin: "autonomous",
        autonomousOutbound: outboundContext,
        autonomousFinalizerToolMenu: [
          { name: OUTBOUND_POST_TOOL_NAME, menuSummary: "Structurally available." },
        ],
      }),
    );

    expect(withAction.system[2]?.text).toContain(
      '<borg_directed_outbound_instruction mode="action_available">',
    );
    expect(withAction.system[2]?.text).toContain("target_session_id");
    expect(withAction.system[0]?.text).not.toContain("borg_directed_outbound_instruction");
    expect(withAction.system[1]?.text).not.toContain("borg_directed_outbound_instruction");

    const withoutAction = build(
      context({
        turnOrigin: "autonomous",
        autonomousOutbound: outboundContext,
        autonomousFinalizerToolMenu: [],
      }),
    );
    expect(allSystemText(withoutAction)).not.toContain("borg_directed_outbound_instruction");
  });

  it("marks only the final static-head block for one-hour caching and preserves block order", () => {
    const planner = build(context());

    expect(planner.system).toHaveLength(3);
    expect(planner.system[0]?.cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    expect(planner.system[1]?.cache_control).toBeUndefined();
    expect(planner.system[2]?.cache_control).toBeUndefined();
    expect(planner.system[0]?.text).toContain("STATIC_HEAD_SENTINEL");
    expect(planner.system[0]?.text).toContain("<borg_planner_pass_contract>");
    expect(planner.system[1]?.text).toContain("<borg_planner_self_digest");
    expect(planner.system[2]?.text).toContain("<borg_planner_turn_state>");
    expect(planner.system.flatMap((block) => block.cache_control ?? [])).toHaveLength(1);

    expect(planner.traceSummary.variant).toBe("compact");
    expect(planner.traceSummary.sections.static_head?.chars).toBe(planner.system[0]?.text.length);
    expect(planner.traceSummary.totalChars).toBe(allSystemText(planner).length);
  });

  it("keeps global goal and commitment membership invariant when the audience changes", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const goals = [
      goal("Global goal\nwith a second source line"),
      goal("Alice-scoped goal", { audience_entity_id: alice }),
      goal("Bob-scoped goal", { audience_entity_id: bob }),
    ];
    const commitments = [
      commitment("Global promise."),
      commitment("Promise made to Alice.", { made_to_entity: alice }),
      commitment("Bob-private boundary.", { restricted_audience: bob }),
    ];
    const executiveFocus = {
      selected_goal: goals[2]!,
      selected_score: null,
      candidates: goals.map((candidate, index) => ({
        goal_id: candidate.id,
        goal: candidate,
        score: 10 - index,
        components: {
          priority: candidate.priority,
          deadline_pressure: 0,
          context_fit: 0,
          progress_debt: 0,
        },
        reason: "global executive ranking",
      })),
      threshold: 0,
      score_basis: {
        score_context: "turn_selection" as const,
        deadline_lookahead_ms: 1,
        progress_debt_stale_ms: 1,
      },
    };
    const throwingEntityRepository = {
      get: () => {
        throw new Error("compact planner builders must not read repositories");
      },
    } as unknown as DeliberationContext["entityRepository"];
    const shared = {
      selfSnapshot: { values: [], goals, traits: [] },
      applicableCommitments: commitments,
      executiveFocus,
      entityRepository: throwingEntityRepository,
    };
    const aliceText = allSystemText(
      build(context({ ...shared, audience: "Alice", audienceEntityId: alice })),
    );
    const bobText = allSystemText(
      build(context({ ...shared, audience: "Bob", audienceEntityId: bob })),
    );

    expect(rowIds(aliceText, "goal")).toEqual(goals.map((entry) => entry.id).sort());
    expect(rowIds(bobText, "goal")).toEqual(goals.map((entry) => entry.id).sort());
    expect(rowIds(aliceText, "c")).toEqual(commitments.map((entry) => entry.id).sort());
    expect(rowIds(bobText, "c")).toEqual(commitments.map((entry) => entry.id).sort());
    expect(aliceText).toContain('s="active"');
    expect(aliceText).toContain('ca="2d ago"');
    expect(aliceText).toContain('x="10.0000"');
    expect(aliceText).toContain('d="Global goal with a second source line" />');
  });

  it("renders goal and commitment fields as lightweight single-line rows", () => {
    const alice = createEntityId();
    const counterparty = createEntityId();
    const indexedGoal = goal("Keep the plan legible", {
      owner_entity_id: alice,
      counterparty_entity_id: counterparty,
      record_version: 7,
    });
    const executiveFocus = {
      selected_goal: indexedGoal,
      selected_score: null,
      candidates: [
        {
          goal_id: indexedGoal.id,
          goal: indexedGoal,
          score: 9,
          components: { priority: 5, deadline_pressure: 1, context_fit: 2, progress_debt: 3 },
          reason: "Highest global score",
        },
      ],
      threshold: 0,
      score_basis: {
        score_context: "turn_selection" as const,
        deadline_lookahead_ms: 1,
        progress_debt_stale_ms: 1,
      },
    };
    const rendered = allSystemText(
      build(
        context({
          selfSnapshot: { values: [], goals: [indexedGoal], traits: [] },
          executiveFocus,
          applicableCommitments: [
            commitment("Keep the first line.\nKeep the second line.", { made_to_entity: alice }),
          ],
        }),
      ),
    );
    const goalRow = selfClosingRows(rendered, "goal")[0]!;
    const detailRow = selfClosingRows(rendered, "goal_detail")[0]!;
    const commitmentRow = selfClosingRows(rendered, "c")[0]!;

    expect(goalRow.length).toBeLessThanOrEqual(250);
    expect(goalRow).toContain(`i="${indexedGoal.id}"`);
    expect(goalRow).toContain('s="active"');
    expect(goalRow).toContain('x="9.0000"');
    expect(goalRow).toContain(`cp="${counterparty}"`);
    expect(goalRow).toContain('dc="relationship_private"');
    expect(goalRow).toContain(`oa="${alice}"`);
    expect(goalRow).toContain(`pt="${alice}"`);
    expect(goalRow).toContain('pub="none"');
    expect(detailRow).toContain('tc="Complete Keep the plan legible"');
    expect(detailRow).toContain(`cp="${counterparty}"`);
    expect(detailRow).toContain('sp="5.0000"');
    expect(detailRow).not.toContain("\n");
    // The write counter is the only witness a goal row has: the table stores no
    // last-written stamp, so ca/pa cannot report that a stored field was edited.
    expect(detailRow).toContain('rv="7"');
    // Detail-only on purpose. The index carries one row per goal against a tight
    // section budget, so widening rv into it is a budget decision, not a rename.
    expect(goalRow).not.toContain("rv=");
    expect(
      commitmentRow.length - "Keep the first line.&#10;Keep the second line.".length,
    ).toBeLessThanOrEqual(400);
    expect(commitmentRow).toContain(`to="${alice}"`);
    expect(commitmentRow).toContain('dc="relationship_private"');
    expect(commitmentRow).toContain('d="Keep the first line.&#10;Keep the second line."');
    expect(commitmentRow).not.toContain("\n");
    expect(rendered).toContain("i=id, s=status, ca=created_age");
    expect(rendered).toContain(
      "cp is the participant the responsibility runs toward, not an owner or audience",
    );
    expect(rendered).toContain("ec=enforcement_class");
  });

  it("keeps the sign of forward-looking stamps that deadline pressure keys on", () => {
    const future = goal("Ship the thing", { target_at: NOW_MS + 3 * 24 * 60 * 60_000 });
    const past = goal("Shipped the thing", { target_at: NOW_MS - 3 * 24 * 60 * 60_000 });
    const untargeted = goal("Never due", { target_at: null });
    const rendered = allSystemText(
      build(
        context({
          selfSnapshot: { values: [], goals: [future, past, untargeted], traits: [] },
          applicableCommitments: [
            commitment("Expires later", { expires_at: NOW_MS + 6 * 60 * 60_000 }),
          ],
        }),
      ),
    );
    const rowFor = (id: string): string =>
      selfClosingRows(rendered, "goal").find((row) => row.includes(`i="${id}"`))!;

    // A future target must not render as an age: the clamp at zero would print "~0s ago",
    // which is indistinguishable from a target that just passed and reads as the most
    // urgent value on the page while meaning the opposite.
    expect(rowFor(future.id)).toContain('ta="in 3d"');
    expect(rowFor(past.id)).toContain('ta="3d ago"');
    expect(rowFor(untargeted.id)).toContain('ta="unknown"');
    expect(rendered).not.toContain('ta="~0s ago"');
    expect(selfClosingRows(rendered, "c")[0]!).toContain('xa="in 6h"');
  });

  it("renders every authority directive in one line while keeping structural payloads exact", () => {
    const payload = `DIRECTIVE_HEAD_${"&".repeat(2_000)}_DIRECTIVE_TAIL`;
    const directives: NonNullable<DeliberationContext["creatorDirectiveBriefing"]>["directives"] = [
      ...Array.from({ length: 15 }, (_, index) => creatorDirective(index, `${index}:${payload}`)),
      {
        renderMode: "private",
        privateKind: "knowledge",
        kind: "subject_fact",
        subjectKind: "entity",
        subjectLabel: "private-subject",
        semanticSlot: null,
        semanticValue: null,
        canonicalFact: `PRIVATE_KNOWLEDGE_HEAD_${"k".repeat(2_000)}_PRIVATE_KNOWLEDGE_TAIL`,
        mentionPolicy: "only_if_topic_raised",
        priority: 500,
        createdAt: NOW_MS,
      },
      {
        renderMode: "private",
        privateKind: "operation",
        kind: "response_policy",
        operationalDirective: `PRIVATE_OPERATION_HEAD_${"o".repeat(2_000)}_PRIVATE_OPERATION_TAIL`,
        priority: 400,
        createdAt: NOW_MS,
      },
      {
        renderMode: "boundary",
        priority: 300,
        createdAt: NOW_MS,
      },
    ];
    const planner = build(
      context({
        creatorDirectiveBriefing: { directives },
      }),
    );
    const text = taggedBlock(allSystemText(planner), "borg_planner_authority_context");
    const rows = selfClosingRows(text, "d");

    expect(text).toContain('directives_total_for_current_audience="18"');
    expect(text).toContain('directives_rendered="18"');
    expect(text).toContain('<creator_directive_index rows_total_for_current_audience="18"');
    expect(text).toContain('rows_omitted_after_current_audience_scope="0"');
    expect(rows).toHaveLength(18);
    expect(rows.map((row) => row.match(/i="([^"]+)"/)?.[1])).toEqual(
      Array.from({ length: 18 }, (_, index) => `cd_${index + 1}`),
    );
    expect(rows.every((row) => !row.includes("\n"))).toBe(true);
    expect(text).toContain("DIRECTIVE_HEAD_");
    expect(text).toContain("_DIRECTIVE_TAIL");
    expect(text).toContain("[ELIDED]");
    expect(text).toContain('sc="c" k="sf" dh="a" sps="not_captured" sk="entity"');
    expect(text).toContain('sl="subject-0"');
    expect(text).toContain('mp="answer_if_asked" pk="cf" px="h:');
    expect(text).toContain('sc="pk" k="sf" dh="pk" sps="not_captured"');
    expect(text).toContain('mp="only_if_topic_raised" pk="cf"');
    expect(text).toContain('sc="po" k="rp" dh="po" sps="not_captured"');
    expect(text).toContain('pk="op"');
    expect(text).toContain('sc="b" k="db" dh="b" sps="not_captured"');
    expect(text).toContain('pk="bp"');
    expect(text.match(/<omitted_count>0<\/omitted_count>/g)).toHaveLength(1);
    expect(planner.traceSummary.sections.authority_and_directives).toMatchObject({
      rowCount: 18,
      truncationCount: 16,
      omissionCount: 0,
    });
    expect(
      planner.traceSummary.sections.authority_and_directives?.estimatedTokens,
    ).toBeLessThanOrEqual(4_000);
  });

  it("keeps the empty directive index count explicitly audience-relative", () => {
    const rendered = taggedBlock(
      allSystemText(build(context({ creatorDirectiveBriefing: null }))),
      "borg_planner_authority_context",
    );

    expect(rendered).toContain(
      '<creator_directive_index status="none" complete_for_current_audience="true" rows_total_for_current_audience="0" rows_omitted_after_current_audience_scope="0" />',
    );
    expect(selfClosingRows(rendered, "d")).toHaveLength(0);
  });

  it("keys exact authority payloads on structural kind and renders every captured scope field", () => {
    const creatorId = createEntityId();
    const allowedId = createEntityId();
    const excludedId = createEntityId();
    const directiveId = createCreatorDirectiveId();
    const operation = `OPERATION_HEAD_${"o".repeat(1_000)}_OPERATION_TAIL`;
    const rendered = allSystemText(
      build(
        context({
          creatorDirectiveBriefing: {
            directives: [
              {
                renderMode: "content",
                kind: "response_policy",
                subjectKind: "entity",
                subjectLabel: "subject",
                semanticSlot: "public_name",
                semanticValue: "slot value that must not select the fact lane",
                canonicalFact: null,
                operationalDirective: operation,
                mentionPolicy: "never_mention",
                priority: 9,
                createdAt: NOW_MS,
                scope: {
                  directiveId,
                  createdByEntityId: creatorId,
                  sourceSessionId: DEFAULT_SESSION_ID,
                  contentScope: "allow_list",
                  allowedEntityIds: [allowedId],
                  excludedEntityIds: [excludedId],
                  subjectMayKnow: null,
                  mentionPolicy: "never_mention",
                  deniedAudienceBehavior: "render_boundary_when_relevant",
                  activationScope: "all_except",
                  activationAllowedEntityIds: [],
                  activationExcludedEntityIds: [excludedId],
                },
              },
            ],
          },
        }),
      ),
    );
    const row = selfClosingRows(rendered, "d")[0]!;

    expect(row).toContain('k="rp"');
    expect(row).toContain('pk="op"');
    expect(row).toContain(`px="f:${operation.length}/${operation.length}"`);
    expect(row).toContain(`v="${operation}"`);
    expect(row).not.toContain("[ELIDED]");
    expect(row).toContain('sps="exact"');
    expect(row).toContain(`di="${directiveId}"`);
    expect(row).toContain(`cb="${creatorId}"`);
    expect(row).toContain(`os="${DEFAULT_SESSION_ID}"`);
    expect(row).toContain('cs="allow_list"');
    expect(row).toContain(`ae="${allowedId}"`);
    expect(row).toContain(`xe="${excludedId}"`);
    expect(row).toContain('smk="null"');
    expect(row).toContain('mp="never_mention"');
    expect(row).toContain('dab="render_boundary_when_relevant"');
    expect(row).toContain('as="all_except"');
    expect(row).toContain('aae="none"');
    expect(row).toContain(`axe="${excludedId}"`);
  });

  it("lets the complete authority structural floor overflow instead of dropping rows", () => {
    const planner = build(
      context({
        creatorDirectiveBriefing: {
          directives: Array.from({ length: 150 }, (_, index) => creatorDirective(index, "x")),
        },
      }),
    );
    const text = taggedBlock(allSystemText(planner), "borg_planner_authority_context");

    expect(text).toContain('directives_total_for_current_audience="150"');
    expect(text).toContain('directives_rendered="150"');
    expect(selfClosingRows(text, "d")).toHaveLength(150);
    expect(text.match(/<omitted_count>0<\/omitted_count>/g)).toHaveLength(1);
    expect(planner.traceSummary.sections.authority_and_directives).toMatchObject({
      rowCount: 150,
      truncationCount: 0,
      omissionCount: 0,
    });
    expect(planner.traceSummary.sections.authority_and_directives?.estimatedTokens).toBeGreaterThan(
      4_000,
    );
  });

  it("matches legacy profile and relational entity membership for each assembled audience context", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const entityIds = [alice, bob];
    const participantProfiles = entityIds.map((entityId, index) =>
      profileContext(entityId, index, entityId),
    );
    const activeParticipants = participantProfiles.map(({ entityId, displayName, role }) => ({
      entityId,
      displayName,
      role,
    }));
    const relationalSlots = entityIds.map((entityId, index) =>
      relationalSlot(entityId, `relationship.slot.${index}`),
    );

    for (const audienceEntityId of entityIds) {
      const assembled = context({
        audienceEntityId,
        activeParticipants,
        participantProfiles,
        relationalSlots,
      });
      const compact = allSystemText(build(assembled));
      const legacy = buildBaseSystemPrompt(assembled, {
        retrievalContextBudget: 1_000,
        semanticContextBudget: 1_000,
        nowMs: NOW_MS,
      });

      expect(
        entityMembership(taggedBlock(compact, "borg_planner_audience_profile_digest"), entityIds),
      ).toEqual(entityMembership(taggedBlock(legacy, "borg_audience_profile"), entityIds));
      expect(
        entityMembership(taggedBlock(compact, "borg_planner_relational_digest"), entityIds),
      ).toEqual(entityMembership(taggedBlock(legacy, "relational_slot_constraints"), entityIds));
    }
  });

  it("aggregates repeated decision derivations structurally and keeps firings separate", () => {
    const unlabeledDerivation = livedEntry({
      id: "decision_unlabeled",
      kind: "self_decision_introspection",
      occurredAt: NOW_MS - 3_000,
      text: "An older derivation without disclosure metadata.",
      outcomeReference: "goal_aaaaaaaaaaaaaaaa",
    });
    delete unlabeledDerivation.state_metadata?.disclosure_label;
    const entries = [
      unlabeledDerivation,
      livedEntry({
        id: "decision_old",
        kind: "self_decision_introspection",
        occurredAt: NOW_MS - 2_000,
        text: "First wording for the settled outcome.",
        outcomeReference: "goal_aaaaaaaaaaaaaaaa",
        disclosureClass: "public",
        stance: "claim",
        beliefEffect: "introduced",
      }),
      livedEntry({
        id: "decision_new",
        kind: "self_decision_introspection",
        occurredAt: NOW_MS - 1_000,
        text: "Different wording from the later derivation.",
        outcomeReference: "goal_aaaaaaaaaaaaaaaa",
        disclosureClass: "sensitive",
        stance: "correction",
        beliefEffect: "revised",
      }),
      livedEntry({
        id: "density",
        kind: "self_decision_density",
        occurredAt: NOW_MS,
        text: "Twenty autonomous reflections fired today.",
      }),
    ];
    const text = allSystemText(build(context({ evidenceLedger: evidenceLedger(entries) })));

    expect(text.match(/<decision_row /g)).toHaveLength(1);
    expect(text).toContain('outcome_ref="goal_aaaaaaaaaaaaaaaa"');
    expect(text).toContain('derivation_count="3"');
    expect(text).toContain("Different wording from the later derivation.");
    expect(text).not.toContain("First wording for the settled outcome.");
    expect(text).toContain('disclosure="disclosure_class=unknown');
    expect(text).toContain('category="firing_volume" kind="self_decision_density"');
    expect(text).toContain(
      'derivation_order="1:decision_unlabeled:2026-08-12T23:59:57.000Z:none:stance=none:belief_effect=none|2:decision_old:2026-08-12T23:59:58.000Z:none:stance=claim:belief_effect=introduced|3:decision_new:2026-08-12T23:59:59.000Z:none:stance=correction:belief_effect=revised"',
    );
  });

  it("combines repeated decision origins in chronology rather than reverse-lexical id order", () => {
    const oldestOrigin = entityIdHelpers.parse("ent_zzzzzzzzzzzzzzzz");
    const newestOrigin = entityIdHelpers.parse("ent_aaaaaaaaaaaaaaaa");
    const entries = [
      livedEntry({
        id: "decision_zzzzzzzzzzzzzzzz",
        kind: "self_decision_introspection",
        occurredAt: NOW_MS - 2_000,
        text: "Oldest derivation.",
        outcomeReference: "goal_chronological_origin",
        disclosureClass: "sensitive",
        originAudienceEntityIds: [oldestOrigin],
      }),
      livedEntry({
        id: "decision_aaaaaaaaaaaaaaaa",
        kind: "self_decision_introspection",
        occurredAt: NOW_MS - 1_000,
        text: "Newest derivation.",
        outcomeReference: "goal_chronological_origin",
        disclosureClass: "sensitive",
        originAudienceEntityIds: [newestOrigin],
      }),
    ];

    const text = allSystemText(build(context({ evidenceLedger: evidenceLedger(entries) })));

    expect(text).toContain(`origin_audience=${oldestOrigin},${newestOrigin}`);
    expect(text).not.toContain(`origin_audience=${newestOrigin},${oldestOrigin}`);
  });

  it("prioritizes structural open loops and expands the lived budget on autonomous turns", () => {
    const outbound = ["attempt-one", "attempt-two"].map(
      (id, index) =>
        ({
          id,
          source_type: "assistant_stream",
          session_scope: "global",
          actor: "assistant",
          trust_rank: 80,
          text: `outbound ${index}`,
          occurred_at: NOW_MS - index,
          stream_index: 20 + index,
          state_metadata: { source_kind: "outbound_attempt", outcome: "unknown" },
        }) as EvidenceLedgerEntry,
    );
    outbound.push(
      {
        id: "attempt-failed",
        source_type: "assistant_stream",
        session_scope: "global",
        actor: "assistant",
        trust_rank: 80,
        text: "failed outbound",
        occurred_at: NOW_MS - 3,
        stream_index: 23,
        state_metadata: { source_kind: "outbound_attempt", outcome: "failed" },
      } as EvidenceLedgerEntry,
      {
        id: "attempt-succeeded",
        source_type: "assistant_stream",
        session_scope: "global",
        actor: "assistant",
        trust_rank: 80,
        text: "completed outbound",
        occurred_at: NOW_MS - 4,
        stream_index: 24,
        state_metadata: { source_kind: "outbound_attempt", outcome: "succeeded" },
      } as EvidenceLedgerEntry,
      {
        id: "attempt-not-made",
        source_type: "assistant_stream",
        session_scope: "global",
        actor: "assistant",
        trust_rank: 80,
        text: "skipped outbound",
        occurred_at: NOW_MS - 5,
        stream_index: 25,
        state_metadata: {
          source_kind: "outbound_attempt",
          status: "not_attempted",
          outcome: "failed",
        },
      } as EvidenceLedgerEntry,
    );
    const source = context({
      turnOrigin: "autonomous",
      workingMemory: {
        ...context().workingMemory,
        pending_actions: [
          { description: "unfinished reach", next_action: "wait", created_at: NOW_MS - 500 },
        ],
      },
      evidenceLedger: {
        ...evidenceLedger(),
        sections: [
          { id: "autobiographical_recall", label: "Autobiographical recall", entries: outbound },
        ],
        audienceStanding: {
          ...evidenceLedger().audienceStanding!,
          recentLivedExperienceEntries: Array.from({ length: 12 }, (_, index) =>
            livedEntry({
              id: `completed-${index}`,
              kind: "cross_session_activity",
              occurredAt: NOW_MS - 10_000 - index,
              text: `completed ${index}`,
            }),
          ),
        },
      } as EvidenceLedger,
    });
    const rendered = build(source);
    const lived = rendered.system.map((block) => block.text).join("\n");

    expect(lived).toContain('autonomous_open_loop_priority="true"');
    expect(lived).toContain('target_tokens="8000"');
    expect(lived.match(/<open_loop_row /g)).toHaveLength(4);
    expect(lived).toContain('id="attempt-one" kind="outbound_attempt"');
    expect(lived).toContain('id="attempt-two" kind="outbound_attempt"');
    expect(lived).toContain(
      'id="attempt-failed" kind="outbound_attempt" status="attempted" outcome="failed"',
    );
    expect(lived).not.toContain('id="attempt-succeeded"');
    expect(lived).not.toContain('id="attempt-not-made"');
    expect(lived.match(/<activity_row /g)).toHaveLength(4);
    expect(lived.indexOf("<open_loops ")).toBeLessThan(lived.indexOf("<firings_and_activity "));
  });

  it("names each lane's cap and its own omission, and caps completed activity lower on a wake", () => {
    const completed = Array.from({ length: 12 }, (_, index) =>
      livedEntry({
        id: `completed-${index}`,
        kind: "cross_session_activity",
        occurredAt: NOW_MS - 10_000 - index,
        text: `completed ${index}`,
      }),
    );
    const lane = (planner: ReturnType<typeof build>, tag: string) => {
      const opening = taggedBlock(allSystemText(planner), "borg_planner_lived_experience_digest")
        .split("\n")
        .find((line) => line.trimStart().startsWith(`<${tag} `));
      return {
        cap: Number(opening?.match(/ cap="(\d+)"/)?.[1]),
        omitted: Number(opening?.match(/ omitted="(\d+)"/)?.[1]),
      };
    };

    const conversational = build(context({ evidenceLedger: evidenceLedger(completed) }));
    const wake = build(
      context({ turnOrigin: "autonomous", evidenceLedger: evidenceLedger(completed) }),
    );

    // The cap is the whole cause of the short lane: rendered saturates it, and
    // omitted is that lane's own residue rather than the digest-wide total.
    const conversationalActivity = lane(conversational, "firings_and_activity");
    const wakeActivity = lane(wake, "firings_and_activity");
    expect(wakeActivity.cap).toBeLessThan(conversationalActivity.cap);
    expect(allSystemText(conversational).match(/<activity_row /g)).toHaveLength(
      conversationalActivity.cap,
    );
    expect(allSystemText(wake).match(/<activity_row /g)).toHaveLength(wakeActivity.cap);
    expect(conversationalActivity.omitted).toBe(completed.length - conversationalActivity.cap);
    expect(wakeActivity.omitted).toBe(completed.length - wakeActivity.cap);

    // The lane a wake widens is the not-done one, and decisions are origin-stable.
    expect(lane(wake, "open_loops").cap).toBeGreaterThan(wakeActivity.cap);
    expect(lane(conversational, "open_loops").cap).toBeNaN();
    expect(lane(wake, "decisions").cap).toBe(lane(conversational, "decisions").cap);
    expect(lane(conversational, "decisions").omitted).toBe(0);

    // The open-loop lane is never queried off an autonomous turn, so a count
    // there would be reporting an unrun query as an empty result. The wake page
    // prints a real 0 for the same nothing; the conversational one must not.
    expect(allSystemText(conversational)).toContain('open_loop_rows_total="not_drawn"');
    expect(allSystemText(wake)).toContain('open_loop_rows_total="0"');

    // Same totals, different aggregate: the trailing count sums lane residues,
    // so it moves with the caps and with which lanes were drawn at all.
    const aggregate = (planner: ReturnType<typeof build>) =>
      Number(
        taggedBlock(allSystemText(planner), "borg_planner_lived_experience_digest").match(
          /<omitted_count>(\d+)<\/omitted_count>/,
        )?.[1],
      );
    expect(aggregate(conversational)).toBe(conversationalActivity.omitted);
    expect(aggregate(wake)).toBe(wakeActivity.omitted);
    expect(aggregate(wake)).toBeGreaterThan(aggregate(conversational));
  });

  it("pins the wake page's mechanism prose against the conversational page's", () => {
    // Only the conversational render of this digest is reachable to a reader who
    // cannot see a wake page, and only it is held in a prompt-surface fixture.
    // Rather than invent a width for the un-watched render, derive it: every
    // sentence readable on the watched page must appear verbatim on the wake
    // page, and the wake page may carry exactly one line that cannot be read
    // from the other side. Wake-only prose then has to be added deliberately.
    const laneTags = ["open_loops", "decisions", "firings_and_activity", "omitted_count"];
    const prose = (planner: ReturnType<typeof build>) =>
      taggedBlock(allSystemText(planner), "borg_planner_lived_experience_digest")
        .split("\n")
        .filter((line) => /^ {2}</.test(line))
        .map((line) => line.trim())
        .filter(
          (line) =>
            !laneTags.some(
              (tag) =>
                line.startsWith(`<${tag}>`) ||
                line.startsWith(`<${tag} `) ||
                line.startsWith(`</${tag}>`),
            ),
        );

    const conversational = prose(build(context({})));
    const wake = prose(build(context({ turnOrigin: "autonomous" })));

    expect(conversational.length).toBeGreaterThan(0);
    for (const line of conversational) expect(wake).toContain(line);

    const wakeOnly = wake.filter((line) => !conversational.includes(line));
    expect(wakeOnly).toHaveLength(1);
    expect(wakeOnly[0]?.startsWith("<autonomous_selection_policy>")).toBe(true);

    // Byte-symmetry is not truth. A line carried by both pages is the same
    // bytes on both, so any rendered value it quotes is being asserted on a
    // page where that value may not be what renders beside it -- and the pin
    // above passes an unscoped assertion exactly as it passes a scoped one.
    // Derive the hazard from the renders instead of naming it: every opening
    // attribute whose value moves with the origin is a slot the two pages
    // disagree about, and shared prose may quote neither side's reading of
    // one. Describing the branch stays open; asserting one side of it in prose
    // both sides carry does not. Only the origin-conditional line, which is
    // rendered on the page it describes, may quote that page's value.
    const openingAttributes = (planner: ReturnType<typeof build>) => {
      const opening =
        taggedBlock(allSystemText(planner), "borg_planner_lived_experience_digest").split(
          "\n",
        )[0] ?? "";
      return new Map<string, string>(
        [...opening.matchAll(/([a-z_]+)="([^"]*)"/g)].map((match) => [
          match[1] as string,
          match[2] as string,
        ]),
      );
    };
    const conversationalAttributes = openingAttributes(build(context({})));
    const wakeAttributes = openingAttributes(build(context({ turnOrigin: "autonomous" })));
    const originVaryingValues = [
      ...new Set([...conversationalAttributes.keys(), ...wakeAttributes.keys()]),
    ]
      .filter((name) => conversationalAttributes.get(name) !== wakeAttributes.get(name))
      .flatMap((name) => [conversationalAttributes.get(name), wakeAttributes.get(name)])
      .filter((value): value is string => value !== undefined && value.length > 0);

    expect(originVaryingValues).toContain("not_drawn");
    for (const line of conversational.filter((entry) => wake.includes(entry))) {
      for (const value of originVaryingValues) {
        expect(line).not.toMatch(
          new RegExp(`(^|[^\\w-])${value.replaceAll(/[.*+?^${}()|[\]\\-]/g, "\\$&")}([^\\w-]|$)`),
        );
      }
    }
  });

  it("combines disclosure fail-closed when the same open loop has two structural sources", () => {
    const questionId = "oq_open_loop";
    const source = context({
      turnOrigin: "autonomous",
      openQuestionsContext: [
        {
          id: questionId,
          question: "What remains unfinished?",
          status: "open",
          urgency: 0.5,
          source: "user",
          audience_entity_id: null,
          goal_id: null,
          created_at: NOW_MS - 2_000,
          last_touched: NOW_MS - 1_000,
          resolution_note: null,
          resolved_at: null,
          abandoned_reason: null,
          provenance: { kind: "manual" },
          disclosure_label: {
            disclosureClass: "public",
            originAudienceEntityIds: [],
            privateToEntityIds: [],
            publicToEntityIds: [],
          },
        } as never,
      ],
      evidenceLedger: {
        ...evidenceLedger(),
        sections: [
          {
            id: "autobiographical_recall",
            label: "Autobiographical recall",
            entries: [
              {
                id: "recalled-question",
                source_type: "system_metadata",
                session_scope: "global",
                actor: "memory",
                trust_rank: 70,
                text: "What remains unfinished?",
                state_metadata: {
                  source_kind: "open_question",
                  open_question_id: questionId,
                  status: "open",
                  occurred_at: NOW_MS,
                },
              },
            ],
          },
        ],
      } as EvidenceLedger,
    });
    const rendered = allSystemText(build(source));

    expect(rendered.match(new RegExp(`id="${questionId}"`, "g"))).toHaveLength(1);
    expect(rendered).toContain(
      `id="${questionId}" kind="open_question" status="open" outcome="pending"`,
    );
    expect(rendered).toContain('disclosure="disclosure_class=unknown');
  });

  it("renders complete indexes, explicit omission counts, and trace budget metrics", () => {
    const goals = Array.from({ length: 10 }, (_, index) =>
      goal(`Goal ${index}`, { priority: index }),
    );
    const entries = [
      ...Array.from({ length: 20 }, (_, index) =>
        livedEntry({
          id: `decision_${index}`,
          kind: "self_decision_introspection",
          occurredAt: NOW_MS - index,
          text: `Decision ${index}`,
          outcomeReference: `structural_outcome_${index}`,
        }),
      ),
      ...Array.from({ length: 18 }, (_, index) =>
        livedEntry({
          id: `activity_${index}`,
          kind: "cross_session_activity",
          occurredAt: NOW_MS - index,
          text: `Activity ${index}`,
        }),
      ),
    ];
    const executiveFocus = {
      selected_goal: goals[0]!,
      selected_score: null,
      candidates: goals.map((candidate, index) => ({
        goal_id: candidate.id,
        goal: candidate,
        score: 100 - index,
        components: { priority: 0, deadline_pressure: 0, context_fit: 0, progress_debt: 0 },
        reason: "ranked globally",
      })),
      threshold: 0,
      score_basis: {
        score_context: "turn_selection" as const,
        deadline_lookahead_ms: 1,
        progress_debt_stale_ms: 1,
      },
    };
    const planner = build(
      context({
        selfSnapshot: { values: [], goals, traits: [] },
        executiveFocus,
        evidenceLedger: evidenceLedger(entries),
      }),
    );
    const text = allSystemText(planner);

    expect(text.match(/<goal /g)).toHaveLength(10);
    expect(text.match(/<goal_detail /g)).toHaveLength(4);
    expect(text).toContain("<omitted_count>6</omitted_count>");
    expect(text.match(/<decision_row /g)).toHaveLength(8);
    expect(text.match(/<activity_row /g)).toHaveLength(8);
    expect(text).toContain("<omitted_count>22</omitted_count>");
    expect(planner.traceSummary.sections.goal_index).toMatchObject({
      rowCount: 14,
      omissionCount: 6,
    });
    expect(planner.traceSummary.sections.lived_experience).toMatchObject({
      rowCount: 16,
      omissionCount: 22,
    });
    expect(planner.traceSummary.totalEstimatedTokens).toBeGreaterThan(0);
  });

  it("keeps complete goal membership for a 98-goal snapshot inside the whole-block target", () => {
    const goals = Array.from({ length: 98 }, (_, index) =>
      goal(`GOAL_${index}_${"d".repeat(512)}`, {
        priority: 98 - index,
        created_at: NOW_MS - (98 - index) * 1_000,
        terminal_condition: `TERMINAL_${index}_${"t".repeat(512)}`,
        progress_notes: `PROGRESS_${index}_${"p".repeat(512)}`,
      }),
    );
    const candidates = goals.map((candidate, index) => ({
      goal_id: candidate.id,
      goal: candidate,
      score: 98 - index,
      components: { priority: 1, deadline_pressure: 1, context_fit: 1, progress_debt: 1 },
      reason: `REASON_${index}_${"r".repeat(512)}`,
    }));
    const topOpenSteps = goals.slice(0, 4).map((candidate, index) => ({
      id: createExecutiveStepId(),
      goal_id: candidate.id,
      description: `STEP_${index}_${"s".repeat(512)}`,
      status: "queued" as const,
      kind: "think" as const,
      due_at: null,
      last_attempt_ts: null,
      created_at: NOW_MS,
      updated_at: NOW_MS,
      provenance: { kind: "manual" as const },
    }));
    const planner = build(
      context({
        selfSnapshot: { values: [], goals, traits: [] },
        executiveFocus: {
          selected_goal: goals[0]!,
          selected_score: candidates[0]!,
          next_step: topOpenSteps[0]!,
          candidate_steps: {
            top_open_steps: topOpenSteps,
            omitted_open_step_count: 94,
          },
          candidates,
          threshold: 0,
          score_basis: {
            score_context: "turn_selection" as const,
            deadline_lookahead_ms: 1,
            progress_debt_stale_ms: 1,
          },
        },
      }),
    );
    const block = taggedBlock(allSystemText(planner), "borg_planner_goal_digest");
    const indexBlock = taggedBlock(block, "goal_index");

    expect(block).toContain(
      `<borg_planner_goal_digest complete_membership="true" rows_total="98" goal_index_rows_rendered="98" membership_order="global_executive_score_desc_then_priority_desc_created_at_asc_id_asc" target_tokens="${PLANNER_GOAL_TARGET_TOKENS}"`,
    );
    expect(indexBlock.match(/<goal /g)).toHaveLength(98);
    expect(indexBlock).toContain("<omitted_count>0</omitted_count>");
    expect(block.match(/<next_step /g)).toHaveLength(4);
    expect(block).not.toContain(' goal_index_not_enumerated_budget="');
    expect(indexBlock).not.toContain("<goal_index_not_enumerated_budget ");
    expect(planner.traceSummary.sections.goal_index?.estimatedTokens).toBeLessThanOrEqual(
      PLANNER_GOAL_TARGET_TOKENS,
    );
  });

  it("drops only a ranked goal-index suffix and includes that remainder in trace omissions", () => {
    const goals = Array.from({ length: 300 }, (_, index) =>
      goal(`GOAL_${index}_${"&".repeat(1_000)}`, {
        priority: 300 - index,
        created_at: NOW_MS - (300 - index) * 1_000,
        terminal_condition: "&".repeat(1_000),
        progress_notes: "&".repeat(1_000),
      }),
    );
    const candidates = goals.map((candidate, index) => ({
      goal_id: candidate.id,
      goal: candidate,
      score: 300 - index,
      components: { priority: 1, deadline_pressure: 1, context_fit: 1, progress_debt: 1 },
      reason: "&".repeat(1_000),
    }));
    const planner = build(
      context({
        selfSnapshot: { values: [], goals, traits: [] },
        executiveFocus: {
          selected_goal: goals[0]!,
          selected_score: candidates[0]!,
          candidates,
          threshold: 0,
          score_basis: {
            score_context: "turn_selection" as const,
            deadline_lookahead_ms: 1,
            progress_debt_stale_ms: 1,
          },
        },
      }),
    );
    const block = taggedBlock(allSystemText(planner), "borg_planner_goal_digest");
    const indexBlock = taggedBlock(block, "goal_index");
    const renderedIds = [...indexBlock.matchAll(/<goal i="([^"]+)"/g)].map((match) => match[1]);
    const omittedIndexCount = goals.length - renderedIds.length;
    const expandedBlock = taggedBlock(block, "top_global_candidates_expanded");
    const expandedOmissionCount = Number(
      /<omitted_count>(\d+)<\/omitted_count>/.exec(expandedBlock)?.[1],
    );

    expect(renderedIds.length).toBeGreaterThanOrEqual(4);
    expect(renderedIds.length).toBeLessThan(goals.length);
    expect(renderedIds).toEqual(goals.slice(0, renderedIds.length).map((entry) => entry.id));
    expect(block).toContain('complete_membership="false"');
    expect(block).toContain(`goal_index_not_enumerated_budget="${omittedIndexCount}"`);
    expect(indexBlock).toContain(
      `<goal_index_not_enumerated_budget total="${omittedIndexCount}" membership_order="global_executive_score_desc_then_priority_desc_created_at_asc_id_asc" />`,
    );
    expect(indexBlock).toContain(`<omitted_count>${omittedIndexCount}</omitted_count>`);
    expect(planner.traceSummary.sections.goal_index).toMatchObject({
      rowCount: renderedIds.length + 4,
      omissionCount: omittedIndexCount + expandedOmissionCount,
    });
    expect(planner.traceSummary.sections.goal_index?.estimatedTokens).toBeLessThanOrEqual(
      PLANNER_GOAL_TARGET_TOKENS,
    );
  });

  it("reports retained statuses and scopes complete membership to the supplied snapshot", () => {
    const mixed = build(
      context({
        selfSnapshot: {
          values: [],
          goals: [goal("Still running"), goal("Retired", { status: "done" }), goal("Also running")],
          traits: [],
        },
      }),
    );
    const mixedText = allSystemText(mixed);

    const mixedBlock = taggedBlock(mixedText, "borg_planner_goal_digest");
    expect(mixedBlock).toContain(
      '<borg_planner_goal_digest complete_membership="true" rows_total="3" goal_index_rows_rendered="3"',
    );
    expect(mixedText).toContain(
      '<goal_index complete_membership="true" rows="3" statuses_present="active,done" description_excerpt_budget_chars=',
    );
    expect(mixedText).toContain(
      "The upstream snapshot is status-scoped; statuses_present names retained statuses",
    );
    expect(mixedBlock).not.toContain(' goal_index_not_enumerated_budget="');
    expect(mixedBlock).not.toContain("<goal_index_not_enumerated_budget ");

    const singleStatus = build(
      context({
        selfSnapshot: { values: [], goals: [goal("Still running")], traits: [] },
      }),
    );

    expect(allSystemText(singleStatus)).toContain(
      '<goal_index complete_membership="true" rows="1" statuses_present="active" description_excerpt_budget_chars=',
    );

    const empty = build(context({ selfSnapshot: { values: [], goals: [], traits: [] } }));

    expect(allSystemText(empty)).toContain(
      '<goal_index complete_membership="true" rows="0" statuses_present="none" description_excerpt_budget_chars=',
    );
  });

  it("renders the progress log oldest first and names pn as an append-only log in the legend", () => {
    const target = goal("Append-ordered progress log", {
      progress_notes: [
        `[${NOW_MS - 5 * 24 * 60 * 60_000}] OLDEST_PROGRESS_ENTRY`,
        `[${NOW_MS - 24 * 60 * 60_000}] ${"m".repeat(2_000)}`,
        `[${NOW_MS - 60_000}] NEWEST_PROGRESS_ENTRY`,
      ].join("\n"),
      last_progress_ts: NOW_MS - 60_000,
    });
    const planner = build(
      context({
        selfSnapshot: { values: [], goals: [target], traits: [] },
        executiveFocus: {
          selected_goal: target,
          selected_score: null,
          candidates: [
            {
              goal_id: target.id,
              goal: target,
              score: 1,
              components: {
                priority: 1,
                deadline_pressure: 1,
                context_fit: 1,
                progress_debt: 1,
              },
              reason: "focus",
            },
          ],
          threshold: 0,
          score_basis: {
            score_context: "turn_selection" as const,
            deadline_lookahead_ms: 1,
            progress_debt_stale_ms: 1,
          },
        },
      }),
    );
    const text = allSystemText(planner);

    // pn keeps the oldest append at its head and the newest at its tail; the elision is the middle.
    expect(text).toContain("OLDEST_PROGRESS_ENTRY");
    expect(text).toContain("NEWEST_PROGRESS_ENTRY");
    expect(text).toContain("HEAD+TAIL EXCERPT");
    expect(text).toContain("pn is an append-ordered log, not a current note");
    expect(text).toContain("so pa dates its tail and never its head");
    expect(text).not.toContain("nothing here rewrites or deletes an entry once written");
    // Appending is the writers' convention; the column itself is replaced whole on every write.
    expect(text).toContain(
      "Appending is a habit of the writers that reach this field rather than a property the field enforces",
    );
    expect(text).toContain("every write replaces the whole column");
    expect(text).toContain(
      "the operator progress writer replaces it with the single note it was handed",
    );
    expect(text).toContain("an entry can be rewritten or dropped by a write from outside a turn");
    expect(text).toContain("rv counts such a write exactly as it counts an append.");
    // A replacing write displaces the entry it drops into the identity event it writes; it does not
    // destroy it. Without this, "dropped" reads as gone and an unread register reads as a limit.
    expect(text).toContain(
      "Those events carry the whole prior row, so a pn entry a later write replaced survives in them, out of reach from this page rather than gone",
    );
    // pn is an excerpt of the column, so its silence is not evidence about the column.
    expect(text).toContain("It is also an excerpt and not the log");
    expect(text).toContain("an entry missing from pn is not evidence it was never written");
    expect(text).toContain("the online turn reflector on user and autonomous turns");
    expect(text).toContain(
      "The reflector appends only when its own structured judgment says the turn made concrete movement; an emission alone does not write progress",
    );
    expect(text).toContain("sdebt uses this same progress_debt_stale_ms denominator");
    expect(text).toContain(
      "The separate executive-focus staleness cadence controls when the stale lane is eligible to wake; it does not rescale progress debt",
    );
    // The excerpt's size is a budget the marker is charged against, so rendered= is not it.
    expect(text).toContain('field_excerpt_budget_chars="240"');
    expect(text).toContain(
      "that budget is spent on the whole excerpt including the marker that announces the cut",
    );
    expect(text).toContain("rendered= cannot be read back as the budget");
    expect(text).toContain(
      "The budget is a fixed size rather than a share, so a longer log earns no more of this page than a short one",
    );
  });

  it("spends the expanded-field budget on the excerpt marker as well as the field text", () => {
    // One budget covers d, tc, pn and er, and it is spent on the whole excerpt -- so the
    // marker's rendered= sits below the printed budget by exactly the marker's own length,
    // and shifts between rows with the digit widths of the three numbers the marker carries.
    // Asserted as an identity against whatever the marker currently costs rather than
    // against a copied constant, so a template change cannot silently break the claim.
    const renderedByLength = [1_897, 40_028, 400_028].map((totalChars) => {
      const target = goal("Budgeted progress log", {
        progress_notes: "p".repeat(totalChars),
        last_progress_ts: NOW_MS - 60_000,
      });
      const text = allSystemText(
        build(
          context({
            selfSnapshot: { values: [], goals: [target], traits: [] },
            executiveFocus: {
              selected_goal: target,
              selected_score: null,
              candidates: [
                {
                  goal_id: target.id,
                  goal: target,
                  score: 1,
                  components: {
                    priority: 1,
                    deadline_pressure: 1,
                    context_fit: 1,
                    progress_debt: 1,
                  },
                  reason: "focus",
                },
              ],
              threshold: 0,
              score_basis: {
                score_context: "turn_selection" as const,
                deadline_lookahead_ms: 1,
                progress_debt_stale_ms: 1,
              },
            },
          }),
        ),
      );

      const budget = Number(/field_excerpt_budget_chars="(\d+)"/.exec(text)?.[1]);
      const progressAttribute = /\spn="([^"]*)"/.exec(text)?.[1];
      expect(progressAttribute).toBeDefined();
      const marker = / \[ELIDED \d+ CHARS; HEAD\+TAIL EXCERPT; rendered=\d+\/total=\d+\] /.exec(
        progressAttribute!,
      )?.[0];
      expect(marker).toBeDefined();
      const rendered = Number(/rendered=(\d+)\//.exec(marker!)?.[1]);
      const elided = Number(/\[ELIDED (\d+) CHARS/.exec(marker!)?.[1]);

      expect(marker).toContain(`total=${totalChars}`);
      expect(rendered + elided).toBe(totalChars);
      expect(rendered + marker!.length).toBe(budget);
      expect(rendered).toBeLessThan(budget);
      return rendered;
    });

    // Three marker widths, so the ceiling is demonstrably not one number the page could be
    // read back for -- which is the whole reason the legend says rendered= is not the budget.
    expect(new Set(renderedByLength).size).toBe(3);
  });

  it("prints the index description budget and closes each container against its own", () => {
    // The same goal's description is cut twice on this page, against two different
    // budgets, so a width reconstructed from one container's residue is not evidence
    // about the other. Both budgets are read off the page rather than copied, so
    // changing either constant cannot make the identity pass by accident.
    const target = goal("d".repeat(4_096));
    const text = allSystemText(
      build(
        context({
          selfSnapshot: { values: [], goals: [target], traits: [] },
          executiveFocus: {
            selected_goal: target,
            selected_score: null,
            candidates: [
              {
                goal_id: target.id,
                goal: target,
                score: 1,
                components: {
                  priority: 1,
                  deadline_pressure: 1,
                  context_fit: 1,
                  progress_debt: 1,
                },
                reason: "focus",
              },
            ],
            threshold: 0,
            score_basis: {
              score_context: "turn_selection" as const,
              deadline_lookahead_ms: 1,
              progress_debt_stale_ms: 1,
            },
          },
        }),
      ),
    );

    const detailedContainerBudget = (open: string, close: string, attribute: string) => {
      const section = text.slice(text.indexOf(open), text.indexOf(close));
      expect(section).not.toBe("");
      const budget = Number(new RegExp(`${attribute}="(\\d+)"`).exec(section)?.[1]);
      const excerpt = /\sd="([^"]*)"/.exec(section)?.[1];
      expect(excerpt).toBeDefined();
      const marker = / \[ELIDED \d+ CHARS; HEAD\+TAIL EXCERPT; rendered=\d+\/total=\d+\] /.exec(
        excerpt!,
      )?.[0];
      expect(marker).toBeDefined();
      const rendered = Number(/rendered=(\d+)\//.exec(marker!)?.[1]);
      expect(rendered + marker!.length).toBe(budget);
      return { budget, rendered };
    };

    const indexSection = text.slice(text.indexOf("<goal_index"), text.indexOf("</goal_index>"));
    const indexBudget = Number(/description_excerpt_budget_chars="(\d+)"/.exec(indexSection)?.[1]);
    const indexExcerpt = /\sd="([^"]*)"/.exec(indexSection)?.[1];
    expect(indexExcerpt).toBeDefined();
    expect(indexExcerpt).toContain("[ELIDED]");
    expect(indexExcerpt).toHaveLength(indexBudget);

    const expanded = detailedContainerBudget(
      "<top_global_candidates_expanded",
      "</top_global_candidates_expanded>",
      "field_excerpt_budget_chars",
    );

    // Two containers, one description, two widths -- which is the whole reason the
    // legend says a budget governs the container that prints it and nothing else.
    expect(indexBudget).not.toBe(expanded.budget);
    expect(indexExcerpt!.length - "[ELIDED]".length).not.toBe(expanded.rendered);
    expect(text).toContain(
      "The one-line index rows carry a d of their own, sized against a separate, smaller, dynamically selected budget",
    );
    expect(text).toContain("A budget printed on one container governs that container alone");
    expect(text).toContain(
      "a number matching it elsewhere is a different budget that happens to agree",
    );
  });

  it("renders each expanded candidate's top open step and reports exact in-scope omissions", () => {
    const goals = Array.from({ length: 7 }, (_, index) => goal(`Goal ${index}`));
    const target = goals[0]!;
    const otherCandidate = goals[1]!;
    const selectedStep = {
      id: "exstep_aaaaaaaaaaaaaaaa" as never,
      goal_id: target.id,
      description: "Top open step of the selected goal",
      status: "doing" as const,
      kind: "think" as const,
      due_at: null,
      last_attempt_ts: null,
      created_at: NOW_MS,
      updated_at: NOW_MS,
      provenance: { kind: "manual" as const },
    };
    const otherStep = {
      ...selectedStep,
      id: "exstep_bbbbbbbbbbbbbbbb" as never,
      goal_id: otherCandidate.id,
      description: "Top open step of the other expanded goal",
      status: "queued" as const,
    };
    const planner = build(
      context({
        selfSnapshot: { values: [], goals, traits: [] },
        executiveFocus: {
          selected_goal: target,
          selected_score: null,
          next_step: selectedStep,
          candidate_steps: {
            top_open_steps: [selectedStep, otherStep],
            omitted_open_step_count: 3,
          },
          candidates: [
            {
              goal_id: target.id,
              goal: target,
              score: 1,
              components: {
                priority: 1,
                deadline_pressure: 1,
                context_fit: 1,
                progress_debt: 1,
              },
              reason: "focus",
            },
            {
              goal_id: otherCandidate.id,
              goal: otherCandidate,
              score: 0.9,
              components: {
                priority: 0.9,
                deadline_pressure: 1,
                context_fit: 1,
                progress_debt: 1,
              },
              reason: "other candidate",
            },
          ],
          threshold: 0,
          score_basis: {
            score_context: "turn_selection" as const,
            deadline_lookahead_ms: 1,
            progress_debt_stale_ms: 1,
          },
        },
      }),
    );
    const text = allSystemText(planner);

    const stepRows = text.match(/<next_step [^\n]+/g) ?? [];
    expect(stepRows).toHaveLength(2);
    expect(stepRows.find((row) => row.match(selectedStep.id))).toContain('sel="true"');
    expect(stepRows.find((row) => row.match(otherStep.id))).toContain('sel="false"');
    expect(text).toContain(
      '<executive_next_step_omitted_count scope="expanded_candidates_top_open">3</executive_next_step_omitted_count>',
    );
    expect(text).toContain("follows top_global_candidates_expanded membership and order");
    expect(text).toContain("Steps of goals outside the expansion are not queried");
    expect(text).toContain("remain uncounted");

    expect(text.match(/<goal /g)).toHaveLength(goals.length);
    expect(text).toContain("<omitted_count>0</omitted_count>");
    expect(text).toContain("complete_membership=true means every rows_total goal has an index row");
  });

  it("reports source-ledger rows excluded by the compact ledger and carries omission guidance", () => {
    const currentEntries = [
      livedEntry({
        id: "current_1",
        kind: "current_message",
        occurredAt: NOW_MS,
        text: "Current message one",
      }),
      livedEntry({
        id: "current_2",
        kind: "current_message",
        occurredAt: NOW_MS,
        text: "Current message two",
      }),
    ];
    const episodeEntries = [
      livedEntry({
        id: "episode_1",
        kind: "episode",
        occurredAt: NOW_MS,
        text: "Episode one",
      }),
      livedEntry({
        id: "episode_2",
        kind: "episode",
        occurredAt: NOW_MS,
        text: "Episode two",
      }),
    ];
    const ledger = {
      ...evidenceLedger(),
      sections: [
        {
          id: "current_user_message" as const,
          label: "Current user",
          entries: currentEntries,
        },
        { id: "episodes" as const, label: "Episodes", entries: episodeEntries },
      ],
    };
    const compactPlannerLedger = buildCompactPlannerLedgerPrompt(ledger);
    const planner = buildCompactPlannerSystemPrompt({
      context: context({ evidenceLedger: ledger }),
      staticPrefix: "STATIC_HEAD_SENTINEL",
      compactPlannerLedger,
    });
    const text = planner.system.map((block) => block.text).join("\n\n");

    expect(text).toContain('<section id="current_user_message" omitted_count="1" />');
    expect(text).toContain('<section id="episodes" omitted_count="2" />');
    expect(text).toContain("<omitted_count>3</omitted_count>");
    expect(text).toContain("I name that limitation in plan.uncertainty");
    expect(text).toContain("conservative about creating NEW follow-up intents");
    expect(planner.traceSummary.sections.compact_evidence_ledger?.omissionCount).toBe(3);
  });

  it("uses visible head+tail excerpts for advisory commitments and never truncates critical directives", () => {
    const advisoryDirective = `ADVISORY_HEAD_${"a".repeat(1_500)}_ADVISORY_TAIL`;
    const criticalDirective = `CRITICAL_HEAD_${"c".repeat(1_500)}_CRITICAL_TAIL`;
    const planner = build(
      context({
        applicableCommitments: [
          commitment(advisoryDirective),
          commitment(criticalDirective, {
            enforcement_class: "critical",
            critical_domain: "safety",
          }),
        ],
      }),
    );
    const text = allSystemText(planner);

    expect(text).toContain("ADVISORY_HEAD_");
    expect(text).toContain("_ADVISORY_TAIL");
    expect(text).toContain("[ELIDED]");
    expect(text).toContain('shape="head+tail"');
    expect(text).toMatch(/ r="\d+" n="\d+" e="\d+"/);
    expect(text).toContain(criticalDirective);
    expect(text).toContain('critical_overflow="false"');
    expect(planner.traceSummary.criticalOverflow).toBe(false);
    expect(planner.traceSummary.sections.commitments?.truncationCount).toBe(1);
    expect(planner.traceSummary.sections.commitments?.omissionCount).toBe(0);
  });

  it("keeps critical membership and reports the exact budget remainder at live row scale", () => {
    const restrictedAudience = createEntityId();
    const critical = [
      commitment("CRITICAL_BUDGET_ALPHA", {
        id: "cmt_0000000000000001" as CommitmentRecord["id"],
        enforcement_class: "critical",
        critical_domain: "privacy",
        priority: -1,
      }),
      commitment("CRITICAL_BUDGET_BETA", {
        id: "cmt_0000000000000002" as CommitmentRecord["id"],
        enforcement_class: "critical",
        critical_domain: "safety",
        priority: -2,
      }),
    ];
    const ordinary = Array.from({ length: 156 }, (_unused, index) =>
      commitment(`ADVISORY_${index}_${"d".repeat(1_000)}`, {
        id: `cmt_${(index + 3).toString(36).padStart(16, "0")}` as CommitmentRecord["id"],
        kind: (["assistant_commitment", "participant_preference", "process_norm"] as const)[
          index % 3
        ],
        priority: 156 - index,
        created_at: NOW_MS - (156 - index) * 1_000,
        restricted_audience: index % 2 === 0 ? restrictedAudience : null,
      }),
    );
    const planner = build(
      context({
        applicableCommitments: [...ordinary, ...critical],
        applicableCommitmentsReadAtMs: NOW_MS - 1_234,
      }),
    );
    const block = taggedBlock(allSystemText(planner), "borg_planner_commitment_digest");
    const renderedIds = new Set([...block.matchAll(/<c i="([^"]+)"/g)].map((match) => match[1]));
    const omitted = ordinary.filter((entry) => !renderedIds.has(entry.id));
    const omittedCount = Number(/membership_not_enumerated_budget="(\d+)"/.exec(block)?.[1]);
    const advisoryBudget = Number(/advisory_excerpt_reserved_chars="(\d+)"/.exec(block)?.[1]);
    const firstRenderedAdvisory = ordinary.find((entry) => renderedIds.has(entry.id));
    const firstRenderedAdvisoryRow =
      firstRenderedAdvisory === undefined
        ? ""
        : (block.match(new RegExp(`<c i="${firstRenderedAdvisory.id}"[^>]+/>`))?.[0] ?? "");
    const retainedDirectiveChars = Number(/ r="(\d+)"/.exec(firstRenderedAdvisoryRow)?.[1]);
    const disclosureBreakdown =
      block.match(/<membership_not_enumerated_by_disclosure_class ([^>]+)\/>/)?.[1] ?? "";
    const kindBreakdown = block.match(/<membership_not_enumerated_by_kind ([^>]+)\/>/)?.[1] ?? "";
    const omittedRelationshipPrivate = omitted.filter(
      (entry) => entry.restricted_audience !== null,
    ).length;
    const omittedUnknown = omitted.length - omittedRelationshipPrivate;

    expect(block).toContain('rows_total="158"');
    expect(block).toContain('rows_total_as_of="2026-08-12T23:59:58.766Z"');
    expect(block).toContain(
      'membership_order="critical_commitments_first_then_priority_desc_created_at_asc_id_asc"',
    );
    expect(block).toContain('complete_membership="false"');
    expect(renderedIds).toContain(critical[0]!.id);
    expect(renderedIds).toContain(critical[1]!.id);
    expect(omitted.length).toBeGreaterThan(0);
    expect(omittedCount).toBe(omitted.length);
    expect(block).toContain(`<membership_not_enumerated_budget total="${omitted.length}">`);
    expect(disclosureBreakdown).toContain(`relationship_private="${omittedRelationshipPrivate}"`);
    expect(disclosureBreakdown).toContain(`unknown="${omittedUnknown}"`);
    for (const kind of [
      "assistant_commitment",
      "participant_preference",
      "process_norm",
    ] as const) {
      expect(kindBreakdown).toContain(
        `${kind}="${omitted.filter((entry) => entry.kind === kind).length}"`,
      );
    }
    expect(firstRenderedAdvisory).toBeDefined();
    expect(firstRenderedAdvisoryRow).toContain("[ELIDED]");
    expect(retainedDirectiveChars + "[ELIDED]".length).toBe(advisoryBudget);
    expect(planner.traceSummary.sections.commitments).toMatchObject({
      estimatedTokens: expect.any(Number),
      omissionCount: omitted.length,
      criticalOverflow: false,
    });
    expect(planner.traceSummary.sections.commitments!.estimatedTokens).toBeLessThanOrEqual(8_000);
  });

  it("detects critical overflow from the actual XML-escaped row", () => {
    const criticalDirective = "&".repeat(15_000);
    const planner = build(
      context({
        applicableCommitments: [
          commitment(criticalDirective, {
            enforcement_class: "critical",
            critical_domain: "safety",
          }),
        ],
      }),
    );
    const commitmentSection = planner.traceSummary.sections.commitments;

    expect(Math.ceil(criticalDirective.length / 4)).toBeLessThan(8_000);
    expect(allSystemText(planner)).toContain("&amp;".repeat(15_000));
    expect(commitmentSection?.estimatedTokens).toBeGreaterThan(8_000);
    expect(commitmentSection?.criticalOverflow).toBe(true);
    expect(planner.traceSummary.criticalOverflow).toBe(true);
  });

  it("allocates advisory excerpts against their actual XML-escaped rows", () => {
    const planner = build(
      context({
        applicableCommitments: Array.from({ length: 30 }, () => commitment("&".repeat(20_000))),
      }),
    );
    const text = allSystemText(planner);
    const budget = Number(text.match(/advisory_excerpt_reserved_chars="(\d+)"/)?.[1]);

    expect(text.match(/<c /g)).toHaveLength(30);
    expect(budget).toBeGreaterThanOrEqual(96);
    expect(budget).toBeLessThan(320);
    expect(planner.traceSummary.sections.commitments?.estimatedTokens).toBeLessThanOrEqual(8_000);
    expect(planner.traceSummary.sections.commitments?.criticalOverflow).toBe(false);
  });

  it("mechanically bounds schema-valid non-critical labels and reports the compact envelope", () => {
    const hugeTraitLabel = `TRAIT_HEAD_${"x".repeat(200_000)}_TRAIT_TAIL`;
    const planner = build(
      context({
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [
            {
              id: createTraitId(),
              label: hugeTraitLabel,
              strength: 0.8,
              last_reinforced: NOW_MS,
              last_decayed: null,
              state: "established",
              established_at: NOW_MS,
              confidence: 0.9,
              last_tested_at: null,
              last_contradicted_at: null,
              support_count: 1,
              contradiction_count: 0,
              evidence_episode_ids: [],
              provenance: { kind: "manual" },
            },
          ],
        },
      }),
    );
    const text = allSystemText(planner);

    expect(text).toContain("TRAIT_HEAD_");
    expect(text).toContain("_TRAIT_TAIL");
    expect(text).toContain("HEAD+TAIL EXCERPT");
    expect(text).not.toContain(hugeTraitLabel);
    expect(planner.traceSummary.sections.durable_self?.truncationCount).toBe(1);
    expect(planner.traceSummary.targetTokens).toBe(COMPACT_PLANNER_TARGET_TOKENS);
    expect(planner.traceSummary.overallOverflow).toBe(false);
  });

  it("keeps a representative bounded high-water surface inside the 20-25K estimated envelope", () => {
    const freeText = `FIELD_HEAD_${"x".repeat(20_000)}_FIELD_TAIL`;
    const goals = Array.from({ length: 8 }, (_, index) =>
      goal(freeText, {
        priority: 10 - index,
        terminal_condition: freeText,
        progress_notes: freeText,
      }),
    );
    const executiveFocus = {
      selected_goal: goals[0]!,
      selected_score: null,
      candidates: goals.map((candidate, index) => ({
        goal_id: candidate.id,
        goal: candidate,
        score: 100 - index,
        components: { priority: 1, deadline_pressure: 1, context_fit: 1, progress_debt: 1 },
        reason: freeText,
      })),
      threshold: 0,
      score_basis: {
        score_context: "turn_selection" as const,
        deadline_lookahead_ms: 1,
        progress_debt_stale_ms: 1,
      },
    };
    const entities = Array.from({ length: 6 }, () => createEntityId());
    const participantProfiles = entities.map((entityId, index) =>
      profileContext(entityId, index, freeText),
    );
    const activeParticipants = participantProfiles.map(({ entityId, displayName, role }) => ({
      entityId,
      displayName,
      role,
    }));
    const relationalSlots = Array.from({ length: 12 }, (_, index) =>
      relationalSlot(entities[index % entities.length]!, freeText),
    );
    const livedEntries = [
      ...Array.from({ length: 10 }, (_, index) =>
        livedEntry({
          id: `high_water_decision_${index}`,
          kind: "self_decision_introspection",
          occurredAt: NOW_MS - index,
          text: freeText,
          outcomeReference: `high_water_outcome_${index}`,
        }),
      ),
      ...Array.from({ length: 10 }, (_, index) =>
        livedEntry({
          id: `high_water_activity_${index}`,
          kind: "cross_session_activity",
          occurredAt: NOW_MS - index,
          text: freeText,
        }),
      ),
    ];
    const socialEntries = Array.from({ length: 8 }, (_, index) =>
      livedEntry({
        id: `high_water_social_${index}`,
        kind: "observed_social_event",
        occurredAt: NOW_MS - index,
        text: freeText,
      }),
    );
    const baseLedger = evidenceLedger(livedEntries);
    const ledger: EvidenceLedger = {
      ...baseLedger,
      audienceStanding: {
        ...baseLedger.audienceStanding!,
        observedEventIntrospectionEntries: socialEntries,
      },
    };
    const values: DeliberationContext["selfSnapshot"]["values"] = Array.from({ length: 4 }, () => ({
      id: createValueId(),
      label: freeText,
      description: freeText,
      priority: 5,
      created_at: NOW_MS,
      last_affirmed: NOW_MS,
      state: "established",
      established_at: NOW_MS,
      confidence: 0.9,
      last_tested_at: null,
      last_contradicted_at: null,
      support_count: 1,
      contradiction_count: 0,
      evidence_episode_ids: [],
      provenance: { kind: "manual" },
    }));
    const traits: DeliberationContext["selfSnapshot"]["traits"] = Array.from({ length: 4 }, () => ({
      id: createTraitId(),
      label: freeText,
      strength: 0.8,
      last_reinforced: NOW_MS,
      last_decayed: null,
      state: "established",
      established_at: NOW_MS,
      confidence: 0.9,
      last_tested_at: null,
      last_contradicted_at: null,
      support_count: 1,
      contradiction_count: 0,
      evidence_episode_ids: [],
      provenance: { kind: "manual" },
    }));
    const planner = build(
      context({
        selfSnapshot: { values, goals, traits: traits },
        executiveFocus,
        applicableCommitments: Array.from({ length: 12 }, () => commitment(freeText)),
        activeParticipants,
        participantProfiles,
        relationalSlots,
        evidenceLedger: ledger,
      }),
    );

    expect(planner.traceSummary.sections.goal_index?.estimatedTokens).toBeLessThanOrEqual(
      PLANNER_GOAL_TARGET_TOKENS,
    );
    expect(planner.traceSummary.sections.commitments?.estimatedTokens).toBeLessThanOrEqual(8_000);
    expect(planner.traceSummary.sections.lived_experience?.estimatedTokens).toBeLessThanOrEqual(
      4_000,
    );
    expect(planner.traceSummary.totalEstimatedTokens).toBeGreaterThanOrEqual(20_000);
    expect(planner.traceSummary.totalEstimatedTokens).toBeLessThanOrEqual(
      COMPACT_PLANNER_TARGET_TOKENS,
    );
    expect(planner.traceSummary.overallOverflow).toBe(false);
    expect(planner.traceSummary.truncationCount).toBeGreaterThan(0);
  });

  it("keeps the complete large authority index and reports total overflow", () => {
    const freeText = `LIVE_HEAD_${"x".repeat(2_000)}_LIVE_TAIL`;
    const goals = Array.from({ length: 109 }, (_, index) =>
      goal(`${index}:${freeText}`, {
        priority: 109 - index,
        terminal_condition: freeText,
        progress_notes: freeText,
      }),
    );
    const executiveFocus = {
      selected_goal: goals[0]!,
      selected_score: null,
      candidates: goals.map((candidate, index) => ({
        goal_id: candidate.id,
        goal: candidate,
        score: 1_000 - index,
        components: { priority: 1, deadline_pressure: 1, context_fit: 1, progress_debt: 1 },
        reason: freeText,
      })),
      threshold: 0,
      score_basis: {
        score_context: "turn_selection" as const,
        deadline_lookahead_ms: 1,
        progress_debt_stale_ms: 1,
      },
    };
    const planner = build(
      context({
        selfSnapshot: { values: [], goals, traits: [] },
        executiveFocus,
        applicableCommitments: Array.from({ length: 126 }, (_, index) =>
          commitment(`${index}:${freeText}`),
        ),
        creatorDirectiveBriefing: {
          directives: Array.from({ length: 100 }, (_, index) =>
            creatorDirective(index, `${index}:${freeText}`),
          ),
        },
      }),
    );

    expect(planner.traceSummary.sections.goal_index?.rowCount).toBe(113);
    expect(planner.traceSummary.sections.commitments?.rowCount).toBeLessThan(126);
    expect(
      (planner.traceSummary.sections.commitments?.rowCount ?? 0) +
        (planner.traceSummary.sections.commitments?.omissionCount ?? 0),
    ).toBe(126);
    expect(planner.traceSummary.sections.authority_and_directives).toMatchObject({
      rowCount: 100,
      omissionCount: 0,
    });
    const authorityRows = selfClosingRows(
      taggedBlock(allSystemText(planner), "borg_planner_authority_context"),
      "d",
    );
    expect(authorityRows).toHaveLength(100);
    expect(Math.max(...authorityRows.map((row) => row.length))).toBeLessThanOrEqual(250);
    const goalBlock = taggedBlock(allSystemText(planner), "borg_planner_goal_digest");
    expect(goalBlock).toContain('complete_membership="true"');
    expect(goalBlock).toContain("<omitted_count>0</omitted_count>");
    expect(planner.traceSummary.sections.goal_index?.estimatedTokens).toBeLessThanOrEqual(
      PLANNER_GOAL_TARGET_TOKENS,
    );
    expect(planner.traceSummary.sections.commitments?.estimatedTokens).toBeLessThanOrEqual(8_000);
    expect(planner.traceSummary.sections.authority_and_directives?.estimatedTokens).toBeGreaterThan(
      4_000,
    );
    expect(planner.traceSummary.totalEstimatedTokens).toBeGreaterThan(
      COMPACT_PLANNER_TARGET_TOKENS,
    );
    expect(planner.traceSummary.overallOverflow).toBe(true);
  });

  it("keeps the generic excerpt shape mechanical and announces every cut", () => {
    const source = `HEAD_${"x".repeat(500)}_TAIL`;
    const excerpt = headTailPlannerExcerpt(source, 120);

    expect(excerpt.truncated).toBe(true);
    expect(excerpt.text).toContain("HEAD_");
    expect(excerpt.text).toContain("_TAIL");
    expect(excerpt.text).toContain("HEAD+TAIL EXCERPT");
    expect(excerpt.text).toContain(`total=${source.length}`);
    expect(excerpt.elidedChars).toBeGreaterThan(0);
  });

  it("renders a zero-length source as an exact zero-character excerpt", () => {
    expect(headTailPlannerExcerpt("", 120)).toEqual({
      text: "",
      truncated: false,
      renderedChars: 0,
      totalChars: 0,
      elidedChars: 0,
    });
  });

  it("keeps astral characters whole across both planner head and tail cuts", () => {
    const source = `a${"😀".repeat(500)}b`;
    const excerpt = headTailPlannerExcerpt(source, 120);
    const hasLoneSurrogate = Array.from(excerpt.text).some((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return codePoint >= 0xd800 && codePoint <= 0xdfff;
    });

    expect(excerpt.text).toMatch(/^a😀/u);
    expect(excerpt.text).toMatch(/😀b$/u);
    expect(hasLoneSurrogate).toBe(false);
  });
});
