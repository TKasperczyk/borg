import { describe, expect, it } from "vitest";

import type { CommitmentRecord } from "../../../memory/commitments/index.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  createCreatorDirectiveId,
  createEntityId,
  createGoalId,
  createRelationalSlotId,
  createSessionId,
  createStreamEntryId,
  createTraitId,
  createValueId,
} from "../../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../../evidence-ledger/index.js";
import type { DeliberationContext, SelfSnapshotGoal } from "../types.js";
import {
  buildCompactFinalizerSystemPrompt,
  COMPACT_FINALIZER_VERIFICATION_RETRIEVAL_BLOCK_ID,
} from "./finalizer-context.js";
import { buildFinalizerSystemPrompt } from "../finalizer.js";
import { TRUSTED_GUIDANCE_PREAMBLE } from "../../prompts/base-identity.js";
import { buildCacheableBaseSystemPromptParts } from "./system-prompt.js";
import { headTailPlannerExcerpt } from "./planner-context.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../../tools/internal/outbound-post-name.js";

const NOW_MS = Date.UTC(2026, 7, 14, 12, 0, 0);

function commitment(
  directive: string,
  overrides: Partial<CommitmentRecord> = {},
): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "boundary",
    kind: "boundary",
    enforcement_class: "critical",
    critical_domain: "privacy",
    directive_family: "terminal_fixture",
    closure_pressure_relevance: "neutral",
    directive,
    priority: 10,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    committed_by_entity_id: null,
    provenance: { kind: "manual" },
    source_stream_entry_ids: [createStreamEntryId()],
    created_at: NOW_MS - 4 * 60 * 60_000,
    updated_at: NOW_MS - 3 * 60 * 60_000,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    canonicalized_by_artifact_entry_id: null,
    last_reinforced_at: NOW_MS - 2 * 60 * 60_000,
    ...overrides,
  };
}

function goal(description: string, audienceEntityId: ReturnType<typeof createEntityId> | null) {
  return {
    id: createGoalId(),
    description,
    terminal_condition: `Complete ${description}`,
    priority: 4,
    parent_goal_id: null,
    status: "active",
    progress_notes: "moving",
    last_progress_ts: NOW_MS - 60_000,
    created_at: NOW_MS - 86_400_000,
    target_at: NOW_MS + 86_400_000,
    audience_entity_id: audienceEntityId,
    owner_entity_id: null,
    provenance: { kind: "manual" },
  } satisfies SelfSnapshotGoal;
}

function ledger(lived: EvidenceLedgerEntry[] = []): EvidenceLedger {
  return {
    sections: [],
    audienceStanding: {
      recentLivedExperienceEntries: lived,
      renderRecentLivedExperience: true,
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
    userMessage: "Terminal pass, please.",
    perception: {
      entities: [],
      mode: "reflective",
      affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
      temporalCue: null,
    },
    retrievalResult: [],
    workingMemory: {
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 3,
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
    evidenceLedger: ledger(),
    ...overrides,
  };
}

function build(inputContext: DeliberationContext, path: "system_1" | "system_2" = "system_2") {
  return buildCompactFinalizerSystemPrompt({
    context: inputContext,
    baseSystemPromptOptions: {
      retrievalContextBudget: 10_000,
      semanticContextBudget: 10_000,
      nowMs: NOW_MS,
    },
    staticHead: "STATIC FINALIZER PROTOCOL",
    path,
    additionalPromptSections: [
      {
        blockId: "borg_evidence_ledger",
        text: "<borg_evidence_ledger>FULL BYTE LEDGER</borg_evidence_ledger>",
      },
      { blockId: "borg_s2_plan", text: "<borg_s2_plan>EXACT PLAN</borg_s2_plan>" },
    ],
  });
}

function text(result: ReturnType<typeof build>): string {
  return result.system.map((block) => block.text).join("\n\n");
}

describe("compact terminal finalizer context", () => {
  it("keeps autonomous outbound availability in the 5m turn-context tier", () => {
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

    expect(withAction.system[3]?.cache_control?.ttl).toBe("5m");
    expect(withAction.system[3]?.text).toContain(
      '<borg_directed_outbound_instruction mode="action_available">',
    );
    expect(
      withAction.system
        .slice(0, 3)
        .map((block) => block.text)
        .join("\n"),
    ).not.toContain("borg_directed_outbound_instruction");

    const withoutAction = build(
      context({
        turnOrigin: "autonomous",
        autonomousOutbound: outboundContext,
        autonomousFinalizerToolMenu: [],
      }),
    );
    expect(text(withoutAction)).not.toContain("borg_directed_outbound_instruction");
    expect(withAction.system.slice(0, 3)).toEqual(withoutAction.system.slice(0, 3));
  });

  it("renders the four cache tiers in order with exactly four breakpoints", () => {
    const result = build(context());
    expect(result.system).toHaveLength(4);
    expect(result.system.map((block) => block.cache_control?.ttl)).toEqual([
      "1h",
      "1h",
      "1h",
      "5m",
    ]);
    expect(result.system[0]?.text).toContain("<borg_terminal_pass_contract>");
    expect(result.system[1]?.text).toContain("<borg_terminal_commitments");
    expect(result.system[2]?.text).toContain("<borg_terminal_audience_durable");
    expect(result.system[3]?.text).toContain("<borg_terminal_relative_age_overlay");
    expect(result.traceSummary.blocks.terminal_turn_context.ttl).toBe("5m");
  });

  it("keeps critical directives exact and visibly annotates advisory head-tail cuts", () => {
    const alice = createEntityId();
    const rows = [
      commitment('Keep <all> & "every" line.\nSecond line.', {
        restricted_audience: alice,
      }),
      commitment(`ADVISORY-HEAD-${"x".repeat(900)}-ADVISORY-TAIL`, {
        enforcement_class: "advisory",
        critical_domain: null,
      }),
    ];
    const rendered = text(build(context({ applicableCommitments: rows })));
    const advisoryExcerpt = headTailPlannerExcerpt(rows[1]!.directive, 480);
    expect([...rendered.matchAll(/<commitment id="([^"]+)"/g)].map((match) => match[1])).toEqual(
      rows.map((row) => row.id),
    );
    expect(rendered).toContain('directive_exact="true"');
    expect(rendered).toContain(`origin_audience=${alice}`);
    expect(rendered).toContain(`private-to=${alice}`);
    expect(rendered).toContain("public-to=none");
    expect(rendered).toContain("Keep &lt;all&gt; &amp; &quot;every&quot; line.&#10;Second line.");
    expect(rendered).toContain('directive_exact="false"');
    expect(rendered).toContain('directive_excerpt_shape="head+tail"');
    expect(rendered).toContain(`directive_included_chars="${advisoryExcerpt.renderedChars}"`);
    expect(rendered).toContain('directive_total_chars="928"');
    expect(rendered).toContain(
      `HEAD+TAIL EXCERPT; rendered=${advisoryExcerpt.renderedChars}/total=928`,
    );
    expect(rendered).toContain("ADVISORY-HEAD-");
    expect(rendered).toContain("-ADVISORY-TAIL");
    expect(rendered).toContain("relationship_private");
    expect(rendered).toContain("omitted_count>0</omitted_count>");
  });

  it("uses structural directive kinds for exact versus visibly excerpted payloads", () => {
    const creatorId = createEntityId();
    const allowedId = createEntityId();
    const excludedId = createEntityId();
    const exactOperation = `PRIVATE-OP-${"o".repeat(800)}-END`;
    const exactSlottedOperation = `SLOTTED-OP-${"s".repeat(800)}-END`;
    const fact = `FACT-HEAD-${"f".repeat(900)}-FACT-TAIL`;
    const scope = {
      directiveId: createCreatorDirectiveId(),
      createdByEntityId: creatorId,
      sourceSessionId: DEFAULT_SESSION_ID,
      contentScope: "allow_list" as const,
      allowedEntityIds: [allowedId],
      excludedEntityIds: [excludedId],
      subjectMayKnow: false,
      mentionPolicy: "never_mention" as const,
      deniedAudienceBehavior: "omit" as const,
      activationScope: "allow_list" as const,
      activationAllowedEntityIds: [allowedId],
      activationExcludedEntityIds: [excludedId],
    };
    const factDirectiveId = createCreatorDirectiveId();
    const boundaryDirectiveId = createCreatorDirectiveId();
    const slottedOperationDirectiveId = createCreatorDirectiveId();
    const rendered = text(
      build(
        context({
          creatorDirectiveBriefing: {
            directives: [
              {
                renderMode: "private",
                privateKind: "operation",
                kind: "routing_instruction",
                operationalDirective: exactOperation,
                priority: 10,
                createdAt: NOW_MS,
                scope,
              },
              {
                renderMode: "content",
                kind: "response_policy",
                subjectKind: "entity",
                subjectLabel: "subject",
                semanticSlot: "public_name",
                semanticValue: "fact-like slot value",
                canonicalFact: null,
                operationalDirective: exactSlottedOperation,
                mentionPolicy: "never_mention",
                priority: 6,
                createdAt: NOW_MS,
                scope: { ...scope, directiveId: slottedOperationDirectiveId },
              },
              {
                renderMode: "content",
                kind: "subject_fact",
                subjectKind: "entity",
                subjectLabel: "subject",
                semanticSlot: null,
                semanticValue: null,
                canonicalFact: fact,
                operationalDirective: null,
                mentionPolicy: "never_mention",
                priority: 5,
                createdAt: NOW_MS,
                scope: { ...scope, directiveId: factDirectiveId },
              },
              {
                renderMode: "boundary",
                priority: 4,
                createdAt: NOW_MS,
                scope: { ...scope, directiveId: boundaryDirectiveId },
              },
            ],
          },
        }),
      ),
    );
    const factExcerpt = headTailPlannerExcerpt(fact, 480);

    expect(rendered).toContain(`payload="${exactOperation}"`);
    expect(rendered).toContain(`payload="${exactSlottedOperation}"`);
    expect(rendered).toContain('payload_kind="operational_directive" payload_status="exact"');
    expect(rendered).toContain('mode="boundary" kind="boundary"');
    expect(rendered).toContain(`directive_id="${boundaryDirectiveId}"`);
    expect(rendered).toContain('payload_kind="boundary_prompt" payload_status="exact"');
    expect(rendered).toContain(`directive_id="${factDirectiveId}"`);
    expect(rendered).toContain('payload_status="head+tail_excerpt"');
    expect(rendered).toContain(`payload_included_chars="${factExcerpt.renderedChars}"`);
    expect(rendered).toContain('payload_total_chars="920"');
    expect(rendered).toContain(
      `HEAD+TAIL EXCERPT; rendered=${factExcerpt.renderedChars}/total=920`,
    );
    expect(rendered).toContain('scope_status="exact"');
    expect(rendered).toContain('content_scope="allow_list"');
    expect(rendered).toContain(`allowed_entity_ids="${allowedId}"`);
    expect(rendered).toContain(`excluded_entity_ids="${excludedId}"`);
    expect(rendered).toContain('mention_policy="never_mention"');
    expect(rendered).toContain('activation_scope="allow_list"');
  });

  it("marks historical directive scope fields unknown instead of exact-empty", () => {
    const rendered = text(
      build(
        context({
          creatorDirectiveBriefing: {
            directives: [
              {
                renderMode: "content",
                kind: "subject_fact",
                subjectKind: "borg_self",
                subjectLabel: "Borg",
                semanticSlot: null,
                semanticValue: null,
                canonicalFact: "Historical captured fact",
                operationalDirective: null,
                mentionPolicy: "answer_if_asked",
                priority: 1,
                createdAt: NOW_MS,
              },
            ],
          },
        }),
      ),
    );

    expect(rendered).toContain('scope_status="not_captured"');
    expect(rendered).toContain('allowed_entity_ids="unknown"');
    expect(rendered).toContain('excluded_entity_ids="unknown"');
    expect(rendered).toContain('activation_allowed_entity_ids="unknown"');
    expect(rendered).toContain('activation_excluded_entity_ids="unknown"');
    expect(rendered).toContain('mention_policy="answer_if_asked"');
  });

  it("never changes global commitment or goal index membership with the audience", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const commitments = [
      commitment("global"),
      commitment("alice", { restricted_audience: alice }),
      commitment("bob", { made_to_entity: bob }),
    ];
    const goals = [goal("global", null), goal("alice", alice), goal("bob", bob)];
    const throwingRepository = {
      get: () => {
        throw new Error("compact terminal rendering must not read repositories");
      },
    } as unknown as DeliberationContext["entityRepository"];
    const render = (audienceEntityId: typeof alice) =>
      build(
        context({
          audienceEntityId,
          entityRepository: throwingRepository,
          applicableCommitments: commitments,
          selfSnapshot: { values: [], goals, traits: [] },
        }),
      );
    const memberships = (surface: string, expression: RegExp) =>
      [...surface.matchAll(expression)].map((match) => match[1]).sort();
    const aliceSurface = render(alice);
    const bobSurface = render(bob);
    expect(memberships(text(aliceSurface), /<commitment id="([^"]+)"/g)).toEqual(
      memberships(text(bobSurface), /<commitment id="([^"]+)"/g),
    );
    expect(memberships(text(aliceSurface), /<goal i="([^"]+)"/g)).toEqual(
      memberships(text(bobSurface), /<goal i="([^"]+)"/g),
    );
    expect(aliceSurface.system[1]?.text).toBe(bobSurface.system[1]?.text);
  });

  it("emits turn-local age overlays for every durable record with relative-age fields", () => {
    const commitments = [commitment("one"), commitment("two")];
    const valueId = createValueId();
    const traitId = createTraitId();
    const ledgerOnlyCommitment: EvidenceLedgerEntry = {
      id: "ledger-only-commitment",
      source_type: "commitment",
      session_scope: "prior_session",
      actor: "memory",
      trust_rank: 70,
      text: "ledger-only exact",
      state_metadata: {
        created_at: new Date(NOW_MS - 6_000).toISOString(),
        last_reinforced_at: new Date(NOW_MS - 3_000).toISOString(),
      },
    };
    const rendered = text(
      build(
        context({
          applicableCommitments: commitments,
          evidenceLedger: {
            ...ledger(),
            audienceStanding: {
              ...ledger().audienceStanding!,
              commitmentEntries: [ledgerOnlyCommitment],
            },
          },
          selfSnapshot: {
            goals: [],
            values: [
              {
                id: valueId,
                label: "care",
                description: "care about exact grounding",
                priority: 1,
                created_at: NOW_MS - 5 * 60 * 60_000,
                last_affirmed: NOW_MS - 60_000,
                state: "established",
                established_at: NOW_MS - 4 * 60 * 60_000,
                confidence: 0.9,
                last_tested_at: null,
                last_contradicted_at: null,
                support_count: 1,
                contradiction_count: 0,
                evidence_episode_ids: [],
                provenance: { kind: "manual" },
              },
            ],
            traits: [
              {
                id: traitId,
                label: "careful",
                strength: 0.8,
                last_reinforced: NOW_MS - 2 * 60_000,
                last_decayed: null,
                state: "established",
                established_at: NOW_MS - 4 * 60 * 60_000,
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
      ),
    );
    const expectedOverlayFields = new Map<string, readonly string[]>([
      ...commitments.map(
        (row) =>
          [
            `commitment:${row.id}`,
            ["created", "updated", "reinforced", "expires", "expired", "revoked"],
          ] as const,
      ),
      [`commitment:${ledgerOnlyCommitment.id}`, ["created", "reinforced"]] as const,
      [
        `value:${valueId}`,
        ["created", "affirmed", "established", "tested", "contradicted"],
      ] as const,
      [
        `trait:${traitId}`,
        ["reinforced", "decayed", "established", "tested", "contradicted"],
      ] as const,
    ]);
    const durableRows = [...rendered.matchAll(/<(commitment|value|trait) id="([^"]+)"[^>]*\/>/g)];
    expect(durableRows).toHaveLength(expectedOverlayFields.size);
    for (const match of durableRows) {
      const tag = match[1]!;
      const id = match[2]!;
      const overlay = rendered.match(new RegExp(`<${tag}_age id="${id}"[^>]*\\/>`))?.[0];
      expect(overlay, `turn overlay for ${tag}:${id}`).toBeDefined();
      for (const field of expectedOverlayFields.get(`${tag}:${id}`) ?? []) {
        expect(overlay, `${tag}:${id}.${field}`).toContain(`${field}="`);
      }
    }
    expect(rendered).toContain('created="4h ago"');
  });

  it("folds standing-ledger commitment fields into the single complete index", () => {
    const canonical = commitment("canonical exact");
    const canonicalEntry: EvidenceLedgerEntry = {
      id: `commitment:${canonical.id}`,
      source_type: "commitment",
      session_scope: "global",
      actor: "memory",
      trust_rank: 82,
      text: canonical.directive,
      value: canonical.directive_family,
      state: "active",
      taint: "none",
      state_metadata: {
        commitment_kind: canonical.kind,
        commitment_type: canonical.type,
        commitment_enforcement_class: canonical.enforcement_class,
        created_at: new Date(canonical.created_at).toISOString(),
        last_reinforced_at: new Date(canonical.last_reinforced_at).toISOString(),
      },
    };
    const participantEntry: EvidenceLedgerEntry = {
      ...canonicalEntry,
      id: "participant_commitment:ent_fixture:com_fixture",
      text: "participant directive exact",
      value: "participant_family",
      trust_rank: 79,
    };
    const rendered = text(
      build(
        context({
          applicableCommitments: [canonical],
          evidenceLedger: {
            ...ledger(),
            audienceStanding: {
              ...ledger().audienceStanding!,
              commitmentEntries: [canonicalEntry, participantEntry],
            },
          },
        }),
      ),
    );
    expect(rendered.match(/<commitment id=/g)).toHaveLength(2);
    expect(rendered).toContain(`ledger_ref="commitment:${canonical.id}"`);
    expect(rendered).toContain('ledger_trust_rank="82"');
    expect(rendered).toContain('id="participant_commitment:ent_fixture:com_fixture"');
    expect(rendered).toContain('directive="participant directive exact"');
    expect(rendered).toContain(
      '<commitment_age id="participant_commitment:ent_fixture:com_fixture"',
    );
  });

  it("renders a field-set union of canonical details and standing-ledger commitment rows", () => {
    const alice = createEntityId();
    const canonical = commitment("union exact", {
      made_to_entity: alice,
      restricted_audience: alice,
      about_entity: alice,
      committed_by_entity_id: alice,
    });
    const entry: EvidenceLedgerEntry = {
      id: `commitment:${canonical.id}`,
      source_type: "commitment",
      session_scope: "prior_session",
      actor: "memory",
      trust_rank: 81,
      text: "distinct ledger projection text",
      value: "distinct_ledger_family",
      state: "active",
      taint: "none",
      persistence_class: "assistant_self_report",
      via_retrieval: true,
      stream_index: 17,
      citation_type: "parent_user_message",
      citations: ["entry:one", "entry:two"],
      state_metadata: {
        disclosure_label: {
          disclosure_class: "relationship_private",
          origin_audience_entity_ids: [alice],
          private_to_entity_ids: [alice],
          public_to_entity_ids: [],
        },
      },
    };
    const result = build(
      context({
        applicableCommitments: [canonical],
        commitmentEntityLabels: { [alice]: "Alice" },
        evidenceLedger: {
          ...ledger(),
          audienceStanding: { ...ledger().audienceStanding!, commitmentEntries: [entry] },
        },
      }),
    );
    const durableRow = result.system[1]!.text.match(/<commitment id="[^"]+"[^>]*\/>/)?.[0];
    const turnRow = result.system[3]!.text.match(/<commitment_age id="[^"]+"[^>]*\/>/)?.[0];
    expect(durableRow).toBeDefined();
    expect(turnRow).toBeDefined();
    const attributes = (row: string) =>
      new Set([...row.matchAll(/\s([a-z_]+)=/g)].map((match) => match[1]));
    const durableFields = attributes(durableRow!);
    const turnFields = attributes(turnRow!);
    const unionFields = new Set([...durableFields, ...turnFields]);
    const legacyCanonicalSemanticFields = [
      "id",
      "ordinal",
      "directive",
      "family",
      "disclosure",
      "kind",
      "type",
      "enforcement_class",
      "critical_domain",
      "created_at",
      "made_to_entity_id",
      "made_to_entity_label",
      "restricted_audience_id",
      "restricted_audience_label",
      "about_entity_id",
      "about_entity_label",
      "committed_by_entity_id",
      "committed_by_entity_label",
      "provenance",
    ] as const;
    const legacyStandingLedgerSemanticFields = [
      "id",
      "status",
      "family",
      "ledger_ref",
      "ledger_source_type",
      "ledger_scope",
      "ledger_actor",
      "ledger_trust_rank",
      "ledger_state",
      "ledger_salience_class",
      "ledger_taint",
      "ledger_value",
      "ledger_text",
      "ledger_state_metadata",
      "persistence_class",
      "via_retrieval",
      "stream_index",
      "citation_type",
      "citations",
      "directive",
      "disclosure",
    ] as const;
    for (const field of [...legacyCanonicalSemanticFields, ...legacyStandingLedgerSemanticFields]) {
      expect(unionFields, `commitment union field ${field}`).toContain(field);
    }
    expect(durableRow).toContain('persistence_class="assistant_self_report"');
    expect(durableRow).toContain('citations="entry:one,entry:two"');
    expect(durableRow).toContain('ledger_text="distinct ledger projection text"');
    expect(durableRow).toContain('ledger_value="distinct_ledger_family"');
    expect(turnRow).toContain('ledger_state_metadata="{&quot;disclosure_label&quot;:');
    expect(turnRow).toContain('made_to_entity_label="Alice"');
  });

  it("combines canonical and ledger disclosure labels fail-closed", () => {
    const alice = createEntityId();
    const canonical = commitment("private", { restricted_audience: alice });
    const entry: EvidenceLedgerEntry = {
      id: `commitment:${canonical.id}`,
      source_type: "commitment",
      session_scope: "global",
      actor: "memory",
      trust_rank: 80,
      text: canonical.directive,
      state: "active",
      // Missing ledger disclosure metadata must contribute `unknown`.
    };
    const result = build(
      context({
        applicableCommitments: [canonical],
        evidenceLedger: {
          ...ledger(),
          audienceStanding: { ...ledger().audienceStanding!, commitmentEntries: [entry] },
        },
      }),
    );
    expect(result.system[1]?.text).toContain("disclosure_class=unknown");
  });

  it("keeps complete relational, social, observed-event, and cross-session membership indexes", () => {
    const alice = createEntityId();
    const disclosure = {
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
        public_to_entity_ids: [],
      },
    };
    const standingEntry = (
      id: string,
      source_type: EvidenceLedgerEntry["source_type"],
    ): EvidenceLedgerEntry => ({
      id,
      source_type,
      session_scope: "prior_session",
      actor: "memory",
      trust_rank: 70,
      text: `${id} payload`,
      state_metadata: disclosure,
    });
    const relational = standingEntry("relational-ledger", "relational_slot");
    const observed = standingEntry("observed-event", "system_metadata");
    observed.text = `HEAD-${"x".repeat(1_000)}-TAIL`;
    const crossSession = standingEntry("cross-session", "assistant_stream");
    const relationalSlotId = createRelationalSlotId();
    const result = build(
      context({
        relationalSlots: [
          {
            id: relationalSlotId,
            subject_entity_id: alice,
            slot_key: "relationship",
            value: "trusted collaborator",
            state: "established",
            evidence_stream_entry_ids: [createStreamEntryId()],
            contradicted_by_stream_entry_ids: [],
            alternate_values: [],
            created_at: NOW_MS - 5_000,
            updated_at: NOW_MS - 1_000,
          },
        ],
        evidenceLedger: {
          ...ledger([crossSession]),
          audienceStanding: {
            ...ledger([crossSession]).audienceStanding!,
            relationalEntries: [relational],
            observedEventIntrospectionEntries: [observed],
          },
        },
      }),
    );
    const turn = result.system[3]!.text;
    expect(turn).toContain(`<relational_slot_row id="${relationalSlotId}"`);
    expect(turn).toContain('<relational_standing_row id="relational-ledger"');
    expect(turn).toContain('<social_standing_row id="observed-event"');
    expect(turn).toContain('<cross_session_row id="cross-session"');
    expect(turn.match(/<omitted_count>0<\/omitted_count>/g)?.length).toBeGreaterThanOrEqual(5);
    for (const rowTag of [
      "relational_slot_row",
      "relational_standing_row",
      "social_standing_row",
      "cross_session_row",
    ]) {
      expect(turn.match(new RegExp(`<${rowTag}[^>]+disclosure="([^"]+)"`))?.[1]).toContain(
        "disclosure_class=relationship_private",
      );
    }
    expect(turn).toContain("HEAD+TAIL EXCERPT");
    expect(result.traceSummary.sections.standing_memory_indexes?.truncationCount).toBeGreaterThan(
      0,
    );
  });

  it("keeps mutable self state and ledger scope out of the one-hour global block", () => {
    const valueId = createValueId();
    const traitId = createTraitId();
    const canonical = commitment("stable exact");
    const makeContext = (confidence: number, scope: EvidenceLedgerEntry["session_scope"]) => {
      const commitmentEntry: EvidenceLedgerEntry = {
        id: `commitment:${canonical.id}`,
        source_type: "commitment",
        session_scope: scope,
        actor: "memory",
        trust_rank: 80,
        text: canonical.directive,
      };
      return context({
        applicableCommitments: [canonical],
        evidenceLedger: {
          ...ledger(),
          audienceStanding: {
            ...ledger().audienceStanding!,
            commitmentEntries: [commitmentEntry],
          },
        },
        selfSnapshot: {
          goals: [],
          values: [
            {
              id: valueId,
              label: "care",
              description: "stable description",
              priority: confidence,
              created_at: NOW_MS - 10_000,
              last_affirmed: NOW_MS - confidence * 1_000,
              state: confidence > 0.5 ? "established" : "candidate",
              established_at: NOW_MS - 9_000,
              confidence,
              last_tested_at: NOW_MS - confidence * 2_000,
              last_contradicted_at: null,
              support_count: Math.round(confidence * 10),
              contradiction_count: 0,
              evidence_episode_ids: [],
              provenance: { kind: "manual" },
            },
          ],
          traits: [
            {
              id: traitId,
              label: "careful",
              strength: confidence,
              last_reinforced: NOW_MS - confidence * 3_000,
              last_decayed: null,
              state: "established",
              established_at: NOW_MS - 9_000,
              confidence,
              last_tested_at: NOW_MS - confidence * 2_000,
              last_contradicted_at: null,
              support_count: Math.round(confidence * 10),
              contradiction_count: 0,
              evidence_episode_ids: [],
              provenance: { kind: "manual" },
            },
          ],
        },
      });
    };
    const first = build(makeContext(0.9, "global"));
    const second = build(makeContext(0.2, "current_session"));
    expect(first.system[1]?.text).toBe(second.system[1]?.text);
    expect(first.system[3]?.text).not.toBe(second.system[3]?.text);
    expect(first.system[1]?.text).not.toContain("ledger_scope=");
    const durableSelf = first.system[1]?.text.match(
      /<borg_terminal_values_traits[\s\S]*?<\/borg_terminal_values_traits>/,
    )?.[0];
    expect(durableSelf).toBeDefined();
    expect(durableSelf).not.toContain("confidence=");
    expect(durableSelf).not.toContain("support_count=");
    expect(durableSelf).not.toContain("last_reinforced=");
    expect(durableSelf).not.toContain("last_tested_at=");
    expect(first.system[3]?.text).toContain('ledger_scope="global"');
  });

  it("imposes evidence-ledger, secondary-retrieval, then S2-plan order on plan-first input", () => {
    const result = buildCompactFinalizerSystemPrompt({
      context: context(),
      baseSystemPromptOptions: {
        retrievalContextBudget: 10_000,
        semanticContextBudget: 10_000,
        nowMs: NOW_MS,
      },
      staticHead: "STATIC FINALIZER PROTOCOL",
      path: "system_2",
      additionalPromptSections: [
        { blockId: "borg_s2_plan", text: "<borg_s2_plan>PLAN</borg_s2_plan>" },
        {
          blockId: "borg_additional_retrieval",
          text: "<borg_additional_retrieval>SECONDARY</borg_additional_retrieval>",
        },
        {
          blockId: "borg_evidence_ledger",
          text: "<borg_evidence_ledger>LEDGER</borg_evidence_ledger>",
        },
      ],
    });
    const turn = result.system[3]!.text;
    expect(turn.indexOf("<borg_evidence_ledger>")).toBeLessThan(
      turn.indexOf("<borg_additional_retrieval>"),
    );
    expect(turn.indexOf("<borg_additional_retrieval>")).toBeLessThan(
      turn.indexOf("<borg_s2_plan>"),
    );
  });

  it("renders production trusted framing and host capabilities exactly once", () => {
    const inputContext = context();
    const baseOptions = {
      retrievalContextBudget: 10_000,
      semanticContextBudget: 10_000,
      nowMs: NOW_MS,
    };
    const cacheable = buildCacheableBaseSystemPromptParts(inputContext, baseOptions);
    const result = buildFinalizerSystemPrompt({
      llmClient: {} as never,
      dispatcher: {} as never,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: cacheable.dynamicContent,
      cacheableSystemPrompt: cacheable,
      initialMessages: [],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_1",
      finalizerSurfaceVariant: "compact",
      compactSurface: { context: inputContext, baseSystemPromptOptions: baseOptions },
    });
    const rendered = result.system.map((block) => block.text).join("\n\n");
    expect(rendered.split(TRUSTED_GUIDANCE_PREAMBLE)).toHaveLength(2);
    expect(rendered.match(/<borg_host_capabilities>/g)).toHaveLength(1);
  });

  it("routes the conversationally scoped policy only for the structural user origin", () => {
    const render = (turnOrigin: unknown) => {
      const inputContext = context({ turnOrigin: turnOrigin as never });
      const baseOptions = {
        retrievalContextBudget: 10_000,
        semanticContextBudget: 10_000,
        nowMs: NOW_MS,
      };
      const cacheable = buildCacheableBaseSystemPromptParts(inputContext, baseOptions);
      return buildFinalizerSystemPrompt({
        llmClient: {} as never,
        dispatcher: {} as never,
        sessionId: DEFAULT_SESSION_ID,
        model: "fake",
        baseSystemPrompt: cacheable.dynamicContent,
        cacheableSystemPrompt: cacheable,
        initialMessages: [],
        userEntryId: undefined,
        maxTokens: 100,
        path: "system_1",
        finalizerSurfaceVariant: "compact_conversational",
        turnOrigin: turnOrigin as never,
        compactSurface: { context: inputContext, baseSystemPromptOptions: baseOptions },
      });
    };

    expect(render("user").traceSummary?.variant).toBe("compact");
    expect(render("autonomous").traceSummary?.variant).toBe("legacy");
    expect(render("directed_outbound").traceSummary?.variant).toBe("legacy");
    expect(render(undefined).traceSummary?.variant).toBe("legacy");
    expect(render("future_origin").traceSummary?.variant).toBe("legacy");
  });

  it("renders scoped autonomous calls byte-identically to explicit legacy", () => {
    const inputContext = context({ turnOrigin: "autonomous" });
    const baseOptions = {
      retrievalContextBudget: 10_000,
      semanticContextBudget: 10_000,
      nowMs: NOW_MS,
    };
    const cacheable = buildCacheableBaseSystemPromptParts(inputContext, baseOptions);
    const base = {
      llmClient: {} as never,
      dispatcher: {} as never,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: cacheable.dynamicContent,
      cacheableSystemPrompt: cacheable,
      initialMessages: [],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_2" as const,
      turnOrigin: "autonomous" as const,
      compactSurface: { context: inputContext, baseSystemPromptOptions: baseOptions },
    };
    const legacy = buildFinalizerSystemPrompt({ ...base, finalizerSurfaceVariant: "legacy" });
    const scoped = buildFinalizerSystemPrompt({
      ...base,
      finalizerSurfaceVariant: "compact_conversational",
    });

    expect(scoped.system).toEqual(legacy.system);
    expect(JSON.stringify(scoped.system)).toBe(JSON.stringify(legacy.system));
    expect(scoped.traceSummary).toEqual(legacy.traceSummary);
  });

  it("preserves full ledger and exact plan bytes and shares the core across S1/S2", () => {
    const input = context();
    const s1 = build(input, "system_1");
    const s2 = build(input, "system_2");
    expect(text(s1)).toBe(text(s2));
    expect(text(s2)).toContain("<borg_evidence_ledger>FULL BYTE LEDGER</borg_evidence_ledger>");
    expect(text(s2)).toContain("<borg_s2_plan>EXACT PLAN</borg_s2_plan>");
    expect(s1.traceSummary.path).toBe("system_1");
    expect(s2.traceSummary.path).toBe("system_2");
  });

  it("keeps the compact-only verification block out of the legacy byte surface", () => {
    const base = {
      llmClient: {} as never,
      dispatcher: {} as never,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: "legacy dynamic",
      cacheableSystemPrompt: { staticPrefix: "legacy static", dynamicContent: "legacy dynamic" },
      initialMessages: [],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_2" as const,
      finalizerSurfaceVariant: "legacy" as const,
    };
    const baseline = buildFinalizerSystemPrompt(base);
    const withCompactOnlySection = buildFinalizerSystemPrompt({
      ...base,
      additionalPromptSections: [
        {
          blockId: COMPACT_FINALIZER_VERIFICATION_RETRIEVAL_BLOCK_ID,
          text: "COMPACT ONLY",
        },
      ],
    });

    expect(withCompactOnlySection.system).toEqual(baseline.system);
    expect(withCompactOnlySection.traceSummary).toEqual(baseline.traceSummary);
  });

  it("keeps decided outcomes aggregated separately from mere firings", () => {
    const decision = (id: string, occurredAt: number): EvidenceLedgerEntry => ({
      id,
      source_type: "system_metadata",
      session_scope: "global",
      actor: "system",
      trust_rank: 70,
      text: "settled outcome",
      planner_metadata: { decision_outcome_ref: "decision:one", decision_summary: "settled" },
      state_metadata: {
        lived_experience_kind: "self_decision_introspection",
        occurred_at: occurredAt,
      },
    });
    const firing: EvidenceLedgerEntry = {
      id: "density",
      source_type: "system_metadata",
      session_scope: "global",
      actor: "system",
      trust_rank: 70,
      text: "many triggers",
      state_metadata: { lived_experience_kind: "self_decision_density", occurred_at: NOW_MS },
    };
    const rendered = text(
      build(
        context({
          evidenceLedger: ledger([decision("a", NOW_MS - 10), decision("b", NOW_MS), firing]),
        }),
      ),
    );
    expect(rendered).toContain('outcome_ref="decision:one" derivation_count="2"');
    expect(rendered).toContain('category="firing_volume"');
  });

  it("keeps regeneration bytes in an unmarked suffix after all four compact markers", () => {
    const inputContext = context();
    const regeneration =
      "<borg_commitment_regeneration_instruction>EXACT REGEN</borg_commitment_regeneration_instruction>";
    const rendered = buildFinalizerSystemPrompt({
      llmClient: {} as never,
      dispatcher: {} as never,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: "legacy dynamic",
      cacheableSystemPrompt: { staticPrefix: "static", dynamicContent: "legacy dynamic" },
      initialMessages: [],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_2",
      finalizerSurfaceVariant: "compact",
      compactSurface: {
        context: inputContext,
        baseSystemPromptOptions: {
          retrievalContextBudget: 10_000,
          semanticContextBudget: 10_000,
          nowMs: NOW_MS,
        },
      },
      additionalPromptSections: [
        { blockId: "borg_evidence_ledger", text: "ledger" },
        { blockId: "borg_commitment_regeneration_instruction", text: regeneration },
      ],
    });
    expect(rendered.system).toHaveLength(5);
    expect(rendered.system.slice(0, 4).every((block) => block.cache_control !== undefined)).toBe(
      true,
    );
    expect(rendered.system[4]).toEqual({ type: "text", text: `\n\n${regeneration}` });
  });
});
