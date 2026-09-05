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
  CROSS_SESSION_ENTRIES_DRAW_SCOPE,
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
  const turnOrigin = inputContext.turnOrigin ?? "user";
  return buildCompactFinalizerSystemPrompt({
    context: inputContext,
    baseSystemPromptOptions: {
      retrievalContextBudget: 10_000,
      semanticContextBudget: 10_000,
      nowMs: NOW_MS,
    },
    staticHead: "STATIC FINALIZER PROTOCOL",
    toolAvailability: {
      turnOrigin,
      participationPolicy: inputContext.participationPolicy ?? "active",
      enabledTerminalEmissions:
        turnOrigin === "autonomous"
          ? ["EmitAnswer", "EmitObserve", "EmitNoOutput", "EmitSelfReport", "EmitContinueThought"]
          : ["EmitAnswer", "EmitObserve", "EmitNoOutput", "EmitSelfReport"],
      outboundPostAvailable:
        inputContext.autonomousFinalizerToolMenu?.some(
          (item) => item.name === OUTBOUND_POST_TOOL_NAME,
        ) ?? false,
    },
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
      '<borg_finalizer_tool_availability turn_origin="autonomous" participation_policy="active" outbound_post="available"',
    );
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
    expect(withoutAction.system[3]?.text).toContain('outbound_post="unavailable"');
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
    // The contract used to say a complete index reports complete="true", which made an element
    // named for completeness read as one that had reported it. A name cannot carry the scope such
    // a claim is true over, so the contract now says a name is never the claim.
    expect(result.system[0]?.text).toContain(
      'A completeness claim rides on a complete="true" attribute beside omitted_count="0"',
    );
    expect(result.system[0]?.text).toContain(
      "An element name is a label and never a claim of coverage, whatever word it contains.",
    );
    expect(result.system[0]?.text).not.toContain("A complete index reports");
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

  it("spends the advisory excerpt budget on the marker as well as the directive text", () => {
    // The budget covers the whole excerpt, marker included, so the per-row ceiling on
    // directive_included_chars sits below it and shifts with the digit widths of the three
    // numbers the marker carries. Asserted as an identity against whatever the marker
    // currently costs rather than against a copied constant.
    const includedByLength = [4_028, 40_028, 400_028].map((totalChars) => {
      const filler = totalChars - "ADVISORY-HEAD-".length - "-ADVISORY-TAIL".length;
      const rendered = text(
        build(
          context({
            applicableCommitments: [
              commitment(`ADVISORY-HEAD-${"x".repeat(filler)}-ADVISORY-TAIL`, {
                enforcement_class: "advisory",
                critical_domain: null,
              }),
            ],
          }),
        ),
      );
      const budget = Number(/advisory_excerpt_budget_chars="(\d+)"/.exec(rendered)?.[1]);
      const included = Number(/directive_included_chars="(\d+)"/.exec(rendered)?.[1]);
      const marker = / \[ELIDED \d+ CHARS; HEAD\+TAIL EXCERPT; rendered=\d+\/total=\d+\] /.exec(
        rendered,
      )?.[0];
      expect(marker).toBeDefined();
      expect(rendered).toContain(`directive_total_chars="${totalChars}"`);
      expect(included + marker!.length).toBe(budget);
      expect(included).toBeLessThan(budget);
      return included;
    });
    expect(new Set(includedByLength).size).toBeGreaterThan(1);

    // A cut that would land mid-character is pulled back, so the identity is an upper
    // bound rather than an equality -- which is why the legend says "at most".
    const astral = text(
      build(
        context({
          applicableCommitments: [
            commitment("\u{1f642}".repeat(1_000), {
              enforcement_class: "advisory",
              critical_domain: null,
            }),
          ],
        }),
      ),
    );
    const astralBudget = Number(/advisory_excerpt_budget_chars="(\d+)"/.exec(astral)?.[1]);
    const astralIncluded = Number(/directive_included_chars="(\d+)"/.exec(astral)?.[1]);
    const astralMarker = / \[ELIDED \d+ CHARS; HEAD\+TAIL EXCERPT; rendered=\d+\/total=\d+\] /.exec(
      astral,
    )?.[0];
    expect(astralMarker).toBeDefined();
    expect(astralIncluded + astralMarker!.length).toBeLessThan(astralBudget);
  });

  it("uses structural directive kinds for exact versus visibly excerpted payloads", () => {
    const creatorId = createEntityId();
    const allowedId = createEntityId();
    const excludedId = createEntityId();
    const exactOperation = `PRIVATE-OP-${"o".repeat(800)}-END`;
    const exactSlottedOperation = `SLOTTED-OP-${"s".repeat(800)}-END`;
    const fact = `FACT-HEAD-${"f".repeat(1_400)}-FACT-TAIL`;
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
    const factExcerpt = headTailPlannerExcerpt(fact, 1_200);
    const directiveIndex =
      rendered.match(/<creator_directive_index[\s\S]*?<\/creator_directive_index>/)?.[0] ?? "";

    expect(directiveIndex).toContain('rows_total_for_current_audience="4"');
    expect(directiveIndex).toContain('rows_omitted_after_current_audience_scope="0"');
    expect(directiveIndex.match(/<creator_directive id_alias=/g)).toHaveLength(4);
    expect(rendered).toContain(`payload="${exactOperation}"`);
    expect(rendered).toContain(`payload="${exactSlottedOperation}"`);
    expect(rendered).toContain('payload_kind="operational_directive" payload_status="exact"');
    expect(rendered).toContain('mode="boundary" kind="boundary"');
    expect(rendered).toContain(`directive_id="${boundaryDirectiveId}"`);
    expect(rendered).toContain('payload_kind="boundary_prompt" payload_status="exact"');
    expect(rendered).toContain(`directive_id="${factDirectiveId}"`);
    expect(rendered).toContain('payload_status="head+tail_excerpt"');
    expect(rendered).toContain(`payload_included_chars="${factExcerpt.renderedChars}"`);
    expect(rendered).toContain(`payload_total_chars="${fact.length}"`);
    expect(rendered).toContain(
      `HEAD+TAIL EXCERPT; rendered=${factExcerpt.renderedChars}/total=${fact.length}`,
    );
    expect(rendered).toContain('scope_status="exact"');
    expect(rendered).toContain('content_scope="allow_list"');
    expect(rendered).toContain(`allowed_entity_ids="${allowedId}"`);
    expect(rendered).toContain(`excluded_entity_ids="${excludedId}"`);
    expect(rendered).toContain('mention_policy="never_mention"');
    expect(rendered).toContain('activation_scope="allow_list"');
  });

  it("keeps the empty directive index count explicitly audience-relative", () => {
    const rendered = text(build(context({ creatorDirectiveBriefing: null })));

    expect(rendered).toContain(
      '<creator_directive_index status="none" complete_for_current_audience="true" rows_total_for_current_audience="0" rows_omitted_after_current_audience_scope="0" />',
    );
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
    const counterparty = createEntityId();
    const goals = [
      goal("global", null),
      { ...goal("alice", alice), counterparty_entity_id: counterparty },
      goal("bob", bob),
    ];
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
    expect(text(aliceSurface)).toContain(`cp="${counterparty}"`);
    expect(text(aliceSurface)).toContain(
      "cp is the participant the responsibility runs toward, not an owner or audience",
    );
    expect(aliceSurface.system[1]?.text).toBe(bobSurface.system[1]?.text);
  });

  it("keeps mutable exact stamps in overlays and derives rather than printing relative ages", () => {
    const scheduledExpiry = NOW_MS + 6 * 60 * 60_000;
    const commitments = [commitment("one", { expires_at: scheduledExpiry }), commitment("two")];
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
    const result = build(
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
    );
    const durable = result.system[1]!.text;
    const overlay = result.system[3]!.text;
    const commitmentBlock = durable.match(
      /<borg_terminal_commitments[\s\S]*?<\/borg_terminal_commitments>/,
    )?.[0];
    expect(commitmentBlock).toBeDefined();
    expect(commitmentBlock).not.toContain("updated_at=");
    expect(commitmentBlock).not.toContain("expires_at=");
    expect(commitmentBlock).not.toContain("expired_at=");
    expect(commitmentBlock).not.toContain("revoked_at=");

    const scheduledOverlay = overlay.match(
      new RegExp(`<commitment_age id="${commitments[0]!.id}"[^>]*\\/>`),
    )?.[0];
    const unscheduledOverlay = overlay.match(
      new RegExp(`<commitment_age id="${commitments[1]!.id}"[^>]*\\/>`),
    )?.[0];
    expect(scheduledOverlay).toContain(
      `updated_at="${new Date(commitments[0]!.updated_at!).toISOString()}"`,
    );
    expect(scheduledOverlay).toContain(`expires_at="${new Date(scheduledExpiry).toISOString()}"`);
    expect(unscheduledOverlay).not.toContain("expires_at=");

    const ledgerOnlyOverlay = overlay.match(
      new RegExp(`<commitment_age id="${ledgerOnlyCommitment.id}"[^>]*\\/>`),
    )?.[0];
    const valueOverlay = overlay.match(new RegExp(`<value_age id="${valueId}"[^>]*\\/>`))?.[0];
    const traitOverlay = overlay.match(new RegExp(`<trait_age id="${traitId}"[^>]*\\/>`))?.[0];
    expect(ledgerOnlyOverlay).toBeDefined();
    expect(valueOverlay).toContain(`last_affirmed_at="${new Date(NOW_MS - 60_000).toISOString()}"`);
    expect(traitOverlay).toContain(
      `last_reinforced_at="${new Date(NOW_MS - 2 * 60_000).toISOString()}"`,
    );
    for (const row of [scheduledOverlay, unscheduledOverlay, ledgerOnlyOverlay]) {
      expect(row).not.toContain("ledger_state_metadata=");
      for (const field of ["created", "updated", "reinforced", "expires", "expired", "revoked"]) {
        expect(row).not.toContain(` ${field}="`);
      }
    }
    for (const field of ["created", "affirmed", "established", "tested", "contradicted"]) {
      expect(valueOverlay).not.toContain(` ${field}="`);
    }
    for (const field of ["reinforced", "decayed", "established", "tested", "contradicted"]) {
      expect(traitOverlay).not.toContain(` ${field}="`);
    }
    expect(result.system[0]?.text).toContain(
      "subtracting it from the borg_current_time current_time_ms value",
    );
    expect(text(result).match(/derive its relative age by subtracting it from/g)).toHaveLength(1);
    expect(text(result)).not.toContain("epoch_ms");
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
    const result = build(
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
    );
    const rendered = text(result);
    expect(rendered.match(/<commitment id=/g)).toHaveLength(2);
    expect(rendered).toContain(`ledger_ref="commitment:${canonical.id}"`);
    expect(rendered).toContain('ledger_trust_rank="82"');
    expect(rendered).toContain('id="participant_commitment:ent_fixture:com_fixture"');
    expect(rendered).toContain('directive="participant directive exact"');
    expect(rendered).toContain(
      '<commitment_age id="participant_commitment:ent_fixture:com_fixture"',
    );
    expect(result.system[3]?.text).toContain(
      'commitment_rows_total="2" commitment_canonical_rows="1" commitment_ledger_only_rows="1"',
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
      "updated_at",
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
      "ledger_salience_class",
      "ledger_taint",
      "ledger_value",
      "ledger_text",
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
    for (const field of [
      "ledger_actor",
      "ledger_trust_rank",
      "ledger_salience_class",
      "ledger_taint",
      "ledger_value",
      "ledger_text",
      "persistence_class",
      "stream_index",
      "citation_type",
      "citations",
      "resolved_disclosure",
    ]) {
      expect(durableRow).not.toContain(`${field}=`);
      expect(turnRow).toContain(`${field}=`);
    }
    expect(turnRow).toContain('persistence_class="assistant_self_report"');
    expect(turnRow).toContain('citations="entry:one,entry:two"');
    expect(turnRow).toContain('ledger_text="distinct ledger projection text"');
    expect(turnRow).toContain('ledger_value="distinct_ledger_family"');
    expect(durableRow).not.toContain("ledger_state=");
    expect(turnRow).not.toContain("ledger_state_metadata=");
    expect(turnRow).toContain('made_to_entity_label="Alice"');
  });

  it("marks missing fields on a present commitment ledger projection", () => {
    const canonical = commitment("projection source exact");
    const entry: EvidenceLedgerEntry = {
      id: `commitment:${canonical.id}`,
      source_type: "commitment",
      session_scope: "global",
      actor: "memory",
      trust_rank: 80,
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
    const durableRow = result.system[1]!.text.match(/<commitment id="[^"]+"[^>]*\/>/)?.[0];
    const turnRow = result.system[3]!.text.match(/<commitment_age id="[^"]+"[^>]*\/>/)?.[0];

    expect(durableRow).not.toContain("ledger_value=");
    expect(durableRow).not.toContain("ledger_text=");
    expect(turnRow).toContain('ledger_value="missing"');
    expect(turnRow).toContain('ledger_text="missing"');
    expect(result.system[1]?.text).toContain(
      'a present projection with no value or text prints "missing" explicitly',
    );
  });

  it("keeps blocks 0-2 stable when commitment stamps and ledger projections change", () => {
    const base = commitment("stable directive");
    const matchingLedgerEntry: EvidenceLedgerEntry = {
      id: `commitment:${base.id}`,
      source_type: "commitment",
      session_scope: "global",
      actor: "memory",
      trust_rank: 82,
      text: base.directive,
      value: base.directive_family,
      state: "active",
      salience_class: "borg_current_turn_action",
      taint: "none",
      persistence_class: "assistant_self_report",
      via_retrieval: false,
      stream_index: 17,
      citation_type: "parent_user_message",
      citations: ["entry:first"],
      state_metadata: {
        disclosure_label: {
          disclosure_class: "public",
          origin_audience_entity_ids: [],
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      },
    };
    const render = (
      updatedAt: number,
      expiresAt: number | null,
      ledgerOverrides: Partial<EvidenceLedgerEntry>,
    ) =>
      build(
        context({
          applicableCommitments: [{ ...base, updated_at: updatedAt, expires_at: expiresAt }],
          evidenceLedger: {
            ...ledger(),
            audienceStanding: {
              ...ledger().audienceStanding!,
              commitmentEntries: [{ ...matchingLedgerEntry, ...ledgerOverrides }],
            },
          },
        }),
      );
    const first = render(NOW_MS - 3_000, null, {});
    const second = render(NOW_MS - 1_000, NOW_MS + 60_000, {
      session_scope: "current_session",
      actor: "assistant",
      trust_rank: 30,
      text: "turn-derived alternate projection",
      value: "turn-derived-family",
      state: "contested",
      salience_class: "completed_recent",
      taint: "contested",
      persistence_class: undefined,
      via_retrieval: true,
      stream_index: 29,
      citation_type: "generated_perception_text",
      citations: ["entry:second", "entry:third"],
      state_metadata: undefined,
    });
    const commitmentBlock = first.system[1]!.text.match(
      /<borg_terminal_commitments[\s\S]*?<\/borg_terminal_commitments>/,
    )?.[0];
    const durableRow = first.system[1]!.text.match(/<commitment id="[^"]+"[^>]*\/>/)?.[0];
    const secondTurnRow = second.system[3]!.text.match(/<commitment_age id="[^"]+"[^>]*\/>/)?.[0];

    expect(JSON.stringify(first.system.slice(0, 3))).toBe(
      JSON.stringify(second.system.slice(0, 3)),
    );
    expect(first.system[3]?.text).not.toBe(second.system[3]?.text);
    expect(commitmentBlock).not.toMatch(/\b(rows_total|canonical_rows|ledger_only_rows)=/);
    expect(first.system[3]?.text).toContain(
      '<borg_terminal_relative_age_overlay complete="true" rows_total="1" commitment_rows_total="1" commitment_canonical_rows="1" commitment_ledger_only_rows="0">',
    );
    expect(first.system[0]?.text).toContain(
      "The commitment membership denominator is commitment_rows_total in the turn-local relative-age overlay.",
    );
    expect(commitmentBlock).toContain(
      "Those counts live in turn block 3 rather than this cacheable block 1",
    );
    expect(durableRow).toContain("disclosure=");
    expect(durableRow).toContain(`ledger_ref="commitment:${base.id}"`);
    expect(durableRow).toContain('ledger_source_type="commitment"');
    for (const field of [
      "canonical_record",
      "updated_at",
      "expires_at",
      "expired_at",
      "revoked_at",
      "ledger_state",
      "ledger_value",
      "ledger_text",
      "ledger_actor",
      "ledger_trust_rank",
      "ledger_salience_class",
      "ledger_taint",
      "ledger_scope",
      "persistence_class",
      "via_retrieval",
      "stream_index",
      "citation_type",
      "citations",
      "resolved_disclosure",
    ]) {
      expect(durableRow).not.toContain(`${field}=`);
    }
    expect(second.system[3]?.text).toContain(
      `updated_at="${new Date(NOW_MS - 1_000).toISOString()}"`,
    );
    expect(second.system[3]?.text).toContain(
      `expires_at="${new Date(NOW_MS + 60_000).toISOString()}"`,
    );
    expect(secondTurnRow).toContain('ledger_actor="assistant"');
    expect(secondTurnRow).toContain('ledger_trust_rank="30"');
    expect(secondTurnRow).toContain('ledger_salience_class="completed_recent"');
    expect(secondTurnRow).toContain('ledger_taint="contested"');
    expect(secondTurnRow).toContain('stream_index="29"');
    expect(secondTurnRow).toContain('citation_type="generated_perception_text"');
    expect(secondTurnRow).toContain('citations="entry:second,entry:third"');
    expect(secondTurnRow).toContain('ledger_state="contested"');
    expect(secondTurnRow).toContain('ledger_value="turn-derived-family"');
    expect(secondTurnRow).toContain('ledger_text="turn-derived alternate projection"');
    expect(secondTurnRow).toContain("resolved_disclosure=");
  });

  it("keeps canonical disclosure durable and resolves ledger disclosure fail-closed in block 3", () => {
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
    const durableRow = result.system[1]!.text.match(/<commitment id="[^"]+"[^>]*\/>/)?.[0];
    const turnRow = result.system[3]!.text.match(/<commitment_age id="[^"]+"[^>]*\/>/)?.[0];
    expect(durableRow).not.toContain("disclosure_class=unknown");
    expect(durableRow).not.toContain("resolved_disclosure=");
    expect(turnRow).toContain('resolved_disclosure="');
    expect(turnRow).toContain("disclosure_class=unknown");
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
    // The observed-event and cross-session draws never filter by audience: they are
    // global lists that the current participants rank, so draw_scope must not claim
    // otherwise. With no roster the two relational draws are unfiltered as well.
    for (const tag of ["relational_slots", "relational_standing", "social_standing"]) {
      expect(turn).toContain(`<${tag} complete="true" rows_total="1" draw_scope="global">`);
    }
    // The cross-session draw is unfiltered by audience and filtered by session: it
    // excludes the current session outright, so it may never claim the global token.
    expect(turn).toContain(
      '<cross_session_entries complete="true" rows_total="1" draw_scope="other_sessions_recent_window">',
    );
    expect(turn).not.toContain(
      '<cross_session_entries complete="true" rows_total="1" draw_scope="global">',
    );
    expect(result.traceSummary.sections.standing_memory_indexes?.truncationCount).toBeGreaterThan(
      0,
    );
  });

  it("names the relational draw as participant-scoped only when a roster constrains it", () => {
    const alice = createEntityId();
    const observed: EvidenceLedgerEntry = {
      id: "observed-event",
      source_type: "system_metadata",
      session_scope: "prior_session",
      actor: "memory",
      trust_rank: 70,
      text: "observed payload",
    };
    const turn = build(
      context({
        activeParticipants: [{ entityId: alice, displayName: "Alice", role: "audience" }],
        evidenceLedger: {
          ...ledger(),
          audienceStanding: {
            ...ledger().audienceStanding!,
            observedEventIntrospectionEntries: [observed],
          },
        },
      }),
    ).system[3]!.text;
    expect(turn).toContain(
      '<relational_slots complete="true" rows_total="0" draw_scope="active_participant_subjects">',
    );
    expect(turn).toContain(
      '<relational_standing complete="true" rows_total="0" draw_scope="active_participant_subjects">',
    );
    // A roster constrains the relational lists; it does not constrain these two.
    expect(turn).toContain('<social_standing complete="true" rows_total="1" draw_scope="global">');
    expect(turn).toContain(
      '<cross_session_entries complete="true" rows_total="0" draw_scope="other_sessions_recent_window">',
    );
  });

  it("names the cross-session draw's own predicate instead of claiming it took everything", () => {
    const turn = build(context({ evidenceLedger: ledger() })).system[3]!.text;
    const scope = turn.match(/<cross_session_entries[^>]*draw_scope="([^"]+)"/)?.[1];
    // The lane filters e.session_id <> currentSessionId, so whatever token it carries,
    // it can never be the one this block defines as filtering by nothing.
    expect(scope).not.toBe("global");
    expect(scope).toBe(CROSS_SESSION_ENTRIES_DRAW_SCOPE);
    // Every token the block prints must be defined where the reader is told to read it.
    const interpretation = turn.match(
      /<borg_terminal_standing_memory_indexes[\s\S]*?<interpretation>([\s\S]*?)<\/interpretation>/,
    )?.[1];
    expect(interpretation).toContain(`${CROSS_SESSION_ENTRIES_DRAW_SCOPE} means`);
    // The reading the old label invited -- a quiet stretch means a quiet stretch.
    expect(interpretation).toContain("not evidence that nothing happened in it");
    expect(interpretation).toContain(
      "the current session is absent from that group because it is the transcript",
    );
  });

  it("leaves the cross-session group unbounded by a shared budget and says its days fold", () => {
    // The group is a union of separately-drawn lanes -- events, self-decisions, day
    // rows, period rows -- each capped upstream on its own. The render site maps one
    // for one, so rows_total is their sum and never one limit's output. If this ever
    // starts truncating, the per-kind counts on the page stop being readable at all.
    const mixed: EvidenceLedgerEntry[] = Array.from({ length: 9 }, (_, index) => ({
      id: `lived-${index}`,
      source_type: "system_metadata",
      session_scope: "prior_session",
      actor: "memory",
      trust_rank: 70,
      text: `lived entry ${index}`,
    }));
    const turn = build(context({ evidenceLedger: ledger(mixed) })).system[3]!.text;
    expect(turn).toContain(
      `<cross_session_entries complete="true" rows_total="9" draw_scope="${CROSS_SESSION_ENTRIES_DRAW_SCOPE}">`,
    );
    for (const index of [0, 8]) {
      expect(turn).toContain(`<cross_session_row id="lived-${index}"`);
    }
    const interpretation = turn.match(
      /<borg_terminal_standing_memory_indexes[\s\S]*?<interpretation>([\s\S]*?)<\/interpretation>/,
    )?.[1];
    // Two bounds the definition used to leave the reader to discover from a hole: the
    // lanes share no budget, so one kind's count says nothing about another's; and
    // older days are carried by a day row while their own events are dropped, so a day
    // present only as a day row is compressed rather than quiet.
    expect(interpretation).toContain("no budget shared between them");
    expect(interpretation).toContain("a compressed day and not a quiet one");
    // "No shared budget" invites the reading that each lane carries its own limit.
    // Two of them carry the same one, so their counts matching is a single number
    // seen twice rather than two lanes agreeing -- and a self-decision's stamp is
    // the end of its turn, not the moment its trigger fired, so the obvious join
    // against wake times lands on the following wake instead of failing.
    expect(interpretation).toContain(
      "the event lane and the self-decision lane are handed one configured cap value rather than two",
    );
    expect(interpretation).toContain(
      "stamped when its decision was recorded at the end of its turn, not when its trigger fired",
    );
  });

  it("orders durable self rows by immutable keys instead of mutable ranking", () => {
    const olderValue = {
      id: "val_zzzzzzzzzzzzzzzz" as ReturnType<typeof createValueId>,
      label: "older value",
      description: "created first",
      priority: 1,
      created_at: NOW_MS - 20_000,
      last_affirmed: NOW_MS - 1_000,
      state: "established" as const,
      established_at: NOW_MS - 19_000,
      confidence: 0.9,
      last_tested_at: null,
      last_contradicted_at: null,
      support_count: 1,
      contradiction_count: 0,
      evidence_episode_ids: [],
      provenance: { kind: "manual" as const },
    };
    const newerValue = {
      ...olderValue,
      id: "val_aaaaaaaaaaaaaaaa" as ReturnType<typeof createValueId>,
      label: "newer value",
      description: "created second",
      created_at: NOW_MS - 10_000,
    };
    const firstTrait = {
      id: "trt_aaaaaaaaaaaaaaaa" as ReturnType<typeof createTraitId>,
      label: "first trait",
      strength: 0.1,
      last_reinforced: NOW_MS - 1_000,
      last_decayed: null,
      state: "established" as const,
      established_at: NOW_MS - 20_000,
      confidence: 0.9,
      last_tested_at: null,
      last_contradicted_at: null,
      support_count: 1,
      contradiction_count: 0,
      evidence_episode_ids: [],
      provenance: { kind: "manual" as const },
    };
    const secondTrait = {
      ...firstTrait,
      id: "trt_zzzzzzzzzzzzzzzz" as ReturnType<typeof createTraitId>,
      label: "second trait",
      established_at: NOW_MS - 10_000,
    };
    const first = build(
      context({
        selfSnapshot: {
          goals: [],
          values: [{ ...newerValue, priority: 9 }, olderValue],
          traits: [{ ...secondTrait, strength: 0.9 }, firstTrait],
        },
      }),
    );
    const second = build(
      context({
        selfSnapshot: {
          goals: [],
          values: [{ ...olderValue, priority: 9 }, newerValue],
          traits: [{ ...firstTrait, strength: 0.9 }, secondTrait],
        },
      }),
    );
    const durable = first.system[1]!.text;

    expect(first.system[1]?.text).toBe(second.system[1]?.text);
    expect(first.system[3]?.text).not.toBe(second.system[3]?.text);
    expect(durable.indexOf(olderValue.id)).toBeLessThan(durable.indexOf(newerValue.id));
    expect(durable.indexOf(firstTrait.id)).toBeLessThan(durable.indexOf(secondTrait.id));

    // The coverage claim is checked against counts the stores report by their own
    // statements, so it cannot stay true for a reason that has stopped holding.
    const durableSelf = (text: string) =>
      text.match(/<borg_terminal_values_traits[\s\S]*?<\/borg_terminal_values_traits>/)?.[0] ?? "";
    const rendered = {
      goals: [],
      values: [olderValue, newerValue],
      traits: [firstTrait, secondTrait],
    };

    const unmeasured = durableSelf(build(context({ selfSnapshot: rendered })).system[1]!.text);
    expect(unmeasured).toContain('complete="unmeasured"');
    expect(unmeasured).not.toContain("<omitted_count>");

    const agreeing = durableSelf(
      build(context({ selfSnapshot: { ...rendered, valuesStoredTotal: 2, traitsStoredTotal: 2 } }))
        .system[1]!.text,
    );
    expect(agreeing).toContain('complete="true"');
    expect(agreeing).toContain("<omitted_count>0</omitted_count>");

    const narrowed = durableSelf(
      build(context({ selfSnapshot: { ...rendered, valuesStoredTotal: 5, traitsStoredTotal: 3 } }))
        .system[1]!.text,
    );
    expect(narrowed).toContain('complete="false"');
    expect(narrowed).toContain("<omitted_count>4</omitted_count>");
    expect(narrowed).toContain('rows_total="4"');
  });

  it("keeps mutable self state and ledger scope out of the one-hour global block", () => {
    const valueId = createValueId();
    const traitId = createTraitId();
    const canonical = commitment("stable exact");
    const makeContext = (
      confidence: number,
      scope: EvidenceLedgerEntry["session_scope"],
      persistenceClass: EvidenceLedgerEntry["persistence_class"],
    ) => {
      const commitmentEntry: EvidenceLedgerEntry = {
        id: `commitment:${canonical.id}`,
        source_type: "commitment",
        session_scope: scope,
        actor: "memory",
        trust_rank: 80,
        text: canonical.directive,
        ...(persistenceClass === undefined ? {} : { persistence_class: persistenceClass }),
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
    const first = build(makeContext(0.9, "global", undefined));
    const second = build(makeContext(0.2, "current_session", "assistant_self_report"));
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
    expect(first.system[1]?.text).not.toContain("persistence_class=");
    expect(first.system[3]?.text).toContain('ledger_scope="global"');
    expect(first.system[3]?.text).toContain('persistence_class="unknown"');
    expect(second.system[3]?.text).toContain('persistence_class="assistant_self_report"');
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
      toolAvailability: {
        turnOrigin: "user",
        participationPolicy: "active",
        enabledTerminalEmissions: ["EmitAnswer", "EmitObserve", "EmitNoOutput", "EmitSelfReport"],
        outboundPostAvailable: false,
      },
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
