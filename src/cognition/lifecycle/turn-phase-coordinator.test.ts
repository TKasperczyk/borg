import { describe, expect, it } from "vitest";

import type {
  DecisionArtifact,
  DecisionArtifactEntry,
  DecisionArtifactEntryKind,
} from "../../memory/decision-artifacts/index.js";
import {
  createDecisionArtifactEntryId,
  createEntityId,
  createStreamEntryId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import { renderEvidenceLedger } from "../evidence-ledger/index.js";
import {
  buildDecisionArtifactLedgerPromptContext,
  shouldSkipDecisionArtifactCompile,
} from "./turn-phase-coordinator.js";

function decisionArtifactEntry(input: {
  audience: DecisionArtifact["audience_entity_id"];
  kind: DecisionArtifactEntryKind;
  source: StreamEntryId;
  index?: number;
}): DecisionArtifactEntry {
  const index = input.index ?? 0;

  return {
    id: createDecisionArtifactEntryId(),
    audience_entity_id: input.audience,
    kind: input.kind,
    text: `${input.kind} decision`,
    owner_entity_id: input.audience,
    provenance_stream_entry_ids: [input.source],
    last_updated_stream_entry_ids: [input.source],
    created_at: 1_000 + index,
    last_updated_at: 1_000 + index,
    superseded_by_id: null,
    rank: index,
    canonicalizes: {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

function decisionArtifact(input: {
  entries?: readonly DecisionArtifactEntry[];
  lastCompiledStreamEntryId?: StreamEntryId | null;
}): DecisionArtifact {
  const source = input.lastCompiledStreamEntryId ?? createStreamEntryId();
  const audience = input.entries?.[0]?.audience_entity_id ?? createEntityId();

  return {
    audience_entity_id: audience,
    record_version: 1,
    created_at: 1_000,
    updated_at: 1_000,
    last_compiled_at: 1_000,
    last_compiled_stream_entry_id: source,
    entries: [...(input.entries ?? [])],
  };
}

function ledgerEntry(input: {
  streamEntryId: StreamEntryId;
  streamIndex: number;
  text: string;
}): EvidenceLedgerEntry {
  return {
    id: `current_session_stream:${input.streamEntryId}`,
    source_type: "current_session_stream",
    session_scope: "current_session",
    actor: "user",
    trust_rank: 95,
    text: input.text,
    taint: "none",
    stream_index: input.streamIndex,
  };
}

function evidenceLedger(entries: readonly EvidenceLedgerEntry[]): EvidenceLedger {
  return {
    transcriptIncluded: true,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
    sections: [
      {
        id: "current_session_transcript",
        label: "2. Current-Session Transcript",
        entries: [...entries],
      },
    ],
  };
}

describe("shouldSkipDecisionArtifactCompile", () => {
  it("skips frame-anomaly turns", () => {
    const skip = shouldSkipDecisionArtifactCompile({
      enabled: true,
      previousArtifact: null,
      perceptionMode: "problem_solving",
      frameAnomaly: {
        status: "ok",
        kind: "frame_assignment_claim",
        confidence: 1,
        rationale: "test",
      },
      closureLoopAssessment: null,
    });

    expect(skip).toMatchObject({
      reason: "frame_anomaly",
      previousActiveEntryCount: 0,
      perceptionMode: "problem_solving",
      frameAnomalyKind: "frame_assignment_claim",
    });
  });

  it("skips idle turns when the previous artifact has no active in-flight decisions", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const previousArtifact = decisionArtifact({
      entries: [decisionArtifactEntry({ audience, source, kind: "locked" })],
      lastCompiledStreamEntryId: source,
    });
    const skip = shouldSkipDecisionArtifactCompile({
      enabled: true,
      previousArtifact,
      perceptionMode: "idle",
      frameAnomaly: null,
      closureLoopAssessment: null,
    });

    expect(skip).toMatchObject({
      reason: "idle_no_active_decisions",
      previousActiveEntryCount: 1,
      perceptionMode: "idle",
    });
  });

  it("does not skip idle turns when a live decision is active", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const previousArtifact = decisionArtifact({
      entries: [decisionArtifactEntry({ audience, source, kind: "live" })],
      lastCompiledStreamEntryId: source,
    });

    expect(
      shouldSkipDecisionArtifactCompile({
        enabled: true,
        previousArtifact,
        perceptionMode: "idle",
        frameAnomaly: null,
        closureLoopAssessment: null,
      }),
    ).toBeNull();
  });
});

describe("buildDecisionArtifactLedgerPromptContext", () => {
  it("renders only delta ledger entries after the previous compile anchor", () => {
    const older = createStreamEntryId();
    const anchor = createStreamEntryId();
    const nextOne = createStreamEntryId();
    const nextTwo = createStreamEntryId();
    const current = createStreamEntryId();
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: older, streamIndex: 0, text: "older transcript" }),
      ledgerEntry({ streamEntryId: anchor, streamIndex: 1, text: "anchor transcript" }),
      ledgerEntry({ streamEntryId: nextOne, streamIndex: 2, text: "new transcript one" }),
      ledgerEntry({ streamEntryId: nextTwo, streamIndex: 3, text: "new transcript two" }),
      ledgerEntry({ streamEntryId: current, streamIndex: 4, text: "current transcript" }),
    ]);
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: decisionArtifact({ lastCompiledStreamEntryId: anchor }),
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 3,
    });

    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("older transcript");
    expect(context.promptVisibleLedger).not.toContain("anchor transcript");
    expect(context.promptVisibleLedger).toContain("new transcript one");
    expect(context.promptVisibleLedger).toContain("new transcript two");
    expect(context.promptVisibleLedger).toContain("current transcript");
  });

  it("falls back to the full ledger when the previous compile anchor is missing", () => {
    const older = createStreamEntryId();
    const current = createStreamEntryId();
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: older, streamIndex: 0, text: "older transcript" }),
      ledgerEntry({ streamEntryId: current, streamIndex: 1, text: "current transcript" }),
    ]);
    const fullPromptVisibleLedger = renderEvidenceLedger(ledger) ?? "";
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: decisionArtifact({ lastCompiledStreamEntryId: createStreamEntryId() }),
      fullPromptVisibleLedger,
      enabled: true,
      minTailPerSection: 3,
    });

    expect(context).toEqual({
      promptVisibleLedger: fullPromptVisibleLedger,
      ledgerMode: "full_fallback",
    });
  });

  it("falls back when the anchor exists only outside the retained current-session window", () => {
    const anchor = createStreamEntryId();
    const retainedOne = createStreamEntryId();
    const retainedTwo = createStreamEntryId();
    const ledger: EvidenceLedger = {
      transcriptIncluded: true,
      transcriptCompacted: true,
      originalTranscriptTokenEstimate: 0,
      compactedTranscriptEntryCount: 2,
      rawPreservedUserTranscriptEntryCount: 0,
      estimatedTokens: 0,
      sections: [
        {
          id: "current_session_transcript",
          label: "2. Current-Session Transcript",
          entries: [
            ledgerEntry({
              streamEntryId: retainedOne,
              streamIndex: 5,
              text: "retained transcript one",
            }),
            ledgerEntry({
              streamEntryId: retainedTwo,
              streamIndex: 6,
              text: "retained transcript two",
            }),
          ],
        },
        {
          id: "retrieved_memory_evidence",
          label: "10. Retrieved Memory Evidence",
          entries: [
            {
              id: "episode:pruned_anchor",
              source_type: "episode",
              session_scope: "current_session",
              actor: "memory",
              trust_rank: 50,
              text: "side metadata for pruned anchor",
              taint: "none",
              stream_index: 1,
              state_metadata: {
                source_stream_ids: [anchor],
              },
            },
          ],
        },
      ],
    };
    const fullPromptVisibleLedger = renderEvidenceLedger(ledger) ?? "";
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: decisionArtifact({ lastCompiledStreamEntryId: anchor }),
      fullPromptVisibleLedger,
      enabled: true,
      minTailPerSection: 1,
    });

    expect(context).toEqual({
      promptVisibleLedger: fullPromptVisibleLedger,
      ledgerMode: "full_fallback",
    });
  });
});
