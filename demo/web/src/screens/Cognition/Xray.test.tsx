import { fireEvent, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import type {
  CommitmentsResponse,
  EvidenceLedger,
  IdentityResponse,
  PromptAssembledResponse,
  SharedStateResponse,
  TurnHistoryRow,
} from "../../api/types";
import type { ApiHookState } from "../../hooks/use-api";
import { initialPhases } from "../../hooks/use-turn-stream";
import { renderWithInspector } from "../../test/inspector";
import { Xray, type XrayTabId } from "./Xray";

type ApiDataState<T> = Pick<ApiHookState<T>, "data" | "loading" | "error">;

function apiState<T>(data: T): ApiDataState<T> {
  return { data, loading: false, error: null };
}

function loadingState<T>(): ApiDataState<T> {
  return { data: null, loading: true, error: null };
}

function errorState<T>(message: string): ApiDataState<T> {
  return { data: null, loading: false, error: new Error(message) };
}

function baseLedger(): EvidenceLedger {
  return {
    sections: [
      {
        id: "episodes",
        label: "episodes",
        entries: [
          {
            id: "episode:ep_recalled111111",
            source_type: "episode",
            session_scope: "prior_session",
            actor: "memory",
            trust_rank: 1,
            text: "retrieved memory kept",
            via_retrieval: true,
          },
          {
            id: "current:transcript",
            source_type: "current_session_stream",
            session_scope: "current_session",
            actor: "user",
            trust_rank: 1,
            text: "transcript-only hidden from memory tab",
          },
        ],
      },
    ],
    sharedState: null,
    transcriptIncluded: true,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 1,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 1,
    estimatedTokens: 42,
  };
}

function identity(openQuestion = true): IdentityResponse {
  return {
    values: [],
    goals: [],
    traits: [],
    open_questions: openQuestion
      ? [
          {
            id: "oq_workbench111111",
            question: "What should Borg clarify next?",
            urgency: 6,
            status: "open",
            goal_id: "goal_workbench111111",
            source: "test",
            created_at: 1,
            last_touched: 2,
            resolved_at: null,
            abandoned_at: null,
            abandoned_reason: null,
            resolution_note: null,
            unresolved_rumination_ticks: 0,
            last_ruminated_at: null,
          },
        ]
      : [],
    growth_markers: [],
    periods: [],
    open_question_events: [],
  };
}

function renderXray(
  overrides: Partial<{
    ledger: EvidenceLedger | undefined;
    sharedStateApi: ApiDataState<SharedStateResponse>;
    commitmentsApi: ApiDataState<CommitmentsResponse>;
    identityApi: ApiDataState<IdentityResponse>;
    promptApi: ApiDataState<PromptAssembledResponse>;
    activeTurnId: string | null;
    replayTurn: TurnHistoryRow | null;
  }> = {},
) {
  function XrayHarness() {
    const [activeTab, setActiveTab] = useState<XrayTabId>("flow");

    return (
      <Xray
        phases={initialPhases()}
        activeTurnId={"activeTurnId" in overrides ? (overrides.activeTurnId ?? null) : "turn_xray"}
        tokenTextByPhase={new Map()}
        detailByPhase={new Map()}
        terminalOutcome={null}
        delibPath={null}
        finalAttempt={1}
        cachedLedger={overrides.ledger ?? baseLedger()}
        audience="alice"
        replayTurn={overrides.replayTurn ?? null}
        sharedStateApi={
          overrides.sharedStateApi ??
          apiState({
            audience: "alice",
            entries: [
              {
                id: "dart_workbench111111",
                audience_entity_id: "ent_alice111111",
                state_key: "demo",
                kind: "live",
                text: "Borg saw shared state text.",
                owner_entity_id: null,
                provenance_stream_entry_ids: ["strm_shared111111"],
                last_updated_stream_entry_ids: ["strm_shared111111"],
                created_at: 1,
                last_updated_at: 2,
                last_updated_turn_global: 1,
                superseded_by_id: null,
                rank: 1,
                canonicalizes: {
                  goal_ids: ["goal_shared111111"],
                  commitment_ids: ["cmt_shared111111"],
                  action_ids: [],
                  open_question_ids: ["oq_shared111111"],
                },
              },
            ],
          })
        }
        commitmentsApi={
          overrides.commitmentsApi ??
          apiState({
            commitments: [
              {
                id: "cmt_workbench111111",
                text: "Keep the commitment visible.",
                type: "rule",
                kind: "assistant_commitment",
                enforcement_class: "critical",
                critical_domain: null,
                state: "active",
                priority: 5,
                directive_family: "demo",
                audience: null,
                made_to: null,
                about: null,
                committed_by: null,
                source: "test",
                source_stream_entry_ids: ["strm_commit111111"],
                created_at: 1,
                expires_at: null,
                expired_at: null,
                revoked_at: null,
                revoked_reason: null,
                superseded_by_id: null,
                canonicalized_by_artifact_entry_id: "dart_commit111111",
                last_reinforced_at: 2,
              },
            ],
          })
        }
        identityApi={overrides.identityApi ?? apiState(identity())}
        promptApi={
          overrides.promptApi ??
          apiState({
            sections: ["identity", "working memory"],
            text: "assembled prompt text",
            segments: [
              {
                id: "identity",
                label: "identity",
                editable_key: null,
                start: 0,
                end: 8,
              },
            ],
          })
        }
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
    );
  }

  renderWithInspector(<XrayHarness />, { inspector: true });
}

describe("Xray workbench tabs", () => {
  it("renders loaded shared state, commitments, open questions, prompt, and raw tabs", () => {
    renderXray();

    fireEvent.click(screen.getByRole("tab", { name: "shared state" }));
    expect(screen.getByText("Borg saw shared state text.")).toBeInTheDocument();
    expect(screen.getByText("live")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "commitments" }));
    expect(screen.getByText("Keep the commitment visible.")).toBeInTheDocument();
    expect(screen.getByText("critical")).toBeInTheDocument();
    expect(screen.getByText("global")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "open questions" }));
    expect(screen.getByText("What should Borg clarify next?")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "jump to goal_workbench111111" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "prompt" }));
    expect(screen.getByText("identity")).toBeInTheDocument();
    expect(screen.getByText("assembled prompt text")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "raw" }));
    expect(screen.getByText("estimatedTokens")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("filters the Memory tab to via_retrieval ledger entries", () => {
    renderXray();

    fireEvent.click(screen.getByRole("tab", { name: "memory" }));

    expect(screen.getByText("retrieved memory kept")).toBeInTheDocument();
    expect(screen.queryByText("transcript-only hidden from memory tab")).not.toBeInTheDocument();
  });

  it("renders loading, error, and empty states for fetched workbench tabs", () => {
    renderXray({
      sharedStateApi: loadingState(),
      commitmentsApi: errorState("commitments failed"),
      identityApi: apiState(identity(false)),
      promptApi: apiState({ sections: [], text: "", segments: [] }),
    });

    fireEvent.click(screen.getByRole("tab", { name: "shared state" }));
    expect(screen.getByText("loading")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "commitments" }));
    expect(screen.getByText("commitments failed")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "open questions" }));
    expect(screen.getByText("no open questions")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "prompt" }));
    expect(screen.getByText("assembled prompt unavailable")).toBeInTheDocument();
  });

  it("renders the compact flow strip while idle outside replay", () => {
    renderXray({ activeTurnId: null, replayTurn: null });

    expect(screen.getByTestId("flow-compact-strip")).toBeInTheDocument();
    expect(screen.queryByLabelText("cognitive turn flow chart")).not.toBeInTheDocument();
  });

  it("focuses honest X-ray tabs from replay FlowChart nodes", () => {
    renderXray({
      replayTurn: {
        turn_id: "turn_replay",
        started_at: 1,
        audience: "alice",
        outcome: "emitted",
        suppression_reason: null,
      },
    });

    fireEvent.click(screen.getByTestId("phase-retrieval"));
    expect(screen.getByText("retrieved memory kept")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "flow" }));
    fireEvent.click(screen.getByTestId("phase-shared"));
    expect(screen.getByText("Borg saw shared state text.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "flow" }));
    fireEvent.click(screen.getByTestId("phase-ledger"));
    expect(screen.getByText("transcript-only hidden from memory tab")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "flow" }));
    fireEvent.click(screen.getByTestId("phase-delib"));
    expect(screen.getByLabelText("cognitive turn flow chart")).toBeInTheDocument();
  });
});
