import type { ReactNode } from "react";

import type {
  CommitmentItem,
  CommitmentsResponse,
  EvidenceLedger,
  EvidenceLedgerEntry,
  IdentityResponse,
  OpenQuestion,
  PromptAssembledResponse,
  SharedStateEntry,
  SharedStateResponse,
  TurnHistoryOutcomeClass,
  TurnHistoryRow,
  TurnTerminalOutcome,
} from "../../api/types";
import { Empty } from "../../components/Empty";
import { ErrorState } from "../../components/ErrorState";
import { IdRef } from "../../components/Inspector/IdRef";
import type { ObjectType } from "../../components/Inspector/inspector-id";
import { JsonValueView } from "../../components/JsonValueView";
import { Loading } from "../../components/Loading";
import { Tag, type TagKind } from "../../components/Tag";
import type { ApiHookState } from "../../hooks/use-api";
import type { PhaseState } from "../../hooks/use-turn-stream";
import { lifecycleLabel, tagKind as lifecycleTagKind } from "../../lib/shared-state-lifecycle";
import { formatTime } from "../../lib/stream-utils";
import { shortId } from "../screen-utils";
import { FlowChart } from "./FlowChart";
import { LedgerView } from "./LedgerView";

export type XrayTabId =
  | "flow"
  | "ledger"
  | "memory"
  | "shared"
  | "commitments"
  | "open_qs"
  | "prompt"
  | "raw";

const XRAY_TABS: readonly { id: XrayTabId; label: string }[] = [
  { id: "flow", label: "flow" },
  { id: "ledger", label: "ledger" },
  { id: "memory", label: "memory" },
  { id: "shared", label: "shared state" },
  { id: "commitments", label: "commitments" },
  { id: "open_qs", label: "open qs" },
  { id: "prompt", label: "prompt" },
  { id: "raw", label: "raw" },
];

type ApiDataState<T> = Pick<ApiHookState<T>, "data" | "loading" | "error">;

export type XrayProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  detailByPhase: Map<string, string[]>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  cachedLedger?: EvidenceLedger;
  audience: string;
  tracePlaceholder?: string | null;
  particleEnabled?: boolean;
  replayTurn?: TurnHistoryRow | null;
  sharedStateApi: ApiDataState<SharedStateResponse>;
  commitmentsApi: ApiDataState<CommitmentsResponse>;
  identityApi: ApiDataState<IdentityResponse>;
  promptApi: ApiDataState<PromptAssembledResponse>;
  activeTab: XrayTabId;
  onTabChange: (tab: XrayTabId) => void;
};

function outcomeKind(outcome: TurnHistoryOutcomeClass): TagKind {
  if (outcome === "emitted" || outcome === "deliberate-silence") {
    return "acc";
  }
  if (outcome === "failed" || outcome === "emission-failed") {
    return "bad";
  }
  if (outcome === "guard-blocked") {
    return "warn";
  }
  if (outcome === "observed") {
    return "info";
  }
  return "";
}

function commitmentStateKind(state: CommitmentItem["state"]): TagKind {
  if (state === "active") {
    return "acc";
  }
  if (state === "revoked") {
    return "bad";
  }
  return "warn";
}

function enforcementKind(enforcement: CommitmentItem["enforcement_class"]): TagKind {
  return enforcement === "critical" ? "bad" : "info";
}

function questionStatusKind(status: OpenQuestion["status"]): TagKind {
  if (status === "open") {
    return "warn";
  }
  if (status === "resolved") {
    return "acc";
  }
  return "";
}

function urgencyKind(urgency: number): TagKind {
  if (urgency >= 8) {
    return "bad";
  }
  if (urgency >= 5) {
    return "warn";
  }
  return "info";
}

function ApiPanel<T>({
  state,
  empty,
  isEmpty,
  children,
}: {
  state: ApiDataState<T>;
  empty: string;
  isEmpty: (data: T) => boolean;
  children: (data: T) => ReactNode;
}) {
  if (state.loading && state.data === null) {
    return <Loading />;
  }
  if (state.error !== null && state.data === null) {
    return <ErrorState>{state.error.message}</ErrorState>;
  }
  if (state.data === null || isEmpty(state.data)) {
    return <Empty>{empty}</Empty>;
  }
  return children(state.data);
}

function IdRefList({ ids, type }: { ids: readonly string[]; type: ObjectType }) {
  if (ids.length === 0) {
    return <span className="dim">none</span>;
  }

  return (
    <>
      {ids.map((id, index) => (
        <span key={id}>
          {index === 0 ? null : ", "}
          <IdRef id={id} type={type} label={shortId(id)} />
        </span>
      ))}
    </>
  );
}

function ReplayContext({ turn }: { turn: TurnHistoryRow | null | undefined }) {
  if (turn === null || turn === undefined) {
    return null;
  }

  return (
    <div className="xray-context" aria-label="Replay turn metadata">
      <span className="xray-context-label">replay</span>
      <IdRef id={turn.turn_id} type="turn" label={shortId(turn.turn_id)} />
      <span className="dim">{formatTime(turn.started_at)}</span>
      <Tag kind={outcomeKind(turn.outcome)}>{turn.outcome}</Tag>
      <Tag>{turn.audience ?? "global"}</Tag>
      {turn.suppression_reason === null ? null : (
        <span className="xray-context-reason">{turn.suppression_reason}</span>
      )}
    </div>
  );
}

function canonicalRefs(entry: SharedStateEntry) {
  return [
    { label: "goals", ids: entry.canonicalizes.goal_ids, type: "goal" as const },
    {
      label: "commitments",
      ids: entry.canonicalizes.commitment_ids,
      type: "commitment" as const,
    },
    { label: "actions", ids: entry.canonicalizes.action_ids, type: "action_record" as const },
    {
      label: "questions",
      ids: entry.canonicalizes.open_question_ids,
      type: "open_question" as const,
    },
  ].filter((item) => item.ids.length > 0);
}

function SharedStateTab({ state }: { state: ApiDataState<SharedStateResponse> }) {
  return (
    <ApiPanel
      state={state}
      empty="no shared state disclosed for this audience"
      isEmpty={(data) => data.entries.length === 0}
    >
      {(data) => (
        <div className="xray-list">
          {data.entries.map((entry) => (
            <article key={entry.id} className="xray-card">
              <div className="xray-card-head">
                <IdRef id={entry.id} type="shared_state_entry" label={shortId(entry.id)} />
                <Tag kind={lifecycleTagKind(entry.kind)}>{lifecycleLabel(entry.kind)}</Tag>
                <Tag>rank {entry.rank}</Tag>
              </div>
              <div className="xray-card-text">{entry.text}</div>
              <div className="xray-fields">
                {canonicalRefs(entry).map((refs) => (
                  <div key={refs.label}>
                    <span className="k">{refs.label}</span>
                    <span className="v">
                      <IdRefList ids={refs.ids} type={refs.type} />
                    </span>
                  </div>
                ))}
                <div>
                  <span className="k">provenance</span>
                  <span className="v">
                    <IdRefList ids={entry.provenance_stream_entry_ids} type="stream_entry" />
                  </span>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </ApiPanel>
  );
}

function CommitmentsTab({ state }: { state: ApiDataState<CommitmentsResponse> }) {
  return (
    <ApiPanel
      state={state}
      empty="no commitments recorded"
      isEmpty={(data) => data.commitments.length === 0}
    >
      {(data) => (
        <div className="xray-list">
          {data.commitments.map((commitment) => (
            <article key={commitment.id} className="xray-card">
              <div className="xray-card-head">
                <IdRef id={commitment.id} type="commitment" label={shortId(commitment.id)} />
                <Tag kind={enforcementKind(commitment.enforcement_class)}>
                  {commitment.enforcement_class}
                </Tag>
                <Tag kind={commitmentStateKind(commitment.state)}>{commitment.state}</Tag>
                <Tag>{commitment.audience ?? "global"}</Tag>
              </div>
              <div className="xray-card-text">{commitment.text}</div>
              <div className="xray-fields">
                <div>
                  <span className="k">kind</span>
                  <span className="v">{commitment.kind}</span>
                </div>
                <div>
                  <span className="k">source</span>
                  <span className="v">
                    <IdRefList ids={commitment.source_stream_entry_ids} type="stream_entry" />
                  </span>
                </div>
                {commitment.canonicalized_by_artifact_entry_id === null ? null : (
                  <div>
                    <span className="k">canonicalized</span>
                    <span className="v">
                      <IdRef
                        id={commitment.canonicalized_by_artifact_entry_id}
                        type="shared_state_entry"
                        label={shortId(commitment.canonicalized_by_artifact_entry_id)}
                      />
                    </span>
                  </div>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </ApiPanel>
  );
}

function OpenQuestionsTab({ state }: { state: ApiDataState<IdentityResponse> }) {
  return (
    <ApiPanel
      state={state}
      empty="no open questions"
      isEmpty={(data) => data.open_questions.length === 0}
    >
      {(data) => (
        <div className="xray-list">
          {data.open_questions.map((question) => (
            <article key={question.id} className="xray-card">
              <div className="xray-card-head">
                <IdRef id={question.id} type="open_question" label={shortId(question.id)} />
                <Tag kind={questionStatusKind(question.status)}>{question.status}</Tag>
                <Tag kind={urgencyKind(question.urgency)}>urgency {question.urgency}</Tag>
              </div>
              <div className="xray-card-text">{question.question}</div>
              <div className="xray-fields">
                <div>
                  <span className="k">goal</span>
                  <span className="v">
                    {question.goal_id === null ? (
                      <span className="dim">none</span>
                    ) : (
                      <IdRef id={question.goal_id} type="goal" label={shortId(question.goal_id)} />
                    )}
                  </span>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </ApiPanel>
  );
}

function PromptTab({ state }: { state: ApiDataState<PromptAssembledResponse> }) {
  return (
    <ApiPanel
      state={state}
      empty="assembled prompt unavailable"
      isEmpty={(data) => data.sections.length === 0 && data.text.length === 0}
    >
      {(data) => (
        <div className="xray-prompt">
          <div className="xray-outline">
            {data.sections.length === 0 ? (
              <Empty>no prompt sections</Empty>
            ) : (
              data.sections.map((section, index) => (
                <div key={`${section}:${index}`} className="xray-outline-row">
                  <span>{index + 1}</span>
                  <span>{section}</span>
                </div>
              ))
            )}
          </div>
          <pre className="xray-prompt-text">{data.text}</pre>
        </div>
      )}
    </ApiPanel>
  );
}

function RawLedgerTab({ ledger }: { ledger?: EvidenceLedger }) {
  if (ledger === undefined) {
    return <Empty>ledger not loaded yet</Empty>;
  }

  return (
    <div className="xray-raw">
      <JsonValueView value={ledger} />
    </div>
  );
}

function retrievalEntry(entry: EvidenceLedgerEntry): boolean {
  return entry.via_retrieval === true;
}

export function Xray({
  phases,
  activeTurnId,
  tokenTextByPhase,
  detailByPhase,
  terminalOutcome,
  delibPath,
  finalAttempt,
  cachedLedger,
  audience,
  tracePlaceholder = null,
  particleEnabled = true,
  replayTurn = null,
  sharedStateApi,
  commitmentsApi,
  identityApi,
  promptApi,
  activeTab,
  onTabChange,
}: XrayProps) {
  return (
    <div className="xray">
      <ReplayContext turn={replayTurn} />
      <div className="xray-tabs" role="tablist" aria-label="What Borg saw">
        {XRAY_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`xray-tab ${activeTab === tab.id ? "active" : ""}`.trim()}
            onClick={() => onTabChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="xray-body">
        {activeTab === "flow" ? (
          tracePlaceholder !== null ? (
            <div className="xray-placeholder" role="status">
              <div className="xray-placeholder-title">historical trace</div>
              <p>{tracePlaceholder}</p>
            </div>
          ) : (
            <FlowChart
              phases={phases}
              activeTurnId={activeTurnId}
              tokenTextByPhase={tokenTextByPhase}
              detailByPhase={detailByPhase}
              terminalOutcome={terminalOutcome}
              delibPath={delibPath}
              finalAttempt={finalAttempt}
              particleEnabled={particleEnabled}
            />
          )
        ) : null}
        {activeTab === "ledger" ? (
          <LedgerView
            turnId={activeTurnId}
            cachedLedger={cachedLedger}
            active={activeTurnId !== null}
            audience={audience}
          />
        ) : null}
        {activeTab === "memory" ? (
          <LedgerView
            turnId={activeTurnId}
            cachedLedger={cachedLedger}
            active={activeTurnId !== null}
            audience={audience}
            entryFilter={retrievalEntry}
            emptyMessage="no retrieved memory in this ledger"
          />
        ) : null}
        {activeTab === "shared" ? <SharedStateTab state={sharedStateApi} /> : null}
        {activeTab === "commitments" ? <CommitmentsTab state={commitmentsApi} /> : null}
        {activeTab === "open_qs" ? <OpenQuestionsTab state={identityApi} /> : null}
        {activeTab === "prompt" ? <PromptTab state={promptApi} /> : null}
        {activeTab === "raw" ? <RawLedgerTab ledger={cachedLedger} /> : null}
      </div>
    </div>
  );
}
