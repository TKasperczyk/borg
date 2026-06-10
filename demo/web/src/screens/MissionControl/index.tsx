import { useMemo, type ReactNode } from "react";

import type { PromptBlockView, ReviewRow } from "../../api/types";
import { CountBadge } from "../../components/CountBadge";
import { Empty } from "../../components/Empty";
import { IdRef } from "../../components/Inspector/IdRef";
import { Orrery } from "../../components/orrery/Orrery";
import { useOrreryData, type OrreryTurnInput } from "../../components/orrery/useOrreryData";
import { Panel } from "../../components/Panel";
import { SeverityChip } from "../../components/SeverityChip";
import { Spark } from "../../components/Spark";
import { Tag, type TagKind } from "../../components/Tag";
import { useInspector } from "../../components/Inspector/inspector-context";
import { moodLabel } from "../../components/StatusBar";
import { useLiveCache } from "../../hooks/use-live-cache";
import type { RouteId, RouteNavigationOptions } from "../../routes";
import { shortId } from "../screen-utils";
import { useAttentionData, type AttentionData } from "./useAttentionData";

export type MissionControlScreenProps = {
  sessionId: string;
  turnStream: OrreryTurnInput;
  onNavigate: (view: RouteId, options?: RouteNavigationOptions) => void;
};

type CardShellProps = {
  id: string;
  title: string;
  badge: string;
  action?: string;
  onAction?: () => void;
  children: ReactNode;
};

function CardShell({ id, title, badge, action = "open", onAction, children }: CardShellProps) {
  return (
    <section className="mc-attention-card" data-testid={id}>
      <Panel
        title={title}
        badge={badge}
        action={onAction === undefined ? undefined : action}
        onAction={onAction}
      >
        {children}
      </Panel>
    </section>
  );
}

function CardNotice({
  loading,
  error,
  empty,
  children,
}: {
  loading: boolean;
  error: string | null;
  empty: boolean;
  children: ReactNode;
}) {
  if (loading) {
    return <Empty>loading</Empty>;
  }
  if (error !== null) {
    return <Empty>error: {error}</Empty>;
  }
  if (empty) {
    return <Empty>{children}</Empty>;
  }
  return null;
}

function Headline({
  count,
  countLabel,
  severity,
  note,
  spark,
}: {
  count: number | null;
  countLabel: string;
  severity: 1 | 2 | 3 | 4;
  note: string;
  spark: readonly number[];
}) {
  return (
    <div className="mc-attention-headline">
      {count === null ? (
        <Tag kind="info">syncing</Tag>
      ) : (
        <CountBadge count={count} severity={severity} label={countLabel} />
      )}
      <SeverityChip rank={severity}>{`rank ${severity}`}</SeverityChip>
      <span className="mc-attention-note">{note}</span>
      {spark.length === 0 ? null : <Spark data={spark} />}
    </div>
  );
}

function Breakdown({ rows }: { rows: Array<{ label: string; count: number; kind?: TagKind }> }) {
  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="mc-breakdown">
      {rows.map((row) => (
        <div className="mc-breakdown-row" key={row.label}>
          <Tag kind={row.kind ?? ""}>{row.label}</Tag>
          <span>{row.count.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}

function InspectButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" className="btn sm ghost" onClick={onClick}>
      {label}
    </button>
  );
}

function ReviewPreviewRows({
  rows,
  onInspect,
}: {
  rows: readonly ReviewRow[];
  onInspect: (row: ReviewRow) => void;
}) {
  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="mc-preview-list">
      {rows.map((row) => (
        <div className="mc-preview-row" key={row.id}>
          <span className="mc-preview-title">review {row.id}</span>
          <Tag>{row.kind.replaceAll("_", " ")}</Tag>
          <InspectButton label="inspect" onClick={() => onInspect(row)} />
        </div>
      ))}
    </div>
  );
}

function ReviewsCard({
  data,
  onNavigate,
  onInspectReview,
}: {
  data: AttentionData["reviews"];
  onNavigate: (view: RouteId, options?: RouteNavigationOptions) => void;
  onInspectReview: (row: ReviewRow) => void;
}) {
  const showNotice = data.loading || data.error !== null || data.groups.length === 0;

  return (
    <CardShell
      id="attention-reviews"
      title="open reviews"
      badge="governance"
      onAction={() => onNavigate("review")}
    >
      <div className="mc-card-body">
        <Headline
          count={data.headlineCount}
          countLabel="open reviews"
          severity={data.severity}
          note={`${data.observedCount.toLocaleString()} rows by kind`}
          spark={data.spark}
        />
        {showNotice ? (
          <CardNotice loading={data.loading} error={data.error} empty={data.groups.length === 0}>
            no open reviews
          </CardNotice>
        ) : (
          <>
            <Breakdown
              rows={data.groups.map((group) => ({ label: group.label, count: group.count }))}
            />
            <ReviewPreviewRows rows={data.previewRows} onInspect={onInspectReview} />
          </>
        )}
      </div>
    </CardShell>
  );
}

function CommitmentsCard({
  data,
  onNavigate,
  onInspect,
}: {
  data: AttentionData["commitments"];
  onNavigate: (view: RouteId, options?: RouteNavigationOptions) => void;
  onInspect: (id: string, hint: unknown) => void;
}) {
  const showNotice = data.loading || data.error !== null || data.groups.length === 0;

  return (
    <CardShell
      id="attention-commitments"
      title="commitments"
      badge="active"
      onAction={() => onNavigate("governance", { governanceTab: "commitments" })}
    >
      <div className="mc-card-body">
        <Headline
          count={data.headlineCount}
          countLabel="active commitments"
          severity={data.severity}
          note={`${data.observedCount.toLocaleString()} active rows`}
          spark={data.spark}
        />
        {showNotice ? (
          <CardNotice loading={data.loading} error={data.error} empty={data.groups.length === 0}>
            no active commitments
          </CardNotice>
        ) : (
          <>
            <Breakdown
              rows={data.groups.map((group) => ({
                label: group.label,
                count: group.count,
                kind: group.enforcement === "critical" ? "warn" : "",
              }))}
            />
            <div className="mc-preview-list">
              {data.previewRows.map((commitment) => (
                <div className="mc-preview-row" key={commitment.id}>
                  <span className="mc-preview-title">{shortId(commitment.id)}</span>
                  <Tag kind={commitment.enforcement_class === "critical" ? "warn" : ""}>
                    {commitment.enforcement_class}
                  </Tag>
                  <InspectButton
                    label="inspect"
                    onClick={() => onInspect(commitment.id, commitment)}
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </CardShell>
  );
}

function DirectiveConflictsCard({
  data,
  onNavigate,
  onInspectReview,
}: {
  data: AttentionData["directiveConflicts"];
  onNavigate: (view: RouteId, options?: RouteNavigationOptions) => void;
  onInspectReview: (row: ReviewRow) => void;
}) {
  const showNotice = data.loading || data.error !== null || data.count === 0;

  return (
    <CardShell
      id="attention-directives"
      title="directive conflicts"
      badge="creator"
      onAction={() => onNavigate("governance", { governanceTab: "shared_state" })}
    >
      <div className="mc-card-body">
        <Headline
          count={data.count}
          countLabel="creator-directive conflicts"
          severity={data.severity}
          note="open reconciliation conflicts"
          spark={data.spark}
        />
        {showNotice ? (
          <CardNotice loading={data.loading} error={data.error} empty={data.count === 0}>
            no creator-directive conflicts
          </CardNotice>
        ) : (
          <div className="mc-preview-list">
            {data.conflicts.map((conflict) => (
              <div className="mc-preview-row" key={conflict.row.id}>
                <span className="mc-preview-title">review {conflict.row.id}</span>
                <Tag kind="purple">{conflict.directiveIds.length} directives</Tag>
                <InspectButton label="inspect" onClick={() => onInspectReview(conflict.row)} />
              </div>
            ))}
          </div>
        )}
      </div>
    </CardShell>
  );
}

function DreamCard({
  data,
  onNavigate,
  onInspectReview,
}: {
  data: AttentionData["dream"];
  onNavigate: (view: RouteId) => void;
  onInspectReview: (row: ReviewRow) => void;
}) {
  const showNotice = data.loading || data.error !== null || data.total === 0;

  return (
    <CardShell
      id="attention-dream"
      title="dream queue"
      badge="offline"
      onAction={() => onNavigate("dream")}
    >
      <div className="mc-card-body">
        <Headline
          count={data.total}
          countLabel="dream extraction and belief revision work"
          severity={data.severity}
          note="pending extraction + belief revision"
          spark={data.spark}
        />
        {showNotice ? (
          <CardNotice loading={data.loading} error={data.error} empty={data.total === 0}>
            no pending dream work
          </CardNotice>
        ) : (
          <>
            <Breakdown
              rows={[
                {
                  label: "extraction episodes",
                  count: data.pendingExtractionEpisodes,
                  kind: "purple",
                },
                { label: "belief revisions", count: data.beliefRevisionCount, kind: "warn" },
              ]}
            />
            <ReviewPreviewRows rows={data.previewRows} onInspect={onInspectReview} />
          </>
        )}
      </div>
    </CardShell>
  );
}

function OutcomesCard({
  data,
  onNavigate,
  onInspect,
}: {
  data: AttentionData["outcomes"];
  onNavigate: (view: RouteId) => void;
  onInspect: (id: string, hint: unknown) => void;
}) {
  const showNotice = data.loading || data.error !== null || data.groups.length === 0;

  return (
    <CardShell
      id="attention-outcomes"
      title="recent outcomes"
      badge="session"
      onAction={() => onNavigate("stream")}
    >
      <div className="mc-card-body">
        <Headline
          count={data.count}
          countLabel="recent suppressed and observed outcomes"
          severity={data.severity}
          note={data.windowed ? `${data.count.toLocaleString()}+ recent window` : "recent window"}
          spark={data.spark}
        />
        {showNotice ? (
          <CardNotice loading={data.loading} error={data.error} empty={data.groups.length === 0}>
            no recent suppressed or observed outcomes
          </CardNotice>
        ) : (
          <>
            <Breakdown
              rows={data.groups.map((group) => ({
                label: group.outcome.label,
                count: group.count,
                kind: group.outcome.tagKind,
              }))}
            />
            <div className="mc-preview-list">
              {data.previewRows.map((row) => (
                <div className="mc-preview-row" key={row.entry.id}>
                  <span className="mc-preview-title">{shortId(row.entry.id)}</span>
                  <Tag kind={row.summary.outcome.tagKind}>{row.summary.outcome.label}</Tag>
                  <InspectButton
                    label="inspect"
                    onClick={() => onInspect(row.entry.id, row.entry)}
                  />
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </CardShell>
  );
}

function PromptsCard({
  data,
  onNavigate,
  onInspect,
}: {
  data: AttentionData["prompts"];
  onNavigate: (view: RouteId) => void;
  onInspect: (block: PromptBlockView) => void;
}) {
  const showNotice = data.loading || data.error !== null || data.count === 0;

  return (
    <CardShell
      id="attention-prompts"
      title="prompt overrides"
      badge="runtime"
      onAction={() => onNavigate("prompts")}
    >
      <div className="mc-card-body">
        <Headline
          count={data.count}
          countLabel="prompt overrides"
          severity={data.severity}
          note="stored prompt blocks"
          spark={data.spark}
        />
        {showNotice ? (
          <CardNotice loading={data.loading} error={data.error} empty={data.count === 0}>
            no prompt overrides
          </CardNotice>
        ) : (
          <div className="mc-preview-list">
            {data.previewRows.map((block) => (
              <div className="mc-preview-row" key={block.key}>
                <span className="mc-preview-title">{block.label}</span>
                <Tag kind="warn">override</Tag>
                <InspectButton label="inspect" onClick={() => onInspect(block)} />
              </div>
            ))}
          </div>
        )}
      </div>
    </CardShell>
  );
}

function AttachmentsCard({ data }: { data: AttentionData["attachments"] }) {
  return (
    <CardShell
      id="attention-attachments"
      title="attachments"
      badge="degraded"
      action="needs backend"
    >
      <div className="mc-card-body">
        <Headline
          count={null}
          countLabel="quarantined or inactive attachments"
          severity={data.severity}
          note="quarantined/inactive status unavailable"
          spark={[]}
        />
        <Empty>{data.note}</Empty>
      </div>
    </CardShell>
  );
}

function StatusStrip({ turnStream }: { turnStream: OrreryTurnInput }) {
  const { stateApi, dreamActivity } = useLiveCache();
  const state = stateApi.data;
  const dreamLabel =
    dreamActivity === null ? "idle" : `${dreamActivity.process} ${dreamActivity.phase}`;

  return (
    <div className="mc-status-strip" data-testid="mission-status-strip">
      <span className="mc-status-seg">
        <span className="k">turn</span>
        {turnStream.activeTurnId === null ? (
          <span className="v">idle</span>
        ) : (
          <IdRef
            id={turnStream.activeTurnId}
            type="turn"
            label={shortId(turnStream.activeTurnId)}
            hint={{ phase: turnStream.lastPhase, outcome: turnStream.terminalOutcome }}
          />
        )}
      </span>
      <span className="mc-status-seg">
        <span className="k">phase</span>
        <span className="v">{turnStream.lastPhase}</span>
      </span>
      <span className="mc-status-seg">
        <span className="k">terminal</span>
        <span className="v">{turnStream.terminalOutcome ?? "—"}</span>
      </span>
      <span className="mc-status-seg">
        <span className="k">mood</span>
        <span className="v">
          <span className="instrument-mood-glyph">◐</span> {moodLabel(state)}
        </span>
      </span>
      <span className="mc-status-seg">
        <span className="k">dream</span>
        <span className="v">{dreamLabel}</span>
      </span>
    </div>
  );
}

export function MissionControlScreen({
  sessionId,
  turnStream,
  onNavigate,
}: MissionControlScreenProps) {
  const inspector = useInspector();
  const turn = useMemo<OrreryTurnInput>(
    () => ({
      activeTurnId: turnStream.activeTurnId,
      lastPhase: turnStream.lastPhase,
      running: turnStream.running,
      terminalOutcome: turnStream.terminalOutcome,
    }),
    [turnStream.activeTurnId, turnStream.lastPhase, turnStream.running, turnStream.terminalOutcome],
  );
  const data = useOrreryData(turn);
  const attention = useAttentionData(sessionId);

  const inspectReview = (row: ReviewRow) => {
    inspector.openObject({ type: "review", id: String(row.id), hint: row });
  };

  return (
    <div className="orr-mission" data-testid="mission-control-screen">
      <main className="orr-mission-main">
        <Orrery size="full" data={data} onNavigate={onNavigate} onInspect={inspector.openObject} />
        <StatusStrip turnStream={turn} />
      </main>
      <aside className="mc-attention-queue" aria-label="attention queue">
        <ReviewsCard
          data={attention.reviews}
          onNavigate={onNavigate}
          onInspectReview={inspectReview}
        />
        <CommitmentsCard
          data={attention.commitments}
          onNavigate={onNavigate}
          onInspect={(id, hint) => inspector.openObject({ type: "commitment", id, hint })}
        />
        <DirectiveConflictsCard
          data={attention.directiveConflicts}
          onNavigate={onNavigate}
          onInspectReview={inspectReview}
        />
        <DreamCard data={attention.dream} onNavigate={onNavigate} onInspectReview={inspectReview} />
        <OutcomesCard
          data={attention.outcomes}
          onNavigate={onNavigate}
          onInspect={(id, hint) => inspector.openObject({ type: "stream_entry", id, hint })}
        />
        <PromptsCard
          data={attention.prompts}
          onNavigate={onNavigate}
          onInspect={(block) =>
            inspector.openObject({ type: "prompt_block", id: block.key, hint: block })
          }
        />
        <AttachmentsCard data={attention.attachments} />
      </aside>
    </div>
  );
}
