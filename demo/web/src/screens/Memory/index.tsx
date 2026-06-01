import { useMemo, useState } from "react";

import {
  getCorrectionReviews,
  getMemoryBand,
  getMemoryBands,
  patchCorrectionReview,
  postCorrectionCorrect,
  postCorrectionForget,
  postSemanticEdgeInvalidate,
} from "../../api/client";
import type {
  EpisodeMemoryItem,
  MemoryBandDetail,
  MemoryBandId,
  MemoryBandSummary,
  ReviewRow,
} from "../../api/types";
import { Modal } from "../../components/Modal";
import { Panel } from "../../components/Panel";
import { Spark } from "../../components/Spark";
import { Tag } from "../../components/Tag";
import { WhyDrawer } from "../../components/WhyDrawer";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import { dateLabel, jsonText, parseJsonPatch, shortId } from "../screen-utils";

const BAND_ORDER: MemoryBandId[] = [
  "episodic",
  "semantic",
  "procedural",
  "affective",
  "self",
  "commitments",
  "social",
  "relational",
];

const BAND_DESCRIPTIONS: Record<MemoryBandId, string> = {
  episodic: "what happened",
  semantic: "what Borg believes",
  procedural: "how Borg solves things",
  affective: "mood and trajectory",
  self: "values, goals, traits, narrative",
  commitments: "scoped promises and boundaries",
  social: "per-entity trust and history",
  relational: "evidence-backed relationship facts",
};

type MemoryCorrectionAction =
  | { kind: "forget"; id: string; title: string }
  | { kind: "correct"; id: string; title: string; patch: string; reason: string }
  | { kind: "invalidate-edge"; id: string; title: string; reason: string; at: string };

function correctionActionKind(id: string): "episode" | "semantic_node" | "semantic_edge" | null {
  if (id.startsWith("ep_")) {
    return "episode";
  }
  if (id.startsWith("semn_")) {
    return "semantic_node";
  }
  if (id.startsWith("seme_")) {
    return "semantic_edge";
  }
  return null;
}

function defaultMemoryPatch(
  row: { title: string; body: string },
  kind: NonNullable<ReturnType<typeof correctionActionKind>>,
): string {
  if (kind === "episode") {
    return JSON.stringify(
      {
        title: row.title,
        narrative: row.body,
      },
      null,
      2,
    );
  }

  if (kind === "semantic_node") {
    return JSON.stringify(
      {
        label: row.title,
        description: row.body,
      },
      null,
      2,
    );
  }

  return "{}";
}

function reviewPatchSummary(row: ReviewRow): string {
  return JSON.stringify(row.refs.patch ?? {}, null, 2);
}

function reviewOperatorReason(row: ReviewRow): string | null {
  const reason = row.refs.operator_reason;
  return typeof reason === "string" && reason.length > 0 ? reason : null;
}

function detailRows(
  detail: MemoryBandDetail,
): Array<{ id: string; title: string; meta: string; body: string }> {
  switch (detail.band) {
    case "episodic":
      return detail.items.map((item) => ({
        id: item.id,
        title: item.title,
        meta: `${dateLabel(item.start_time)} · ${item.audience ?? "global"} · ${item.source_count} src`,
        body: item.narrative,
      }));
    case "semantic":
      return [
        ...detail.nodes.map((node) => ({
          id: node.id,
          title: node.label,
          meta: `${node.kind} · ${node.status} · ${node.source_count} src`,
          body: node.description,
        })),
        ...detail.edges.map((edge) => ({
          id: edge.id,
          title: `${edge.from_node_id} --${edge.relation}-> ${edge.to_node_id}`,
          meta: `edge · confidence ${edge.confidence.toFixed(2)} · ${edge.source_count} src`,
          body: edge.invalidated_reason ?? "active edge",
        })),
      ];
    case "procedural":
      return detail.items.map((skill) => ({
        id: skill.id,
        title: skill.applies_when,
        meta: `alpha ${skill.alpha.toFixed(1)} · beta ${skill.beta.toFixed(1)} · ${skill.sample_count} samples`,
        body: skill.approach,
      }));
    case "affective":
      return detail.history.map((point) => ({
        id: String(point.id),
        title: `${formatTime(point.ts)} · valence ${point.valence.toFixed(2)}`,
        meta: `arousal ${point.arousal.toFixed(2)} · ${point.trigger_reason ?? "no trigger"}`,
        body: jsonText(point.provenance),
      }));
    case "self":
      return [
        ...detail.values.map((value) => ({
          id: value.id,
          title: value.label,
          meta: `value · confidence ${value.confidence.toFixed(2)}`,
          body: value.description,
        })),
        ...detail.goals.map((goal) => ({
          id: goal.id,
          title: goal.description,
          meta: `goal · ${goal.status} · priority ${goal.priority.toFixed(2)}`,
          body: goal.progress_notes ?? "no progress notes",
        })),
        ...detail.traits.map((trait) => ({
          id: trait.id,
          title: trait.label,
          meta: `trait · confidence ${trait.confidence.toFixed(2)}`,
          body: `${trait.support_count} support · ${trait.contradiction_count} contradiction`,
        })),
        ...detail.open_questions.map((question) => ({
          id: question.id,
          title: question.question,
          meta: `open_question · ${question.status} · urgency ${question.urgency.toFixed(2)}`,
          body: question.resolution_note ?? question.abandoned_reason ?? "unresolved",
        })),
      ];
    case "commitments":
      return detail.items.map((commitment) => ({
        id: commitment.id,
        title: commitment.text,
        meta: `${commitment.state} · ${commitment.enforcement_class} · ${commitment.audience ?? "global"}`,
        body: `${commitment.type} · ${commitment.kind}`,
      }));
    case "social":
      return detail.items.map((profile) => ({
        id: profile.entity_id,
        title: profile.name ?? profile.entity_id,
        meta: `trust ${profile.trust.toFixed(2)} · ${profile.history_count} interactions`,
        body: `attachment ${profile.attachment.toFixed(2)} · commitments ${profile.commitment_count}`,
      }));
    case "relational":
      return detail.items.map((slot) => ({
        id: slot.id,
        title: slot.slot,
        meta: `${slot.state} · ${slot.sources_count} src · ${slot.alternate_count} alternates`,
        body: slot.value,
      }));
  }
}

export function MemoryScreen({ sessionId }: { sessionId: string }) {
  const api = useApi(() => getMemoryBands({ session: sessionId }), [sessionId]);
  const [activeBand, setActiveBand] = useState<MemoryBandId | null>(null);

  if (activeBand !== null) {
    return (
      <MemoryDrill
        band={activeBand}
        sessionId={sessionId}
        back={() => setActiveBand(null)}
        onMemoryChanged={api.refetch}
      />
    );
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading memory bands</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  const bands = BAND_ORDER.map(
    (id, index) =>
      api.data?.bands.find((band) => band.id === id) ?? {
        id,
        n: String(index + 1).padStart(2, "0"),
        name: id,
        desc: BAND_DESCRIPTIONS[id],
        count: 0,
        growth: [1, 1, 1],
        stats: [],
      },
  );

  return (
    <div className="bands">
      <div className="bands-head">
        <h1>memory::bands</h1>
        <div className="desc">
          raw memory store browser · audience scoping applies during retrieval/evidence ledger
        </div>
      </div>
      <div className="bands-grid">
        {bands.map((band) => (
          <BandCard key={band.id} band={band} onClick={() => setActiveBand(band.id)} />
        ))}
      </div>
      <div className="divider" style={{ marginTop: 22 }}>
        governance
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Panel title="identity governance" badge="guards">
          <div style={{ padding: 12 }}>
            <div className="panel-note">
              identity-bearing writes are guarded by provenance, confidence, and review routing.
            </div>
            <div className="props">
              <div className="row">
                <span className="k">bands</span>
                <span className="v">8</span>
              </div>
              <div className="row">
                <span className="k">review hint</span>
                <span className="v">see dream for belief-revision rows</span>
              </div>
            </div>
          </div>
        </Panel>
        <Panel title="correction queue" badge="open">
          <CorrectionQueuePanel onResolved={api.refetch} />
        </Panel>
      </div>
    </div>
  );
}

function BandCard({ band, onClick }: { band: MemoryBandSummary; onClick: () => void }) {
  return (
    <div className="band-card" onClick={onClick}>
      <div className="head">
        <span>band {band.n ?? "—"}</span>
        <span className="n">{band.count.toLocaleString()}</span>
      </div>
      <div className="name">{band.name}</div>
      <div className="desc-line">{band.desc ?? BAND_DESCRIPTIONS[band.id]}</div>
      <Spark data={band.growth ?? [1, 1, 1]} />
      <div className="stat-row">
        {band.stats.slice(0, 2).map((stat) => (
          <div key={stat.k} className="stat">
            <div className="k">{stat.k}</div>
            <div className="v">{stat.v}</div>
          </div>
        ))}
      </div>
      <div className="explore">browse ▸</div>
    </div>
  );
}

function MemoryDrill({
  band,
  sessionId,
  back,
  onMemoryChanged,
}: {
  band: MemoryBandId;
  sessionId: string;
  back: () => void;
  onMemoryChanged: () => Promise<void>;
}) {
  const api = useApi(() => getMemoryBand(band, { session: sessionId }), [band, sessionId]);
  const rows = useMemo(() => (api.data === null ? [] : detailRows(api.data)), [api.data]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [whyId, setWhyId] = useState<string | null>(null);
  const [action, setAction] = useState<MemoryCorrectionAction | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const selected = rows.find((row) => row.id === selectedId) ?? rows[0] ?? null;
  const selectedCorrectionKind = selected === null ? null : correctionActionKind(selected.id);

  async function refetchAfterMemoryCorrection(): Promise<void> {
    // Invalidates GET /api/memory/bands and GET /api/memory/bands/:band.
    // Semantic node/edge changes also affect GET /api/semantic/graph; Graph is not live-wired this sprint.
    await Promise.all([api.refetch(), onMemoryChanged()]);
  }

  async function runMemoryAction(label: string, callback: () => Promise<void>): Promise<void> {
    setBusy(label);
    setOperatorError(null);
    try {
      await callback();
      setAction(null);
      await refetchAfterMemoryCorrection();
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function submitMemoryAction(): Promise<void> {
    if (action === null) {
      return;
    }

    if (action.kind === "forget") {
      await runMemoryAction("forget", async () => {
        await postCorrectionForget(action.id);
      });
      return;
    }

    if (action.kind === "correct") {
      await runMemoryAction("correct", async () => {
        const patch = parseJsonPatch(action.patch);
        await postCorrectionCorrect(action.id, {
          patch,
          ...(action.reason.trim().length === 0 ? {} : { reason: action.reason.trim() }),
        });
      });
      return;
    }

    await runMemoryAction("invalidate-edge", async () => {
      const parsedAt = action.at.trim().length === 0 ? undefined : Number(action.at);
      if (parsedAt !== undefined && !Number.isFinite(parsedAt)) {
        throw new Error("at must be a finite number");
      }
      await postSemanticEdgeInvalidate(action.id, {
        ...(parsedAt === undefined ? {} : { at: parsedAt }),
        ...(action.reason.trim().length === 0 ? {} : { reason: action.reason.trim() }),
      });
    });
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <span className="btn sm ghost" style={{ cursor: "pointer" }} onClick={back}>
          ← memory
        </span>
        <h1>{band} memory</h1>
        <span className="desc">{BAND_DESCRIPTIONS[band]}</span>
        <span className="spacer"></span>
        <Tag>{rows.length} rows</Tag>
      </div>
      {operatorError === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          {operatorError}
        </div>
      )}
      <div className="band-detail" style={{ flex: 1 }}>
        <div className="list">
          <div
            style={{
              padding: "8px 14px",
              borderBottom: "1px solid var(--line)",
              display: "flex",
              gap: 8,
              alignItems: "center",
              fontSize: 10.5,
              color: "var(--text-mute)",
            }}
          >
            <span>{rows.length} visible</span>
            <span style={{ flex: 1 }}></span>
            <span>sort: backend ▾</span>
          </div>
          {api.loading && rows.length === 0 ? <div className="notice">loading {band}</div> : null}
          {api.error !== null ? <div className="notice bad">{api.error.message}</div> : null}
          {rows.map((row) => (
            <div
              key={row.id}
              className={`list-row ${row.id === selected?.id ? "selected" : ""}`}
              onClick={() => setSelectedId(row.id)}
            >
              <div className="ttl">{row.title}</div>
              <div className="meta">
                <span>[{shortId(row.id)}]</span>
                <span>·</span>
                <span>{row.meta}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="detail">
          {selected === null ? (
            <div className="notice">no records in this band</div>
          ) : (
            <>
              <h2>{selected.title}</h2>
              <div className="meta-line">
                <span>[{selected.id}]</span>
                <span>·</span>
                <span>{selected.meta}</span>
              </div>
              <div className="divider">body</div>
              <div
                style={{
                  fontFamily: "var(--sans)",
                  color: "var(--text-dim)",
                  fontSize: 13,
                  lineHeight: 1.6,
                }}
              >
                {selected.body}
              </div>
              {api.data === null ? null : (
                <BandSpecificDetail detail={api.data} selectedId={selected.id} />
              )}
            </>
          )}
        </div>
        <div
          className="panel"
          style={{
            borderLeft: "1px solid var(--line)",
            borderTop: 0,
            borderRight: 0,
            borderBottom: 0,
          }}
        >
          <div className="panel-header">
            <span className="title">properties</span>
          </div>
          <div className="panel-body pad">
            <div className="props">
              <div className="row">
                <span className="k">band</span>
                <span className="v">{band}</span>
              </div>
              <div className="row">
                <span className="k">id</span>
                <span className="v">{selected?.id ?? "—"}</span>
              </div>
              <div className="row">
                <span className="k">rows</span>
                <span className="v">{rows.length}</span>
              </div>
              <div className="row">
                <span className="k">source policy</span>
                <span className="v">source-linked records only</span>
              </div>
            </div>
            <div className="divider" style={{ marginTop: 16 }}>
              operations
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {selected !== null && selectedCorrectionKind !== null ? (
                <>
                  <button
                    className="btn sm"
                    disabled={busy !== null}
                    onClick={() => setWhyId(selected.id)}
                  >
                    why
                  </button>
                  {selectedCorrectionKind === "semantic_edge" ? (
                    <button
                      className="btn sm ghost"
                      disabled={busy !== null}
                      onClick={() =>
                        setAction({
                          kind: "invalidate-edge",
                          id: selected.id,
                          title: selected.title,
                          reason: "",
                          at: "",
                        })
                      }
                    >
                      invalidate
                    </button>
                  ) : (
                    <>
                      <button
                        className="btn sm ghost"
                        disabled={busy !== null}
                        onClick={() =>
                          setAction({ kind: "forget", id: selected.id, title: selected.title })
                        }
                      >
                        forget
                      </button>
                      <button
                        className="btn sm ghost"
                        disabled={busy !== null}
                        onClick={() =>
                          setAction({
                            kind: "correct",
                            id: selected.id,
                            title: selected.title,
                            patch: defaultMemoryPatch(selected, selectedCorrectionKind),
                            reason: "",
                          })
                        }
                      >
                        correct
                      </button>
                    </>
                  )}
                </>
              ) : (
                <span className="dim" style={{ fontSize: 11 }}>
                  no correction actions for this row
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
      <WhyDrawer open={whyId !== null} id={whyId} onClose={() => setWhyId(null)} />
      <Modal
        open={action !== null}
        title={action === null ? "correction" : `${action.kind} ${action.id}`}
        onClose={() => {
          if (busy === null) {
            setAction(null);
          }
        }}
        footer={
          <>
            <button
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={() => setAction(null)}
            >
              cancel
            </button>
            <button
              className="btn sm primary"
              disabled={busy !== null}
              onClick={() => void submitMemoryAction()}
            >
              {busy === null
                ? action?.kind === "invalidate-edge"
                  ? "invalidate"
                  : action?.kind === "correct"
                    ? "queue"
                    : action?.kind
                : "saving"}
            </button>
          </>
        }
      >
        {action?.kind === "forget" ? (
          <div className="modal-form">
            <div className="dim">{action.title}</div>
          </div>
        ) : null}
        {action?.kind === "correct" ? (
          <div className="modal-form">
            <div className="dim">{action.title}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                value={action.reason}
                onChange={(event) => setAction({ ...action, reason: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>json patch</span>
              <textarea
                value={action.patch}
                onChange={(event) => setAction({ ...action, patch: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {action?.kind === "invalidate-edge" ? (
          <div className="modal-form">
            <div className="dim">{action.title}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                value={action.reason}
                onChange={(event) => setAction({ ...action, reason: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>at ms</span>
              <input
                type="number"
                value={action.at}
                onChange={(event) => setAction({ ...action, at: event.target.value })}
              />
            </label>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}

function CorrectionQueuePanel({ onResolved }: { onResolved: () => Promise<void> }) {
  const api = useApi(getCorrectionReviews, []);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState<number | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const rows = api.data?.rows ?? [];

  async function resolveCorrection(row: ReviewRow, action: "accept" | "reject"): Promise<void> {
    setBusy(row.id);
    setOperatorError(null);
    try {
      await patchCorrectionReview(row.id, {
        action,
        ...(notes[row.id]?.trim() ? { note: notes[row.id]!.trim() } : {}),
      });
      setNotes((current) => {
        const next = { ...current };
        delete next[row.id];
        return next;
      });
      // Invalidates GET /api/correction/reviews plus any band touched by the accepted patch.
      await Promise.all([api.refetch(), onResolved()]);
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading corrections</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div style={{ padding: 12 }}>
      {operatorError === null ? null : (
        <div className="notice bad" style={{ padding: 8 }}>
          {operatorError}
        </div>
      )}
      {rows.length === 0 ? (
        <div className="panel-note">No pending corrections.</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {rows.map((row) => {
            const operatorReason = reviewOperatorReason(row);
            return (
              <div key={row.id} className="item">
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <Tag>{String(row.refs.target_type ?? "target")}</Tag>
                  <span className="acc">{String(row.refs.target_id ?? "unknown")}</span>
                  <span className="dim">review {row.id}</span>
                </div>
                <div style={{ color: "var(--text-dim)", fontSize: 11.5, marginTop: 6 }}>
                  {String(row.refs.prompt_summary ?? row.reason)}
                </div>
                {operatorReason === null ? null : (
                  <div style={{ color: "var(--text)", fontSize: 11.5, marginTop: 6 }}>
                    {operatorReason}
                  </div>
                )}
                <pre className="why-pre" style={{ marginTop: 8 }}>
                  {reviewPatchSummary(row)}
                </pre>
                <label className="modal-field" style={{ marginTop: 8 }}>
                  <span>note</span>
                  <input
                    value={notes[row.id] ?? ""}
                    onChange={(event) => setNotes({ ...notes, [row.id]: event.target.value })}
                  />
                </label>
                <div className="operator-actions" style={{ marginTop: 8 }}>
                  <button
                    className="btn sm primary"
                    disabled={busy !== null}
                    onClick={() => void resolveCorrection(row, "accept")}
                  >
                    accept
                  </button>
                  <button
                    className="btn sm ghost"
                    disabled={busy !== null}
                    onClick={() => void resolveCorrection(row, "reject")}
                  >
                    reject
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function BandSpecificDetail({
  detail,
  selectedId,
}: {
  detail: MemoryBandDetail;
  selectedId: string;
}) {
  if (detail.band === "episodic") {
    const episode = detail.items.find((item) => item.id === selectedId) as
      | EpisodeMemoryItem
      | undefined;
    if (episode === undefined) {
      return null;
    }
    return (
      <>
        <div className="divider">citations</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {episode.source_stream_ids.map((id) => (
            <span key={id} className="tag info">
              {id}
            </span>
          ))}
        </div>
      </>
    );
  }

  if (detail.band === "affective") {
    return (
      <>
        <div className="divider">current mood</div>
        <div className="props">
          <div className="row">
            <span className="k">valence</span>
            <span className="v">{detail.current.valence.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">arousal</span>
            <span className="v">{detail.current.arousal.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">updated</span>
            <span className="v">{formatTime(detail.current.updated_at)}</span>
          </div>
        </div>
      </>
    );
  }

  if (detail.band === "relational") {
    return (
      <>
        <div className="divider">state counts</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {Object.entries(detail.counts).map(([state, count]) => (
            <Tag key={state}>
              {state} {count}
            </Tag>
          ))}
        </div>
      </>
    );
  }

  return null;
}
