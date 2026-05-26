import { useMemo, useState } from "react";

import { getMemoryBand, getMemoryBands } from "../../api/client";
import type {
  EpisodeMemoryItem,
  MemoryBandDetail,
  MemoryBandId,
  MemoryBandSummary,
} from "../../api/types";
import { Panel } from "../../components/Panel";
import { Spark } from "../../components/Spark";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import { dateLabel, jsonText, shortId } from "../screen-utils";

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
            <div style={{ fontSize: 11.5, color: "var(--text-dim)", marginBottom: 10 }}>
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
        <Panel title="review queue" badge="hint">
          <div style={{ padding: 12 }}>
            <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>
              P2 keeps review visibility light here; dream renders belief-revision rows in detail.
            </div>
          </div>
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
}: {
  band: MemoryBandId;
  sessionId: string;
  back: () => void;
}) {
  const api = useApi(() => getMemoryBand(band, { session: sessionId }), [band, sessionId]);
  const rows = useMemo(() => (api.data === null ? [] : detailRows(api.data)), [api.data]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = rows.find((row) => row.id === selectedId) ?? rows[0] ?? null;

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
              <button className="btn sm" disabled title="v1 read-only">
                view stream chain
              </button>
              <button className="btn sm ghost" disabled title="v1 read-only">
                flag for review
              </button>
            </div>
          </div>
        </div>
      </div>
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
