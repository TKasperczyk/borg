import { useMemo, useState } from "react";

import { getIdentity } from "../../api/client";
import type { OpenQuestion } from "../../api/types";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { clamp01, dateLabel } from "../screen-utils";

type QuestionFilter = "all" | OpenQuestion["status"];

function questionTag(status: OpenQuestion["status"]) {
  if (status === "open") {
    return "acc";
  }
  if (status === "resolved") {
    return "info";
  }
  return "warn";
}

export function IdentityScreen() {
  const api = useApi(getIdentity, []);
  const [questionFilter, setQuestionFilter] = useState<QuestionFilter>("all");
  const identity = api.data;
  const questions = useMemo(() => {
    const all = identity?.open_questions ?? [];
    return questionFilter === "all" ? all : all.filter((question) => question.status === questionFilter);
  }, [identity?.open_questions, questionFilter]);
  const currentPeriod = identity?.periods.find((period) => period.end_ts === null) ?? identity?.periods[0] ?? null;
  const activeGoals = identity?.goals.filter((goal) => goal.status === "active").length ?? 0;

  if (api.loading && identity === null) {
    return <div className="notice">loading identity</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  if (identity === null) {
    return <div className="notice">no identity records</div>;
  }

  return (
    <div className="identity">
      <div className="id-hero">
        <div>
          <div
            style={{
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "var(--text-mute)",
              marginBottom: 8
            }}
          >
            self::current
          </div>
          <div className="stamp">
            borg <span className="acc">·</span> v89 identity substrate
          </div>
          <div className="quote">
            values, goals, traits, open questions, growth markers, and autobiography.
          </div>
          <div className="quote-attr">
            current period: {currentPeriod?.label ?? "none"} · {dateLabel(currentPeriod?.start_ts)}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="mini-stat">
            <div className="k">autobiographical periods</div>
            <div className="v tab-num">{identity.periods.length}</div>
            <div className="sub">
              current: <span className="acc">{currentPeriod?.label ?? "—"}</span>
            </div>
          </div>
          <div className="mini-stat">
            <div className="k">growth markers</div>
            <div className="v tab-num">{identity.growth_markers.length}</div>
            <div className="sub">evidence-backed changes</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="mini-stat">
            <div className="k">values · goals · traits</div>
            <div className="v tab-num">
              {identity.values.length} · {identity.goals.length} · {identity.traits.length}
            </div>
            <div className="sub">{activeGoals} active goals</div>
          </div>
          <div className="mini-stat">
            <div className="k">open-question events</div>
            <div className="v tab-num">{identity.open_question_events.length}</div>
            <div className="sub">create · resolve · abandon · bump</div>
          </div>
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 5" }}>
        <div className="h">
          <span className="ttl">values</span>
          <span className="n">{identity.values.length}</span>
          <span style={{ flex: 1 }}></span>
          <span className="dim" style={{ textTransform: "none" }}>
            preserved across turns
          </span>
        </div>
        <div className="body">
          {identity.values.map((value) => (
            <div key={value.id} className="item">
              <div
                style={{
                  color: "var(--text)",
                  fontFamily: "var(--sans)",
                  fontSize: 12.5,
                  lineHeight: 1.45,
                  marginBottom: 6
                }}
              >
                {value.description}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="bar-meter" style={{ flex: 1 }}>
                  <div className="fill" style={{ width: `${clamp01(value.confidence) * 100}%` }}></div>
                </div>
                <span className="dim tab-num" style={{ fontSize: 10.5, width: 34, textAlign: "right" }}>
                  {value.confidence.toFixed(2)}
                </span>
                <span className="dim" style={{ fontSize: 10.5, whiteSpace: "nowrap" }}>
                  {value.support_count} src · {dateLabel(value.created_at)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 4" }}>
        <div className="h">
          <span className="ttl">goals</span>
          <span className="n">{activeGoals} active</span>
        </div>
        <div className="body">
          {identity.goals.map((goal) => (
            <div key={goal.id} className="item">
              <div style={{ color: "var(--text)", fontFamily: "var(--sans)", fontSize: 13, marginBottom: 4 }}>
                {goal.description}
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <Tag kind={goal.status === "active" ? "acc" : ""} dot>
                  {goal.status}
                </Tag>
                <span className="dim" style={{ fontSize: 10.5 }}>
                  priority {goal.priority.toFixed(2)}
                </span>
                <span className="dim" style={{ fontSize: 10.5 }}>
                  since {dateLabel(goal.created_at)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 3" }}>
        <div className="h">
          <span className="ttl">open questions</span>
          <span className="n">{questions.length}</span>
        </div>
        <div className="body">
          <div className="filter-pills" style={{ marginBottom: 8 }}>
            {(["all", "open", "resolved", "abandoned"] as const).map((filter) => (
              <span
                key={filter}
                className={`pill ${questionFilter === filter ? "on" : ""}`}
                onClick={() => setQuestionFilter(filter)}
              >
                {filter}
              </span>
            ))}
          </div>
          {questions.map((question) => (
            <div key={question.id} className="item">
              <div
                style={{
                  color: "var(--text-dim)",
                  fontFamily: "var(--sans)",
                  fontSize: 12,
                  lineHeight: 1.5,
                  marginBottom: 4
                }}
              >
                {question.question}
              </div>
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", fontSize: 10.5 }}>
                <Tag kind={questionTag(question.status)} dot>
                  {question.status}
                </Tag>
                <span className="dim">urg {question.urgency.toFixed(2)}</span>
                <span className="dim">touched {dateLabel(question.last_touched)}</span>
                {question.resolved_at === null ? null : <span className="info">resolved {dateLabel(question.resolved_at)}</span>}
                {question.abandoned_at === null ? null : (
                  <span className="warn">abandoned {dateLabel(question.abandoned_at)}</span>
                )}
                {question.last_ruminated_at === null ? null : (
                  <span className="purple">bumped {dateLabel(question.last_ruminated_at)}</span>
                )}
              </div>
              <div className="bar-meter" style={{ marginTop: 4 }}>
                <div
                  className={`fill ${question.urgency > 0.6 ? "warn" : ""}`}
                  style={{ width: `${clamp01(question.urgency) * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 5" }}>
        <div className="h">
          <span className="ttl">traits</span>
          <span className="n">{identity.traits.length}</span>
        </div>
        <div className="body">
          {identity.traits.map((trait) => (
            <div key={trait.id} className="item" style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
              <span style={{ color: "var(--text-dim)", fontFamily: "var(--sans)", fontSize: 12.5, lineHeight: 1.5 }}>
                {trait.label}
              </span>
              <span className="dim" style={{ fontSize: 10.5, whiteSpace: "nowrap" }}>
                {trait.support_count} obs
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 7" }}>
        <div className="h">
          <span className="ttl">growth markers</span>
          <span className="n">{identity.growth_markers.length}</span>
        </div>
        <div className="body">
          <div className="timeline">
            {identity.growth_markers.map((marker) => (
              <div key={marker.id} className={`ev ${marker.confidence < 0.6 ? "warn" : ""}`}>
                <div className="t">{dateLabel(marker.ts)} · {marker.source_process}</div>
                <div className="x">{marker.what_changed}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 12" }}>
        <div className="h">
          <span className="ttl">autobiographical periods</span>
          <span className="n">{identity.periods.length}</span>
        </div>
        <div className="body" style={{ padding: 0 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${Math.max(identity.periods.length, 1)}, 1fr)`,
              borderBottom: "1px solid var(--line)"
            }}
          >
            {identity.periods.map((period, index) => {
              const current = period.id === currentPeriod?.id;
              return (
                <div
                  key={period.id}
                  style={{
                    padding: 14,
                    borderRight: index === identity.periods.length - 1 ? "0" : "1px solid var(--line)",
                    background: current ? "oklch(0.84 0.155 142 / 0.05)" : "transparent"
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                      color: current ? "var(--acc)" : "var(--text-mute)",
                      marginBottom: 6
                    }}
                  >
                    period {index + 1}
                    {current ? " · current" : ""}
                  </div>
                  <div style={{ color: "var(--text)", fontSize: 14, fontWeight: 500, marginBottom: 4 }}>
                    {period.label}
                  </div>
                  <div className="dim" style={{ fontSize: 10.5, marginBottom: 8 }}>
                    {dateLabel(period.start_ts)} to {period.end_ts === null ? "present" : dateLabel(period.end_ts)}
                  </div>
                  <div style={{ color: "var(--text-dim)", fontFamily: "var(--sans)", fontSize: 12, lineHeight: 1.55 }}>
                    {period.narrative}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
