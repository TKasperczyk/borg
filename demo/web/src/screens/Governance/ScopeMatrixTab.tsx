import type {
  CommitmentItem,
  CreatorDirectiveItem,
  ReviewRow,
  SessionParticipationPolicy,
  SessionRecord,
  SharedStateEntryKind,
} from "../../api/types";
import { IdRef } from "../../components/Inspector/IdRef";
import { PolicyValue } from "../../components/PolicyValue";
import { Tag } from "../../components/Tag";
import { lifecycleLabel, tagKind } from "../../lib/shared-state-lifecycle";
import { shortId } from "../screen-utils";
import type { DirectiveSupportData } from "./directive-support";

type ScopeMatrixTabProps = {
  sessions: readonly SessionRecord[];
  commitments: readonly CommitmentItem[];
  directives: readonly CreatorDirectiveItem[];
  supportData: DirectiveSupportData | null;
  reviews?: readonly ReviewRow[] | null;
  reviewsLoading?: boolean;
  reviewsError?: Error | null;
};

type AudienceRollup = {
  audience: string;
  sessions: SessionRecord[];
  policyCounts: Map<SessionParticipationPolicy, number>;
  matchingCommitments: CommitmentItem[];
  sharedKindCounts: Map<SharedStateEntryKind, number>;
  linkedOpenReviews: number | null;
};

const POLICY_ORDER: readonly SessionParticipationPolicy[] = [
  "active",
  "paused",
  "observing",
  "muted",
];

function commitmentDisplayState(
  commitment: CommitmentItem,
): "active" | "revoked" | "expired" | "superseded" {
  return commitment.superseded_by_id === null ? commitment.state : "superseded";
}

function commitmentsForAudience(
  commitments: readonly CommitmentItem[],
  audience: string,
): CommitmentItem[] {
  return commitments.filter(
    (commitment) => commitment.audience === null || commitment.audience === audience,
  );
}

function sharedEntriesForAudience(supportData: DirectiveSupportData | null, audience: string) {
  return supportData?.sharedAudiences.find((row) => row.audience === audience)?.entries ?? [];
}

function collectRefStrings(value: unknown, into: string[]): void {
  if (typeof value === "string") {
    into.push(value);
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectRefStrings(item, into);
    }
    return;
  }

  if (value !== null && typeof value === "object") {
    for (const item of Object.values(value)) {
      collectRefStrings(item, into);
    }
  }
}

function linkedReviewCount(reviews: readonly ReviewRow[], ids: readonly string[]): number {
  if (ids.length === 0) {
    return 0;
  }

  const idSet = new Set(ids);
  return reviews.filter((review) => {
    const refs: string[] = [];
    collectRefStrings(review.refs, refs);
    return refs.some((ref) => idSet.has(ref));
  }).length;
}

function PolicyDistribution({
  policyCounts,
}: {
  policyCounts: ReadonlyMap<SessionParticipationPolicy, number>;
}) {
  const labels = POLICY_ORDER.flatMap((policy) => {
    const count = policyCounts.get(policy) ?? 0;
    return count === 0 ? [] : [{ policy, count }];
  });
  if (labels.length === 0) {
    return <>none</>;
  }
  return (
    <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {labels.map(({ policy, count }) => (
        <span key={policy}>
          <PolicyValue domain="participation_policy" value={policy} />{" "}
          <span className="dim">x{count}</span>
        </span>
      ))}
    </span>
  );
}

function lifecycleCounts(
  entries: ReturnType<typeof sharedEntriesForAudience>,
): Map<SharedStateEntryKind, number> {
  const counts = new Map<SharedStateEntryKind, number>();
  for (const entry of entries) {
    counts.set(entry.kind, (counts.get(entry.kind) ?? 0) + 1);
  }
  return counts;
}

function lifecycleCountLabel(counts: ReadonlyMap<SharedStateEntryKind, number>): string {
  if (counts.size === 0) {
    return "none";
  }

  return [...counts]
    .sort(([left], [right]) => lifecycleLabel(left).localeCompare(lifecycleLabel(right)))
    .map(([kind, count]) => `${lifecycleLabel(kind)}:${count}`)
    .join(" · ");
}

function idsForSessionScope(input: {
  session: SessionRecord;
  commitments: readonly CommitmentItem[];
  directives: readonly CreatorDirectiveItem[];
}): string[] {
  const ids = [input.session.session_id];
  if (input.session.audience_entity_id !== null) {
    ids.push(input.session.audience_entity_id);
  }
  for (const commitment of commitmentsForAudience(
    input.commitments,
    input.session.audience_label,
  )) {
    ids.push(commitment.id);
  }
  for (const directive of input.directives) {
    if (directive.source_session_id === input.session.session_id) {
      ids.push(directive.id);
    }
    if (
      input.session.audience_entity_id !== null &&
      directive.subject_entity_id === input.session.audience_entity_id
    ) {
      ids.push(directive.id);
    }
  }
  return ids;
}

function audienceRollups({
  sessions,
  commitments,
  supportData,
  reviews,
}: {
  sessions: readonly SessionRecord[];
  commitments: readonly CommitmentItem[];
  supportData: DirectiveSupportData | null;
  reviews: readonly ReviewRow[] | null;
}): AudienceRollup[] {
  const grouped = new Map<string, SessionRecord[]>();
  for (const session of sessions) {
    const current = grouped.get(session.audience_label) ?? [];
    current.push(session);
    grouped.set(session.audience_label, current);
  }

  return [...grouped]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([audience, audienceSessions]) => {
      const policyCounts = new Map<SessionParticipationPolicy, number>();
      for (const session of audienceSessions) {
        policyCounts.set(
          session.participation_policy,
          (policyCounts.get(session.participation_policy) ?? 0) + 1,
        );
      }
      const matchingCommitments = commitmentsForAudience(commitments, audience);
      const sharedKindCounts = lifecycleCounts(sharedEntriesForAudience(supportData, audience));
      const linkedOpenReviews =
        reviews === null
          ? null
          : linkedReviewCount(reviews, [
              ...audienceSessions.map((session) => session.session_id),
              ...audienceSessions.flatMap((session) =>
                session.audience_entity_id === null ? [] : [session.audience_entity_id],
              ),
              ...matchingCommitments.map((commitment) => commitment.id),
            ]);

      return {
        audience,
        sessions: audienceSessions,
        policyCounts,
        matchingCommitments,
        sharedKindCounts,
        linkedOpenReviews,
      };
    });
}

export function ScopeMatrixTab({
  sessions,
  commitments,
  directives,
  supportData,
  reviews,
  reviewsLoading = false,
  reviewsError = null,
}: ScopeMatrixTabProps) {
  const showReviews = reviews !== undefined && reviewsError === null;
  const rollups = audienceRollups({
    sessions,
    commitments,
    supportData,
    reviews: showReviews && reviews !== undefined ? reviews : null,
  });

  return (
    <div className="governance-panel scope-matrix">
      <div className="page-head">
        <span className="desc">
          read-only operator policy labels over the current session/shared-state window
        </span>
        <span className="spacer"></span>
        {supportData?.audienceDiscoveryTruncated ? (
          <Tag kind="warn">audience discovery hit 1000-session cap</Tag>
        ) : (
          <Tag>current getSessions window</Tag>
        )}
      </div>
      <div className="page-body scope-matrix-body">
        <div className="notice" style={{ marginBottom: 12 }}>
          disclosure scope, participation policy, and commitment audience are labels for operator
          review. They are not recall gates or output controls.
          {reviewsError === null
            ? ""
            : " Linked open reviews are omitted because reviews failed to load."}
        </div>
        <div className="capability-note" style={{ marginBottom: 12 }}>
          dream-impact and full entity inventory not shown here
        </div>

        <section className="scope-section" aria-label="session scope matrix">
          <div className="scope-section-head">
            <h2>session rows</h2>
            <span className="dim">
              {sessions.length.toLocaleString()} rows · shared-state counts from discovered
              audiences
            </span>
          </div>
          <div style={{ overflow: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ minWidth: 160 }}>label</th>
                  <th style={{ width: 140 }}>audience</th>
                  <th style={{ width: 120 }}>policy</th>
                  <th style={{ width: 100 }}>role</th>
                  <th style={{ width: 90 }}>status</th>
                  <th style={{ width: 120 }}>latest turn</th>
                  <th style={{ width: 90, textAlign: "right" }}>messages</th>
                  <th style={{ width: 130 }}>commitments</th>
                  <th style={{ width: 130 }}>shared-state</th>
                  {showReviews ? <th style={{ width: 130 }}>reviews</th> : null}
                </tr>
              </thead>
              <tbody>
                {sessions.map((session) => {
                  const matchingCommitments = commitmentsForAudience(
                    commitments,
                    session.audience_label,
                  );
                  const activeCommitments = matchingCommitments.filter(
                    (commitment) => commitmentDisplayState(commitment) === "active",
                  );
                  const criticalCommitments = activeCommitments.filter(
                    (commitment) => commitment.enforcement_class === "critical",
                  );
                  const sharedEntries = sharedEntriesForAudience(
                    supportData,
                    session.audience_label,
                  );
                  const linkedReviews =
                    showReviews && reviews !== undefined && reviews !== null
                      ? linkedReviewCount(
                          reviews,
                          idsForSessionScope({ session, commitments, directives }),
                        )
                      : null;

                  return (
                    <tr key={session.session_id}>
                      <td>
                        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                          <IdRef
                            id={session.session_id}
                            type="session"
                            label={session.label}
                            hint={session}
                          />
                          <span className="dim" style={{ fontSize: 10.5 }}>
                            {session.source_type}
                          </span>
                        </div>
                      </td>
                      <td>
                        <span title={session.audience_label}>{session.audience_label}</span>
                      </td>
                      <td>
                        <PolicyValue
                          domain="participation_policy"
                          value={session.participation_policy}
                        />
                      </td>
                      <td>{session.audience_role}</td>
                      <td>
                        <Tag kind={session.status === "active" ? "acc" : ""}>{session.status}</Tag>
                      </td>
                      <td>
                        {session.last_turn_id === null ? (
                          <span className="mute">—</span>
                        ) : (
                          <IdRef
                            id={session.last_turn_id}
                            type="turn"
                            label={shortId(session.last_turn_id)}
                          />
                        )}
                      </td>
                      <td className="tab-num" style={{ textAlign: "right" }}>
                        {session.message_count.toLocaleString()}
                      </td>
                      <td>
                        {activeCommitments.length} active / {criticalCommitments.length} critical
                      </td>
                      <td>
                        {sharedEntries.length} rows
                        {supportData === null ? <span className="dim"> · loading</span> : null}
                      </td>
                      {showReviews ? (
                        <td>
                          {reviewsLoading && reviews === null ? (
                            <span className="dim">loading</span>
                          ) : (
                            `${linkedReviews ?? 0} linked open`
                          )}
                        </td>
                      ) : null}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className="scope-section" aria-label="audience rollups">
          <div className="scope-section-head">
            <h2>audience rollups</h2>
            <span className="dim">
              policy distribution · active critical commitments · lifecycle counts
            </span>
          </div>
          <div className="scope-rollups">
            {rollups.map((rollup) => {
              const activeCommitments = rollup.matchingCommitments.filter(
                (commitment) => commitmentDisplayState(commitment) === "active",
              );
              const criticalCommitments = activeCommitments.filter(
                (commitment) => commitment.enforcement_class === "critical",
              );

              return (
                <article key={rollup.audience} className="scope-rollup">
                  <div className="scope-rollup-head">
                    <strong>{rollup.audience}</strong>
                    <Tag>{rollup.sessions.length} sessions</Tag>
                  </div>
                  <div className="props">
                    <div className="row">
                      <span className="k">policies</span>
                      <span className="v">
                        <PolicyDistribution policyCounts={rollup.policyCounts} />
                      </span>
                    </div>
                    <div className="row">
                      <span className="k">commitments</span>
                      <span className="v">
                        {activeCommitments.length} active / {criticalCommitments.length} critical
                      </span>
                    </div>
                    <div className="row">
                      <span className="k">shared lifecycle</span>
                      <span className="v">{lifecycleCountLabel(rollup.sharedKindCounts)}</span>
                    </div>
                    {showReviews ? (
                      <div className="row">
                        <span className="k">reviews</span>
                        <span className="v">
                          {reviewsLoading && reviews === null
                            ? "loading"
                            : `${rollup.linkedOpenReviews ?? 0} linked open`}
                        </span>
                      </div>
                    ) : null}
                  </div>
                  {rollup.sharedKindCounts.size === 0 ? null : (
                    <div className="scope-lifecycle-tags">
                      {[...rollup.sharedKindCounts].map(([kind, count]) => (
                        <Tag key={kind} kind={tagKind(kind)}>
                          {lifecycleLabel(kind)} {count}
                        </Tag>
                      ))}
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
