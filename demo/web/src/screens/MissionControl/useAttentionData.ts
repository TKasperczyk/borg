import { useCallback, useEffect, useMemo } from "react";

import { getCommitments, getDreamState, getPrompts, getReviews, getStream } from "../../api/client";
import type {
  CommitmentEnforcement,
  CommitmentItem,
  PromptBlockView,
  ReviewKind,
  ReviewRow,
  StreamEntry,
} from "../../api/types";
import type { SeverityRank } from "../../components/SeverityChip";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi, type ApiHookState } from "../../hooks/use-api";
import { useLiveCache } from "../../hooks/use-live-cache";
import { streamOutcomeSummary, type StreamOutcomeDescriptor } from "../../lib/stream-outcomes";

const OUTCOME_LIMIT = 24;

type SourceState = {
  loading: boolean;
  error: string | null;
};

export type AttentionReviewGroup = {
  kind: ReviewKind;
  label: string;
  count: number;
  rows: ReviewRow[];
};

export type AttentionCommitmentGroup = {
  enforcement: CommitmentEnforcement;
  label: string;
  count: number;
  rows: CommitmentItem[];
};

export type AttentionDirectiveConflict = {
  row: ReviewRow;
  directiveIds: string[];
};

export type AttentionOutcomeItem = {
  entry: StreamEntry;
  summary: NonNullable<ReturnType<typeof streamOutcomeSummary>>;
};

export type AttentionOutcomeGroup = {
  outcome: StreamOutcomeDescriptor;
  count: number;
  rows: AttentionOutcomeItem[];
};

export type AttentionData = {
  reviews: SourceState & {
    headlineCount: number | null;
    observedCount: number;
    severity: SeverityRank;
    groups: AttentionReviewGroup[];
    previewRows: ReviewRow[];
  };
  commitments: SourceState & {
    headlineCount: number | null;
    observedCount: number;
    severity: SeverityRank;
    groups: AttentionCommitmentGroup[];
    previewRows: CommitmentItem[];
  };
  directiveConflicts: SourceState & {
    count: number;
    severity: SeverityRank;
    conflicts: AttentionDirectiveConflict[];
  };
  dream: SourceState & {
    pendingExtractionEpisodes: number;
    beliefRevisionCount: number;
    total: number;
    severity: SeverityRank;
    beliefRevisionRows: ReviewRow[];
    previewRows: ReviewRow[];
  };
  outcomes: SourceState & {
    count: number;
    windowed: boolean;
    severity: SeverityRank;
    groups: AttentionOutcomeGroup[];
    previewRows: AttentionOutcomeItem[];
  };
  prompts: SourceState & {
    count: number;
    severity: SeverityRank;
    blocks: PromptBlockView[];
    previewRows: PromptBlockView[];
  };
  attachments: {
    degraded: true;
    severity: SeverityRank;
    note: string;
  };
};

function sourceState<T>(api: ApiHookState<T>): SourceState {
  return {
    loading: api.loading && api.data === null,
    error: api.error?.message ?? null,
  };
}

function severityForCount(count: number, highAt = 5, criticalAt = 10): SeverityRank {
  if (count <= 0) {
    return 1;
  }
  if (count >= criticalAt) {
    return 4;
  }
  if (count >= highAt) {
    return 3;
  }
  return 2;
}

function reviewKindLabel(kind: ReviewKind): string {
  return kind.replaceAll("_", " ");
}

function groupReviewsByKind(rows: readonly ReviewRow[]): AttentionReviewGroup[] {
  const groups = new Map<ReviewKind, ReviewRow[]>();
  for (const row of rows) {
    groups.set(row.kind, [...(groups.get(row.kind) ?? []), row]);
  }

  return [...groups.entries()]
    .map(([kind, groupedRows]) => ({
      kind,
      label: reviewKindLabel(kind),
      count: groupedRows.length,
      rows: groupedRows,
    }))
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
}

function groupCommitmentsByEnforcement(
  rows: readonly CommitmentItem[],
): AttentionCommitmentGroup[] {
  const groups = new Map<CommitmentEnforcement, CommitmentItem[]>();
  for (const row of rows) {
    groups.set(row.enforcement_class, [...(groups.get(row.enforcement_class) ?? []), row]);
  }

  return [...groups.entries()]
    .map(([enforcement, groupedRows]) => ({
      enforcement,
      label: enforcement,
      count: groupedRows.length,
      rows: groupedRows,
    }))
    .sort((left, right) => {
      if (left.enforcement === right.enforcement) {
        return 0;
      }
      return left.enforcement === "critical" ? -1 : 1;
    });
}

function directiveIds(row: ReviewRow): string[] {
  const value = row.refs.directive_ids;
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function directiveConflicts(rows: readonly ReviewRow[]): AttentionDirectiveConflict[] {
  return rows
    .filter((row) => row.refs.subkind === "conflict")
    .map((row) => ({
      row,
      directiveIds: directiveIds(row),
    }));
}

function groupOutcomes(rows: readonly AttentionOutcomeItem[]): AttentionOutcomeGroup[] {
  const groups = new Map<string, AttentionOutcomeItem[]>();
  const descriptors = new Map<string, StreamOutcomeDescriptor>();
  for (const row of rows) {
    const key = row.summary.outcome.outcomeClass;
    groups.set(key, [...(groups.get(key) ?? []), row]);
    descriptors.set(key, row.summary.outcome);
  }

  return [...groups.entries()]
    .flatMap(([key, groupedRows]) => {
      const first = groupedRows[0];
      if (first === undefined) {
        return [];
      }

      return [
        {
          outcome: descriptors.get(key) ?? first.summary.outcome,
          count: groupedRows.length,
          rows: groupedRows,
        },
      ];
    })
    .sort(
      (left, right) =>
        right.count - left.count || left.outcome.label.localeCompare(right.outcome.label),
    );
}

export function useAttentionData(sessionId: string): AttentionData {
  const live = useLiveEventsContext();
  const { counts } = useLiveCache();
  const reviewsApi = useApi(() => getReviews({ openOnly: true }), []);
  const commitmentsApi = useApi(() => getCommitments({ state: "active" }), []);
  const directiveReviewsApi = useApi(
    () => getReviews({ openOnly: true, kind: "creator_directive_reconciliation" }),
    [],
  );
  const dreamApi = useApi(getDreamState, []);
  const outcomesApi = useApi(
    () =>
      getStream({
        session: sessionId,
        kinds: ["agent_suppressed", "agent_observed"],
        limit: OUTCOME_LIMIT,
      }),
    [sessionId],
  );
  const promptsApi = useApi(getPrompts, []);
  const refetchReviews = reviewsApi.refetch;
  const refetchCommitments = commitmentsApi.refetch;
  const refetchDirectiveReviews = directiveReviewsApi.refetch;
  const refetchDream = dreamApi.refetch;
  const refetchOutcomes = outcomesApi.refetch;
  const refetchPrompts = promptsApi.refetch;

  const refetchAttention = useCallback(async () => {
    await Promise.all([
      refetchReviews(),
      refetchCommitments(),
      refetchDirectiveReviews(),
      refetchDream(),
      refetchOutcomes(),
      refetchPrompts(),
    ]);
  }, [
    refetchCommitments,
    refetchDirectiveReviews,
    refetchDream,
    refetchOutcomes,
    refetchPrompts,
    refetchReviews,
  ]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (
        frame.type === "maintenance:tick" ||
        frame.type === "dream:process:started" ||
        frame.type === "dream:process:completed" ||
        frame.type === "stream:append" ||
        frame.type === "borg:reset"
      ) {
        void refetchAttention();
      }
    });
  }, [live, refetchAttention]);

  return useMemo<AttentionData>(() => {
    const reviewRows = reviewsApi.data?.rows ?? [];
    const reviewGroups = groupReviewsByKind(reviewRows);
    const activeCommitments = commitmentsApi.data?.commitments ?? [];
    const commitmentGroups = groupCommitmentsByEnforcement(activeCommitments);
    const conflicts = directiveConflicts(directiveReviewsApi.data?.rows ?? []);
    const beliefRevisionRows = dreamApi.data?.belief_revision_rows ?? [];
    const pendingExtractionEpisodes = dreamApi.data?.pending_extraction_episodes ?? 0;
    const outcomeRows =
      outcomesApi.data?.entries.flatMap<AttentionOutcomeItem>((entry) => {
        const summary = streamOutcomeSummary(entry);
        return summary === null ? [] : [{ entry, summary }];
      }) ?? [];
    const outcomeGroups = groupOutcomes(outcomeRows);
    const promptBlocks = promptsApi.data?.blocks.filter((block) => block.overridden) ?? [];

    return {
      reviews: {
        ...sourceState(reviewsApi),
        headlineCount: counts?.open_reviews ?? null,
        observedCount: reviewRows.length,
        severity: severityForCount(counts?.open_reviews ?? reviewRows.length),
        groups: reviewGroups,
        previewRows: reviewRows.slice(0, 3),
      },
      commitments: {
        ...sourceState(commitmentsApi),
        headlineCount: counts?.commitments ?? null,
        observedCount: activeCommitments.length,
        severity: activeCommitments.some(
          (commitment) => commitment.enforcement_class === "critical",
        )
          ? severityForCount(counts?.commitments ?? activeCommitments.length, 3, 8)
          : severityForCount(counts?.commitments ?? activeCommitments.length),
        groups: commitmentGroups,
        previewRows: activeCommitments.slice(0, 3),
      },
      directiveConflicts: {
        ...sourceState(directiveReviewsApi),
        count: conflicts.length,
        severity: conflicts.length > 0 ? 4 : 1,
        conflicts: conflicts.slice(0, 4),
      },
      dream: {
        ...sourceState(dreamApi),
        pendingExtractionEpisodes,
        beliefRevisionCount: beliefRevisionRows.length,
        total: pendingExtractionEpisodes + beliefRevisionRows.length,
        severity: severityForCount(pendingExtractionEpisodes + beliefRevisionRows.length),
        beliefRevisionRows,
        previewRows: beliefRevisionRows.slice(0, 3),
      },
      outcomes: {
        ...sourceState(outcomesApi),
        count: outcomeRows.length,
        windowed:
          outcomesApi.data?.next_cursor !== null && outcomesApi.data?.next_cursor !== undefined,
        severity: outcomeRows.some((row) => row.summary.outcome.outcomeClass === "emission-failed")
          ? 3
          : severityForCount(outcomeRows.length),
        groups: outcomeGroups,
        previewRows: outcomeRows.slice(0, 3),
      },
      prompts: {
        ...sourceState(promptsApi),
        count: promptBlocks.length,
        severity: severityForCount(promptBlocks.length),
        blocks: promptBlocks,
        previewRows: promptBlocks.slice(0, 3),
      },
      attachments: {
        degraded: true,
        severity: 2,
        note: "needs backend: add a list/count endpoint for quarantined or inactive attachments; current status lookup requires explicit attachment ids",
      },
    };
  }, [
    commitmentsApi,
    counts?.commitments,
    counts?.open_reviews,
    directiveReviewsApi,
    dreamApi,
    outcomesApi,
    promptsApi,
    reviewsApi,
  ]);
}
