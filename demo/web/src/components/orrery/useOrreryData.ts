import { useCallback, useEffect, useMemo } from "react";

import {
  getCommitments,
  getCreatorDirectives,
  getDreamState,
  getMemoryBands,
  getReviews,
} from "../../api/client";
import type {
  CommitmentEnforcement,
  DreamProcessSummary,
  MaintenanceTickFrame,
  MemoryBandId,
  ReviewKind,
  ReviewRow,
  StateSnapshot,
  TurnTerminalOutcome,
  WsState,
} from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi, type ApiHookState } from "../../hooks/use-api";
import { useLiveCache, type DreamActivity } from "../../hooks/use-live-cache";

export type OrreryTurnInput = {
  activeTurnId: string | null;
  lastPhase: string;
  running: boolean;
  terminalOutcome: TurnTerminalOutcome | null;
};

export type OrreryMemoryBand = {
  id: MemoryBandId;
  name: string;
  count: number;
  countIsLowerBound: boolean;
};

export type OrreryDreamSatellite = {
  name: string;
  label: string;
  enabled: boolean;
  lastStatus: DreamProcessSummary["last_status"];
  running: boolean;
  phase: DreamActivity["phase"] | null;
};

export type OrreryFault = {
  id: number;
  kind: ReviewKind;
  reason: string;
};

export type OrreryGovernance = {
  commitments: Record<CommitmentEnforcement, number> & { total: number };
  directives: {
    active: number;
    total: number;
  };
};

export type OrreryRuntime = {
  wsState: WsState;
  connectionCount: number;
  counts: StateSnapshot["counts"] | null;
  dreamActivity: DreamActivity | null;
  lastMaintenanceTick: MaintenanceTickFrame | null;
};

export type OrreryViewModel = {
  loading: boolean;
  error: string | null;
  memoryBands: OrreryMemoryBand[];
  dream: {
    satellites: OrreryDreamSatellite[];
    runningCount: number;
  };
  governance: OrreryGovernance;
  reviews: {
    openCount: number;
    severity: "idle" | "warn" | "bad";
    faults: OrreryFault[];
  };
  stream: OrreryTurnInput & {
    state: "idle" | "running" | "terminal";
  };
  runtime: OrreryRuntime;
};

const EMPTY_GOVERNANCE: OrreryGovernance = {
  commitments: { critical: 0, advisory: 0, total: 0 },
  directives: { active: 0, total: 0 },
};

function firstError(apis: readonly ApiHookState<unknown>[]): string | null {
  for (const api of apis) {
    if (api.error !== null) {
      return api.error.message;
    }
  }
  return null;
}

function isLoadingInitial(apis: readonly ApiHookState<unknown>[]): boolean {
  return apis.some((api) => api.loading && api.data === null);
}

function processLabel(name: string): string {
  return name.replaceAll("-", " ");
}

export function useOrreryData(turn: OrreryTurnInput): OrreryViewModel {
  const live = useLiveEventsContext();
  const { counts, dreamActivity, lastMaintenanceTick, wsState, connectionCount } = useLiveCache();
  const memoryApi = useApi(() => getMemoryBands(), []);
  const dreamApi = useApi(getDreamState, []);
  const reviewsApi = useApi(() => getReviews({ openOnly: true }), []);
  const commitmentsApi = useApi(() => getCommitments(), []);
  const directivesApi = useApi(() => getCreatorDirectives(), []);
  const refetchMemory = memoryApi.refetch;
  const refetchDream = dreamApi.refetch;
  const refetchReviews = reviewsApi.refetch;
  const refetchCommitments = commitmentsApi.refetch;
  const refetchDirectives = directivesApi.refetch;

  const refetchSubstrate = useCallback(async () => {
    await Promise.all([
      refetchMemory(),
      refetchDream(),
      refetchReviews(),
      refetchCommitments(),
      refetchDirectives(),
    ]);
  }, [refetchCommitments, refetchDirectives, refetchDream, refetchMemory, refetchReviews]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (
        frame.type === "maintenance:tick" ||
        frame.type === "dream:process:started" ||
        frame.type === "dream:process:completed" ||
        frame.type === "borg:reset"
      ) {
        void refetchSubstrate();
      }
    });
  }, [live, refetchSubstrate]);

  return useMemo<OrreryViewModel>(() => {
    const apis: readonly ApiHookState<unknown>[] = [
      memoryApi,
      dreamApi,
      reviewsApi,
      commitmentsApi,
      directivesApi,
    ];
    const memoryBands: OrreryMemoryBand[] =
      memoryApi.data?.bands.map((band) => ({
        id: band.id,
        name: band.name,
        count: band.count,
        countIsLowerBound: band.count_is_lower_bound ?? false,
      })) ?? [];

    const dreamProcesses = [...(dreamApi.data?.processes ?? [])];
    if (
      dreamActivity !== null &&
      !dreamProcesses.some((process) => process.name === dreamActivity.process)
    ) {
      dreamProcesses.push({
        name: dreamActivity.process as DreamProcessSummary["name"],
        description: "",
        last_run_at: null,
        last_status: null,
        last_audit_id: null,
        budget: null,
        enabled: true,
      });
    }

    const satellites = dreamProcesses.map<OrreryDreamSatellite>((process) => {
      const running = dreamActivity?.process === process.name;
      return {
        name: process.name,
        label: processLabel(process.name),
        enabled: process.enabled,
        lastStatus: process.last_status,
        running,
        phase: running ? (dreamActivity?.phase ?? null) : null,
      };
    });

    const activeCommitments =
      commitmentsApi.data?.commitments.filter((commitment) => commitment.state === "active") ?? [];
    const critical = activeCommitments.filter(
      (commitment) => commitment.enforcement_class === "critical",
    ).length;
    const advisory = activeCommitments.filter(
      (commitment) => commitment.enforcement_class === "advisory",
    ).length;
    const directives = directivesApi.data?.directives ?? [];
    const activeDirectives = directives.filter((directive) => directive.status === "active").length;
    const faults: OrreryFault[] =
      reviewsApi.data?.rows.map((row: ReviewRow) => ({
        id: row.id,
        kind: row.kind,
        reason: row.reason,
      })) ?? [];
    const openCount = faults.length;

    return {
      loading: isLoadingInitial(apis),
      error: firstError(apis),
      memoryBands,
      dream: {
        satellites,
        runningCount: satellites.filter((satellite) => satellite.running).length,
      },
      governance:
        commitmentsApi.data === null && directivesApi.data === null
          ? EMPTY_GOVERNANCE
          : {
              commitments: {
                critical,
                advisory,
                total: activeCommitments.length,
              },
              directives: {
                active: activeDirectives,
                total: directives.length,
              },
            },
      reviews: {
        openCount,
        severity: openCount === 0 ? "idle" : openCount >= 5 ? "bad" : "warn",
        faults,
      },
      stream: {
        ...turn,
        state: turn.running ? "running" : turn.terminalOutcome === null ? "idle" : "terminal",
      },
      runtime: {
        wsState,
        connectionCount,
        counts,
        dreamActivity,
        lastMaintenanceTick,
      },
    };
  }, [
    commitmentsApi,
    counts,
    directivesApi,
    dreamActivity,
    dreamApi,
    lastMaintenanceTick,
    memoryApi,
    reviewsApi,
    turn,
    wsState,
    connectionCount,
  ]);
}
