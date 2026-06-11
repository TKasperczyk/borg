import { createContext, type ReactNode, useContext } from "react";

import { fetchState } from "../api/client";
import type { ApiState } from "../api/types";
import { useQuery } from "../api/useQuery";

type StateQuery = {
  data: ApiState | undefined;
  error: Error | undefined;
  loading: boolean;
  refetch: () => void;
};

const StateContext = createContext<StateQuery | null>(null);

export function StateProvider({
  children,
  sessionId = null,
}: {
  children: ReactNode;
  sessionId?: string | null;
}) {
  const state = useQuery(`state:${sessionId ?? ""}`, () => fetchState(sessionId ?? undefined));
  return <StateContext.Provider value={state}>{children}</StateContext.Provider>;
}

export function useAppState(): StateQuery {
  const state = useContext(StateContext);
  if (state === null) {
    throw new Error("useAppState must be used within StateProvider");
  }

  return state;
}
