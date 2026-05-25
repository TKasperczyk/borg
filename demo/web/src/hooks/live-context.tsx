import { createContext, useContext } from "react";

import type { LiveEvents } from "./use-live-events";

const LiveEventsContext = createContext<LiveEvents | null>(null);

export const LiveEventsProvider = LiveEventsContext.Provider;

export function useLiveEventsContext(): LiveEvents {
  const value = useContext(LiveEventsContext);
  if (value === null) {
    throw new Error("LiveEventsContext is not available");
  }
  return value;
}
