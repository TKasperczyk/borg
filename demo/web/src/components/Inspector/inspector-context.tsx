import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import type { RouteId } from "../../routes";
import { objectRegistry, type InspectorTab } from "./inspector-registry";
import type { ObjectType } from "./inspector-id";

export type InspectorTarget = {
  type: ObjectType;
  id: string;
  presetTab?: InspectorTab;
  hint?: unknown;
};

export type InspectorContextValue = {
  target: InspectorTarget | null;
  targets: readonly InspectorTarget[];
  openObject: (target: InspectorTarget) => void;
  back: () => void;
  close: () => void;
  openInSourceScreen: () => void;
  canBack: boolean;
  sessionId: string;
  audience: string;
};

type InspectorProviderProps = {
  children: ReactNode;
  setView: (view: RouteId) => void;
  setSessionId: (sessionId: string) => void;
  sessionId: string;
  audience: string;
};

const InspectorContext = createContext<InspectorContextValue | null>(null);

function sameTarget(left: InspectorTarget, right: InspectorTarget): boolean {
  return left.type === right.type && left.id === right.id;
}

export function InspectorProvider({
  children,
  setView,
  setSessionId,
  sessionId,
  audience,
}: InspectorProviderProps) {
  const [targets, setTargets] = useState<InspectorTarget[]>([]);
  const target = targets.at(-1) ?? null;

  const openObject = useCallback((nextTarget: InspectorTarget) => {
    setTargets((current) => {
      const currentTarget = current.at(-1);
      if (currentTarget !== undefined && sameTarget(currentTarget, nextTarget)) {
        return [...current.slice(0, -1), nextTarget];
      }
      return [...current, nextTarget];
    });
  }, []);

  const back = useCallback(() => {
    setTargets((current) => (current.length <= 1 ? current : current.slice(0, -1)));
  }, []);

  const close = useCallback(() => {
    setTargets([]);
  }, []);

  const openInSourceScreen = useCallback(() => {
    const currentTarget = targets.at(-1);
    if (currentTarget === undefined) {
      return;
    }

    if (currentTarget.type === "session") {
      setSessionId(currentTarget.id);
    }

    const route = objectRegistry[currentTarget.type].sourceRoute;
    if (route !== null) {
      setView(route);
    }
  }, [setSessionId, setView, targets]);

  const value = useMemo<InspectorContextValue>(
    () => ({
      target,
      targets,
      openObject,
      back,
      close,
      openInSourceScreen,
      canBack: targets.length > 1,
      sessionId,
      audience,
    }),
    [audience, back, close, openInSourceScreen, openObject, sessionId, target, targets],
  );

  return <InspectorContext.Provider value={value}>{children}</InspectorContext.Provider>;
}

export function useInspector(): InspectorContextValue {
  const value = useContext(InspectorContext);
  if (value === null) {
    throw new Error("InspectorContext is not available");
  }
  return value;
}
