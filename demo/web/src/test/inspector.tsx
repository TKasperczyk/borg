import { render, type RenderOptions } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { vi } from "vitest";

import { Inspector } from "../components/Inspector/Inspector";
import { InspectorProvider } from "../components/Inspector/inspector-context";

type RenderWithInspectorOptions = Omit<RenderOptions, "wrapper"> & {
  inspector?: boolean;
  sessionId?: string;
  audience?: string;
};

export function renderWithInspector(
  ui: ReactElement,
  {
    inspector = false,
    sessionId = "default",
    audience = "alice",
    ...renderOptions
  }: RenderWithInspectorOptions = {},
) {
  const setView = vi.fn();
  const setSessionId = vi.fn();

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <InspectorProvider
        setView={setView}
        setSessionId={setSessionId}
        sessionId={sessionId}
        audience={audience}
      >
        {children}
        {inspector ? <Inspector /> : null}
      </InspectorProvider>
    );
  }

  return {
    ...render(ui, { ...renderOptions, wrapper: Wrapper }),
    setView,
    setSessionId,
  };
}
