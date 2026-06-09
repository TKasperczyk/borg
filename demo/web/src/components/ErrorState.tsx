import type { ReactNode } from "react";

export function ErrorState({ children }: { children: ReactNode }) {
  return <div className="notice bad">{children}</div>;
}
