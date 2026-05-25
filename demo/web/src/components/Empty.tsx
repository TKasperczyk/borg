import type { ReactNode } from "react";

export function Empty({ children }: { children: ReactNode }) {
  return <div className="notice">{children}</div>;
}
