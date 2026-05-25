import type { ReactNode } from "react";

export function Divider({ children }: { children: ReactNode }) {
  return <div className="divider">{children}</div>;
}
