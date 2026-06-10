import type { ReactNode } from "react";

export function Loading({ children = "loading" }: { children?: ReactNode }) {
  return <div className="notice loading">{children}</div>;
}
