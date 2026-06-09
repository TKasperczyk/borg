import type { ReactNode } from "react";

export type TagKind =
  | "acc"
  | "warn"
  | "bad"
  | "info"
  | "purple"
  | "solid"
  | "sev-1"
  | "sev-2"
  | "sev-3"
  | "sev-4"
  | "";

export type TagProps = {
  kind?: TagKind;
  dot?: boolean;
  children: ReactNode;
};

export function Tag({ kind = "", dot = false, children }: TagProps) {
  return (
    <span className={`tag ${kind}`}>
      {dot ? <span className={`dot ${kind}`}></span> : null}
      {children}
    </span>
  );
}
