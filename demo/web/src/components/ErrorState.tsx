import type { ReactNode } from "react";

export function ErrorState({ children, onRetry }: { children: ReactNode; onRetry?: () => void }) {
  return (
    <div className="notice error bad" role="alert">
      <span className="error-state-glyph" aria-hidden="true">
        !
      </span>
      <span className="error-state-message">{children}</span>
      {onRetry === undefined ? null : (
        <button type="button" className="btn sm ghost" onClick={onRetry}>
          retry
        </button>
      )}
    </div>
  );
}
