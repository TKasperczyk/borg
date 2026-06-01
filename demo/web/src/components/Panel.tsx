import type { ReactNode } from "react";

export type PanelProps = {
  title: string;
  badge?: string;
  action?: string;
  onAction?: () => void;
  children: ReactNode;
  className?: string;
};

export function Panel({ title, badge, action, onAction, children, className }: PanelProps) {
  return (
    <div className={`panel ${className ?? ""}`}>
      <div className="panel-header">
        <span className="title">{title}</span>
        {badge === undefined ? null : <span className="badge">{badge}</span>}
        <span className="spacer"></span>
        {action === undefined ? null : (
          <button type="button" className="action" onClick={onAction}>
            {action}
          </button>
        )}
      </div>
      <div className="panel-body">{children}</div>
    </div>
  );
}
