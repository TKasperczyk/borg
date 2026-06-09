import { Component, type ErrorInfo, type ReactNode } from "react";

import { ErrorState } from "./ErrorState";

type AppErrorBoundaryProps = {
  children: ReactNode;
  resetKey?: unknown;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
};

type AppErrorBoundaryState = {
  error: Error | null;
};

export class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error(error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  componentDidUpdate(prevProps: AppErrorBoundaryProps): void {
    if (prevProps.resetKey !== this.props.resetKey && this.state.error !== null) {
      this.setState({ error: null });
    }
  }

  private retry = (): void => {
    this.setState({ error: null });
  };

  private reload = (): void => {
    window.location.reload();
  };

  render() {
    if (this.state.error !== null) {
      return (
        <ErrorState>
          <div role="alert" className="app-error-boundary">
            <div>screen crashed</div>
            <div className="dim">{this.state.error.message}</div>
            <div className="operator-actions" style={{ marginTop: 10 }}>
              <button type="button" className="btn sm" onClick={this.retry}>
                retry
              </button>
              <button type="button" className="btn sm ghost" onClick={this.reload}>
                reload
              </button>
            </div>
          </div>
        </ErrorState>
      );
    }

    return this.props.children;
  }
}
