import { useState } from "react";

import type { EvidenceLedger, TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";
import { FlowChart } from "./FlowChart";
import { LedgerView } from "./LedgerView";

export type XrayProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  detailByPhase: Map<string, string[]>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  cachedLedger?: EvidenceLedger;
  audience: string;
  tracePlaceholder?: string | null;
};

export function Xray({
  phases,
  activeTurnId,
  tokenTextByPhase,
  detailByPhase,
  terminalOutcome,
  delibPath,
  finalAttempt,
  cachedLedger,
  audience,
  tracePlaceholder = null,
}: XrayProps) {
  const [view, setView] = useState<"flow" | "ledger">("flow");

  return (
    <div className="xray">
      {tracePlaceholder !== null ? (
        <div className="xray-placeholder" role="status">
          <div className="xray-placeholder-title">historical trace</div>
          <p>{tracePlaceholder}</p>
        </div>
      ) : (
        <>
          <div className="xray-tabs" role="tablist" aria-label="Cognition replay views">
            <button
              type="button"
              role="tab"
              aria-selected={view === "flow"}
              className={`xray-tab ${view === "flow" ? "active" : ""}`.trim()}
              onClick={() => setView("flow")}
            >
              flow
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={view === "ledger"}
              className={`xray-tab ${view === "ledger" ? "active" : ""}`.trim()}
              onClick={() => setView("ledger")}
            >
              ledger
            </button>
          </div>
          <div className="xray-body">
            {view === "flow" ? (
              <FlowChart
                phases={phases}
                activeTurnId={activeTurnId}
                tokenTextByPhase={tokenTextByPhase}
                detailByPhase={detailByPhase}
                terminalOutcome={terminalOutcome}
                delibPath={delibPath}
                finalAttempt={finalAttempt}
              />
            ) : (
              <LedgerView
                turnId={activeTurnId}
                cachedLedger={cachedLedger}
                active={activeTurnId !== null}
                audience={audience}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
