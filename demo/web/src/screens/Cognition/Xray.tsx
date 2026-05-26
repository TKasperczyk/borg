import type { TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";
import { FlowChart } from "./FlowChart";

export type XrayProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  terminalOutcome: TurnTerminalOutcome | null;
};

export function Xray({ phases, activeTurnId, tokenTextByPhase, terminalOutcome }: XrayProps) {
  return (
    <div className="xray">
      <FlowChart
        phases={phases}
        activeTurnId={activeTurnId}
        tokenTextByPhase={tokenTextByPhase}
        terminalOutcome={terminalOutcome}
      />
    </div>
  );
}
