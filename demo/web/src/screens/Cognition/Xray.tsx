import type { TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";
import { FlowChart } from "./FlowChart";

export type XrayProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
};

export function Xray({
  phases,
  activeTurnId,
  tokenTextByPhase,
  terminalOutcome,
  delibPath,
  finalAttempt,
}: XrayProps) {
  return (
    <div className="xray">
      <FlowChart
        phases={phases}
        activeTurnId={activeTurnId}
        tokenTextByPhase={tokenTextByPhase}
        terminalOutcome={terminalOutcome}
        delibPath={delibPath}
        finalAttempt={finalAttempt}
      />
    </div>
  );
}
