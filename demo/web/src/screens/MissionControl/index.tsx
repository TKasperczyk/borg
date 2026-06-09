import { useMemo } from "react";

import { Orrery } from "../../components/orrery/Orrery";
import { useOrreryData, type OrreryTurnInput } from "../../components/orrery/useOrreryData";
import { useInspector } from "../../components/Inspector/inspector-context";
import type { RouteId } from "../../routes";

export type MissionControlScreenProps = {
  turnStream: OrreryTurnInput;
  onNavigate: (view: RouteId) => void;
};

export function MissionControlScreen({ turnStream, onNavigate }: MissionControlScreenProps) {
  const inspector = useInspector();
  const turn = useMemo<OrreryTurnInput>(
    () => ({
      activeTurnId: turnStream.activeTurnId,
      lastPhase: turnStream.lastPhase,
      running: turnStream.running,
      terminalOutcome: turnStream.terminalOutcome,
    }),
    [turnStream.activeTurnId, turnStream.lastPhase, turnStream.running, turnStream.terminalOutcome],
  );
  const data = useOrreryData(turn);

  return (
    <div className="orr-mission" data-testid="mission-control-screen">
      {/* Fleet and attention-queue columns land in the next Mission Control slice. */}
      <aside className="orr-mission-column" aria-label="fleet placeholder" />
      <main className="orr-mission-main">
        <Orrery size="full" data={data} onNavigate={onNavigate} onInspect={inspector.openObject} />
      </main>
      <aside className="orr-mission-column" aria-label="attention placeholder" />
    </div>
  );
}
