import { useEffect, useRef, useState } from "react";

import type { EvidenceLedger, SharedStateEntry } from "../../api/types";
import type { PhaseState, TailEvent } from "../../hooks/use-turn-stream";
import { LedgerView } from "./LedgerView";
import { PhasesView } from "./PhasesView";
import { SharedSnippet } from "./SharedSnippet";
import { TailView } from "./TailView";

export type XrayTab = "phases" | "ledger" | "shared" | "tail";

export type XrayProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  ledger?: EvidenceLedger;
  sharedEntries: readonly SharedStateEntry[];
  audience: string;
  tailEvents: readonly TailEvent[];
};

export function Xray({ phases, activeTurnId, ledger, sharedEntries, audience, tailEvents }: XrayProps) {
  const [xrayTab, setXrayTab] = useState<XrayTab>("phases");
  const tailRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (tailRef.current !== null) {
      tailRef.current.scrollTop = 0;
    }
  }, [tailEvents]);

  return (
    <div className="xray">
      <div className="xray-tabs">
        {[
          { id: "phases", label: "lifecycle" },
          { id: "ledger", label: "ledger" },
          { id: "shared", label: "shared" },
          { id: "tail", label: "tail" }
        ].map((tab) => (
          <div
            key={tab.id}
            className={`xray-tab ${xrayTab === tab.id ? "active" : ""}`}
            onClick={() => setXrayTab(tab.id as XrayTab)}
          >
            {tab.label}
          </div>
        ))}
      </div>
      <div className="xray-body">
        {xrayTab === "phases" ? <PhasesView phases={phases} activeTurnId={activeTurnId} /> : null}
        {xrayTab === "ledger" ? (
          <LedgerView turnId={activeTurnId} cachedLedger={ledger} active={xrayTab === "ledger"} />
        ) : null}
        {xrayTab === "shared" ? <SharedSnippet entries={sharedEntries} audience={audience} /> : null}
        {xrayTab === "tail" ? <TailView events={tailEvents} ref={tailRef} /> : null}
      </div>
    </div>
  );
}
