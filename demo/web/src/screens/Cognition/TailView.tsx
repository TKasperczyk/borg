import { forwardRef } from "react";

import { Empty } from "../../components/Empty";
import type { TailEvent } from "../../hooks/use-turn-stream";

export type TailViewProps = {
  events: readonly TailEvent[];
};

export const TailView = forwardRef<HTMLDivElement, TailViewProps>(function TailView({ events }, ref) {
  return (
    <div className="tail" ref={ref}>
      {events.length === 0 ? (
        <Empty>live events will appear here</Empty>
      ) : (
        events.map((event) => (
          <div key={event.id} className={`tail-row kind-${event.kind}${event.isNew ? " new" : ""}`}>
            <span className="t">{event.ts}</span>
            <span className="k">{event.kind}</span>
            <span className="v">{event.body}</span>
          </div>
        ))
      )}
    </div>
  );
});
