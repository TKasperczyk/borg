import type { BuilderSectionContext } from "../builder-context.js";
import { DISCOURSE_TRUST_RANK, addEntry } from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";

export function addDiscourseStateSection(context: BuilderSectionContext): void {
  const discourseState = context.input.workingMemory.discourse_state;

  addEntry(context.buckets, "closure_discourse_state", {
    id: "discourse_state:working_memory",
    source_type: "system_metadata",
    session_scope: "current_session",
    actor: "system",
    trust_rank: DISCOURSE_TRUST_RANK,
    text: `mode=${context.input.workingMemory.mode}; turn_counter=${context.input.workingMemory.turn_counter}`,
    state: context.input.workingMemory.mode ?? undefined,
    taint: "none",
  });

  if (discourseState?.closure_loop !== undefined && discourseState.closure_loop !== null) {
    addEntry(context.buckets, "closure_discourse_state", {
      id: "discourse_state:closure_loop",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: DISCOURSE_TRUST_RANK,
      text: discourseState.closure_loop.reason,
      value: discourseState.closure_loop.source_stream_entry_ids.join(", "),
      state: discourseState.closure_loop.status,
      taint: "none",
      ...persistenceClassFromProvenance(
        { streamEntryIds: discourseState.closure_loop.source_stream_entry_ids },
        context.resolver,
      ),
    });
  }

  const stopState = discourseState?.stop_until_substantive_content;

  if (stopState !== undefined && stopState !== null) {
    addEntry(context.buckets, "closure_discourse_state", {
      id: "discourse_constraint:stop_until_substantive_content",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: DISCOURSE_TRUST_RANK,
      text: stopState.reason,
      value: stopState.provenance,
      state: "active",
      taint: "none",
    });
  }
}
