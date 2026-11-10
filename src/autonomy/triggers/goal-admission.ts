import type { GoalAdmission, GoalInputEvent } from "../../executive/goal-presentation.js";
import type { SourceStreamAudienceDisclosureResolver } from "../../memory/common/source-stream-audience-disclosure.js";
import {
  memoryDisclosureLabelFromMetadata,
  unknownMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import type { GoalRecord } from "../../memory/self/types.js";
import type { StreamEntry } from "../../stream/types.js";

type GoalAdmissionInputPayload =
  | (Exclude<GoalInputEvent, { kind: "associated_inbound" }> &
      ReturnType<typeof memoryDisclosurePayloadFields>)
  | (Extract<GoalInputEvent, { kind: "associated_inbound" }> & {
      timestamp: number | null;
      sender_entity_id: StreamEntry["sender_entity_id"];
      session_id: StreamEntry["session_id"] | null;
      audience: string | null;
    } & ReturnType<typeof memoryDisclosurePayloadFields>);

export type GoalAdmissionPayload = Omit<GoalAdmission, "new_inputs"> & {
  new_inputs: GoalAdmissionInputPayload[];
};

/** Render known source handles only; missing records remain explicit, unknown-labelled handles. */
export function goalAdmissionPayload(
  admission: GoalAdmission,
  goal: GoalRecord,
  resolver?: SourceStreamAudienceDisclosureResolver,
): GoalAdmissionPayload {
  const entryIds = admission.new_inputs.flatMap((event) =>
    event.kind === "associated_inbound" ? [event.entry_id] : [],
  );
  const sources = entryIds.length === 0 ? undefined : resolver?.readStreamEntries(entryIds);
  return {
    ...admission,
    new_inputs: admission.new_inputs.map((event): GoalAdmissionInputPayload => {
      if (event.kind === "associated_inbound") {
        const source = sources?.get(event.entry_id);
        const entry = source?.entry;
        return {
          ...event,
          timestamp: entry?.timestamp ?? null,
          sender_entity_id: source?.indexRecord.sender_entity_id ?? null,
          session_id: entry?.session_id ?? null,
          audience: entry?.audience ?? null,
          ...memoryDisclosurePayloadFields(
            entry === undefined
              ? unknownMemoryDisclosureLabel()
              : (resolver?.resolveStreamEntryLabel(entry) ?? unknownMemoryDisclosureLabel()),
          ),
        };
      }
      return {
        ...event,
        ...memoryDisclosurePayloadFields(
          event.kind === "block_ended"
            ? (memoryDisclosureLabelFromMetadata(event.disclosure_label) ??
                unknownMemoryDisclosureLabel())
            : goalMemoryDisclosureLabel(goal),
        ),
      };
    }),
  };
}
