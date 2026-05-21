import type {
  RelationshipEvidenceStreamEntryTrustValidator,
  SyncRelationshipEvidenceStreamEntryTrustValidator,
} from "../memory/source-trust.js";
import { streamEntryIdHelpers, type StreamEntryId } from "../util/ids.js";
import { participantRosterRelationalSlotIds, type ParticipantRoster } from "./perception/index.js";
import { protectedRelationshipLabelsInText } from "./prompts/relationship-labels.js";

export type RelationshipEvidenceRejection = {
  id: string;
  reason:
    | "invalid_id"
    | "not_in_source_bundle"
    | "missing"
    | "not_user_msg"
    | "untrusted"
    | "unavailable";
};

export type RelationshipLabelGroundingCheck = {
  protectedLabels: string[];
  grounded: boolean;
  acceptedRelationalSlotIds: string[];
  acceptedStreamEntryIds: StreamEntryId[];
  rejectedRelationalSlotIds: string[];
  rejectedStreamEntryIds: RelationshipEvidenceRejection[];
};

type RelationshipLabelGroundingInput = {
  text: string;
  participantRoster?: ParticipantRoster | null;
  relationshipEvidenceRelationalSlotIds?: readonly string[];
  relationshipEvidenceStreamEntryIds?: readonly string[];
  allowedRelationshipEvidenceStreamEntryIds?: ReadonlySet<StreamEntryId> | null;
};

type RelationshipLabelGroundingBase = Omit<RelationshipLabelGroundingCheck, "grounded"> & {
  streamEntryIdsForTrust: StreamEntryId[];
};

function relationshipLabelGroundingBase(
  input: RelationshipLabelGroundingInput,
): RelationshipLabelGroundingBase {
  const protectedLabels = [...new Set(protectedRelationshipLabelsInText(input.text))];
  const suppliedRelationalSlotIds = input.relationshipEvidenceRelationalSlotIds ?? [];
  const suppliedStreamEntryIds = input.relationshipEvidenceStreamEntryIds ?? [];
  const groundedRelationalSlotIds = participantRosterRelationalSlotIds(input.participantRoster);
  const acceptedRelationalSlotIds: string[] = [];
  const rejectedRelationalSlotIds: string[] = [];
  const acceptedStreamEntryIds: StreamEntryId[] = [];
  const rejectedStreamEntryIds: RelationshipEvidenceRejection[] = [];
  const streamEntryIdsForTrust: StreamEntryId[] = [];

  for (const id of suppliedRelationalSlotIds) {
    if (groundedRelationalSlotIds.has(id)) {
      acceptedRelationalSlotIds.push(id);
    } else {
      rejectedRelationalSlotIds.push(id);
    }
  }

  for (const id of suppliedStreamEntryIds) {
    if (!streamEntryIdHelpers.is(id)) {
      rejectedStreamEntryIds.push({
        id,
        reason: "invalid_id",
      });
      continue;
    }

    if (
      input.allowedRelationshipEvidenceStreamEntryIds !== undefined &&
      input.allowedRelationshipEvidenceStreamEntryIds !== null &&
      !input.allowedRelationshipEvidenceStreamEntryIds.has(id)
    ) {
      rejectedStreamEntryIds.push({
        id,
        reason: "not_in_source_bundle",
      });
      continue;
    }

    streamEntryIdsForTrust.push(id);
  }

  return {
    protectedLabels,
    acceptedRelationalSlotIds,
    acceptedStreamEntryIds,
    rejectedRelationalSlotIds,
    rejectedStreamEntryIds,
    streamEntryIdsForTrust,
  };
}

function finishRelationshipLabelGrounding(
  base: RelationshipLabelGroundingBase,
): RelationshipLabelGroundingCheck {
  return {
    protectedLabels: base.protectedLabels,
    grounded:
      base.protectedLabels.length === 0 ||
      base.acceptedRelationalSlotIds.length > 0 ||
      base.acceptedStreamEntryIds.length > 0,
    acceptedRelationalSlotIds: base.acceptedRelationalSlotIds,
    acceptedStreamEntryIds: base.acceptedStreamEntryIds,
    rejectedRelationalSlotIds: base.rejectedRelationalSlotIds,
    rejectedStreamEntryIds: base.rejectedStreamEntryIds,
  };
}

export function checkRelationshipLabelGrounding(
  input: RelationshipLabelGroundingInput & {
    relationshipEvidenceStreamEntryTrust?: SyncRelationshipEvidenceStreamEntryTrustValidator;
  },
): RelationshipLabelGroundingCheck {
  const base = relationshipLabelGroundingBase(input);

  for (const id of base.streamEntryIdsForTrust) {
    const trust = input.relationshipEvidenceStreamEntryTrust?.(id);

    if (trust?.allowed === true) {
      base.acceptedStreamEntryIds.push(id);
      continue;
    }

    base.rejectedStreamEntryIds.push({
      id,
      reason: trust?.reason ?? "unavailable",
    });
  }

  return finishRelationshipLabelGrounding(base);
}

export async function checkRelationshipLabelGroundingAsync(
  input: RelationshipLabelGroundingInput & {
    relationshipEvidenceStreamEntryTrust?: RelationshipEvidenceStreamEntryTrustValidator;
  },
): Promise<RelationshipLabelGroundingCheck> {
  const base = relationshipLabelGroundingBase(input);

  for (const id of base.streamEntryIdsForTrust) {
    const trust = await input.relationshipEvidenceStreamEntryTrust?.(id);

    if (trust?.allowed === true) {
      base.acceptedStreamEntryIds.push(id);
      continue;
    }

    base.rejectedStreamEntryIds.push({
      id,
      reason: trust?.reason ?? "unavailable",
    });
  }

  return finishRelationshipLabelGrounding(base);
}
