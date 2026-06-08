import type {
  RelationshipEvidenceStreamEntryTrustValidator,
  SyncRelationshipEvidenceStreamEntryTrustValidator,
} from "../source-trust.js";
import { streamEntryIdHelpers, type StreamEntryId } from "../../util/ids.js";
import type { RelationshipClaim } from "./relationship-claims.js";
import {
  participantRosterRelationalSlotIds,
  type ParticipantRosterRelationshipEvidence,
} from "./relationship-evidence.js";

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

export type RelationshipClaimGroundingCheck = {
  claims: RelationshipClaim[];
  ungroundedClaims: RelationshipClaim[];
  grounded: boolean;
  acceptedRelationalSlotIds: string[];
  acceptedStreamEntryIds: StreamEntryId[];
  rejectedRelationalSlotIds: string[];
  rejectedStreamEntryIds: RelationshipEvidenceRejection[];
};

type RelationshipClaimGroundingInput = {
  claims?: readonly RelationshipClaim[];
  participantRoster?: ParticipantRosterRelationshipEvidence | null;
  allowedRelationshipEvidenceStreamEntryIds?: ReadonlySet<StreamEntryId> | null;
};

type ClaimEvidenceCheck = {
  claim: RelationshipClaim;
  acceptedRelationalSlotIds: string[];
  acceptedStreamEntryIds: StreamEntryId[];
  rejectedRelationalSlotIds: string[];
  rejectedStreamEntryIds: RelationshipEvidenceRejection[];
  streamEntryIdsForTrust: StreamEntryId[];
};

type RelationshipClaimGroundingBase = Omit<
  RelationshipClaimGroundingCheck,
  "grounded" | "ungroundedClaims"
> & {
  claimChecks: ClaimEvidenceCheck[];
};

function pushUnique<T>(target: T[], value: T): void {
  if (!target.some((existing) => existing === value)) {
    target.push(value);
  }
}

function relationshipClaimGroundingBase(
  input: RelationshipClaimGroundingInput,
): RelationshipClaimGroundingBase {
  const claims = [...(input.claims ?? [])];
  const groundedRelationalSlotIds = participantRosterRelationalSlotIds(input.participantRoster);
  const claimChecks: ClaimEvidenceCheck[] = [];
  const acceptedRelationalSlotIds: string[] = [];
  const acceptedStreamEntryIds: StreamEntryId[] = [];
  const rejectedRelationalSlotIds: string[] = [];
  const rejectedStreamEntryIds: RelationshipEvidenceRejection[] = [];

  for (const claim of claims) {
    const claimCheck: ClaimEvidenceCheck = {
      claim,
      acceptedRelationalSlotIds: [],
      acceptedStreamEntryIds: [],
      rejectedRelationalSlotIds: [],
      rejectedStreamEntryIds: [],
      streamEntryIdsForTrust: [],
    };

    for (const id of claim.evidence_relational_slot_ids) {
      if (groundedRelationalSlotIds.has(id)) {
        pushUnique(claimCheck.acceptedRelationalSlotIds, id);
        pushUnique(acceptedRelationalSlotIds, id);
      } else {
        pushUnique(claimCheck.rejectedRelationalSlotIds, id);
        pushUnique(rejectedRelationalSlotIds, id);
      }
    }

    for (const id of claim.evidence_stream_entry_ids) {
      if (!streamEntryIdHelpers.is(id)) {
        const rejection = {
          id,
          reason: "invalid_id" as const,
        };
        claimCheck.rejectedStreamEntryIds.push(rejection);
        rejectedStreamEntryIds.push(rejection);
        continue;
      }

      if (
        input.allowedRelationshipEvidenceStreamEntryIds !== undefined &&
        input.allowedRelationshipEvidenceStreamEntryIds !== null &&
        !input.allowedRelationshipEvidenceStreamEntryIds.has(id)
      ) {
        const rejection = {
          id,
          reason: "not_in_source_bundle" as const,
        };
        claimCheck.rejectedStreamEntryIds.push(rejection);
        rejectedStreamEntryIds.push(rejection);
        continue;
      }

      claimCheck.streamEntryIdsForTrust.push(id);
    }

    claimChecks.push(claimCheck);
  }

  return {
    claims,
    claimChecks,
    acceptedRelationalSlotIds,
    acceptedStreamEntryIds,
    rejectedRelationalSlotIds,
    rejectedStreamEntryIds,
  };
}

function finishRelationshipClaimGrounding(
  base: RelationshipClaimGroundingBase,
): RelationshipClaimGroundingCheck {
  const ungroundedClaims = base.claimChecks
    .filter(
      (check) =>
        check.claim.requires_grounding &&
        check.acceptedRelationalSlotIds.length === 0 &&
        check.acceptedStreamEntryIds.length === 0,
    )
    .map((check) => check.claim);

  return {
    claims: base.claims,
    ungroundedClaims,
    grounded: ungroundedClaims.length === 0,
    acceptedRelationalSlotIds: base.acceptedRelationalSlotIds,
    acceptedStreamEntryIds: base.acceptedStreamEntryIds,
    rejectedRelationalSlotIds: base.rejectedRelationalSlotIds,
    rejectedStreamEntryIds: base.rejectedStreamEntryIds,
  };
}

export function checkRelationshipClaimGrounding(
  input: RelationshipClaimGroundingInput & {
    relationshipEvidenceStreamEntryTrust?: SyncRelationshipEvidenceStreamEntryTrustValidator;
  },
): RelationshipClaimGroundingCheck {
  const base = relationshipClaimGroundingBase(input);

  for (const check of base.claimChecks) {
    for (const id of check.streamEntryIdsForTrust) {
      const trust = input.relationshipEvidenceStreamEntryTrust?.(id);

      if (trust?.allowed === true) {
        pushUnique(check.acceptedStreamEntryIds, id);
        pushUnique(base.acceptedStreamEntryIds, id);
        continue;
      }

      const rejection = {
        id,
        reason: trust?.reason ?? "unavailable",
      };
      check.rejectedStreamEntryIds.push(rejection);
      base.rejectedStreamEntryIds.push(rejection);
    }
  }

  return finishRelationshipClaimGrounding(base);
}

export async function checkRelationshipClaimGroundingAsync(
  input: RelationshipClaimGroundingInput & {
    relationshipEvidenceStreamEntryTrust?: RelationshipEvidenceStreamEntryTrustValidator;
  },
): Promise<RelationshipClaimGroundingCheck> {
  const base = relationshipClaimGroundingBase(input);

  for (const check of base.claimChecks) {
    for (const id of check.streamEntryIdsForTrust) {
      const trust = await input.relationshipEvidenceStreamEntryTrust?.(id);

      if (trust?.allowed === true) {
        pushUnique(check.acceptedStreamEntryIds, id);
        pushUnique(base.acceptedStreamEntryIds, id);
        continue;
      }

      const rejection = {
        id,
        reason: trust?.reason ?? "unavailable",
      };
      check.rejectedStreamEntryIds.push(rejection);
      base.rejectedStreamEntryIds.push(rejection);
    }
  }

  return finishRelationshipClaimGrounding(base);
}
