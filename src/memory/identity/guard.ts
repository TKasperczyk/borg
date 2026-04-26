import { isEpisodeProvenance, type Provenance } from "../common/provenance.js";

export type IdentityGuardState = {
  state?: "candidate" | "established";
};

export type IdentityGuardDecision = {
  allowed: boolean;
  requires_review: boolean;
};

export class IdentityGuard {
  evaluateChange(input: {
    current: IdentityGuardState | null;
    provenance: Provenance;
    throughReview?: boolean;
  }): IdentityGuardDecision {
    if (input.current === null) {
      return {
        allowed: true,
        requires_review: false,
      };
    }

    if (input.throughReview === true) {
      return {
        allowed: true,
        requires_review: false,
      };
    }

    if (input.current.state !== "established") {
      return {
        allowed: true,
        requires_review: false,
      };
    }

    if (isEpisodeProvenance(input.provenance)) {
      return {
        allowed: true,
        requires_review: false,
      };
    }

    return {
      allowed: false,
      requires_review: true,
    };
  }
}
