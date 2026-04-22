import { isEpisodeProvenance, type Provenance } from "../common/provenance.js";

export type IdentityGuardState = {
  state?: "candidate" | "established";
  provenance?: Provenance;
};

export type IdentityGuardDecision = {
  allowed: boolean;
  requires_review: boolean;
  overwrite_without_review: boolean;
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
        overwrite_without_review: false,
      };
    }

    if (input.throughReview === true) {
      return {
        allowed: true,
        requires_review: false,
        overwrite_without_review: false,
      };
    }

    if (
      input.current.state !== "established" ||
      input.current.provenance === undefined ||
      !isEpisodeProvenance(input.current.provenance)
    ) {
      return {
        allowed: true,
        requires_review: false,
        overwrite_without_review: false,
      };
    }

    if (input.provenance.kind === "system" || input.provenance.kind === "offline") {
      return {
        allowed: true,
        requires_review: false,
        overwrite_without_review: true,
      };
    }

    return {
      allowed: false,
      requires_review: true,
      overwrite_without_review: false,
    };
  }
}
