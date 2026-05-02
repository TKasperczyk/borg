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
    changeKind?: "open_question_resolution";
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

    if (
      input.changeKind === "open_question_resolution" &&
      input.provenance.kind === "online_reflector" &&
      (input.provenance.evidence_episode_ids.length > 0 ||
        input.provenance.evidence_stream_entry_ids.length > 0)
    ) {
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
