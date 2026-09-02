import { isRecentLivedExperienceSpineKind } from "../../../memory/activity/index.js";
import type { BuilderSectionContext } from "../builder-context.js";
import { buildRecentLivedExperienceLedgerEntry } from "../recent-lived-experience.js";
import { addEntry, setSectionFraming } from "../section-buckets.js";
import { SELF_DECISION_LABEL_SCOPE_FRAMING } from "../types.js";

const RECENT_LIVED_EXPERIENCE_FRAMING =
  "Recent lived experience is a session-agnostic chronological surface. It shows density, labels, and self-private disclosure metadata for intervening activity, never verbatim other-audience message text. " +
  `${SELF_DECISION_LABEL_SCOPE_FRAMING} This section carries no action records, so a label here is the whole of what it says and not the whole of what the turn did.`;

export function addRecentLivedExperienceSection(context: BuilderSectionContext): void {
  const rows = context.input.recentLivedExperience ?? [];

  if (context.input.renderRecentLivedExperience !== true || rows.length === 0) {
    return;
  }

  setSectionFraming(context.buckets, "recent_lived_experience", {
    text: RECENT_LIVED_EXPERIENCE_FRAMING,
    counts: {
      // Was `entries`, which named the same number but read as one subset among the others
      // rather than as the population the other two are subsets of.
      rows_assembled: rows.length,
      spine: rows.filter((row) => isRecentLivedExperienceSpineKind(row.kind)).length,
      density: rows.filter(
        (row) =>
          row.kind === "cross_session_activity_density" || row.kind === "self_decision_density",
      ).length,
    },
  });

  for (const [index, row] of rows.entries()) {
    addEntry(
      context.buckets,
      "recent_lived_experience",
      buildRecentLivedExperienceLedgerEntry({
        row,
        index,
        audienceEntityId: context.input.audienceEntityId,
      }),
    );
  }
}
