import { parseReviewProvenance } from "../../common/provenance.js";
import type { Episode, EpisodePatch } from "../../episodic/index.js";
import type { EpisodeId } from "../../../util/ids.js";
import type { JsonValue } from "../../../util/json-value.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";

type EpisodeFieldValues = Record<string, JsonValue>;

function jsonFieldValue(value: unknown): JsonValue {
  return JSON.parse(JSON.stringify(value ?? null) ?? "null") as JsonValue;
}

// The embedding is left out: these patches never carry one, and a vector would
// dwarf the fields the event exists to keep.
function changedEpisodeFields(
  before: Episode,
  after: Episode,
): { old: EpisodeFieldValues; new: EpisodeFieldValues } {
  const oldFields: EpisodeFieldValues = {};
  const newFields: EpisodeFieldValues = {};
  const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();

  for (const key of keys) {
    if (key === "embedding") {
      continue;
    }

    const oldValue = jsonFieldValue(before[key as keyof Episode]);
    const newValue = jsonFieldValue(after[key as keyof Episode]);

    if (JSON.stringify(oldValue) !== JSON.stringify(newValue)) {
      oldFields[key] = oldValue;
      newFields[key] = newValue;
    }
  }

  return { old: oldFields, new: newFields };
}

// An accepted repair overwrites episode fields in place, and the review item keeps
// only the patch. Record every changed field before and after, as correction
// applies do, so the replaced values stay recoverable from identity_events.
export async function updateEpisodeFromReview(input: {
  ctx: ReviewHandlerContext;
  item: ReviewQueueItem;
  episodeId: EpisodeId;
  patch: EpisodePatch;
  action: string;
}): Promise<Episode | null> {
  const { ctx, item, episodeId, patch, action } = input;
  const episodicRepository = ctx.episodicRepository!;
  const identityEvents = ctx.identityEventRepository;
  const before =
    identityEvents === undefined
      ? null
      : await episodicRepository.get(episodeId, { includeArchived: true });
  const updated = await episodicRepository.update(episodeId, patch);

  if (updated === null || identityEvents === undefined || before === null) {
    return updated;
  }

  const existing = identityEvents.findByReviewKey({
    reviewItemId: item.id,
    recordType: "episode",
    recordId: episodeId,
    action,
  });

  if (existing === null) {
    const changed = changedEpisodeFields(before, updated);

    identityEvents.record({
      record_type: "episode",
      record_id: episodeId,
      action,
      old_value: changed.old,
      new_value: changed.new,
      reason: item.reason,
      provenance: parseReviewProvenance(item.refs),
      review_item_id: item.id,
    });
  }

  return updated;
}
