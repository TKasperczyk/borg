import type { TurnOrigin } from "../../contracts/cognitive-contracts.js";
import type { StreamEntryIndexRepository } from "../../stream/index.js";
import { StorageError, ToolError } from "../../util/errors.js";
import { episodeIdHelpers, streamEntryIdHelpers } from "../../util/ids.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  unknownMemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import type { EpisodicRepository } from "../episodic/index.js";
import type { IdentityService } from "../identity/service.js";
import { openQuestionResolutionCorrectionInputSchema } from "../self/types.js";
import {
  RESOLUTION_CORRECTION_TOOL_NAME,
  openQuestionResolutionIdentity,
  type OpenQuestionResolutionIdentity,
} from "../self/resolution-corrections.js";
import type { OpenQuestionsRepository } from "../self/open-questions.js";
import type { z } from "zod";

export type CorrectOpenQuestionResolutionIdentityService = Pick<
  IdentityService,
  "correctOpenQuestionResolution"
>;

export async function correctOpenQuestionResolutionThroughIdentityService(options: {
  input: z.infer<typeof openQuestionResolutionCorrectionInputSchema>;
  identityService: CorrectOpenQuestionResolutionIdentityService;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "get">;
  expectedResolution?: OpenQuestionResolutionIdentity;
  episodicRepository: Pick<EpisodicRepository, "getMany">;
  entryIndex: Pick<StreamEntryIndexRepository, "lookupEntriesById">;
  turnId?: string;
  turnOrigin: TurnOrigin;
  path: "tool" | "reflector";
}) {
  const { open_question_id, ...correction } = openQuestionResolutionCorrectionInputSchema.parse(
    options.input,
  );
  // Direct calls capture before the first await. Reflection supplies the identity
  // captured from its presented block, before the model call instead.
  let expectedResolution = options.expectedResolution;
  if (expectedResolution === undefined) {
    const target = options.openQuestionsRepository.get(open_question_id);
    if (target === null) {
      throw new StorageError(`Unknown open question id: ${open_question_id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }
    if (target.status !== "resolved") {
      throw new StorageError("Corrections require a resolved question. Resolve it first.", {
        code: "OPEN_QUESTION_CORRECTION_NOT_RESOLVED",
      });
    }
    expectedResolution = openQuestionResolutionIdentity(target);
  }
  const ids = [...new Set(correction.evidence.map((item) => item.id))];
  const episodeIds = ids.filter(episodeIdHelpers.is);
  const streamEntryIds = ids.filter(streamEntryIdHelpers.is);
  // Like resolve: archived episodes and inactive stream entries still exist.
  const episodes = await options.episodicRepository.getMany(episodeIds);
  const foundEpisodes = new Set(episodes.map((episode) => episode.id));
  const streamEntries = options.entryIndex.lookupEntriesById(streamEntryIds);
  const missingIds = [
    ...episodeIds.filter((id) => !foundEpisodes.has(id)),
    ...streamEntryIds.filter((id) => !streamEntries.has(id)),
  ];
  if (missingIds.length > 0) {
    throw new ToolError(
      `Citation ids do not name stored records: ${missingIds.join(", ")}. Re-cite or remove these citations.`,
      { code: "OPEN_QUESTION_CITATION_NOT_FOUND" },
    );
  }
  const disclosureLabel = combineMemoryDisclosureLabels([
    ...episodes.map(memoryDisclosureLabelFromEpisodeAccess),
    ...streamEntryIds.map(() => unknownMemoryDisclosureLabel()),
  ]);
  return options.identityService.correctOpenQuestionResolution(
    open_question_id,
    correction,
    {
      author: "borg",
      turn_id: options.turnId ?? null,
      turn_kind: options.turnOrigin,
      path: options.path,
      provenance: {
        kind: "online_reflector",
        process: options.path === "tool" ? RESOLUTION_CORRECTION_TOOL_NAME : "reflector",
        evidence_episode_ids: episodeIds,
        evidence_stream_entry_ids: streamEntryIds,
      },
    },
    disclosureLabel,
    { expectedResolution },
  );
}
