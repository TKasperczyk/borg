import { z } from "zod";

import { DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET } from "../../cognition/deliberation/constants.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../../memory/common/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  openQuestionIdSchema,
  type OpenQuestion,
  type OpenQuestionRumination,
} from "../../memory/self/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { OpenQuestionId } from "../../util/ids.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import type { ToolDefinition } from "../dispatcher.js";

const RUMINATION_PAYLOAD_STATUSES = ["exact", "check_not_completed_budget"] as const;
const DEFAULT_RUMINATIONS_LIMIT = 10;
const MAX_RUMINATIONS_LIMIT = 50;

const openQuestionsRuminationsInputSchema = z
  .object({
    since: z.iso.datetime({ offset: true }),
    until: z.iso.datetime({ offset: true }),
    open_question_id: openQuestionIdSchema.optional(),
    limit: z.number().int().positive().max(MAX_RUMINATIONS_LIMIT).optional(),
  })
  .strict()
  .superRefine((input, context) => {
    if (Date.parse(input.since) > Date.parse(input.until)) {
      context.addIssue({
        code: "custom",
        message: "since must be earlier than or equal to until",
        path: ["since"],
      });
    }
  });

const ruminationForCognitionSchema = z
  .object({
    handle: z.string().min(1),
    open_question_id: openQuestionIdSchema,
    question: z.string().nullable(),
    question_status: z.string().nullable(),
    question_unresolved_rumination_ticks: z.number().int().nonnegative().nullable(),
    note: z.string().nullable(),
    payload_status: z.enum(RUMINATION_PAYLOAD_STATUSES),
    payload_included_chars: z.number().int().nonnegative(),
    payload_total_chars: z.number().int().nonnegative(),
    tensions: z.array(z.string()),
    connected_open_question_ids: z.array(openQuestionIdSchema),
    source_process: z.string().min(1),
    created_at: z.number().int().finite(),
    created_at_iso: z.iso.datetime({ offset: true }),
    relative_age: z.string().min(1),
    disclosure: z.string().min(1),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  })
  .strict();

const openQuestionsRuminationsOutputSchema = z
  .object({
    ruminations: z.array(ruminationForCognitionSchema),
    has_more: z.boolean(),
  })
  .strict();

export type OpenQuestionsRuminationsToolOptions = {
  listRuminations: (input: {
    sinceMs: number;
    untilMs: number;
    openQuestionId?: OpenQuestionId;
    limit: number;
  }) => OpenQuestionRumination[] | Promise<OpenQuestionRumination[]>;
  getOpenQuestion: (id: OpenQuestionId) => OpenQuestion | null | Promise<OpenQuestion | null>;
  clock?: Clock;
};

type RuminationForCognition = z.infer<typeof ruminationForCognitionSchema>;
type OpenQuestionsRuminationsOutput = z.infer<typeof openQuestionsRuminationsOutputSchema>;

function resultFitsBudget(output: OpenQuestionsRuminationsOutput, maxTokens: number): boolean {
  return estimatePromptTokens(JSON.stringify(output)) <= maxTokens;
}

function withoutPayload(row: RuminationForCognition): RuminationForCognition {
  return {
    ...row,
    question: null,
    note: null,
    payload_status: "check_not_completed_budget",
    payload_included_chars: 0,
  };
}

export function createOpenQuestionsRuminationsTool(
  options: OpenQuestionsRuminationsToolOptions,
): ToolDefinition<
  z.infer<typeof openQuestionsRuminationsInputSchema>,
  z.infer<typeof openQuestionsRuminationsOutputSchema>
> {
  const clock = options.clock ?? new SystemClock();

  return {
    name: "tool.openQuestions.ruminations",
    description:
      "Browse the rumination notes my offline mind-maintenance wrote against my open questions, by inclusive created-at range and optionally one question id. Notes survive the question closing, so this reaches questions I later resolved and questions the loop abandoned for me. A question is abandoned that way when its still-open passes reach the no-traction threshold and no episode created after it cites it and no action against it is active: a deterministic dismissal taken without a model call, so for those rows these notes are the only record of the reasoning. question_unresolved_rumination_ticks is that counter as the store holds it now, for the question rather than for this note: a note that narrates which pass it is, is narrating, and the counter is what the dismissal actually reads. It has no text query: I choose the dates and the question.",
    menuSummary:
      "Browse my offline rumination notes on open questions by created-at range, including questions that have since resolved or been abandoned.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: openQuestionsRuminationsInputSchema,
    outputSchema: openQuestionsRuminationsOutputSchema,
    async invoke(input) {
      const sinceMs = Date.parse(input.since);
      const untilMs = Date.parse(input.until);
      const limit = input.limit ?? DEFAULT_RUMINATIONS_LIMIT;
      const candidates = await options.listRuminations({
        sinceMs,
        untilMs,
        ...(input.open_question_id === undefined
          ? {}
          : { openQuestionId: input.open_question_id }),
        limit: limit + 1,
      });
      const pageCandidates = candidates.slice(0, limit);
      const moreInStore = candidates.length > pageCandidates.length;
      const nowMs = clock.now();
      const ruminations: RuminationForCognition[] = [];

      for (let index = 0; index < pageCandidates.length; index += 1) {
        const candidate = pageCandidates[index]!;
        const question = await options.getOpenQuestion(candidate.open_question_id);
        // The note is my own offline reasoning, so it is self-private; the question it was written
        // against carries its own disclosure label. Combine both and let the combination fail closed
        // rather than letting the note's label stand in for the question's.
        const label = combineMemoryDisclosureLabels([
          selfPrivateMemoryDisclosureLabel(),
          question === null
            ? unknownMemoryDisclosureLabel()
            : openQuestionMemoryDisclosureLabel(question),
        ]);
        const payloadChars = candidate.note.length + (question?.question.length ?? 0);
        const row: RuminationForCognition = {
          handle: `rumination:${candidate.id}`,
          open_question_id: candidate.open_question_id,
          question: question?.question ?? null,
          question_status: question?.status ?? null,
          question_unresolved_rumination_ticks: question?.unresolved_rumination_ticks ?? null,
          note: candidate.note,
          payload_status: "exact",
          payload_included_chars: payloadChars,
          payload_total_chars: payloadChars,
          tensions: candidate.tensions,
          connected_open_question_ids: candidate.connected_open_question_ids,
          source_process: candidate.source_process,
          created_at: candidate.created_at,
          created_at_iso: new Date(candidate.created_at).toISOString(),
          relative_age: formatRelativeAge(candidate.created_at, nowMs),
          ...memoryDisclosurePayloadFields(label),
        };
        const hasMoreAfterCandidate = index + 1 < pageCandidates.length || moreInStore;
        const exactOutput = {
          ruminations: [...ruminations, row],
          has_more: hasMoreAfterCandidate,
        } satisfies OpenQuestionsRuminationsOutput;

        if (resultFitsBudget(exactOutput, DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET)) {
          ruminations.push(row);
          continue;
        }

        if (ruminations.length > 0) {
          return { ruminations, has_more: true };
        }

        return { ruminations: [withoutPayload(row)], has_more: hasMoreAfterCandidate };
      }

      return {
        ruminations,
        has_more: ruminations.length < pageCandidates.length || moreInStore,
      };
    },
  };
}
