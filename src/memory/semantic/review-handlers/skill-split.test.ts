import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import type { EpisodeId, SkillId } from "../../../util/ids.js";
import {
  type ReviewHandlerContext,
  type ReviewQueueItem,
  type SkillSplitReviewHandler,
  type SkillSplitReviewPayload,
} from "../review-queue.js";
import { createSkillSplitReviewQueueHandler } from "./skill-split.js";

const originalSkillId = "skl_aaaaaaaaaaaaaaaa" as SkillId;
const newSkillId = "skl_bbbbbbbbbbbbbbbb" as SkillId;
const sourceEpisodeId = "ep_aaaaaaaaaaaaaaaa" as EpisodeId;

const payload = {
  target_type: "skill",
  target_id: originalSkillId,
  original_skill_id: originalSkillId,
  proposed_children: [
    {
      label: "Debug comparison",
      problem: "Debug comparison",
      approach: "Compare the failing state.",
      context_stats: [
        {
          skill_id: originalSkillId,
          context_key: "code_debugging:typescript:self",
          alpha: 2,
          beta: 1,
          attempts: 1,
          successes: 1,
          failures: 0,
          last_used: 1_000,
          last_successful: 1_000,
          updated_at: 1_000,
        },
      ],
    },
    {
      label: "Planning comparison",
      problem: "Planning comparison",
      approach: "Compare the roadmap state.",
      context_stats: [
        {
          skill_id: originalSkillId,
          context_key: "planning:roadmap:self",
          alpha: 1,
          beta: 2,
          attempts: 1,
          successes: 0,
          failures: 1,
          last_used: 1_000,
          last_successful: null,
          updated_at: 1_000,
        },
      ],
    },
  ],
  rationale: "The contexts diverge.",
  evidence_summary: {
    source_episode_ids: [sourceEpisodeId],
    divergence: 0.5,
    min_posterior_mean: 0.25,
    max_posterior_mean: 0.75,
    buckets: [
      {
        context_key: "code_debugging:typescript:self",
        posterior_mean: 0.75,
        alpha: 2,
        beta: 1,
        attempts: 1,
        successes: 1,
        failures: 0,
        last_used: 1_000,
        last_successful: 1_000,
      },
      {
        context_key: "planning:roadmap:self",
        posterior_mean: 0.25,
        alpha: 1,
        beta: 2,
        attempts: 1,
        successes: 0,
        failures: 1,
        last_used: 1_000,
        last_successful: null,
      },
    ],
  },
  cooldown: {
    proposed_at: 1_000,
    claimed_at: 900,
    claim_expires_at: 1_800_900,
    split_cooldown_days: 7,
    split_claim_stale_sec: 1_800,
    last_split_attempt_at: null,
    split_failure_count: 0,
    last_split_error: null,
  },
} satisfies SkillSplitReviewPayload;

const item = {
  id: 1,
  kind: "skill_split",
  refs: payload,
  reason: "split proposed",
  created_at: 1_000,
  resolved_at: null,
  resolution: null,
} satisfies ReviewQueueItem;

const ctx = {
  clock: new FixedClock(2_000),
} as unknown as ReviewHandlerContext;

describe("skill split review handler", () => {
  it("uses strict skill split refs and external transaction scope", () => {
    const handler = createSkillSplitReviewQueueHandler({
      accept: vi.fn(),
      reject: vi.fn(),
    } as unknown as SkillSplitReviewHandler);

    expect(handler.refsSchema.parse(payload)).toEqual(payload);
    expect(() => handler.refsSchema.parse({ ...payload, extra: true })).toThrow();
    expect(
      handler.transactionScope({
        item,
        refs: payload,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("external");
    expect([...handler.allowedResolutions]).toEqual(["accept", "reject"]);
  });

  it("records applied accept metadata", async () => {
    const handler = createSkillSplitReviewQueueHandler({
      accept: vi.fn(() => ({
        status: "applied" as const,
        newSkillIds: [newSkillId],
      })),
      reject: vi.fn(),
    } as SkillSplitReviewHandler);

    const outcome = await handler.apply({
      item,
      refs: payload,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx,
    });

    expect(outcome).toEqual({
      refs: expect.objectContaining({
        review_resolution: {
          decision: "accept",
          applied_at: 2_000,
          new_skill_ids: ["skl_bbbbbbbbbbbbbbbb"],
        },
      }),
    });
  });

  it("turns stale accepts into rejected review outcomes", async () => {
    const handler = createSkillSplitReviewQueueHandler({
      accept: vi.fn(() => ({
        status: "rejected",
        reason: "Skill already superseded",
      })),
      reject: vi.fn(),
    } as SkillSplitReviewHandler);

    const outcome = await handler.apply({
      item,
      refs: payload,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx,
    });

    expect(outcome).toEqual({
      finalResolution: {
        decision: "reject",
        reason: "Skill already superseded",
      },
      refs: expect.objectContaining({
        review_resolution: expect.objectContaining({
          decision: "reject",
          requested_decision: "accept",
        }),
      }),
    });
  });
});
