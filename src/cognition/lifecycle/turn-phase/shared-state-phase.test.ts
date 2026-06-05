import { describe, expect, it, vi } from "vitest";

import type { ActionRecord } from "../../../memory/actions/index.js";
import type { ActionRecordListFilter } from "../../../memory/actions/repository.js";
import {
  createActionId,
  createEntityId,
  createStreamEntryId,
  type EntityId,
} from "../../../util/ids.js";
import { listActionCandidatesForCognition } from "../../evidence-ledger/action-threads.js";

import { selectCurrentAudienceSharedStateActionCandidatesForCanonicalization } from "./shared-state-phase.js";

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.updated_at ?? 1_000;

  return {
    id: createActionId(),
    description: "Track an action",
    actor: "borg",
    audience_entity_id: null,
    goal_id: null,
    open_question_id: null,
    state: "scheduled",
    confidence: 0.9,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [createStreamEntryId()],
    created_at: nowMs,
    updated_at: nowMs,
    considering_at: null,
    committed_at: null,
    scheduled_at: nowMs,
    completed_at: null,
    not_done_at: null,
    expired_at: null,
    archived_at: null,
    unknown_at: null,
    canonicalized_by_artifact_entry_id: null,
    session_scope: null,
    session_anchor_id: null,
    last_referenced_at_ms: nowMs,
    last_referenced_turn_counter: null,
    ...overrides,
  };
}

function makeRepository(actions: readonly ActionRecord[]) {
  return {
    get: vi.fn(() => null),
    list: vi.fn((filter: ActionRecordListFilter = {}) =>
      actions
        .filter(
          (action) =>
            (filter.state === undefined || action.state === filter.state) &&
            (filter.states === undefined || filter.states.includes(action.state)) &&
            (filter.actor === undefined || action.actor === filter.actor) &&
            (filter.recallAllAudiences === true ||
              !("audienceEntityId" in filter) ||
              (filter.audienceEntityId === null
                ? action.audience_entity_id === null
                : action.audience_entity_id === filter.audienceEntityId)),
        )
        .slice(0, filter.limit ?? actions.length),
    ),
    update: vi.fn(),
  };
}

describe("shared-state action canonicalization candidates", () => {
  it("keeps other-audience-private participant actions out of audience artifact canonicalization", () => {
    const alice = createEntityId();
    const bob = createEntityId();
    const alicePrivateAction = makeAction({
      description: "Prepare Alice private launch note",
      actor: alice,
      audience_entity_id: alice,
      updated_at: 3_000,
    });
    const bobAudienceAction = makeAction({
      description: "Prepare Bob channel note",
      audience_entity_id: bob,
      updated_at: 2_000,
    });
    const globalAction = makeAction({
      description: "Prepare public maintenance note",
      audience_entity_id: null,
      updated_at: 1_000,
    });
    const actionRepository = makeRepository([
      alicePrivateAction,
      bobAudienceAction,
      globalAction,
    ]);

    const cognitionCandidates = listActionCandidatesForCognition({
      actionRepository,
      audienceEntityId: bob,
      activeParticipants: [
        {
          entityId: alice,
          displayName: "Alice",
          role: "participant",
        },
      ],
      limit: 10,
    });
    const canonicalizationCandidates =
      selectCurrentAudienceSharedStateActionCandidatesForCanonicalization({
        actionRepository,
        audienceEntityId: bob,
        activeParticipants: [
          {
            entityId: alice,
            displayName: "Alice",
            role: "participant",
          },
        ],
      });

    expect(cognitionCandidates.map((candidate) => candidate.record.id)).toContain(
      alicePrivateAction.id,
    );
    expect(
      cognitionCandidates.find((candidate) => candidate.record.id === alicePrivateAction.id)
        ?.disclosureLabel,
    ).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [alice],
    });
    expect((canonicalizationCandidates.candidates ?? []).map((candidate) => candidate.id)).toEqual([
      bobAudienceAction.id,
      globalAction.id,
    ]);
    const candidateById = new Map(
      (canonicalizationCandidates.candidates ?? []).map((candidate) => [candidate.id, candidate]),
    );
    expect(candidateById.get(bobAudienceAction.id)).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [bob],
      },
    });
    expect(candidateById.get(bobAudienceAction.id)?.disclosure).toContain(
      "disclosure_class=relationship_private",
    );
    expect(candidateById.get(globalAction.id)).toMatchObject({
      disclosure_label: {
        disclosure_class: "self_private",
      },
    });
    expect(actionRepository.list).not.toHaveBeenCalledWith(
      expect.objectContaining({
        actor: alice as EntityId,
      }),
    );
  });
});
