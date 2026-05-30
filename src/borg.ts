// Borg public facade: exposes stable APIs while setup details live in focused modules.

import type {
  BorgEnqueueMessageInput,
  BorgEnqueueMessageResult,
  TurnInput,
  TurnResult,
} from "./cognition/index.js";
import { createBorgFacades } from "./borg/facade.js";
import { closeBorgDependencies } from "./borg/lifecycle.js";
import { openBorgDependencies } from "./borg/open.js";
import type { BorgFacades } from "./borg/public-facade.js";
import type { BorgDependencies, BorgOpenOptions } from "./borg/types.js";
import {
  expireSessionScopedActions,
  rolloverNextSessionActions,
} from "./memory/lifecycle-ops/index.js";
import { DEFAULT_SESSION_ID, type SessionId } from "./util/ids.js";

export type {
  BorgDreamOptions,
  BorgDreamRunner,
  BorgEpisodeGetOptions,
  BorgEpisodeSearchOptions,
  BorgOpenOptions,
} from "./borg/types.js";
export type { BorgEnqueueMessageInput, BorgEnqueueMessageResult } from "./cognition/index.js";

/**
 * Public Borg library facade.
 *
 * Construct with `Borg.open()`, call `turn()` for conversation turns, and use
 * the exposed band facades for memory access, maintenance, correction, and
 * operational state. Repository construction and storage wiring stay internal.
 */
export class Borg {
  readonly stream: BorgFacades["stream"];
  readonly episodic: BorgFacades["episodic"];
  readonly self: BorgFacades["self"];
  readonly skills: BorgFacades["skills"];
  readonly mood: BorgFacades["mood"];
  readonly actions: BorgFacades["actions"];
  readonly social: BorgFacades["social"];
  readonly entities: BorgFacades["entities"];
  readonly sharedState: BorgFacades["sharedState"];
  readonly attachments: BorgFacades["attachments"];
  readonly semantic: BorgFacades["semantic"];
  readonly relationalSlots: BorgFacades["relationalSlots"];
  readonly commitments: BorgFacades["commitments"];
  readonly creatorDirectives: BorgFacades["creatorDirectives"];
  readonly identity: BorgFacades["identity"];
  readonly correction: BorgFacades["correction"];
  readonly review: BorgFacades["review"];
  readonly audit: BorgFacades["audit"];
  readonly dream: BorgFacades["dream"];
  readonly autonomy: BorgFacades["autonomy"];
  readonly maintenance: BorgFacades["maintenance"];
  readonly inbox: BorgFacades["inbox"];
  readonly workmem: BorgFacades["workmem"];
  readonly prompts: BorgFacades["prompts"];
  readonly sessions: BorgFacades["sessions"];

  private constructor(private readonly deps: BorgDependencies) {
    const facades = createBorgFacades(deps) as unknown as BorgFacades;

    this.stream = facades.stream;
    this.episodic = facades.episodic;
    this.self = facades.self;
    this.skills = facades.skills;
    this.mood = facades.mood;
    this.actions = facades.actions;
    this.social = facades.social;
    this.entities = facades.entities;
    this.sharedState = facades.sharedState;
    this.attachments = facades.attachments;
    this.semantic = facades.semantic;
    this.relationalSlots = facades.relationalSlots;
    this.commitments = facades.commitments;
    this.creatorDirectives = facades.creatorDirectives;
    this.identity = facades.identity;
    this.correction = facades.correction;
    this.review = facades.review;
    this.audit = facades.audit;
    this.dream = facades.dream;
    this.autonomy = facades.autonomy;
    this.maintenance = facades.maintenance;
    this.inbox = facades.inbox;
    this.workmem = facades.workmem;
    this.prompts = facades.prompts;
    this.sessions = facades.sessions;
  }

  /**
   * Run one user or autonomous turn through perception, retrieval,
   * deliberation, generation, persistence, and post-turn maintenance hooks.
   */
  turn(input: TurnInput): Promise<TurnResult> {
    return this.deps.turnOrchestrator.run(input);
  }

  /**
   * Durably enqueue an inbound chat message without running a cognition turn.
   */
  enqueueMessage(input: BorgEnqueueMessageInput): Promise<BorgEnqueueMessageResult> {
    return this.deps.messageEnqueuer.enqueueMessage(input);
  }

  /**
   * Close a session-scoped action window, expiring current-session actions and
   * optionally rolling eligible next-session actions forward.
   */
  endSession(
    sessionId: SessionId = DEFAULT_SESSION_ID,
    options: { nextSessionId?: SessionId } = {},
  ): void {
    const rollover =
      options.nextSessionId === undefined
        ? null
        : rolloverNextSessionActions({
            fromSessionId: sessionId,
            toSessionId: options.nextSessionId,
            repository: this.deps.actionRepository,
            nowMs: this.deps.clock.now(),
            tracer: this.deps.tracer,
          });
    const expired = expireSessionScopedActions({
      sessionId,
      repository: this.deps.actionRepository,
      nowMs: this.deps.clock.now(),
      tracer: this.deps.tracer,
    });
    const expiredCount = expired.value?.expiredActionIds.length ?? 0;
    const expirationConflictCount = expired.value?.conflictedActionIds.length ?? 0;
    const promotedCount = rollover?.value?.promotedActionIds.length ?? 0;
    const rolloverConflictCount = rollover?.value?.conflictedActionIds.length ?? 0;

    if (this.deps.tracer.enabled) {
      this.deps.tracer.emit("session.completed", {
        turnId: `session_end:${sessionId}`,
        session_id: sessionId,
        next_session_id: options.nextSessionId,
        actions_expired_at_session_close: expiredCount,
        actions_expiration_conflict_count: expirationConflictCount,
        actions_promoted_to_next_session: promotedCount,
        actions_rollover_conflict_count: rolloverConflictCount,
      });
    }
  }

  /** Open Borg with default or caller-supplied configuration and clients. */
  static async open(options: BorgOpenOptions = {}): Promise<Borg> {
    return new Borg(await openBorgDependencies(options));
  }

  /** Close underlying storage and runtime resources. */
  async close(): Promise<void> {
    await closeBorgDependencies(this.deps);
  }
}
