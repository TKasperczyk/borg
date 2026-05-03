// Borg public facade: exposes stable APIs while setup details live in focused modules.

import type { TurnInput, TurnResult } from "./cognition/index.js";
import { createBorgFacades } from "./borg/facade.js";
import type { BorgFacades } from "./borg/facade-types.js";
import { closeBorgDependencies } from "./borg/lifecycle.js";
import { openBorgDependencies } from "./borg/open.js";
import type { BorgDependencies, BorgOpenOptions } from "./borg/types.js";

export type {
  BorgDreamOptions,
  BorgDreamRunner,
  BorgEpisodeGetOptions,
  BorgEpisodeSearchOptions,
  BorgOpenOptions,
} from "./borg/types.js";

export class Borg {
  readonly stream: BorgFacades["stream"];
  readonly episodic: BorgFacades["episodic"];
  readonly self: BorgFacades["self"];
  readonly skills: BorgFacades["skills"];
  readonly mood: BorgFacades["mood"];
  readonly actions: BorgFacades["actions"];
  readonly social: BorgFacades["social"];
  readonly semantic: BorgFacades["semantic"];
  readonly commitments: BorgFacades["commitments"];
  readonly identity: BorgFacades["identity"];
  readonly correction: BorgFacades["correction"];
  readonly review: BorgFacades["review"];
  readonly audit: BorgFacades["audit"];
  readonly dream: BorgFacades["dream"];
  readonly autonomy: BorgFacades["autonomy"];
  readonly maintenance: BorgFacades["maintenance"];
  readonly workmem: BorgFacades["workmem"];

  private constructor(private readonly deps: BorgDependencies) {
    const facades = createBorgFacades(deps);

    this.stream = facades.stream;
    this.episodic = facades.episodic;
    this.self = facades.self;
    this.skills = facades.skills;
    this.mood = facades.mood;
    this.actions = facades.actions;
    this.social = facades.social;
    this.semantic = facades.semantic;
    this.commitments = facades.commitments;
    this.identity = facades.identity;
    this.correction = facades.correction;
    this.review = facades.review;
    this.audit = facades.audit;
    this.dream = facades.dream;
    this.autonomy = facades.autonomy;
    this.maintenance = facades.maintenance;
    this.workmem = facades.workmem;
  }

  turn(input: TurnInput): Promise<TurnResult> {
    return this.deps.turnOrchestrator.run(input);
  }

  static async open(options: BorgOpenOptions = {}): Promise<Borg> {
    return new Borg(await openBorgDependencies(options));
  }

  async close(): Promise<void> {
    await closeBorgDependencies(this.deps);
  }
}
