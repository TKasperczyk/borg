import { customAlphabet } from "nanoid";

const ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789";
const ID_LENGTH = 16;
const DEFAULT_SESSION_LITERAL = "default";
const createNanoId = customAlphabet(ID_ALPHABET, ID_LENGTH);

export type BrandedId<BrandName extends string> = string & {
  readonly __brand: BrandName;
};

export type StreamEntryId = BrandedId<"StreamEntryId">;
export type SessionId = BrandedId<"SessionId">;
export type EpisodeId = BrandedId<"EpisodeId">;
export type GoalId = BrandedId<"GoalId">;
export type ValueId = BrandedId<"ValueId">;

export const DEFAULT_SESSION_ID = DEFAULT_SESSION_LITERAL as SessionId;

export type IdHelpers<BrandName extends string> = {
  readonly pattern: RegExp;
  create(): BrandedId<BrandName>;
  is(value: string): value is BrandedId<BrandName>;
  parse(value: string): BrandedId<BrandName>;
};

export function createIdHelpers<BrandName extends string>(prefix: string): IdHelpers<BrandName> {
  const pattern = new RegExp(`^${prefix}_[${ID_ALPHABET}]{${ID_LENGTH}}$`);

  return {
    pattern,
    create: () => `${prefix}_${createNanoId()}` as BrandedId<BrandName>,
    is: (value: string): value is BrandedId<BrandName> => pattern.test(value),
    parse: (value: string): BrandedId<BrandName> => {
      if (!pattern.test(value)) {
        throw new TypeError(`Invalid ${prefix} identifier: ${value}`);
      }

      return value as BrandedId<BrandName>;
    },
  };
}

export const streamEntryIdHelpers = createIdHelpers<"StreamEntryId">("strm");
export const sessionIdHelpers = createIdHelpers<"SessionId">("sess");
export const episodeIdHelpers = createIdHelpers<"EpisodeId">("ep");
export const goalIdHelpers = createIdHelpers<"GoalId">("goal");
export const valueIdHelpers = createIdHelpers<"ValueId">("val");

export const createStreamEntryId = (): StreamEntryId => streamEntryIdHelpers.create();
export const createSessionId = (): SessionId => sessionIdHelpers.create();
export const createEpisodeId = (): EpisodeId => episodeIdHelpers.create();
export const createGoalId = (): GoalId => goalIdHelpers.create();
export const createValueId = (): ValueId => valueIdHelpers.create();

export function isSessionId(value: string): value is SessionId {
  return value === DEFAULT_SESSION_LITERAL || sessionIdHelpers.is(value);
}

export function parseSessionId(value: string): SessionId {
  if (value === DEFAULT_SESSION_LITERAL) {
    return DEFAULT_SESSION_ID;
  }

  return sessionIdHelpers.parse(value);
}

export function parseEpisodeId(value: string): EpisodeId {
  return episodeIdHelpers.parse(value);
}

export function parseGoalId(value: string): GoalId {
  return goalIdHelpers.parse(value);
}

export function parseValueId(value: string): ValueId {
  return valueIdHelpers.parse(value);
}
