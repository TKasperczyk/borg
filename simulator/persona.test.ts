import { describe, expect, it } from "vitest";

import { PersonaSession } from "./persona.js";
import { tomPersona } from "./personas/tom.js";

const FIRST_TOM_TURN = "first Tom turn";
const SECOND_TOM_TURN = "second Tom turn";
const FIRST_BORG_REPLY = "Borg replied.";
const SECOND_BORG_REPLY = "Borg replied again.";

describe("PersonaSession", () => {
  it("generates mock persona messages in sequence", async () => {
    const persona = new PersonaSession({
      persona: tomPersona,
      mock: true,
      mockMessages: [FIRST_TOM_TURN, SECOND_TOM_TURN],
    });

    const first = await persona.prepareNextTurn(null);
    persona.commit(first, FIRST_BORG_REPLY);
    const second = await persona.prepareNextTurn(FIRST_BORG_REPLY);
    persona.commit(second, SECOND_BORG_REPLY);
    const third = await persona.prepareNextTurn(SECOND_BORG_REPLY);

    expect(first.message).toBe(FIRST_TOM_TURN);
    expect(second.message).toBe(SECOND_TOM_TURN);
    expect(third.message).toBe(FIRST_TOM_TURN);
  });

  it("does not advance mock history when a draft is rolled back", async () => {
    const persona = new PersonaSession({
      persona: tomPersona,
      mock: true,
      mockMessages: [FIRST_TOM_TURN, SECOND_TOM_TURN],
    });

    const draft = await persona.prepareNextTurn(null);
    persona.rollback(draft);
    const retry = await persona.prepareNextTurn(null);

    expect(draft.message).toBe(FIRST_TOM_TURN);
    expect(retry.message).toBe(FIRST_TOM_TURN);
  });
});
