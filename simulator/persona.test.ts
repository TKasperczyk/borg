import { describe, expect, it } from "vitest";

import { PersonaSession } from "./persona.js";
import { tomPersona } from "./personas/tom.js";

describe("PersonaSession", () => {
  it("generates mock persona messages in sequence", async () => {
    const persona = new PersonaSession({
      persona: tomPersona,
      mock: true,
      mockMessages: ["first Tom turn", "second Tom turn"],
    });

    await expect(persona.nextTurn(null)).resolves.toBe("first Tom turn");
    await expect(persona.nextTurn("Borg replied.")).resolves.toBe("second Tom turn");
    await expect(persona.nextTurn("Borg replied again.")).resolves.toBe("first Tom turn");
  });
});
