import { afterEach, describe, expect, it, vi } from "vitest";

import type { BorgDependencies } from "./types.js";
import { closeBorgDependencies } from "./lifecycle.js";

describe("borg lifecycle", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("attempts LanceDB close when SQLite close throws", async () => {
    const sqliteError = new Error("sqlite close failed");
    const lanceClose = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(console, "error").mockImplementation(() => {});

    const deps = {
      autonomyScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      maintenanceScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      sqlite: {
        close: vi.fn(() => {
          throw sqliteError;
        }),
      },
      lance: {
        close: lanceClose,
      },
    } as unknown as BorgDependencies;

    await expect(closeBorgDependencies(deps)).rejects.toThrow(AggregateError);
    expect(lanceClose).toHaveBeenCalledTimes(1);
  });
});
