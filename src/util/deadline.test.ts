import { afterEach, expect, it, vi } from "vitest";
import { runWithDeadline } from "./deadline.js";
import { sleep } from "./clock.js";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

it("keeps an absolute deadline when an event-loop hold makes a result beat the timer", async () => {
  let now = 0;
  vi.spyOn(performance, "now").mockImplementation(() => now);
  await expect(
    runWithDeadline({
      timeoutMs: 100,
      timeoutError: () => new Error("deadline"),
      run: async () => {
        now = 150;
        return "late result";
      },
    }),
  ).rejects.toThrow("deadline");
});

it("bounds uncooperative work and handles its late rejection", async () => {
  vi.useFakeTimers();
  let reject!: (error: Error) => void;
  let signal!: AbortSignal;
  const result = runWithDeadline({
    timeoutMs: 30,
    timeoutError: () => new Error("deadline"),
    run: async (value) => {
      signal = value;
      return await new Promise((_, fail) => {
        reject = fail;
      });
    },
  });
  const rejected = expect(result).rejects.toThrow("deadline");
  await vi.advanceTimersByTimeAsync(30);
  await rejected;
  expect(signal.aborted).toBe(true);
  reject(new Error("late transport rejection"));
  await Promise.resolve();
  expect(vi.getTimerCount()).toBe(0);
});

it("cancels retry sleep through a parent signal and cleans both timers", async () => {
  vi.useFakeTimers();
  const controller = new AbortController();
  const result = runWithDeadline({
    timeoutMs: 100,
    signal: controller.signal,
    timeoutError: () => new Error("deadline"),
    run: (signal) => sleep(90, signal),
  });
  const rejected = expect(result).rejects.toThrow("caller deadline");
  controller.abort(new Error("caller deadline"));
  await rejected;
  expect(vi.getTimerCount()).toBe(0);
});
