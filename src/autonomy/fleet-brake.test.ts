import { describe, expect, it, vi } from "vitest";

import {
  DEFAULT_FLEET_BRAKE_OPTIONS,
  emptyFleetBrakeMetadata,
  fleetBrakeCooldownUntilMs,
  fleetBrakeErrorPausedUntilMs,
  readFleetBrakeMetadata,
} from "./fleet-brake.js";

describe("autonomy fleet brake", () => {
  it("tolerates absent and partial metadata by filling missing fields with zeros", () => {
    expect(readFleetBrakeMetadata(null)).toEqual(emptyFleetBrakeMetadata());
    expect(readFleetBrakeMetadata({ metadata: null })).toEqual(emptyFleetBrakeMetadata());
    expect(
      readFleetBrakeMetadata({
        metadata: {
          empty_streak: 4,
          last_wake_ts: 1_000,
        },
      }),
    ).toEqual({
      ...emptyFleetBrakeMetadata(),
      empty_streak: 4,
      last_wake_ts: 1_000,
    });
  });

  it.each([
    ["non-object", []],
    ["negative", { empty_streak: -1 }],
    ["fractional", { error_streak: 1.5 }],
    ["wrong type", { bypass_count: "3" }],
  ])("reports malformed %s metadata once and falls back to zeros", (_label, metadata) => {
    const notifyError = vi.fn();

    expect(readFleetBrakeMetadata({ metadata }, notifyError)).toEqual(emptyFleetBrakeMetadata());
    expect(notifyError).toHaveBeenCalledTimes(1);
  });

  it("never propagates a notification failure from the tolerant reader", () => {
    expect(() =>
      readFleetBrakeMetadata({ metadata: { empty_streak: "bad" } }, () => {
        throw new Error("observer unavailable");
      }),
    ).not.toThrow();
  });

  it("derives finite escalating operational cooldowns and caps them at six hours", () => {
    const anchor = 10_000;

    expect(
      fleetBrakeCooldownUntilMs(
        { ...emptyFleetBrakeMetadata(), empty_streak: 4, last_wake_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBeNull();
    expect(
      fleetBrakeCooldownUntilMs(
        { ...emptyFleetBrakeMetadata(), empty_streak: 5, last_wake_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBe(anchor + 30 * 60 * 1_000);
    expect(
      fleetBrakeCooldownUntilMs(
        { ...emptyFleetBrakeMetadata(), empty_streak: 6, last_wake_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBe(anchor + 60 * 60 * 1_000);
    expect(
      fleetBrakeCooldownUntilMs(
        { ...emptyFleetBrakeMetadata(), empty_streak: 20, last_wake_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBe(anchor + 6 * 60 * 60 * 1_000);
  });

  it("derives the three-error circuit pause with a finite thirty-minute cap", () => {
    const anchor = 20_000;

    expect(
      fleetBrakeErrorPausedUntilMs(
        { ...emptyFleetBrakeMetadata(), error_streak: 2, last_error_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBeNull();
    expect(
      fleetBrakeErrorPausedUntilMs(
        { ...emptyFleetBrakeMetadata(), error_streak: 3, last_error_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBe(anchor + 5 * 60 * 1_000);
    expect(
      fleetBrakeErrorPausedUntilMs(
        { ...emptyFleetBrakeMetadata(), error_streak: 10, last_error_ts: anchor },
        DEFAULT_FLEET_BRAKE_OPTIONS,
      ),
    ).toBe(anchor + 30 * 60 * 1_000);
  });

  it("is restart-stable because derivation depends only on durable metadata", () => {
    const metadata = {
      ...emptyFleetBrakeMetadata(),
      empty_streak: 7,
      last_wake_ts: 50_000,
      error_streak: 4,
      last_error_ts: 60_000,
    };

    expect(fleetBrakeCooldownUntilMs(metadata, DEFAULT_FLEET_BRAKE_OPTIONS)).toBe(
      fleetBrakeCooldownUntilMs({ ...metadata }, DEFAULT_FLEET_BRAKE_OPTIONS),
    );
    expect(fleetBrakeErrorPausedUntilMs(metadata, DEFAULT_FLEET_BRAKE_OPTIONS)).toBe(
      fleetBrakeErrorPausedUntilMs({ ...metadata }, DEFAULT_FLEET_BRAKE_OPTIONS),
    );
  });
});
