import { dateTimeLabel, dayLabel, hm, hms, humanMs, relativeAge, relativeDay } from "./time";

describe("time formatting", () => {
  it("formats local clock times", () => {
    const date = new Date(2026, 5, 11, 9, 5, 3, 4);

    expect(hm(date)).toBe("09:05");
    expect(hms(date)).toBe("09:05:03.004");
  });

  it("formats local day labels", () => {
    expect(dayLabel(new Date(2026, 5, 11, 9, 5))).toBe("JUN 11");
  });

  it("formats compact relative ages", () => {
    const now = new Date(2026, 5, 11, 12, 0);

    expect(relativeAge(new Date(2026, 5, 11, 11, 54), now)).toBe("6m");
    expect(relativeAge(new Date(2026, 5, 11, 6, 0), now)).toBe("6h");
    expect(relativeAge(new Date(2026, 5, 9, 12, 0), now)).toBe("2d");
  });

  it("formats calendar-relative days", () => {
    const now = new Date(2026, 5, 11, 12, 0);

    expect(relativeDay(new Date(2026, 5, 11, 0, 30), now)).toBe("Today");
    expect(relativeDay(new Date(2026, 5, 11, 23, 30), now)).toBe("Today");
    expect(relativeDay(new Date(2026, 5, 10, 23, 30), now)).toBe("Yesterday");
    expect(relativeDay(new Date(2026, 5, 9, 12, 0), now)).toBe("JUN 9");
    expect(relativeDay(new Date(2025, 5, 9, 12, 0), now)).toBe("JUN 9 2025");
  });

  it("formats relative date + time labels", () => {
    const now = new Date(2026, 5, 11, 12, 0);

    expect(dateTimeLabel(new Date(2026, 5, 11, 15, 0), now)).toBe("Today 15:00");
    expect(dateTimeLabel(new Date(2026, 5, 10, 15, 0), now)).toBe("Yesterday 15:00");
    expect(dateTimeLabel(new Date(2026, 5, 9, 9, 5), now)).toBe("JUN 9 09:05");
  });

  it("formats maintenance intervals", () => {
    expect(humanMs(250)).toBe("250ms");
    expect(humanMs(1_500)).toBe("1.5s");
    expect(humanMs(60_000)).toBe("1m");
    expect(humanMs(3_660_000)).toBe("1h 1m");
  });
});
