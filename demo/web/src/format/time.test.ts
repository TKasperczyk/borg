import { dayLabel, hm, hms, relativeAge } from "./time";

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
});
