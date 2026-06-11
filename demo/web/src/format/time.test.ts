import { dayLabel, hm, hms } from "./time";

describe("time formatting", () => {
  it("formats local clock times", () => {
    const date = new Date(2026, 5, 11, 9, 5, 3, 4);

    expect(hm(date)).toBe("09:05");
    expect(hms(date)).toBe("09:05:03.004");
  });

  it("formats local day labels", () => {
    expect(dayLabel(new Date(2026, 5, 11, 9, 5))).toBe("JUN 11");
  });
});
