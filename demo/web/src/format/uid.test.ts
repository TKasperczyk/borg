import { newId } from "./uid";

describe("newId", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uses getRandomValues v4 fallback when randomUUID is unavailable", () => {
    const getRandomValues = vi.fn((bytes: Uint8Array) => {
      bytes.fill(0x11);
      return bytes;
    });
    vi.stubGlobal("crypto", { getRandomValues });

    expect(newId()).toMatch(/^11111111-1111-4111-9111-111111111111$/);
    expect(getRandomValues).toHaveBeenCalledOnce();
  });
});
