import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ParticleField } from "./ParticleField";

function installCanvasMocks() {
  const gradient = { addColorStop: vi.fn() };
  const context = {
    arc: vi.fn(),
    beginPath: vi.fn(),
    clearRect: vi.fn(),
    createRadialGradient: vi.fn(() => gradient),
    fill: vi.fn(),
    fillRect: vi.fn(),
    setTransform: vi.fn(),
  } as unknown as CanvasRenderingContext2D;

  vi.spyOn(window.HTMLCanvasElement.prototype, "getContext").mockImplementation(
    () => context as never,
  );
  vi.spyOn(window.HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    () =>
      ({
        bottom: 660,
        height: 660,
        left: 0,
        right: 1200,
        toJSON: () => ({}),
        top: 0,
        width: 1200,
        x: 0,
        y: 0,
      }) as DOMRect,
  );

  class ResizeObserverStub {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
  }

  vi.stubGlobal("ResizeObserver", ResizeObserverStub);
}

function installReducedMotion(matches: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches,
      media: "(prefers-reduced-motion: reduce)",
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  );
}

describe("ParticleField", () => {
  beforeEach(() => {
    installCanvasMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders no idle canvas and schedules no animation frame when target is null", () => {
    const requestFrame = vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1);

    const { container } = render(<ParticleField enabled target={null} />);

    expect(container.querySelector(".fc-particles")).toBeNull();
    expect(requestFrame).not.toHaveBeenCalled();
  });

  it("schedules animation only while active and cancels on unmount", () => {
    const requestFrame = vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 12);
    const cancelFrame = vi
      .spyOn(window, "cancelAnimationFrame")
      .mockImplementation(() => undefined);

    const { container, unmount } = render(<ParticleField enabled target={{ x: 200, y: 125 }} />);

    expect(container.querySelector(".fc-particles")).toBeInTheDocument();
    expect(requestFrame).toHaveBeenCalledTimes(1);

    unmount();

    expect(cancelFrame).toHaveBeenCalledWith(12);
  });

  it("honors reduced motion by rendering no canvas and scheduling no frame", () => {
    installReducedMotion(true);
    const requestFrame = vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1);

    const { container } = render(<ParticleField enabled target={{ x: 200, y: 125 }} />);

    expect(container.querySelector(".fc-particles")).toBeNull();
    expect(requestFrame).not.toHaveBeenCalled();
  });
});
