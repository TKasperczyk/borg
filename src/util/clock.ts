export type Clock = {
  now(): number;
};

export class SystemClock implements Clock {
  now(): number {
    return Date.now();
  }
}

export class FixedClock implements Clock {
  constructor(private readonly value: number) {}

  now(): number {
    return this.value;
  }
}

export class ManualClock implements Clock {
  constructor(private value = 0) {}

  now(): number {
    return this.value;
  }

  set(value: number): void {
    this.value = value;
  }

  advance(deltaMs: number): number {
    this.value += deltaMs;
    return this.value;
  }
}

export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
