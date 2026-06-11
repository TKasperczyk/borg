type Listener = (event: unknown) => void;

export class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readonly sent: string[] = [];
  readyState = MockWebSocket.CONNECTING;
  private readonly listeners = new Map<string, Set<Listener>>();

  constructor(readonly url: string) {}

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    const fn: Listener =
      typeof listener === "function"
        ? (event) => listener(event as Event)
        : (event) => listener.handleEvent(event as Event);
    const listeners = this.listeners.get(type) ?? new Set<Listener>();
    listeners.add(fn);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    const listeners = this.listeners.get(type);
    if (listeners === undefined) {
      return;
    }

    for (const candidate of listeners) {
      const source = typeof listener === "function" ? listener : listener.handleEvent;
      if (candidate === source) {
        listeners.delete(candidate);
      }
    }
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.emit("close", {});
  }

  open(): void {
    this.readyState = MockWebSocket.OPEN;
    this.emit("open", {});
  }

  receive(frame: unknown): void {
    this.emit("message", { data: JSON.stringify(frame) });
  }

  error(): void {
    this.emit("error", {});
  }

  private emit(type: string, event: unknown): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }
}

export function installMockWebSocket(): { instances: MockWebSocket[] } {
  const instances: MockWebSocket[] = [];

  class InstalledMockWebSocket extends MockWebSocket {
    constructor(url: string) {
      super(url);
      instances.push(this);
    }
  }

  vi.stubGlobal("WebSocket", InstalledMockWebSocket);
  return { instances };
}
