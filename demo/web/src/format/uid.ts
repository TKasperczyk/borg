function randomHex(bytes: Uint8Array): string {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function randomBytes(length: number): Uint8Array | null {
  const cryptoLike = globalThis.crypto;
  if (cryptoLike?.getRandomValues === undefined) {
    return null;
  }

  const bytes = new Uint8Array(length);
  cryptoLike.getRandomValues(bytes);
  return bytes;
}

export function newId(): string {
  if (globalThis.crypto?.randomUUID !== undefined) {
    return globalThis.crypto.randomUUID();
  }

  const bytes = randomBytes(16);
  if (bytes !== null) {
    bytes[6] = (bytes[6]! & 0x0f) | 0x40;
    bytes[8] = (bytes[8]! & 0x3f) | 0x80;
    const hex = randomHex(bytes);
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(
      16,
      20,
    )}-${hex.slice(20)}`;
  }

  return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}
