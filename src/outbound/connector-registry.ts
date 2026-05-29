import type { SessionSourceType } from "../sessions/index.js";
import { ToolError } from "../util/errors.js";

import type { MessageConnector } from "./types.js";

export class MessageConnectorRegistry {
  private readonly connectors = new Map<SessionSourceType, MessageConnector>();

  constructor(connectors: readonly MessageConnector[] = []) {
    for (const connector of connectors) {
      this.register(connector);
    }
  }

  register(connector: MessageConnector): this {
    if (this.connectors.has(connector.sourceType)) {
      throw new ToolError(`Outbound connector already registered: ${connector.sourceType}`, {
        code: "OUTBOUND_CONNECTOR_ALREADY_REGISTERED",
      });
    }

    this.connectors.set(connector.sourceType, connector);
    return this;
  }

  get(sourceType: SessionSourceType): MessageConnector | null {
    return this.connectors.get(sourceType) ?? null;
  }

  has(sourceType: SessionSourceType): boolean {
    return this.connectors.has(sourceType);
  }

  sourceTypes(): SessionSourceType[] {
    return [...this.connectors.keys()].sort();
  }
}
