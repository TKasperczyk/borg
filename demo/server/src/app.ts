import { Buffer } from "node:buffer";

import { createNodeWebSocket } from "@hono/node-ws";
import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import {
  DEFAULT_SESSION_ID,
  STREAM_ENTRY_KINDS,
  VERSION,
  type AttachmentId,
  type Borg,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryKind,
} from "borg";
import { z } from "zod";

import type { LiveBridge } from "./live.js";

type CursorPayload = {
  ts: number;
  entryId: string;
};

const cursorPayloadSchema = z.object({
  ts: z.number().finite(),
  entryId: z.string().min(1),
});

const csvKindsSchema = z
  .string()
  .optional()
  .transform((value, ctx) => {
    if (value === undefined || value.trim().length === 0) {
      return undefined;
    }

    const kinds = value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    const parsed = z.array(z.enum(STREAM_ENTRY_KINDS)).safeParse(kinds);

    if (!parsed.success) {
      ctx.addIssue({
        code: "custom",
        message: `kind must be one or more of ${STREAM_ENTRY_KINDS.join(",")}`,
      });
      return z.NEVER;
    }

    return parsed.data;
  });

const limitSchema = z.coerce.number().int().min(1).max(500);

const streamQuerySchema = z.object({
  kind: csvKindsSchema,
  audience: z.string().min(1).optional(),
  limit: limitSchema.default(50),
  before: z
    .string()
    .optional()
    .transform((value, ctx) => {
      if (value === undefined || value.length === 0) {
        return undefined;
      }

      const parsed = decodeCursor(value);

      if (parsed === null) {
        ctx.addIssue({ code: "custom", message: "before is not a valid stream cursor" });
        return z.NEVER;
      }

      return parsed;
    }),
});

const audienceQuerySchema = z.object({
  audience: z.string().min(1).optional(),
});

const auditQuerySchema = z.object({
  limit: limitSchema.default(50),
});

const attachmentQuerySchema = z.object({
  audience: z.string().min(1).nullable().optional(),
});

const turnBodySchema = z.object({
  message: z.string().min(1),
  audience: z.string().min(1),
  stakes: z.enum(["low", "medium", "high"]).optional(),
});

function parseRequest<T>(schema: z.ZodType<T>, value: unknown): T {
  const parsed = schema.safeParse(value);

  if (!parsed.success) {
    throw new HTTPException(400, { message: parsed.error.message });
  }

  return parsed.data;
}

async function parseJsonBody(c: Context): Promise<unknown> {
  try {
    return await c.req.json();
  } catch {
    throw new HTTPException(400, { message: "Malformed JSON body" });
  }
}

function jsonError(status: number, message: string): Response {
  return new Response(
    JSON.stringify({
      error: {
        status,
        message,
      },
    }),
    {
      status,
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
}

function encodeCursor(entry: Pick<StreamEntry, "timestamp" | "id">): string {
  const payload: CursorPayload = {
    ts: entry.timestamp,
    entryId: entry.id,
  };

  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
}

function decodeCursor(cursor: string): StreamCursor | null {
  try {
    const raw = Buffer.from(cursor, "base64url").toString("utf8");
    const parsed = cursorPayloadSchema.parse(JSON.parse(raw));
    return {
      ts: parsed.ts,
      entryId: parsed.entryId as StreamCursor["entryId"],
    };
  } catch {
    return null;
  }
}

async function readStream(input: {
  borg: Borg;
  kinds?: readonly StreamEntryKind[];
  audience?: string;
  limit: number;
  before?: StreamCursor;
}): Promise<{ entries: StreamEntry[]; next_cursor: string | null }> {
  const collected: StreamEntry[] = [];
  const reader = input.borg.stream.reader();

  for await (const entry of reader.iterate({
    kinds: input.kinds,
    untilCursor: input.before,
  })) {
    if (input.audience !== undefined && entry.audience !== input.audience) {
      continue;
    }

    collected.push(entry);
  }

  const cursorIndex =
    input.before === undefined
      ? -1
      : collected.findIndex(
          (entry) => entry.timestamp === input.before?.ts && entry.id === input.before.entryId,
        );
  const beforeCursor = cursorIndex === -1 ? collected : collected.slice(0, cursorIndex);
  const page = beforeCursor.slice(-(input.limit + 1));
  const entries = page.length > input.limit ? page.slice(1) : page;
  const next_cursor =
    page.length > input.limit && entries[0] !== undefined ? encodeCursor(entries[0]) : null;

  return { entries, next_cursor };
}

async function countTurns(borg: Borg): Promise<number> {
  let count = 0;

  for await (const entry of borg.stream.reader().iterate({ kinds: ["user_msg"] })) {
    if (entry.turn_status !== "aborted") {
      count += 1;
    }
  }

  return count;
}

function listAudiences(borg: Borg): string[] {
  return [
    ...new Set(
      borg.stream
        .tail(500)
        .flatMap((entry) => (entry.audience === undefined ? [] : [entry.audience])),
    ),
  ].sort();
}

function sumRecord(record: Record<string, number>): number {
  return Object.values(record).reduce((sum, value) => sum + value, 0);
}

async function memoryBands(borg: Borg) {
  const episodes = await borg.episodic.list({ limit: 500 });
  const semanticCounts = borg.semantic.nodes.countByStatus();
  const procedural = borg.skills.list(500);
  const moodHistory = borg.mood.history(DEFAULT_SESSION_ID, { limit: 500 });
  const values = borg.self.values.list();
  const goals = borg.self.goals.list();
  const traits = borg.self.traits.list();
  const openQuestions = borg.self.openQuestions.list({ status: "open" });
  const growthMarkers = borg.self.growthMarkers.list({ limit: 500 });
  const periods = borg.self.autobiographical.listPeriods({ limit: 500 });
  const relationalCounts = borg.relationalSlots.countByState();
  const audiences = listAudiences(borg);

  return [
    { id: "episodic", name: "Episodic", count: episodes.items.length, stats: [] },
    {
      id: "semantic",
      name: "Semantic",
      count: sumRecord(semanticCounts),
      stats: Object.entries(semanticCounts).map(([k, v]) => ({ k, v })),
    },
    { id: "procedural", name: "Procedural", count: procedural.length, stats: [] },
    { id: "affective", name: "Affective", count: moodHistory.length, stats: [] },
    {
      id: "self",
      name: "Self",
      count:
        values.length +
        goals.length +
        traits.length +
        openQuestions.length +
        growthMarkers.length +
        periods.length,
      stats: [
        { k: "values", v: values.length },
        { k: "goals", v: goals.length },
        { k: "traits", v: traits.length },
        { k: "open_questions", v: openQuestions.length },
        { k: "growth_markers", v: growthMarkers.length },
        { k: "periods", v: periods.length },
      ],
    },
    { id: "commitments", name: "Commitments", count: borg.commitments.countActive(), stats: [] },
    { id: "social", name: "Social", count: audiences.length, stats: [] },
    {
      id: "relational",
      name: "Relational",
      count: sumRecord(relationalCounts),
      stats: Object.entries(relationalCounts).map(([k, v]) => ({ k, v })),
    },
  ];
}

function selfSnapshot(borg: Borg) {
  return {
    values: borg.self.values.list(),
    goals: borg.self.goals.list(),
    traits: borg.self.traits.list(),
    open_questions: borg.self.openQuestions.list({ status: "open" }),
    growth_markers: borg.self.growthMarkers.list({ limit: 100 }),
    periods: borg.self.autobiographical.listPeriods({ limit: 100 }),
  };
}

export function createDemoServerApp(input: {
  borg: Borg;
  live: LiveBridge;
  corsOrigins?: readonly string[];
}) {
  const app = new Hono();
  const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });
  const allowedOrigins = input.corsOrigins ?? ["http://localhost:5173"];

  app.onError((error) => {
    if (error instanceof HTTPException) {
      return jsonError(error.status, error.message);
    }

    console.error(error instanceof Error ? error.message : String(error));
    return jsonError(500, "Internal Server Error");
  });

  app.use(
    "/api/*",
    cors({
      origin: (origin) => (allowedOrigins.includes(origin) ? origin : (allowedOrigins[0] ?? "")),
    }),
  );

  app.get(
    "/api/live",
    upgradeWebSocket(() => ({
      onOpen: (_event, ws) => input.live.broadcaster.add(ws),
      onClose: (_event, ws) => input.live.broadcaster.remove(ws),
      onError: (_event, ws) => input.live.broadcaster.remove(ws),
    })),
  );

  app.get("/api/state", async (c) => {
    const auditRows = input.borg.audit.list();

    return c.json({
      active_session: DEFAULT_SESSION_ID,
      audiences: listAudiences(input.borg),
      counts: {
        turns: await countTurns(input.borg),
        commitments: input.borg.commitments.countActive(),
        open_qs: input.borg.self.openQuestions.list({ status: "open" }).length,
        dream_audit_rows: auditRows.length,
      },
      current_mood: input.borg.mood.current(DEFAULT_SESSION_ID),
      version: VERSION,
    });
  });

  app.get("/api/stream", async (c) => {
    const query = parseRequest(streamQuerySchema, c.req.query());
    return c.json(
      await readStream({
        borg: input.borg,
        kinds: query.kind,
        audience: query.audience,
        limit: query.limit,
        before: query.before,
      }),
    );
  });

  app.get("/api/turns/:id/ledger", (c) => {
    const turnId = c.req.param("id");
    const ledger = input.live.ledgerCache.get(turnId);

    if (ledger === undefined) {
      throw new HTTPException(404, { message: "ledger not found" });
    }

    return c.json({ turn_id: turnId, ledger });
  });

  app.get("/api/memory/bands", async (c) => c.json({ bands: await memoryBands(input.borg) }));

  app.get("/api/memory/bands/:id", async (c) => {
    const band = c.req.param("id");

    if (band === "episodic") {
      return c.json({ band, ...(await input.borg.episodic.list({ limit: 50 })) });
    }

    if (band === "semantic") {
      return c.json({
        band,
        nodes: await input.borg.semantic.nodes.list({ limit: 50 }),
        edges: input.borg.semantic.edges.list().slice(0, 50),
      });
    }

    if (band === "commitments") {
      return c.json({ band, items: input.borg.commitments.list({ activeOnly: true }) });
    }

    if (band === "self") {
      return c.json({ band, ...selfSnapshot(input.borg) });
    }

    return c.json({ band, items: [], note: "not yet implemented" });
  });

  app.get("/api/commitments", (c) => {
    const query = parseRequest(audienceQuerySchema, c.req.query());

    if (query.audience !== undefined) {
      const entity = input.borg.entities.find(query.audience);

      if (entity === null) {
        return c.json({ commitments: [] });
      }

      return c.json({
        commitments: input.borg.commitments.list({
          activeOnly: true,
          audience: entity.canonical_name,
        }),
      });
    }

    return c.json({ commitments: input.borg.commitments.list({ activeOnly: true }) });
  });

  app.get("/api/shared-state", (c) => {
    const query = parseRequest(audienceQuerySchema, c.req.query());
    const audience = query.audience ?? "self";
    return c.json({ audience, entries: input.borg.sharedState.listEntriesForAudience(audience) });
  });

  app.get("/api/identity", (c) => c.json(selfSnapshot(input.borg)));

  app.get("/api/dream/audit", (c) => {
    const query = parseRequest(auditQuerySchema, c.req.query());
    return c.json({ rows: input.borg.audit.list().slice(0, query.limit) });
  });

  app.get("/api/attachments/:id/bytes", (c) => {
    const query = parseRequest(attachmentQuerySchema, c.req.query());
    const result = input.borg.attachments.getBytes(c.req.param("id") as AttachmentId, {
      audience: query.audience,
    });

    if (result === null) {
      throw new HTTPException(404, { message: "attachment not found" });
    }

    return new Response(result.bytes, {
      status: 200,
      headers: {
        "Content-Type": result.mediaType,
        "Content-Length": String(result.bytes.byteLength),
      },
    });
  });

  app.post("/api/turn", async (c) => {
    const body = parseRequest(turnBodySchema, await parseJsonBody(c));
    const result = await input.borg.turn({
      userMessage: body.message,
      audience: body.audience,
      stakes: body.stakes,
    });

    return c.json({ turn_id: result.turn_id, ok: true });
  });

  return { app, injectWebSocket };
}
