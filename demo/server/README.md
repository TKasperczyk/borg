# Borg Demo Server

LAN-only Hono backend for the visual demo. Build Borg first with `pnpm --filter borg build`, then run `BORG_DATA_DIR=/path/to/data pnpm dev:server`; the server exposes REST under `/api` and unauthenticated live WebSocket events at `/api/live`.
