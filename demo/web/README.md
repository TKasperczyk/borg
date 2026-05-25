# Borg Demo Web

Vite + React console for the P0 demo server.

Run the backend first:

```sh
BORG_DATA_DIR=/path/to/data pnpm dev:server
```

Then run the frontend:

```sh
pnpm dev:web
```

The app defaults to `http://localhost:7740` for REST and derives `ws://localhost:7740` for live events. Override with:

```sh
VITE_BORG_API_BASE=http://localhost:7740
VITE_BORG_WS_BASE=ws://localhost:7740
```

Build and tests:

```sh
pnpm --filter @borg/demo-web test
pnpm --filter @borg/demo-web build
```
