# Image for the borg memory sidecar service (scripts/memory-sidecar.ts).
#
# Multi-stage: a builder compiles the native deps (better-sqlite3, @lancedb/lancedb)
# in-image for this Node ABI, then a slim runtime carries only the result. Native
# deps MUST be compiled in-image — never copy host node_modules (a NODE_MODULE_VERSION
# mismatch crashes at import; we hit exactly that locally with Node 26 vs a v141
# binding). Both stages are node:22 on Debian 12, so the compiled .node addon's ABI
# (NODE_MODULE_VERSION 127) and glibc match across the COPY.
#
# Build with: docker build --network=host (the build network can't resolve the npm
# registry otherwise).

# ---- builder: toolchain + native compile -------------------------------------
FROM node:22-bookworm AS builder

# node-gyp toolchain for the better-sqlite3 native build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 make g++ \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g pnpm@11.3.0 node-gyp

WORKDIR /app

# Full source: this is a pnpm workspace, so the lockfile + workspace package.jsons
# must all be present for a --frozen-lockfile install. tsx is a devDependency
# (the CMD runs `pnpm exec tsx`), so dev deps are required — do NOT use --prod.
COPY . .

# Install without lifecycle scripts (pnpm 11 hard-errors on un-approved build
# scripts otherwise), then compile better-sqlite3's native addon explicitly for
# this image's Node. esbuild's platform binary and @lancedb/lancedb's prebuilt
# napi binary are regular (optional) dependency packages, fetched during install.
RUN pnpm install --frozen-lockfile --ignore-scripts \
 && cd node_modules/.pnpm/better-sqlite3@*/node_modules/better-sqlite3 \
 && npm_config_build_from_source=true node-gyp rebuild

# ---- runtime: slim base, compiled result only --------------------------------
FROM node:22-bookworm-slim

# No apt needed: Node ships its own root CA bundle (tls.rootCertificates) for
# public TLS, and the private play-root-ca is trusted at runtime via
# NODE_EXTRA_CA_CERTS — so ca-certificates is unnecessary. No pnpm needed either:
# the entrypoint runs tsx directly via Node's loader (`node --import tsx`), which
# resolves the tsx devDependency from node_modules and does NOT go through pnpm's
# build-script verification gate (which hard-errors after an --ignore-scripts
# install). OpenShift runs this under restricted-v2 with an arbitrary non-root UID
# and an auto-assigned fsGroup; the process only writes to the fsGroup-owned PVC
# (BORG_DATA_ROOT) and /tmp, and /app is world-readable, so no USER/chown is needed.

WORKDIR /app

# Carry the built workspace (source + node_modules with the compiled native addon).
COPY --from=builder /app /app

ENV NODE_ENV=production \
    BORG_MEMORY_HOST=0.0.0.0 \
    BORG_MEMORY_PORT=8088

EXPOSE 8088
CMD ["node", "--import", "tsx", "scripts/memory-sidecar.ts"]
