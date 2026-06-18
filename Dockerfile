# Image for the borg memory sidecar service (scripts/memory-sidecar.ts).
#
# Multi-stage: a builder installs dependencies for Node 22, then a slim runtime
# carries the result. Dependencies MUST be installed in-image — never copy host
# node_modules, because platform-specific package layouts can differ.
#
# Build with: docker build --network=host (the build network can't resolve the npm
# registry otherwise).

# ---- builder: install dependencies -------------------------------------------
FROM node:22-bookworm AS builder

WORKDIR /app

# Full source: tsx is a devDependency (the CMD runs `node --import tsx`), so dev
# deps are required — do NOT use --prod.
COPY . .

RUN npm ci

# ---- runtime: slim base, compiled result only --------------------------------
FROM node:22-bookworm-slim

# No apt needed: Node ships its own root CA bundle (tls.rootCertificates) for
# public TLS, and the private play-root-ca is trusted at runtime via
# NODE_EXTRA_CA_CERTS — so ca-certificates is unnecessary. OpenShift runs this
# under restricted-v2 with an arbitrary non-root UID and an auto-assigned fsGroup;
# the process only writes to the fsGroup-owned PVC (BORG_DATA_ROOT) and /tmp, and
# /app is world-readable, so no USER/chown is needed.

WORKDIR /app

# Carry the installed workspace.
COPY --from=builder /app /app

ENV NODE_ENV=production \
    BORG_MEMORY_HOST=0.0.0.0 \
    BORG_MEMORY_PORT=8088

EXPOSE 8088
CMD ["node", "--import", "tsx", "scripts/memory-sidecar.ts"]
