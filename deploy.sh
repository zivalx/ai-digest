#!/usr/bin/env bash
# Run ON the server, from /opt/ai-digest:  /opt/ai-digest/deploy.sh
# TLS, basic auth and backups live in the ops repo (/opt/zivalx-ops); this only updates the app.
#
# Needs next to this checkout a checkout of the collectors library (the "connectors" package):
#   /opt/ai-digest   this repo
#   /opt/collectors  github.com/zivalx/collectors, branch with the transcript-segment and lazy
#                    Whisper commits (feat/channel-intel-feeder at the time of writing)
# The build copies its source into backend/collectors_src for the Docker build and removes it
# afterwards. On a dev Mac the layout is ../collectors/collectors (the inner dir is the git root).
#
# Environment (all optional):
#   COLLECTORS_SRC       path of the collectors checkout (default: auto-detected as above)
#   DEPLOY_NO_PULL=1     do not git pull
#   DEPLOY_PREPARE_ONLY=1  only prepare (and clean up) the build context, then stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${SCRIPT_DIR}"
COMPOSE="docker compose -f docker-compose.prod.yml"
ZBACKUP="/opt/zivalx-ops/backup/zbackup"
DUMP_DIR="/var/backups/ai-digest"
BUILD_CONTEXT="${APP_DIR}/backend/collectors_src"

cd "${APP_DIR}"

find_collectors() {
    if [ -n "${COLLECTORS_SRC:-}" ]; then
        echo "${COLLECTORS_SRC}"
    elif [ -f "${APP_DIR}/../collectors/pyproject.toml" ]; then
        echo "${APP_DIR}/../collectors"
    elif [ -f "${APP_DIR}/../collectors/collectors/pyproject.toml" ]; then
        echo "${APP_DIR}/../collectors/collectors"
    fi
}

COLLECTORS="$(find_collectors)"
if [ -z "${COLLECTORS}" ] || [ ! -f "${COLLECTORS}/pyproject.toml" ] || [ ! -d "${COLLECTORS}/src/connectors" ]; then
    echo "ERROR: collectors checkout not found (looked for ../collectors and ../collectors/collectors)." >&2
    echo "       On the server: git clone https://github.com/zivalx/collectors /opt/collectors" >&2
    echo "       (then check out the branch named at the top of this script), or set COLLECTORS_SRC." >&2
    exit 1
fi

if [ "${DEPLOY_NO_PULL:-0}" != "1" ] && [ -d "${APP_DIR}/.git" ]; then
    echo "==> Pulling latest code"
    git pull --ff-only
fi

echo "==> Preparing the build context from ${COLLECTORS}"
if [ -d "${COLLECTORS}/.git" ] || git -C "${COLLECTORS}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "    collectors commit: $(git -C "${COLLECTORS}" rev-parse --short HEAD)" \
         "($(git -C "${COLLECTORS}" branch --show-current 2>/dev/null || true))"
fi
rm -rf "${BUILD_CONTEXT}"
mkdir -p "${BUILD_CONTEXT}"
trap 'rm -rf "${BUILD_CONTEXT}"' EXIT
# Only what pip needs: the package, pyproject and the README it names. No tests, docs, caches.
tar -C "${COLLECTORS}" \
    --exclude=.git --exclude=.venv --exclude=build --exclude=output --exclude=docs \
    --exclude=examples --exclude=tests --exclude=__pycache__ --exclude='*.egg-info' \
    --exclude=.pytest_cache --exclude=.coverage --exclude=.env --exclude='.env.*' \
    -cf - . | tar -C "${BUILD_CONTEXT}" -xf -

if [ "${DEPLOY_PREPARE_ONLY:-0}" = "1" ]; then
    echo "==> DEPLOY_PREPARE_ONLY set: build context ready, stopping (it is removed on exit)"
    ls "${BUILD_CONTEXT}"
    exit 0
fi

echo "==> Checking the edge network"
if ! docker network inspect edge >/dev/null 2>&1; then
    echo "ERROR: docker network 'edge' does not exist." >&2
    echo "       Bring up the ops edge first: docker compose -f /opt/zivalx-ops/edge/compose.yml up -d" >&2
    exit 1
fi

if [ ! -f "${APP_DIR}/.env" ]; then
    echo "ERROR: ${APP_DIR}/.env is missing: copy .env.example and fill it in (mode 0600)." >&2
    exit 1
fi

echo "==> Pre-deploy database dump"
if [ "$(docker inspect -f '{{.State.Running}}' ai-digest-db 2>/dev/null || true)" != "true" ]; then
    echo "    ai-digest-db is not running yet (first deploy?) - skipping the dump"
elif [ -x "${ZBACKUP}" ]; then
    "${ZBACKUP}" dump ai-digest
else
    mkdir -p "${DUMP_DIR}"
    dump_file="${DUMP_DIR}/predeploy-$(date +%F-%H%M).dump"
    if ! docker exec ai-digest-db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "${dump_file}"; then
        rm -f "${dump_file}"
        echo "ERROR: pre-deploy dump failed; not deploying" >&2
        exit 1
    fi
    echo "    wrote ${dump_file}"
fi

echo "==> Building and starting services (the first build downloads Node and Python packages)"
${COMPOSE} up -d --build

echo "==> Waiting for deep health..."
for _ in $(seq 1 90); do
    if docker exec ai-digest-app python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health/deep')" >/dev/null 2>&1; then
        echo "==> Healthy"
        exit 0
    fi
    sleep 2
done

echo "==> WARNING: deep health not passing after 180s - check logs:" >&2
echo "    cd ${APP_DIR} && ${COMPOSE} logs --tail=50 digest" >&2
exit 1
