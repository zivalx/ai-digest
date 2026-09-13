#!/bin/bash
# Prep build context (collectors source must be a sibling dir) then deploy.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COLLECTORS_SRC="$SCRIPT_DIR/../collectors/collectors"
BUILD_CONTEXT="$SCRIPT_DIR/backend/collectors_src"

echo "Copying collectors source into build context..."
rm -rf "$BUILD_CONTEXT"
cp -r "$COLLECTORS_SRC" "$BUILD_CONTEXT"

echo "Building and starting ai-digest..."
docker compose -f "$SCRIPT_DIR/docker-compose.prod.yml" up --build -d

echo "Cleaning up collectors copy..."
rm -rf "$BUILD_CONTEXT"

echo "Done. Logs: docker compose -f docker-compose.prod.yml logs -f app"
