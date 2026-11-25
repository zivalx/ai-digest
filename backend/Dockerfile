# ai-digest production image: FastAPI backend (feeder scheduler inside) + the built React UI.
#
# BUILD CONTEXT IS THE REPO ROOT (the UI lives in ../frontend):
#   docker build -f backend/Dockerfile .          (deploy.sh / docker-compose.prod.yml do this)
# backend/collectors_src/ must exist first: deploy.sh copies ../collectors/collectors there.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG NODE_IMAGE=node:20-alpine

# ---- UI build: static files only, the Node toolchain is not in the final image -------------
FROM ${NODE_IMAGE} AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/public ./public
COPY frontend/src ./src
# Empty API URL = same origin (the page and /api are served by the same app behind the edge).
# The heap cap keeps the build inside a 4 GB box that is also running two other apps.
ENV REACT_APP_API_URL="" \
    GENERATE_SOURCEMAP=false \
    NODE_OPTIONS=--max-old-space-size=1024
RUN npm run build

# ---- App -----------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}

WORKDIR /app

# ffmpeg: only used on the optional Whisper path (YOUTUBE_WHISPER_MODEL); libpq/gcc: build deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -r digest && chown digest:digest /app && mkdir /data && chown digest:digest /data

# Python deps first (layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# collectors library ("connectors" package), copied in by deploy.sh from ../collectors/collectors
COPY backend/collectors_src/ /tmp/collectors_src/
RUN pip install --no-cache-dir "/tmp/collectors_src[all]" && rm -rf /tmp/collectors_src

# App source (only what runs: no tests, no collectors_src, no legacy digest_runner) and the built UI
COPY --chown=digest:digest backend/src ./src
COPY --chown=digest:digest backend/alembic ./alembic
COPY --chown=digest:digest backend/alembic.ini backend/init_db.py ./
COPY --from=ui --chown=digest:digest /ui/build /app/ui

USER digest
ENV UI_DIR=/app/ui \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

# One worker only: the feeder scheduler lives in this process (more workers = more feeders).
# --proxy-headers: the edge terminates TLS, so redirects must keep https and the real host.
# `exec` lets docker's SIGTERM reach uvicorn (graceful stop).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn src.main:app --host 0.0.0.0 --port 8001 --proxy-headers --forwarded-allow-ips='*'"]
