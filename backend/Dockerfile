FROM python:3.12-slim

WORKDIR /app

# System deps for yt-dlp, faster-whisper (ffmpeg), and asyncpg (libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install collectors package (copied in by deploy.sh from ../collectors/collectors)
COPY collectors_src/ /tmp/collectors_src/
RUN pip install --no-cache-dir "/tmp/collectors_src[all]" && rm -rf /tmp/collectors_src

# App source
COPY . .

# Run migrations then start
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8001"]
