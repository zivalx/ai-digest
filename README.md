# Daigest - Multi-Source AI Summarization Platform

Daigest is a powerful platform for collecting data from multiple sources (Reddit, YouTube, Telegram, Twitter, Google News, Google Trends) and generating AI-powered summaries using LangChain and various LLM providers.

## Features

- **Multi-Source Data Collection**: Reddit, YouTube, Telegram, Twitter, GNews, PyTrends
- **LLM-Powered Summarization**: Support for OpenAI, Anthropic, and local models via LangChain
- **Cycle Management**: Track and manage collection + summarization cycles
- **Cost Tracking**: Monitor token usage and API costs
- **React Frontend**: User-friendly interface for configuration and viewing summaries
- **PostgreSQL Storage**: Persistent storage of collected data and summaries

## Architecture

```
┌─────────────────────────────────────┐
│  Frontend (React + Material-UI)    │
│  - Source Configuration             │
│  - Cycle History                    │
│  - Summary Viewer                   │
└──────────────┬──────────────────────┘
               │ REST API
┌──────────────┴──────────────────────┐
│  Backend (FastAPI + SQLAlchemy)     │
│  - Collection Orchestrator          │
│  - LangChain Summary Service        │
│  - PostgreSQL Database              │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│  Connectors Library (External)      │
│  RedditCollector | YouTubeCollector │
│  TelegramCollector | TwitterCollector│
│  GNewsCollector | PyTrendsCollector  │
└─────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL 13+ (or SQLite for development)
- API keys for desired sources and LLM providers

### Backend Setup

1. **Clone the repository and navigate to backend:**

```bash
cd backend
```

2. **Create and activate virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env and fill in your API keys and database URL
```

5. **Initialize the database:**

```bash
# Create tables
python init_db.py

# Optional: Seed with sample data
python init_db.py --seed

# Reset database (WARNING: deletes all data)
python init_db.py --reset
```

6. **Run the backend server:**

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. **Navigate to frontend directory:**

```bash
cd frontend
```

2. **Install dependencies:**

```bash
npm install
```

3. **Configure API URL (optional):**

Create `.env.local` file:

```bash
REACT_APP_API_URL=http://localhost:8000
```

4. **Start the development server:**

```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

### 1. Configure Data Sources

Go to the "Configure Sources" tab and add source configurations:

- **Source Type**: Select from Reddit, YouTube, Telegram, Twitter, GNews, or PyTrends
- **Credential Reference**: Environment variable prefix (e.g., `REDDIT_CLIENT_1`)
- **Collection Parameters**: Specify what to collect (subreddits, channels, keywords, etc.)

### 2. Trigger Collection Cycles

Use the API or frontend to create a new cycle:

**API Example:**

```bash
curl -X POST http://localhost:8000/api/cycles/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Weekly Tech Digest",
    "sources": [
      {
        "source_type": "reddit",
        "credential_ref": "REDDIT_CLIENT_1",
        "collect_spec": {
          "subreddits": ["python", "programming"],
          "max_posts": 50,
          "sort": "hot"
        }
      }
    ],
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini"
  }'
```

### 3. View Summaries

Navigate to "Cycle History & Summaries" tab to:

- View all past cycles
- See collection metrics (items collected, time taken, data size)
- Read AI-generated summaries
- Track costs and token usage

## API Endpoints

### Cycles

- `POST /api/cycles/` - Create new cycle
- `GET /api/cycles/` - List all cycles (paginated)
- `GET /api/cycles/{id}` - Get cycle details + summary
- `DELETE /api/cycles/{id}` - Delete cycle

### Source Configurations

- `POST /api/configs/` - Create source config
- `GET /api/configs/` - List all configs
- `GET /api/configs/{id}` - Get specific config
- `PUT /api/configs/{id}` - Update config
- `DELETE /api/configs/{id}` - Delete config

### Health Check

- `GET /` - API status
- `GET /health` - Detailed health check

## Configuration Reference

### Source-Specific Parameters

#### Reddit
```json
{
  "subreddits": ["python", "programming"],
  "sort": "hot",  // or "new", "top", "rising"
  "max_posts": 50,
  "include_comments": false
}
```

#### YouTube
```json
{
  "channels": ["@Veritasium", "@VSauce"],
  "max_videos": 10,
  "days_back": 7,
  "use_transcript_api": true
}
```

#### Telegram
```json
{
  "channels": ["channel1", "channel2"],
  "max_messages": 200,
  "include_replies": true
}
```

#### Twitter
```json
{
  "query": "python OR programming",
  "max_results": 100
}
```

#### Google News
```json
{
  "query": "artificial intelligence",
  "language": "en",
  "max_results": 10,
  "sort_by": "publishedAt"
}
```

#### Google Trends
```json
{
  "keywords": ["bitcoin", "ethereum"],
  "timeframe": "today 3-m",
  "geo": "US"
}
```

## Database Schema

### Tables

- **cycles**: Collection/summarization runs
- **collected_data**: Raw data from connectors (JSONB)
- **summaries**: AI-generated summaries with metadata
- **source_configs**: Reusable source configurations

### Migrations

Using Alembic for database migrations:

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## LLM Providers

### OpenAI

```python
llm_provider="openai"
llm_model="gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo"
```

### Anthropic (Claude)

```python
llm_provider="anthropic"
llm_model="claude-3-5-sonnet-20241022"  # or "claude-3-haiku-20240307"
```

## Cost Tracking

Daigest automatically tracks:

- Input/output tokens
- Estimated cost per summary (based on pricing table)
- Generation time
- Data size

View costs in the Summary Viewer or query the database:

```sql
SELECT llm_provider, model_name, SUM(cost_usd) AS total_cost
FROM summaries
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY llm_provider, model_name;
```

## Deployment (Hetzner, behind the zivalx edge)

ai-digest is the third app on the shared box, after Navigator and trender. TLS, basic auth,
backups and the watchdog belong to the `zivalx-ops` repo (`runbooks/first-deploy.md`,
`runbooks/round-2026-11.md`); this repo only ships the app.

- `docker-compose.prod.yml`: `ai-digest-db` (Postgres 16) and `ai-digest-app` (FastAPI, the
  feeder scheduler, and the built UI at `/`). No published ports; the app joins the external
  `edge` network, so the edge reaches it as `ai-digest-app:8001` and it reaches trender as
  `http://trender-app:8000`. The UI is `https://digest.zivalx.com` (Sources page).
- `.env.example`: every setting, with comments. On the server it becomes `/opt/ai-digest/.env`
  (0600). `TRENDER_INGEST_TOKEN` must equal trender's.
- `deploy.sh` (run on the server from `/opt/ai-digest`): pull, pre-deploy dump, build, wait for
  `/health/deep`. The build needs the collectors library checked out at `/opt/collectors`
  (it copies the source into `backend/collectors_src` and removes it afterwards).
- `GET /health/deep`: `{"status": "ok|degraded|down", "checks": {...}}` for the watchdog. It
  checks the database and the age of the last finished feeder cycle (limit 2 x interval + 30 min).
- The container runs `alembic upgrade head` on every start, then uvicorn with one worker (the
  feeder lives in that process).

### Catch-up backfill

The feeder looks at the newest few items of each source every 6 hours. After a gap, run once:

```
docker exec ai-digest-app python -m src.backfill --days 30 --dry-run   # what would be processed
docker exec ai-digest-app python -m src.backfill --days 30             # do it
docker exec ai-digest-app python -m src.backfill --days 30 --source "Finance YT" --max-items 80
```

Capped at 60 days and 200 items per channel. It reuses the feeder's ledger, so reruns skip what
is done and nothing is pushed twice. The YouTube collector takes the newest N videos per channel
and does not itself stop at a date, so the real reach is N videos per channel (default 40), then
filtered to the window. LLM calls cost money: dry-run first. See the docstring in
`backend/src/backfill.py`.

## Development

### Code Quality

```bash
cd backend

# Format code
black src/

# Lint
ruff check src/

# Type check
mypy src/
```

## Troubleshooting

### Database Connection Issues

- Verify `DATABASE_URL` in `.env`
- Ensure PostgreSQL is running
- Check database credentials

### API Key Errors

- Verify API keys in `.env`
- Ensure credential references match environment variable names
- Check API quotas and rate limits

### Frontend Not Loading

- Verify backend is running on port 8000
- Check CORS configuration in `backend/src/main.py`
- Clear browser cache

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Issues: https://github.com/yourusername/daigest/issues
- Documentation: https://daigest.readthedocs.io

---

**Built with:**
- FastAPI
- SQLAlchemy
- LangChain
- React
- Material-UI
- Connectors Library
