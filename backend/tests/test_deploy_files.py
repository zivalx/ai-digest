"""
Static checks on the production deploy files (no Docker needed).

The shape must match Navigator and trender on the shared box: fixed container names, no
published ports, the external `edge` network, a project-prefixed volume, an env template that
lists everything the code reads.
"""
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "backend" / "src"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((REPO / "docker-compose.prod.yml").read_text())


def test_container_names_are_fixed_and_match_the_ops_repo(compose):
    assert compose["name"] == "ai-digest"
    assert compose["services"]["db"]["container_name"] == "ai-digest-db"
    assert compose["services"]["digest"]["container_name"] == "ai-digest-app"


def test_no_service_publishes_ports(compose):
    for name, svc in compose["services"].items():
        assert "ports" not in svc, name


def test_app_is_on_the_external_edge_network_and_db_is_not(compose):
    assert compose["networks"]["edge"] == {"external": True}
    assert "edge" in compose["services"]["digest"]["networks"]
    assert "edge" not in (compose["services"]["db"].get("networks") or [])


def test_service_is_not_called_app_because_trender_already_is(compose):
    assert "app" not in compose["services"]


def test_restart_healthchecks_and_limits(compose):
    for name, svc in compose["services"].items():
        assert svc["restart"] == "unless-stopped", name
        assert "healthcheck" in svc, name
        assert svc["mem_limit"], name
    app = compose["services"]["digest"]
    assert app["depends_on"]["db"]["condition"] == "service_healthy"
    assert "/health" in " ".join(app["healthcheck"]["test"])


def test_volume_name_is_stable_and_project_prefixed(compose):
    assert compose["volumes"]["pgdata"]["name"] == "ai-digest_pgdata"
    assert "pgdata:/var/lib/postgresql/data" in compose["services"]["db"]["volumes"]


def test_env_file_is_dot_env_and_feeder_defaults_on(compose):
    app = compose["services"]["digest"]
    assert app["env_file"] == ".env"
    assert app["environment"]["FEEDER_ENABLED"] == "${FEEDER_ENABLED:-true}"
    assert app["environment"]["DATABASE_URL"].startswith("postgresql+asyncpg://")
    # the password is required, not defaulted
    assert ":?" in compose["services"]["db"]["environment"]["POSTGRES_PASSWORD"]


def test_db_credentials_match_what_the_backup_reads(compose):
    """zbackup reads POSTGRES_USER / POSTGRES_DB from the db container's own environment."""
    env = compose["services"]["db"]["environment"]
    assert set(env) >= {"POSTGRES_USER", "POSTGRES_DB", "POSTGRES_PASSWORD"}


def test_dockerfile_copies_only_paths_that_exist():
    text = (REPO / "backend" / "Dockerfile").read_text()
    for src in re.findall(r"^COPY (?:--\S+ )*(\S+(?: \S+)*?) \S+$", text, re.M):
        for part in src.split():
            if part.startswith("backend/collectors_src") or part.startswith("/"):
                continue  # created by deploy.sh / a path inside another build stage
            assert (REPO / part.rstrip("/")).exists(), part
    assert "alembic upgrade head" in text
    assert "--proxy-headers" in text
    assert "USER digest" in text


def test_dockerignore_keeps_secrets_out_of_the_context():
    lines = (REPO / ".dockerignore").read_text().splitlines()
    for pattern in ("**/.env", "**/.env.*", "**/*.session"):
        assert pattern in lines
    assert "backend/collectors_src" not in lines  # the build needs it


def test_requirements_do_not_pull_the_wrong_connectors_package():
    reqs = [l for l in (REPO / "backend" / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    assert not [l for l in reqs if l.lower().startswith("connectors")]


def test_env_template_lists_every_variable_the_code_reads():
    template = (REPO / ".env.example").read_text()
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", template, re.M))
    read = set()
    for path in SRC.rglob("*.py"):
        read |= set(re.findall(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]+)"', path.read_text()))
    # Derived per source config (<credential_ref>_API_ID ...) and not settings:
    read -= {"DATABASE_URL", "TELEGRAM_SESSION_PATH", "UI_DIR"}  # set by compose / the Dockerfile
    assert read <= documented, sorted(read - documented)
    for needed in ("TRENDER_BASE_URL", "TRENDER_INGEST_TOKEN", "POSTGRES_PASSWORD",
                   "FEEDER_ENABLED", "YOUTUBE_WHISPER_MODEL", "ANTHROPIC_API_KEY"):
        assert needed in documented


def test_env_template_points_at_trender_over_the_internal_network():
    assert re.search(r"^TRENDER_BASE_URL=http://trender-app:8000$",
                     (REPO / ".env.example").read_text(), re.M)
