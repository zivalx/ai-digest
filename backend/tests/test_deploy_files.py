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


# --- optional local-only corporate CA build secret (same pattern as Navigator and trender) ---------

DOCKERFILE = REPO / "backend" / "Dockerfile"


def _logical_lines(path):
    """Instructions with comments dropped and backslash continuations joined."""
    out, cur = [], ""
    for ln in path.read_text().splitlines():
        if ln.lstrip().startswith("#"):
            continue
        cur = f"{cur} {ln.strip()}" if cur else ln
        if cur.endswith("\\"):
            cur = cur[:-1].rstrip()
            continue
        out.append(cur)
        cur = ""
    return out


def _download_runs():
    return [ln for ln in _logical_lines(DOCKERFILE)
            if ln.startswith("RUN ") and ("pip install" in ln or "npm ci" in ln)]


def test_every_download_run_mounts_the_optional_corp_ca_secret():
    runs = _download_runs()
    assert len(runs) == 3, runs  # npm ci, requirements.txt, collectors
    for ln in runs:
        assert "--mount=type=secret,id=corp_ca,required=false" in ln
        assert "/run/secrets/corp_ca" in ln


def test_pip_runs_build_a_temp_bundle_from_system_roots_and_clean_up():
    for ln in (r for r in _download_runs() if "pip install" in r):
        assert "/etc/ssl/certs/ca-certificates.crt /run/secrets/corp_ca > /tmp/ca-bundle.pem" in ln
        assert "trap 'rm -f /tmp/ca-bundle.pem' EXIT" in ln
        assert "|| exit 1" in ln
        assert "PIP_CERT=/tmp/ca-bundle.pem" in ln


def test_npm_uses_node_extra_ca_certs_from_the_secret():
    (npm,) = [r for r in _download_runs() if "npm ci" in r]
    assert "NODE_EXTRA_CA_CERTS=/run/secrets/corp_ca" in npm


def test_no_cert_copied_into_the_image_or_set_in_env():
    for ln in _logical_lines(DOCKERFILE):
        if re.match(r"(COPY|ADD)\b", ln):
            assert not re.search(r"\.(pem|crt|cer)\b|corp_ca", ln), ln
        if re.match(r"(ENV|ARG)\b", ln):
            assert not re.search(r"corp_ca|SSL_CERT|PIP_CERT|CA_CERTS|\.pem", ln, re.I), ln
    text = DOCKERFILE.read_text()
    assert "update-ca-certificates" not in text
    assert "/usr/local/share/ca-certificates" not in text


def test_deploy_script_passes_the_secret_only_when_corp_ca_is_set():
    text = (REPO / "deploy.sh").read_text()
    assert text.count("--secret") == 1
    _, rest = text.split('if [ -n "${CORP_CA:-}" ]; then', 1)
    branch, other = rest.split("\nelse\n", 1)
    assert '--secret id=corp_ca,src="${CORP_CA}"' in branch
    assert "--secret" not in other
    assert "up -d --build" in other and "up -d --build" not in branch
