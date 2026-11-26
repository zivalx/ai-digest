"""App wiring: deep health route and serving the built UI from the same origin."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import main


def _ui_dir(tmp_path):
    (tmp_path / "static").mkdir()
    (tmp_path / "index.html").write_text("<html><body>sources ui</body></html>")
    (tmp_path / "static" / "app.js").write_text("console.log(1)")
    return tmp_path


def test_mount_ui_serves_index_and_assets_next_to_the_api(tmp_path):
    app = FastAPI()

    @app.get("/api/ping")
    def ping():
        return {"pong": True}

    assert main.mount_ui(app, str(_ui_dir(tmp_path))) is True
    client = TestClient(app)
    assert "sources ui" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/ping").json() == {"pong": True}  # API routes win over the mount


def test_mount_ui_without_a_build_does_nothing(tmp_path):
    app = FastAPI()
    assert main.mount_ui(app, str(tmp_path / "missing")) is False
    assert main.mount_ui(app, None) is False
    assert main.mount_ui(app, str(tmp_path)) is False  # a directory with no index.html


def test_deep_health_route_maps_down_to_503(monkeypatch):
    import src.health as health

    async def fake(**kwargs):
        return {"status": "down", "checks": {"database": {"ok": False, "detail": "unreachable"}}}

    monkeypatch.setattr(health, "deep_health", fake)
    client = TestClient(main.app)
    response = client.get("/health/deep")
    assert response.status_code == 503
    assert response.json()["status"] == "down"


def test_deep_health_route_ok_is_200(monkeypatch):
    import src.health as health

    async def fake(**kwargs):
        return {"status": "ok", "checks": {}}

    monkeypatch.setattr(health, "deep_health", fake)
    assert TestClient(main.app).get("/health/deep").status_code == 200


def test_cors_origins_come_from_env(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert main.cors_origins() == ["http://localhost:3000", "http://localhost:3001"]
    monkeypatch.setenv("CORS_ORIGINS", "https://digest.zivalx.com, https://x.example")
    assert main.cors_origins() == ["https://digest.zivalx.com", "https://x.example"]
