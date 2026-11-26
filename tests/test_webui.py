"""Unit and integration tests for the schema-driven Config WebUI."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from spectra.__main__ import (
    _load_webui_config_for_standalone,
    _run_bot,
    _run_webui_only,
    main,
)
from spectra.config import AppConfig, WebUIConfig, load_config, save_config
from spectra.webui import WebUIServer, create_webui_app


async def test_static_assets_and_path_traversal_blocked(tmp_path: Path) -> None:
    app = create_webui_app(WebUIConfig(), config_path=tmp_path / "config.json")
    client = app.test_client()

    index_resp = await client.get("/")
    assert index_resp.status_code == 200
    assert "text/html" in index_resp.content_type
    assert index_resp.headers["X-Content-Type-Options"] == "nosniff"
    assert index_resp.headers["X-Frame-Options"] == "DENY"
    html = await index_resp.get_data(as_text=True)
    assert "Spectra-bot Config Studio" in html
    assert "/static/webui.js" in html
    assert "/static/webui.css" in html

    css_resp = await client.get("/static/webui.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.content_type

    js_resp = await client.get("/static/webui.js")
    assert js_resp.status_code == 200
    assert "javascript" in js_resp.content_type

    # Path traversal and unknown assets must be rejected cleanly with 404.
    for bad_path in (
        "/static/../config.py",
        "/static/..%2Fconfig.py",
        "/static/__init__.py",
        "/static/nonexistent.js",
    ):
        resp = await client.get(bad_path)
        assert resp.status_code == 404


async def test_meta_is_public_and_never_leaks_secrets(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    save_config(
        AppConfig(
            llm={"api_key": "sk-super-secret-llm"},
            webui={"auth_token": "tok-super-secret"},
        ),
        cfg_path,
    )
    app = create_webui_app(
        WebUIConfig(auth_token="tok-super-secret"),
        config_path=cfg_path,
        bot_running=True,
    )
    client = app.test_client()

    resp = await client.get("/api/meta")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-store"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"

    body_text = await resp.get_data(as_text=True)
    assert "sk-super-secret-llm" not in body_text
    assert "tok-super-secret" not in body_text

    data = await resp.get_json()
    assert data == {
        "version": data["version"],
        "config_path": str(cfg_path),
        "auth_required": True,
        "bot_running": True,
    }
    assert isinstance(data["version"], str) and data["version"]


async def test_auth_token_enforced_across_all_config_endpoints(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    token = "my-secret-webui-token"
    app = create_webui_app(WebUIConfig(auth_token=token), config_path=cfg_path)
    client = app.test_client()

    endpoints: list[tuple[str, str, dict[str, Any] | None]] = [
        ("GET", "/api/config/schema", None),
        ("GET", "/api/config/defaults", None),
        ("GET", "/api/config", None),
        ("POST", "/api/config/validate", AppConfig().model_dump(mode="json")),
        ("PUT", "/api/config", AppConfig().model_dump(mode="json")),
    ]

    for method, path, payload in endpoints:
        # Missing token -> 401
        resp = await client.open(path, method=method, json=payload)
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        assert resp.headers["Cache-Control"] == "no-store"

        # Wrong token -> 401
        resp_bad = await client.open(
            path,
            method=method,
            json=payload,
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp_bad.status_code == 401

        # Valid Authorization: Bearer -> 200
        resp_bearer = await client.open(
            path,
            method=method,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp_bearer.status_code == 200

        # Valid X-Spectra-Token -> 200
        resp_header = await client.open(
            path,
            method=method,
            json=payload,
            headers={"X-Spectra-Token": token},
        )
        assert resp_header.status_code == 200


async def test_schema_defaults_and_missing_config_behavior(tmp_path: Path) -> None:
    cfg_path = tmp_path / "missing.json"
    app = create_webui_app(WebUIConfig(), config_path=cfg_path)
    client = app.test_client()

    schema_resp = await client.get("/api/config/schema")
    assert schema_resp.status_code == 200
    assert schema_resp.headers["Cache-Control"] == "no-store"
    schema = await schema_resp.get_json()
    assert schema == AppConfig.model_json_schema()
    assert "webui" in schema["properties"]

    defaults_resp = await client.get("/api/config/defaults")
    assert defaults_resp.status_code == 200
    assert defaults_resp.headers["Cache-Control"] == "no-store"
    assert await defaults_resp.get_json() == AppConfig().model_dump(mode="json")

    cfg_resp = await client.get("/api/config")
    assert cfg_resp.status_code == 200
    assert cfg_resp.headers["Cache-Control"] == "no-store"
    body = await cfg_resp.get_json()
    assert body == {
        "exists": False,
        "valid": True,
        "config": AppConfig().model_dump(mode="json"),
        "errors": [],
    }
    assert not cfg_path.exists()


async def test_validate_and_put_config_round_trip_and_errors(tmp_path: Path) -> None:
    cfg_path = tmp_path / "nested" / "config.json"
    app = create_webui_app(WebUIConfig(), config_path=cfg_path)
    client = app.test_client()

    # Validate valid config without writing disk.
    updated = AppConfig(
        napcat={"bot_name": "SpectraStudio", "port": 6299},
        llm={"model": "gpt-5", "api_key": "sk-test-123"},
        webui={"port": 6205},
    ).model_dump(mode="json")

    val_resp = await client.post("/api/config/validate", json=updated)
    assert val_resp.status_code == 200
    assert val_resp.headers["Cache-Control"] == "no-store"
    val_body = await val_resp.get_json()
    assert val_body["valid"] is True
    assert val_body["normalized"]["napcat"]["bot_name"] == "SpectraStudio"
    assert not cfg_path.exists()

    # Validate invalid payload -> structured 422 with loc/msg/type and no secret echo.
    bad_payload = AppConfig().model_dump(mode="json")
    bad_payload["napcat"]["port"] = 70000
    bad_payload["session"]["window_keep_messages"] = 60
    bad_payload["session"]["window_max_messages"] = 60
    bad_payload["llm"]["api_key"] = "sk-do-not-echo-me"
    bad_payload["unknown_extra"] = 1

    bad_val_resp = await client.post("/api/config/validate", json=bad_payload)
    assert bad_val_resp.status_code == 422
    bad_text = await bad_val_resp.get_data(as_text=True)
    assert "sk-do-not-echo-me" not in bad_text
    bad_val_body = await bad_val_resp.get_json()
    assert bad_val_body["valid"] is False
    locs = [tuple(err["loc"]) for err in bad_val_body["errors"]]
    assert ("napcat", "port") in locs
    assert ("session",) in locs
    assert ("unknown_extra",) in locs
    for err in bad_val_body["errors"]:
        assert set(err.keys()) == {"loc", "msg", "type"}

    # PUT invalid payload -> 422 and does not create or overwrite disk file.
    bad_put_resp = await client.put("/api/config", json=bad_payload)
    assert bad_put_resp.status_code == 422
    assert bad_put_resp.headers["Cache-Control"] == "no-store"
    bad_put_body = await bad_put_resp.get_json()
    assert bad_put_body["saved"] is False
    assert not cfg_path.exists()

    # PUT valid payload -> atomic save + restart_required: true.
    put_resp = await client.put("/api/config", json=updated)
    assert put_resp.status_code == 200
    assert put_resp.headers["Cache-Control"] == "no-store"
    put_body = await put_resp.get_json()
    assert put_body["saved"] is True
    assert put_body["restart_required"] is True
    assert put_body["config"]["napcat"]["bot_name"] == "SpectraStudio"
    assert cfg_path.is_file()
    assert load_config(cfg_path).napcat.bot_name == "SpectraStudio"
    assert load_config(cfg_path).llm.model == "gpt-5"

    # Subsequent GET /api/config reflects persisted file.
    get_resp = await client.get("/api/config")
    assert get_resp.status_code == 200
    get_body = await get_resp.get_json()
    assert get_body["exists"] is True
    assert get_body["valid"] is True
    assert get_body["config"]["napcat"]["port"] == 6299


async def test_corrupted_config_recovery_responses(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    app = create_webui_app(WebUIConfig(), config_path=cfg_path)
    client = app.test_client()

    # 1. Invalid JSON syntax on disk
    cfg_path.write_text('{"llm": {"model": "gpt-4o",}', encoding="utf-8")
    resp = await client.get("/api/config")
    assert resp.status_code == 422
    body = await resp.get_json()
    assert body["exists"] is True
    assert body["valid"] is False
    assert body["error_type"] == "json_decode_error"
    assert body["raw_text"] == '{"llm": {"model": "gpt-4o",}'
    assert body["errors"][0]["type"] == "json_invalid"

    # 2. Valid JSON, invalid schema on disk
    cfg_path.write_text(
        json.dumps({"napcat": {"port": 99999}, "llm": {"api_key": "sk-secret"}}),
        encoding="utf-8",
    )
    resp2 = await client.get("/api/config")
    assert resp2.status_code == 422
    body2 = await resp2.get_json()
    assert body2["exists"] is True
    assert body2["valid"] is False
    assert body2["error_type"] == "validation_error"
    assert body2["raw_config"]["napcat"]["port"] == 99999
    assert any(err["loc"] == ["napcat", "port"] for err in body2["errors"])

    # 3. Invalid UTF-8 bytes on disk
    cfg_path.write_bytes(b"\xff\xfe\x00\x01")
    resp3 = await client.get("/api/config")
    assert resp3.status_code == 422
    body3 = await resp3.get_json()
    assert body3["error_type"] == "unicode_decode_error"


async def test_request_body_validation_media_type_and_size_limit(
    tmp_path: Path,
) -> None:
    app = create_webui_app(
        WebUIConfig(),
        config_path=tmp_path / "config.json",
        max_body_bytes=256,
    )
    client = app.test_client()

    # Non-JSON Content-Type -> 415
    resp_text = await client.post(
        "/api/config/validate",
        data="hello",
        headers={"Content-Type": "text/plain"},
    )
    assert resp_text.status_code == 415

    # Malformed JSON -> 400
    resp_bad_json = await client.post(
        "/api/config/validate",
        data="{invalid",
        headers={"Content-Type": "application/json"},
    )
    assert resp_bad_json.status_code == 400

    # Non-object JSON -> 422
    resp_array = await client.post("/api/config/validate", json=[1, 2, 3])
    assert resp_array.status_code == 422

    # Oversized body -> 413
    oversized = json.dumps({"llm": {"model": "x" * 512}})
    resp_large = await client.put(
        "/api/config",
        data=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert resp_large.status_code == 413


async def test_cross_origin_mutating_requests_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    app = create_webui_app(WebUIConfig(), config_path=cfg_path)
    client = app.test_client()
    payload = AppConfig().model_dump(mode="json")

    # Cross-origin POST / PUT and null Origin must be rejected with 403.
    for bad_origin in ("https://evil.example", "http://127.0.0.1:9999", "null"):
        resp_post = await client.post(
            "/api/config/validate",
            json=payload,
            headers={"Origin": bad_origin},
        )
        assert resp_post.status_code == 403

        resp_put = await client.put(
            "/api/config",
            json=payload,
            headers={"Origin": bad_origin},
        )
        assert resp_put.status_code == 403
        assert not cfg_path.exists()

    # Matching same-origin header succeeds.
    resp_ok = await client.put(
        "/api/config",
        json=payload,
        headers={"Origin": "http://localhost", "Host": "localhost"},
    )
    assert resp_ok.status_code == 200
    assert cfg_path.is_file()


async def test_webui_server_ephemeral_port_lifecycle_and_http_requests(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "live_config.json"
    server = WebUIServer(
        WebUIConfig(host="127.0.0.1", auth_token="live-token"),
        config_path=cfg_path,
        bot_running=False,
        port=0,
    )
    assert server.bound_port is None
    assert server.url is None

    async with server:
        assert server.is_running is True
        assert server.bound_host == "127.0.0.1"
        assert server.bound_port is not None and server.bound_port > 0
        assert server.url == f"http://127.0.0.1:{server.bound_port}"

        async with httpx.AsyncClient(base_url=server.url) as client:
            html_resp = await client.get("/")
            assert html_resp.status_code == 200
            assert "Spectra-bot Config Studio" in html_resp.text

            meta_resp = await client.get("/api/meta")
            assert meta_resp.status_code == 200
            assert meta_resp.json()["auth_required"] is True
            assert meta_resp.json()["bot_running"] is False

            unauth_resp = await client.get("/api/config")
            assert unauth_resp.status_code == 401

            headers = {"Authorization": "Bearer live-token"}
            cfg_resp = await client.get("/api/config", headers=headers)
            assert cfg_resp.status_code == 200
            payload = cfg_resp.json()["config"]
            payload["napcat"]["bot_name"] = "LiveSavedBot"

            put_resp = await client.put("/api/config", json=payload, headers=headers)
            assert put_resp.status_code == 200
            assert put_resp.json()["saved"] is True
            assert load_config(cfg_path).napcat.bot_name == "LiveSavedBot"

    assert server.is_running is False
    # Idempotent close
    await server.close()
    # Closed server refuses restart
    with pytest.raises(RuntimeError, match="already been closed"):
        await server.start()


async def test_webui_non_loopback_empty_token_logs_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = WebUIServer(
        WebUIConfig(host="0.0.0.0", auth_token=""),
        config_path=tmp_path / "config.json",
        port=0,
    )
    with caplog.at_level(logging.WARNING, logger="spectra.webui"):
        async with server:
            assert server.bound_port is not None and server.bound_port > 0

    assert any(
        "non-loopback" in record.getMessage() and "auth_token" in record.getMessage()
        for record in caplog.records
    )


def test_standalone_webui_config_recovery_preserves_valid_webui_section(
    tmp_path: Path,
) -> None:
    # Missing file -> defaults
    assert _load_webui_config_for_standalone(tmp_path / "missing.json") == WebUIConfig()

    # Broken other section, valid webui section -> preserves auth_token/port
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(
        json.dumps(
            {
                "napcat": {"port": 999999},
                "webui": {"host": "127.0.0.1", "port": 6211, "auth_token": "keep-me"},
            }
        ),
        encoding="utf-8",
    )
    recovered = _load_webui_config_for_standalone(partial_path)
    assert recovered.port == 6211
    assert recovered.auth_token == "keep-me"

    # Unparseable JSON -> falls back to loopback defaults
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{broken json", encoding="utf-8")
    assert _load_webui_config_for_standalone(broken_path) == WebUIConfig()


async def test_cli_webui_only_mode_runs_and_shuts_down(tmp_path: Path) -> None:
    cfg_path = tmp_path / "standalone.json"
    ready = asyncio.Event()
    holder: dict[str, WebUIServer] = {}

    def on_ready(server: WebUIServer) -> None:
        holder["server"] = server
        ready.set()

    task = asyncio.create_task(
        _run_webui_only(cfg_path, host="127.0.0.1", port=0, ready_callback=on_ready)
    )
    try:
        async with asyncio.timeout(5.0):
            await ready.wait()
        server = holder["server"]
        assert server.url is not None
        async with httpx.AsyncClient(base_url=server.url) as client:
            resp = await client.get("/api/meta")
            assert resp.status_code == 200
            assert resp.json()["bot_running"] is False
            assert resp.json()["config_path"] == str(cfg_path)
        server.request_shutdown()
        async with asyncio.timeout(5.0):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_main_dispatches_webui_only_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "config.json"
    called_with: list[Path] = []

    async def fake_run_webui_only(path: Path) -> None:
        called_with.append(path)

    monkeypatch.setattr("spectra.__main__._run_webui_only", fake_run_webui_only)
    rc = main(["--webui-only", "--config", str(cfg_path)])
    assert rc == 0
    assert called_with == [cfg_path]


async def test_run_bot_starts_and_closes_webui_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path = tmp_path / "config.json"
    save_config(AppConfig(webui={"enabled": True, "host": "127.0.0.1"}), cfg_path)

    events: list[str] = []

    class FakeBot:
        def __init__(self, config: AppConfig) -> None:
            self.config = config
            self._stop = asyncio.Event()

        def request_shutdown(self) -> None:
            events.append("bot_shutdown")
            self._stop.set()

        async def run(self) -> None:
            events.append("bot_run")
            await asyncio.sleep(0)

    class FakeWebUIServer:
        def __init__(
            self,
            config: WebUIConfig,
            *,
            config_path: Path,
            bot_running: bool,
        ) -> None:
            assert bot_running is True
            assert config_path == cfg_path
            self._stop = asyncio.Event()

        async def start(self) -> None:
            events.append("webui_start")

        def request_shutdown(self) -> None:
            events.append("webui_shutdown")
            self._stop.set()

        async def wait(self) -> None:
            await self._stop.wait()

        async def close(self) -> None:
            events.append("webui_close")

    monkeypatch.setattr("spectra.__main__.SpectraBot", FakeBot)
    monkeypatch.setattr("spectra.__main__.WebUIServer", FakeWebUIServer)

    await _run_bot(cfg_path)
    assert events == ["webui_start", "bot_run", "webui_shutdown", "webui_close"]

    # When webui.enabled is False, WebUIServer must NOT be constructed.
    events.clear()
    save_config(AppConfig(webui={"enabled": False}), cfg_path)

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("WebUIServer should not be created when disabled")

    monkeypatch.setattr("spectra.__main__.WebUIServer", fail_if_called)
    await _run_bot(cfg_path)
    assert events == ["bot_run"]
