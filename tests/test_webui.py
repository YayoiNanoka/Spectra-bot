"""Unit and integration tests for the schema-driven Config WebUI."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
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
from spectra.db import Database
from spectra.personas import PersonaStore
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
    assert '<html lang="zh-CN"' in html
    assert "Spectra-bot 配置台" in html
    assert 'id="prompt-modal"' in html
    assert 'id="mode-personas"' in html
    assert 'id="personas-view"' in html
    assert 'id="persona-modal"' in html
    assert "人格预设管理" in html
    assert "会话专属人格绑定" in html
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
    assert schema["$defs"]["VisionConfig"]["properties"]["prompt"]["x_widget"] == (
        "textarea"
    )
    assert (
        schema["$defs"]["VisionConfig"]["properties"]["montage_prompt_note"][
            "x_collapsed"
        ]
        is True
    )
    assert schema["$defs"]["JevConfig"]["properties"]["prompts"]["x_collapsed"] is True
    assert (
        schema["$defs"]["MemoryConfig"]["properties"]["prompts"]["x_collapsed"] is True
    )
    assert "datetime_system_prompt" in schema["$defs"]["SessionConfig"]["properties"]

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
        vision={
            "prompt": "自定义识图指令",
            "montage_prompt_note": "自定义动图拼图说明",
        },
        jev={
            "prompts": {
                "reply_instructions": "自定义插话指令",
                "rerank_instructions": "候选 {index}（保留 {raw}）",
            }
        },
        session={"datetime_system_prompt": False},
        memory={
            "prompts": {
                "group_consolidation_prompt": "自定义群聊记忆系统提示词",
                "private_consolidation_prompt": "自定义私聊记忆系统提示词",
            }
        },
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
    assert load_config(cfg_path).vision.montage_prompt_note == "自定义动图拼图说明"
    assert load_config(cfg_path).jev.prompts.reply_instructions == "自定义插话指令"
    assert load_config(cfg_path).session.datetime_system_prompt is False
    assert (
        load_config(cfg_path).memory.prompts.group_consolidation_prompt
        == "自定义群聊记忆系统提示词"
    )

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
            assert "Spectra-bot 配置台" in html_resp.text

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

    sentinel_persona_store = object()

    class FakeBot:
        def __init__(self, config: AppConfig) -> None:
            self.config = config
            self._stop = asyncio.Event()
            self.persona_store: Any = None

        async def start(self) -> None:
            events.append("bot_start")
            self.persona_store = sentinel_persona_store

        def request_shutdown(self) -> None:
            events.append("bot_shutdown")
            self._stop.set()

        async def run(self) -> None:
            events.append("bot_run")
            await asyncio.sleep(0)

        async def close(self) -> None:
            events.append("bot_close")

    class FakeWebUIServer:
        def __init__(
            self,
            config: WebUIConfig,
            *,
            config_path: Path,
            bot_running: bool,
            persona_store: Any = None,
        ) -> None:
            assert bot_running is True
            assert config_path == cfg_path
            assert persona_store is sentinel_persona_store
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
    assert events == [
        "bot_start",
        "webui_start",
        "bot_run",
        "webui_shutdown",
        "webui_close",
        "bot_close",
    ]

    # When webui.enabled is False, WebUIServer must NOT be constructed.
    events.clear()
    save_config(AppConfig(webui={"enabled": False}), cfg_path)

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("WebUIServer should not be created when disabled")

    monkeypatch.setattr("spectra.__main__.WebUIServer", fail_if_called)
    await _run_bot(cfg_path)
    assert events == ["bot_run", "bot_close"]


async def test_non_rfc8259_nan_and_infinity_rejected_everywhere(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    app = create_webui_app(WebUIConfig(), config_path=cfg_path)
    client = app.test_client()

    # 1. Request payloads containing bare Infinity / NaN / -Infinity -> 400
    for bad_body in (
        '{"vision": {"timeout_sec": Infinity}}',
        '{"vision": {"timeout_sec": NaN}}',
        '{"llm": {"timeout_sec": -Infinity}}',
    ):
        resp_val = await client.post(
            "/api/config/validate",
            data=bad_body,
            headers={"Content-Type": "application/json"},
        )
        assert resp_val.status_code == 400

        resp_put = await client.put(
            "/api/config",
            data=bad_body,
            headers={"Content-Type": "application/json"},
        )
        assert resp_put.status_code == 400
        assert not cfg_path.exists()

    # 2. Disk config with bare Infinity / NaN -> 422 recovery (not 200 invalid JSON)
    cfg_path.write_text('{"vision": {"timeout_sec": Infinity}}', encoding="utf-8")
    get_resp = await client.get("/api/config")
    assert get_resp.status_code == 422
    body = await get_resp.get_json()
    assert body["valid"] is False
    assert body["error_type"] == "json_decode_error"
    assert body["raw_text"] == '{"vision": {"timeout_sec": Infinity}}'


async def test_auth_token_whitespace_normalized_and_warning_enforced(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg_path = tmp_path / "config.json"

    # Whitespace-padded token must not lock out trimmed bearer requests.
    app_padded = create_webui_app(
        WebUIConfig(auth_token="  padded-token  "),
        config_path=cfg_path,
    )
    client_padded = app_padded.test_client()
    assert (await client_padded.get("/api/config")).status_code == 401
    ok_resp = await client_padded.get(
        "/api/config",
        headers={"Authorization": "Bearer padded-token"},
    )
    assert ok_resp.status_code == 200

    # Whitespace-only token counts as empty: auth_required=False and non-loopback warns.
    app_blank = create_webui_app(
        WebUIConfig(auth_token="   "),
        config_path=cfg_path,
    )
    meta = await (await app_blank.test_client().get("/api/meta")).get_json()
    assert meta["auth_required"] is False

    server = WebUIServer(
        WebUIConfig(host="0.0.0.0", auth_token="   "),
        config_path=cfg_path,
        port=0,
    )
    with caplog.at_level(logging.WARNING, logger="spectra.webui"):
        async with server:
            assert server.bound_port is not None and server.bound_port > 0
    assert any("non-loopback" in rec.getMessage() for rec in caplog.records)


async def test_webui_server_concurrent_start_is_idempotent(tmp_path: Path) -> None:
    server = WebUIServer(
        WebUIConfig(host="127.0.0.1"),
        config_path=tmp_path / "config.json",
        port=0,
    )
    try:
        await asyncio.gather(server.start(), server.start())
        assert server.is_running is True
        assert server.bound_port is not None and server.bound_port > 0
    finally:
        await server.close()


@pytest.mark.skipif(shutil.which("node") is None, reason="node runtime not available")
def test_frontend_schema_nullable_widget_and_error_helpers_in_node() -> None:
    webui_js = (
        Path(__file__).resolve().parent.parent / "spectra" / "static" / "webui.js"
    )
    schema_json = json.dumps(AppConfig.model_json_schema())
    defaults_json = json.dumps(AppConfig().model_dump(mode="json"))

    script = f"""
import assert from "node:assert/strict";
import {{
  formatEnumLabel,
  localizeValidationError,
  unwrapSchema,
  valuesEqual,
  isTextareaField,
  getTextareaRows,
  isRenderedFieldPath,
  hasErrorsUnderPath,
  countCustomizedFields,
  countCustomizedEntries,
  partitionSectionFields,
  fieldMatchesSearch,
  defaultValueForNonScalarNullable,
  parseNumberInput,
  parseListItemValue,
  serializeStringInput,
  serializeNullableEnumSelect,
  serializeNullableBoolSelect,
  validateKeyValueEntries,
  getEffectiveValue,
  collectRootAndUnknownErrors,
  collectUnmatchedSectionErrors,
  collectLeafFieldErrors,
  groupDialogsIntoPairs,
  flattenDialogPairs,
  validatePersonaDraft,
  serializePersonaExport,
  parsePersonaImportJson,
  formatBindingTimestamp,
}} from {json.dumps(webui_js.as_uri())};

const rootSchema = {schema_json};
const defaults = {defaults_json};

// 1. Nullable schema unwrapping & non-scalar fallback must never return null
const nullableObjSchema = {{
  anyOf: [
    {{ type: "object", properties: {{ a: {{ type: "integer", default: 3 }} }} }},
    {{ type: "null" }},
  ],
  default: null,
}};
const unwrappedObj = unwrapSchema(nullableObjSchema, rootSchema);
assert.equal(unwrappedObj.nullable, true);
assert.deepEqual(
  defaultValueForNonScalarNullable(unwrappedObj.schema, rootSchema),
  {{ a: 3 }},
);

const nullableArrSchema = {{
  anyOf: [{{ type: "array", items: {{ type: "string" }} }}, {{ type: "null" }}],
  default: null,
}};
const unwrappedArr = unwrapSchema(nullableArrSchema, rootSchema);
assert.deepEqual(
  defaultValueForNonScalarNullable(unwrappedArr.schema, rootSchema),
  [],
);

// 2. Scalar nullable serialization
assert.deepEqual(
  parseNumberInput("", "integer", {{ nullable: true }}),
  {{ ok: true, value: null }},
);
assert.equal(parseNumberInput("", "integer", {{ nullable: false }}).ok, false);
assert.equal(
  parseNumberInput("", "integer", {{ nullable: true, badInput: true }}).ok,
  false,
);
assert.equal(parseNumberInput("12.7", "integer", {{ nullable: false }}).ok, false);
assert.deepEqual(
  parseNumberInput("42", "integer", {{ nullable: true }}),
  {{ ok: true, value: 42 }},
);
assert.deepEqual(
  parseNumberInput("1.5", "number", {{ nullable: true }}),
  {{ ok: true, value: 1.5 }},
);
assert.equal(parseListItemValue("abc", "integer").ok, false);
assert.equal(parseListItemValue("1.5", "integer").ok, false);
assert.deepEqual(parseListItemValue("7", "integer"), {{ ok: true, value: 7 }});

assert.equal(serializeStringInput("", true), null);
assert.equal(serializeStringInput("", false), "");
assert.equal(serializeStringInput("hi", true), "hi");

assert.equal(serializeNullableEnumSelect(0, ["low", "high"]), null);
assert.equal(serializeNullableEnumSelect(1, ["low", "high"]), "low");
assert.equal(serializeNullableEnumSelect(2, ["low", "high"]), "high");

assert.equal(serializeNullableBoolSelect(""), null);
assert.equal(serializeNullableBoolSelect("true"), true);
assert.equal(serializeNullableBoolSelect("false"), false);

// 3. Key-Value validation catches blank keys with values & duplicates
assert.equal(validateKeyValueEntries([{{ key: "", value: "secret" }}]).ok, false);
const dupEntries = [
  {{ key: "X-A", value: "1" }},
  {{ key: "X-A", value: "2" }},
];
assert.equal(validateKeyValueEntries(dupEntries).ok, false);
assert.deepEqual(
  validateKeyValueEntries([
    {{ key: "", value: "" }},
    {{ key: "X-A", value: "1" }},
  ]),
  {{ ok: true, value: {{ "X-A": "1" }} }},
);

// 4. Partial draftConfig falls back to defaults while preserving explicit null
const partialDraft = {{
  napcat: {{ port: 99999 }},
  custom: {{ nullable_field: null }},
}};
assert.equal(
  getEffectiveValue(partialDraft, defaults, ["vision", "enabled"], {{}}),
  true,
);
assert.equal(
  getEffectiveValue(partialDraft, defaults, ["gate", "proactive_mode"], {{}}),
  "on",
);
assert.equal(
  getEffectiveValue(
    partialDraft,
    defaults,
    ["custom", "nullable_field"],
    {{ default: "x" }},
  ),
  null,
);

// 5. All Pydantic loc error shapes stay visible
const errorsByPath = new Map([
  ["unknown_extra", ["Extra inputs are not permitted"]],
  ["llm.typo_field", ["Extra inputs are not permitted"]],
  ["access.admin_users.0", ["Input should be a valid string"]],
  ["llm.custom_headers.X-Bad", ["Invalid header value"]],
]);
const rootMsgs = collectRootAndUnknownErrors(errorsByPath, rootSchema);
assert.equal(rootMsgs.length, 1);
assert.match(rootMsgs[0], /unknown_extra/);

const llmSection = unwrapSchema(rootSchema.properties.llm, rootSchema).schema;
const unmatchedLlm = collectUnmatchedSectionErrors(
  "llm",
  llmSection,
  rootSchema,
  errorsByPath,
);
assert.equal(unmatchedLlm.length, 1);
assert.match(unmatchedLlm[0], /llm\\.typo_field/);

const arrayLeaf = collectLeafFieldErrors(
  "access.admin_users",
  errorsByPath,
  new Map(),
  true,
);
assert.equal(arrayLeaf.length, 1);
assert.match(arrayLeaf[0], /\\[0\\]/);

const mapLeaf = collectLeafFieldErrors(
  "llm.custom_headers",
  errorsByPath,
  new Map(),
  true,
);
assert.equal(mapLeaf.length, 1);
assert.match(mapLeaf[0], /\\[X-Bad\\]/);

// 6. Chinese enum label formatting & validation error localization
assert.equal(formatEnumLabel("blacklist"), "黑名单 (blacklist)");
assert.equal(formatEnumLabel("shadow"), "影子模式 (shadow)");
assert.equal(formatEnumLabel("baidu_ai_search"), "百度 AI 搜索 (baidu_ai_search)");
assert.equal(formatEnumLabel("auto"), "自动识别 (auto)");
assert.equal(formatEnumLabel("always"), "始终发送 (always)");
assert.equal(formatEnumLabel("never"), "永不发送 (never)");
assert.equal(formatEnumLabel("custom_unknown"), "custom_unknown");
assert.ok(rootSchema.properties.embedding, "embedding section must exist in schema");

assert.equal(
  localizeValidationError({{
    type: "extra_forbidden",
    msg: "Extra inputs are not permitted",
  }}),
  "不允许包含未定义的配置项",
);
assert.equal(
  localizeValidationError({{
    type: "less_than_equal",
    msg: "Input should be less than or equal to 65535",
  }}),
  "数值必须小于等于 65535",
);
assert.match(
  localizeValidationError({{
    type: "value_error",
    msg: "Value error, Invalid IANA timezone: 'Bad/TZ'",
  }}),
  /时区/,
);
assert.match(
  localizeValidationError({{
    type: "value_error",
    msg: "Value error, window_keep_messages must be strictly less than " +
      "window_max_messages",
  }}),
  /保留消息数/,
);
assert.match(
  localizeValidationError({{
    type: "value_error",
    msg: "Value error, group_idle_timeout_minutes must be strictly less than " +
      "group_hard_timeout_hours * 60",
  }}),
  /群聊冷场空闲阈值/,
);
assert.match(
  localizeValidationError({{
    type: "value_error",
    msg: "Value error, private_idle_timeout_minutes must be strictly less than " +
      "private_hard_timeout_hours * 60",
  }}),
  /私聊冷场空闲阈值/,
);
assert.equal(
  localizeValidationError({{
    type: "value_error",
    msg: "Value error, 自定义中文校验错误",
  }}),
  "自定义中文校验错误",
);
assert.equal(
  localizeValidationError({{
    type: "path_type",
    msg: "Input is not a valid path for <class 'pathlib.Path'>",
  }}),
  "请输入有效的文件路径",
);
assert.equal(
  localizeValidationError({{
    type: "literal_error",
    msg: "Input should be 'blacklist' or 'whitelist'",
  }}),
  "取值无效，可选值：'blacklist' 或 'whitelist'",
);
assert.equal(
  localizeValidationError({{
    type: "literal_error",
    msg: "Input should be 'auto', 'system_one' or 'openai_decision'",
  }}),
  "取值无效，可选值：'auto'、'system_one' 或 'openai_decision'",
);
assert.equal(
  localizeValidationError({{ type: "io_error", msg: "配置路径不是常规文件。" }}),
  "配置路径不是常规文件。",
);

// 7. Prompt editors, reset-to-default comparison, and collapsible helpers
assert.equal(valuesEqual("abc", "abc"), true);
assert.equal(valuesEqual("abc", "def"), false);
assert.equal(valuesEqual(["a", "b"], ["a", "b"]), true);
assert.equal(valuesEqual(undefined, "a"), false);

const visionSection = unwrapSchema(rootSchema.properties.vision, rootSchema).schema;
const visionPromptSchema = unwrapSchema(
  visionSection.properties.prompt,
  rootSchema,
).schema;
const montageSchema = unwrapSchema(
  visionSection.properties.montage_prompt_note,
  rootSchema,
).schema;
assert.equal(isTextareaField(visionPromptSchema, "prompt", ""), true);
assert.equal(getTextareaRows(visionPromptSchema), 3);
assert.equal(isTextareaField(montageSchema, "montage_prompt_note", ""), true);
assert.equal(montageSchema.x_collapsed, true);

const {{ regularEntries, collapsedLeafEntries }} = partitionSectionFields(
  visionSection.properties,
  rootSchema,
);
assert.deepEqual(
  collapsedLeafEntries.map(([k]) => k),
  ["montage_prompt_note"],
);
assert.equal(
  regularEntries.some(([k]) => k === "montage_prompt_note"),
  false,
);

const jevSection = unwrapSchema(rootSchema.properties.jev, rootSchema).schema;
const jevPromptsSchema = unwrapSchema(jevSection.properties.prompts, rootSchema).schema;
assert.equal(jevPromptsSchema.x_collapsed, true);
assert.equal(
  isRenderedFieldPath(["prompts", "rerank_instructions"], jevSection, rootSchema),
  true,
);
assert.equal(
  isRenderedFieldPath(["prompts", "nonexistent"], jevSection, rootSchema),
  false,
);
assert.equal(
  fieldMatchesSearch(
    "prompts",
    jevPromptsSchema,
    jevSection.title,
    "rerank",
    rootSchema,
  ),
  true,
);
assert.equal(
  fieldMatchesSearch(
    "prompts",
    jevPromptsSchema,
    jevSection.title,
    "记忆精排",
    rootSchema,
  ),
  true,
);

const memorySection = unwrapSchema(rootSchema.properties.memory, rootSchema).schema;
const memoryPromptsSchema = unwrapSchema(
  memorySection.properties.prompts,
  rootSchema,
).schema;
assert.equal(memoryPromptsSchema.x_collapsed, true);
assert.equal(
  getTextareaRows(
    unwrapSchema(memoryPromptsSchema.properties.group_consolidation_prompt, rootSchema)
      .schema,
  ),
  10,
);
assert.equal(
  isRenderedFieldPath(
    ["prompts", "private_consolidation_prompt"],
    memorySection,
    rootSchema,
  ),
  true,
);

const modifiedDraft = JSON.parse(JSON.stringify(defaults));
modifiedDraft.vision.montage_prompt_note = "已自定义动图说明";
modifiedDraft.jev.prompts.reply_instructions = "已自定义插话";
modifiedDraft.jev.prompts.rerank_instructions = "已自定义精排 {{index}}";

assert.equal(
  countCustomizedEntries(
    collapsedLeafEntries,
    modifiedDraft,
    defaults,
    rootSchema,
    ["vision"],
  ),
  1,
);
assert.equal(
  countCustomizedFields(
    jevPromptsSchema,
    modifiedDraft,
    defaults,
    rootSchema,
    ["jev", "prompts"],
  ),
  2,
);
assert.equal(
  countCustomizedFields(
    memoryPromptsSchema,
    modifiedDraft,
    defaults,
    rootSchema,
    ["memory", "prompts"],
  ),
  0,
);

const nestedErrs = new Map([["jev.prompts.reply_instructions", ["无效"]]]);
assert.equal(hasErrorsUnderPath("jev.prompts", nestedErrs, new Map()), true);
assert.equal(hasErrorsUnderPath("memory.prompts", nestedErrs, new Map()), false);

// 8. Persona modal helpers and JSON import/export round-trip compatibility
const pairs = groupDialogsIntoPairs(["你好", "你好呀！", "再见", "拜拜~"]);
assert.deepEqual(pairs, [
  {{ user: "你好", assistant: "你好呀！" }},
  {{ user: "再见", assistant: "拜拜~" }},
]);
assert.deepEqual(
  flattenDialogPairs([...pairs, {{ user: "   ", assistant: "" }}]),
  {{ ok: true, value: ["你好", "你好呀！", "再见", "拜拜~"] }},
);
assert.equal(
  flattenDialogPairs([{{ user: "只有用户", assistant: "   " }}]).ok,
  false,
);
assert.equal(
  validatePersonaDraft({{
    persona_id: "bad/id",
    system_prompt: "prompt",
    pairs: [],
    custom_error_message: "",
    isEdit: false,
  }}).ok,
  false,
);
const exported = serializePersonaExport({{
  persona_id: "catgirl",
  system_prompt: "你是猫娘",
  begin_dialogs: ["你好", "喵~"],
  custom_error_message: "出错啦喵",
  created_at: 100,
  updated_at: 200,
}});
assert.deepEqual(Object.keys(exported).sort(), [
  "begin_dialogs",
  "custom_error_message",
  "persona_id",
  "system_prompt",
]);
const imported = parsePersonaImportJson(JSON.stringify(exported));
assert.deepEqual(imported, {{ ok: true, value: exported }});
assert.equal(
  parsePersonaImportJson(
    JSON.stringify({{
      persona_id: "odd",
      system_prompt: "sys",
      begin_dialogs: ["only-user"],
    }}),
  ).ok,
  false,
);
assert.equal(
  parsePersonaImportJson(
    JSON.stringify({{
      persona_id: "../escape",
      system_prompt: "sys",
    }}),
  ).ok,
  false,
);
assert.equal(formatBindingTimestamp(0), "--");
assert.match(
  formatBindingTimestamp(1700000000),
  /^\\d{{4}}-\\d{{2}}-\\d{{2}} \\d{{2}}:\\d{{2}}$/,
);
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Node test failed:\n{proc.stderr}\n{proc.stdout}"


async def test_persona_api_crud_bindings_auth_and_csrf(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("初始默认人设", encoding="utf-8")
    token = "persona-secret-token"
    headers = {"Authorization": f"Bearer {token}"}

    async with Database(":memory:") as db:
        store = PersonaStore(db, prompts_dir=prompts_dir)
        await store.init()

        app = create_webui_app(
            WebUIConfig(auth_token=token),
            config_path=cfg_path,
            persona_store=store,
        )
        client = app.test_client()

        # 1. Auth enforced on GET and mutating endpoints.
        assert (await client.get("/api/personas")).status_code == 401
        assert (
            await client.post(
                "/api/personas",
                json={"persona_id": "p1", "system_prompt": "hi"},
            )
        ).status_code == 401

        # 2. Cross-origin CSRF rejected on POST/PUT/DELETE.
        csrf_headers = {**headers, "Origin": "https://evil.example"}
        assert (
            await client.post(
                "/api/personas",
                json={"persona_id": "p1", "system_prompt": "hi"},
                headers=csrf_headers,
            )
        ).status_code == 403
        assert (
            await client.put(
                "/api/personas/default",
                json={"system_prompt": "hijack"},
                headers=csrf_headers,
            )
        ).status_code == 403
        assert (
            await client.delete("/api/personas/p1", headers=csrf_headers)
        ).status_code == 403
        assert (
            await client.put(
                "/api/persona-bindings",
                json={"scope": "group", "target_id": "100", "persona_id": "default"},
                headers=csrf_headers,
            )
        ).status_code == 403
        assert (
            await client.delete(
                "/api/persona-bindings/group/100",
                headers=csrf_headers,
            )
        ).status_code == 403

        # 3. GET /api/personas returns seeded default persona.
        list_resp = await client.get("/api/personas", headers=headers)
        assert list_resp.status_code == 200
        assert list_resp.headers["Cache-Control"] == "no-store"
        list_data = await list_resp.get_json()
        assert list_data["default_persona_id"] == "default"
        assert [p["persona_id"] for p in list_data["personas"]] == ["default"]
        assert list_data["personas"][0]["system_prompt"] == "初始默认人设"
        assert list_data["bindings"] == []

        # 4. POST /api/personas: 201 created, 409 duplicate, 422 invalid.
        create_resp = await client.post(
            "/api/personas",
            json={
                "persona_id": "coder",
                "system_prompt": "你是程序员助手。",
                "begin_dialogs": ["hi", "hello!"],
                "custom_error_message": "编译失败了",
            },
            headers=headers,
        )
        assert create_resp.status_code == 201
        created = (await create_resp.get_json())["persona"]
        assert created["persona_id"] == "coder"
        assert created["begin_dialogs"] == ["hi", "hello!"]
        assert created["custom_error_message"] == "编译失败了"

        dup_resp = await client.post(
            "/api/personas",
            json={"persona_id": "coder", "system_prompt": "duplicate"},
            headers=headers,
        )
        assert dup_resp.status_code == 409

        bad_dialogs_resp = await client.post(
            "/api/personas",
            json={
                "persona_id": "bad_odd",
                "system_prompt": "sys",
                "begin_dialogs": ["only-one"],
            },
            headers=headers,
        )
        assert bad_dialogs_resp.status_code == 422

        bad_id_resp = await client.post(
            "/api/personas",
            json={"persona_id": "bad/slash", "system_prompt": "sys"},
            headers=headers,
        )
        assert bad_id_resp.status_code == 422

        # 5. PUT /api/personas/<id>: 200 updated, 404 missing, 422 invalid.
        upd_resp = await client.put(
            "/api/personas/coder",
            json={
                "system_prompt": "你是高级程序员助手。",
                "begin_dialogs": ["问1", "答1", "问2", "答2"],
                "custom_error_message": "",
            },
            headers={"X-Spectra-Token": token},
        )
        assert upd_resp.status_code == 200
        assert (await upd_resp.get_json())["persona"]["system_prompt"] == (
            "你是高级程序员助手。"
        )
        assert (
            await client.put(
                "/api/personas/missing",
                json={"system_prompt": "x"},
                headers=headers,
            )
        ).status_code == 404
        assert (
            await client.put(
                "/api/personas/coder",
                json={"system_prompt": "   "},
                headers=headers,
            )
        ).status_code == 422

        # 6. PUT & DELETE /api/persona-bindings.
        bind_resp = await client.put(
            "/api/persona-bindings",
            json={"scope": "group", "target_id": "50001", "persona_id": "coder"},
            headers=headers,
        )
        assert bind_resp.status_code == 200
        assert (await bind_resp.get_json())["binding"]["session_key"] == "group:50001"

        assert (
            await client.put(
                "/api/persona-bindings",
                json={"scope": "group", "target_id": "50002", "persona_id": "missing"},
                headers=headers,
            )
        ).status_code == 404
        assert (
            await client.put(
                "/api/persona-bindings",
                json={"scope": "bad", "target_id": "50002", "persona_id": "coder"},
                headers=headers,
            )
        ).status_code == 422
        assert (
            await client.put(
                "/api/persona-bindings",
                json={"scope": "group", "target_id": "500 02", "persona_id": "coder"},
                headers=headers,
            )
        ).status_code == 422

        # Update config.json default_persona_id -> GET /api/personas reflects it.
        cfg_payload = AppConfig(session={"default_persona_id": "coder"}).model_dump(
            mode="json"
        )
        assert (
            await client.put("/api/config", json=cfg_payload, headers=headers)
        ).status_code == 200
        list_after_cfg = await (
            await client.get("/api/personas", headers=headers)
        ).get_json()
        assert list_after_cfg["default_persona_id"] == "coder"
        assert len(list_after_cfg["bindings"]) == 1

        # 7. DELETE /api/personas/default -> 400; deleting "coder" -> 200 + cascade!
        assert (
            await client.delete("/api/personas/default", headers=headers)
        ).status_code == 400
        assert (
            await client.delete("/api/personas/nonexistent", headers=headers)
        ).status_code == 404

        del_resp = await client.delete("/api/personas/coder", headers=headers)
        assert del_resp.status_code == 200
        assert (await del_resp.get_json())["deleted"] is True
        assert await store.list_bindings() == []

        # Deleting missing binding returns 404.
        assert (
            await client.delete(
                "/api/persona-bindings/group/50001",
                headers=headers,
            )
        ).status_code == 404

        # Bind default then explicitly unbind via DELETE.
        assert (
            await client.put(
                "/api/persona-bindings",
                json={
                    "scope": "private",
                    "target_id": "20002",
                    "persona_id": "default",
                },
                headers=headers,
            )
        ).status_code == 200
        unbind_resp = await client.delete(
            "/api/persona-bindings/private/20002",
            headers=headers,
        )
        assert unbind_resp.status_code == 200
        assert (await unbind_resp.get_json())["deleted"] is True


async def test_standalone_webui_persona_store_db_lifecycle_no_resource_warning(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "custom_prompts"
    prompts_dir.mkdir()
    (prompts_dir / "default.md").write_text("独立 WebUI 种子人设", encoding="utf-8")

    cfg_path = tmp_path / "standalone.json"
    save_config(
        AppConfig(
            db={"path": Path("data/standalone.db")},
            session={"prompts_dir": Path("custom_prompts")},
        ),
        cfg_path,
    )

    server = WebUIServer(
        WebUIConfig(host="127.0.0.1"),
        config_path=cfg_path,
        bot_running=False,
        port=0,
    )
    async with server:
        assert server.url is not None
        async with httpx.AsyncClient(base_url=server.url) as client:
            resp = await client.get("/api/personas")
            assert resp.status_code == 200
            body = resp.json()
            assert body["personas"][0]["persona_id"] == "default"
            assert body["personas"][0]["system_prompt"] == "独立 WebUI 种子人设"

            create = await client.post(
                "/api/personas",
                json={
                    "persona_id": "helper",
                    "system_prompt": "简洁助手",
                    "begin_dialogs": ["你好", "在呢！"],
                    "custom_error_message": "出错啦",
                },
            )
            assert create.status_code == 201

            bind = await client.put(
                "/api/persona-bindings",
                json={"scope": "group", "target_id": "88888", "persona_id": "helper"},
            )
            assert bind.status_code == 200

    # Relative db.path was resolved against config_path.parent and closed cleanly.
    expected_db = tmp_path / "data" / "standalone.db"
    assert expected_db.is_file()

    # Re-opening verifies persistence across restarts.
    async with Database(expected_db) as db:
        store = PersonaStore(db, prompts_dir=prompts_dir)
        await store.init()
        eff = await store.resolve_effective_persona(group_id="88888", user_id="10001")
        assert eff.persona_id == "helper"
        assert eff.system_prompt == "简洁助手"
        assert eff.begin_dialogs == ("你好", "在呢！")
        assert eff.custom_error_message == "出错啦"
