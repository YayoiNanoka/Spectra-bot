"""Schema-driven Config WebUI HTTP server backed by Quart and Hypercorn."""

from __future__ import annotations

import asyncio
import hmac
import importlib.metadata
import importlib.resources
import ipaddress
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HypercornConfig
from hypercorn.config import Sockets
from hypercorn.logging import Logger as HypercornLogger
from pydantic import ValidationError
from quart import Quart, Response, jsonify, request

from spectra.config import AppConfig, WebUIConfig, save_config

logger = logging.getLogger(__name__)

MAX_CONFIG_BODY_BYTES = 1 * 1024 * 1024  # 1 MiB
_PACKAGE_VERSION_FALLBACK = "0.1.0"

_STATIC_ASSETS: dict[str, tuple[str, str]] = {
    "webui.html": ("webui.html", "text/html; charset=utf-8"),
    "webui.css": ("webui.css", "text/css; charset=utf-8"),
    "webui.js": ("webui.js", "text/javascript; charset=utf-8"),
}


def _get_package_version() -> str:
    """Return the installed ``spectra-bot`` package version."""
    try:
        return importlib.metadata.version("spectra-bot")
    except importlib.metadata.PackageNotFoundError:
        return _PACKAGE_VERSION_FALLBACK


def _is_loopback_host(host: str) -> bool:
    """Return ``True`` if ``host`` refers strictly to a loopback interface."""
    cleaned = host.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def _format_bind_address(host: str, port: int) -> str:
    """Format ``host`` and ``port`` into a Hypercorn bind string."""
    cleaned = host.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if ":" in cleaned:
        return f"[{cleaned}]:{port}"
    return f"{cleaned}:{port}"


def _format_http_url(host: str, port: int) -> str:
    """Format a bound ``(host, port)`` pair as an HTTP URL."""
    return f"http://{_format_bind_address(host, port)}"


def _load_static_asset(filename: str) -> tuple[bytes, str] | None:
    """Load an allowlisted static asset from ``spectra.static`` resources."""
    asset = _STATIC_ASSETS.get(filename)
    if asset is None:
        return None
    resource_name, content_type = asset
    resource = importlib.resources.files("spectra.static").joinpath(resource_name)
    return resource.read_bytes(), content_type


def _extract_request_token() -> str:
    """Extract a bearer or custom header token from the active Quart request."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header:
        scheme, _, value = auth_header.partition(" ")
        if scheme.lower() == "bearer":
            return value.strip()
    return request.headers.get("X-Spectra-Token", "").strip()


def _serialize_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    """Serialize Pydantic validation errors without echoing raw input secrets."""
    errors: list[dict[str, Any]] = []
    for err in exc.errors(include_url=False):
        loc = [
            part if isinstance(part, (str, int)) else str(part)
            for part in err.get("loc", ())
        ]
        errors.append(
            {
                "loc": loc,
                "msg": str(err.get("msg", "Invalid value")),
                "type": str(err.get("type", "value_error")),
            }
        )
    return errors


def _read_config_file_state(config_path: Path) -> tuple[int, dict[str, Any]]:
    """Read and validate ``config_path`` from disk for ``GET /api/config``."""
    if not config_path.exists():
        return (
            200,
            {
                "exists": False,
                "valid": True,
                "config": AppConfig().model_dump(mode="json"),
                "errors": [],
            },
        )

    if not config_path.is_file():
        return (
            422,
            {
                "exists": True,
                "valid": False,
                "error_type": "invalid_path",
                "detail": f"Configuration path is not a regular file: {config_path}",
                "raw_text": "",
                "raw_config": None,
                "errors": [
                    {
                        "loc": [],
                        "msg": "Configuration path is not a regular file.",
                        "type": "io_error",
                    }
                ],
            },
        )

    file_size = config_path.stat().st_size
    if file_size > MAX_CONFIG_BODY_BYTES:
        return (
            422,
            {
                "exists": True,
                "valid": False,
                "error_type": "file_too_large",
                "detail": (
                    f"Configuration file exceeds maximum supported size "
                    f"({MAX_CONFIG_BODY_BYTES} bytes)."
                ),
                "raw_text": "",
                "raw_config": None,
                "errors": [
                    {
                        "loc": [],
                        "msg": "Configuration file exceeds size limit.",
                        "type": "value_error",
                    }
                ],
            },
        )

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return (
            422,
            {
                "exists": True,
                "valid": False,
                "error_type": "unicode_decode_error",
                "detail": f"Configuration file is not valid UTF-8: {exc}",
                "raw_text": "",
                "raw_config": None,
                "errors": [
                    {
                        "loc": [],
                        "msg": f"Invalid UTF-8 encoding: {exc}",
                        "type": "unicode_decode",
                    }
                ],
            },
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return (
            422,
            {
                "exists": True,
                "valid": False,
                "error_type": "json_decode_error",
                "detail": (
                    f"Invalid JSON syntax at line {exc.lineno}, "
                    f"column {exc.colno}: {exc.msg}"
                ),
                "raw_text": raw_text,
                "raw_config": None,
                "errors": [
                    {
                        "loc": [],
                        "msg": (
                            f"Invalid JSON at line {exc.lineno}, "
                            f"column {exc.colno}: {exc.msg}"
                        ),
                        "type": "json_invalid",
                    }
                ],
            },
        )

    try:
        cfg = AppConfig.model_validate(parsed)
    except ValidationError as exc:
        return (
            422,
            {
                "exists": True,
                "valid": False,
                "error_type": "validation_error",
                "detail": "Configuration file failed schema validation.",
                "raw_text": raw_text,
                "raw_config": parsed if isinstance(parsed, dict) else None,
                "errors": _serialize_validation_errors(exc),
            },
        )

    return (
        200,
        {
            "exists": True,
            "valid": True,
            "config": cfg.model_dump(mode="json"),
            "errors": [],
        },
    )


class _WebUIHypercornLogger(HypercornLogger):
    """Minimal Hypercorn logger that signals when the server starts listening."""

    def __init__(self, config: HypercornConfig) -> None:
        super().__init__(config)
        self._on_listening: Callable[[], None] | None = getattr(
            config, "_on_listening", None
        )

    async def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        if message.startswith("Running on ") and self._on_listening is not None:
            self._on_listening()


class _BoundHypercornConfig(HypercornConfig):
    """Hypercorn config subclass capturing actual bound sockets for introspection."""

    def __init__(
        self,
        *,
        on_sockets_bound: Callable[[Sockets], None],
        on_listening: Callable[[], None],
    ) -> None:
        super().__init__()
        self._on_sockets_bound = on_sockets_bound
        self._on_listening = on_listening
        self.logger_class = _WebUIHypercornLogger

    def create_sockets(self) -> Sockets:
        sockets = super().create_sockets()
        self._on_sockets_bound(sockets)
        return sockets


def create_webui_app(
    webui_config: WebUIConfig,
    *,
    config_path: Path | str = Path("config.json"),
    bot_running: bool = False,
    max_body_bytes: int = MAX_CONFIG_BODY_BYTES,
) -> Quart:
    """Build the Quart ASGI application for the Spectra-bot Config WebUI.

    Args:
        webui_config: Validated :class:`spectra.config.WebUIConfig`.
        config_path: Filesystem path to the target ``config.json`` file.
        bot_running: Whether the WebUI is running alongside an active bot.
        max_body_bytes: Maximum accepted JSON request body size in bytes.

    Returns:
        Configured :class:`quart.Quart` ASGI application.
    """
    resolved_config_path = Path(config_path)
    expected_token = webui_config.auth_token
    auth_required = bool(expected_token)
    write_lock = asyncio.Lock()
    version = _get_package_version()

    app = Quart(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = max_body_bytes
    app.config["JSON_SORT_KEYS"] = False

    def _unauthorized_response() -> tuple[Response, int]:
        resp = jsonify(
            {
                "error": "Unauthorized",
                "detail": "Valid authentication token is required.",
            }
        )
        resp.headers["WWW-Authenticate"] = "Bearer"
        return resp, 401

    def _check_auth() -> tuple[Response, int] | None:
        if not auth_required:
            return None
        provided = _extract_request_token()
        if not provided or not hmac.compare_digest(
            provided.encode("utf-8"),
            expected_token.encode("utf-8"),
        ):
            return _unauthorized_response()
        return None

    def _check_same_origin() -> tuple[Response, int] | None:
        origin = request.headers.get("Origin", "").strip()
        if not origin:
            return None
        parsed = urlsplit(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return (
                jsonify(
                    {
                        "error": "Forbidden",
                        "detail": "Cross-origin requests are not allowed.",
                    }
                ),
                403,
            )
        if parsed.netloc.lower() != request.host.lower():
            return (
                jsonify(
                    {
                        "error": "Forbidden",
                        "detail": "Cross-origin requests are not allowed.",
                    }
                ),
                403,
            )
        return None

    async def _parse_json_object() -> tuple[
        dict[str, Any] | None, Response | None, int
    ]:
        if not request.is_json:
            return (
                None,
                jsonify(
                    {
                        "error": "Unsupported Media Type",
                        "detail": "Request Content-Type must be application/json.",
                    }
                ),
                415,
            )

        content_length = request.content_length
        if content_length is not None and content_length > max_body_bytes:
            return (
                None,
                jsonify(
                    {
                        "error": "Payload Too Large",
                        "detail": (
                            f"Request body exceeds maximum allowed size "
                            f"({max_body_bytes} bytes)."
                        ),
                    }
                ),
                413,
            )

        raw_bytes = await request.get_data(cache=False, as_text=False)
        if len(raw_bytes) > max_body_bytes:
            return (
                None,
                jsonify(
                    {
                        "error": "Payload Too Large",
                        "detail": (
                            f"Request body exceeds maximum allowed size "
                            f"({max_body_bytes} bytes)."
                        ),
                    }
                ),
                413,
            )

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return (
                None,
                jsonify(
                    {
                        "error": "Invalid JSON",
                        "detail": f"Request body could not be parsed as JSON: {exc}",
                    }
                ),
                400,
            )

        if not isinstance(payload, dict):
            return (
                None,
                jsonify(
                    {
                        "valid": False,
                        "saved": False,
                        "errors": [
                            {
                                "loc": [],
                                "msg": "Input should be a valid JSON object.",
                                "type": "model_type",
                            }
                        ],
                    }
                ),
                422,
            )

        return payload, None, 200

    @app.after_request
    async def _apply_security_headers(response: Response) -> Response:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        elif request.path == "/":
            response.headers.setdefault("Cache-Control", "no-store")
        else:
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.errorhandler(413)
    async def _handle_payload_too_large(_: Exception) -> tuple[Response, int]:
        return (
            jsonify(
                {
                    "error": "Payload Too Large",
                    "detail": (
                        f"Request body exceeds maximum allowed size "
                        f"({max_body_bytes} bytes)."
                    ),
                }
            ),
            413,
        )

    @app.get("/")
    async def _serve_index() -> Response:
        loaded = _load_static_asset("webui.html")
        assert loaded is not None
        data, content_type = loaded
        return Response(data, status=200, mimetype=content_type)

    @app.get("/static/<path:filename>")
    async def _serve_static(filename: str) -> Response | tuple[Response, int]:
        loaded = _load_static_asset(filename)
        if loaded is None:
            return jsonify({"error": "Not Found"}), 404
        data, content_type = loaded
        return Response(data, status=200, mimetype=content_type)

    @app.get("/api/meta")
    async def _get_meta() -> Response:
        return jsonify(
            {
                "version": version,
                "config_path": str(resolved_config_path),
                "auth_required": auth_required,
                "bot_running": bot_running,
            }
        )

    @app.get("/api/config/schema")
    async def _get_schema() -> Response | tuple[Response, int]:
        denied = _check_auth()
        if denied is not None:
            return denied
        return jsonify(AppConfig.model_json_schema())

    @app.get("/api/config/defaults")
    async def _get_defaults() -> Response | tuple[Response, int]:
        denied = _check_auth()
        if denied is not None:
            return denied
        return jsonify(AppConfig().model_dump(mode="json"))

    @app.get("/api/config")
    async def _get_config() -> Response | tuple[Response, int]:
        denied = _check_auth()
        if denied is not None:
            return denied
        try:
            status_code, body = await asyncio.to_thread(
                _read_config_file_state,
                resolved_config_path,
            )
        except OSError as exc:
            logger.warning("Failed to read configuration file: %s", exc)
            return (
                jsonify(
                    {
                        "error": "Read Error",
                        "detail": f"Unable to read configuration file: {exc}",
                    }
                ),
                500,
            )
        return jsonify(body), status_code

    @app.post("/api/config/validate")
    async def _validate_config() -> Response | tuple[Response, int]:
        csrf_denied = _check_same_origin()
        if csrf_denied is not None:
            return csrf_denied
        denied = _check_auth()
        if denied is not None:
            return denied
        payload, err_resp, err_status = await _parse_json_object()
        if err_resp is not None or payload is None:
            assert err_resp is not None
            return err_resp, err_status

        try:
            cfg = AppConfig.model_validate(payload)
        except ValidationError as exc:
            return (
                jsonify(
                    {
                        "valid": False,
                        "errors": _serialize_validation_errors(exc),
                    }
                ),
                422,
            )
        return jsonify({"valid": True, "normalized": cfg.model_dump(mode="json")}), 200

    @app.put("/api/config")
    async def _put_config() -> Response | tuple[Response, int]:
        csrf_denied = _check_same_origin()
        if csrf_denied is not None:
            return csrf_denied
        denied = _check_auth()
        if denied is not None:
            return denied
        payload, err_resp, err_status = await _parse_json_object()
        if err_resp is not None or payload is None:
            assert err_resp is not None
            return err_resp, err_status

        try:
            cfg = AppConfig.model_validate(payload)
        except ValidationError as exc:
            return (
                jsonify(
                    {
                        "saved": False,
                        "errors": _serialize_validation_errors(exc),
                    }
                ),
                422,
            )

        try:
            async with write_lock:
                await asyncio.to_thread(save_config, cfg, resolved_config_path)
        except OSError as exc:
            logger.warning("Failed to save configuration file: %s", exc)
            return (
                jsonify(
                    {
                        "saved": False,
                        "error": "Write Error",
                        "detail": f"Unable to save configuration file: {exc}",
                    }
                ),
                500,
            )

        logger.info("Saved updated configuration to %s", resolved_config_path)
        return (
            jsonify(
                {
                    "saved": True,
                    "restart_required": True,
                    "config": cfg.model_dump(mode="json"),
                }
            ),
            200,
        )

    return app


class WebUIServer:
    """Async lifecycle wrapper serving the Config WebUI via Hypercorn.

    Args:
        config: Validated :class:`spectra.config.WebUIConfig`.
        config_path: Target ``config.json`` path read and written by the API.
        bot_running: Whether the server is running alongside ``SpectraBot``.
        host: Optional bind host override (defaults to ``config.host``).
        port: Optional bind port override (defaults to ``config.port``; ``0``
            binds an ephemeral OS-assigned port for testing).
    """

    def __init__(
        self,
        config: WebUIConfig,
        *,
        config_path: Path | str = Path("config.json"),
        bot_running: bool = False,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._config = config
        self._config_path = Path(config_path)
        self._bot_running = bot_running
        self._bind_host = config.host if host is None else host
        self._bind_port = config.port if port is None else port
        if not (0 <= self._bind_port <= 65535):
            raise ValueError(
                f"WebUI bind port must be between 0 and 65535, got {self._bind_port}."
            )

        self._app = create_webui_app(
            config,
            config_path=self._config_path,
            bot_running=bot_running,
        )
        self._shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._serve_task: asyncio.Task[None] | None = None
        self._bound_host: str | None = None
        self._bound_port: int | None = None
        self._started = False
        self._closed = False

    @property
    def app(self) -> Quart:
        """Return the underlying :class:`quart.Quart` ASGI app."""
        return self._app

    @property
    def config(self) -> WebUIConfig:
        """Return the active :class:`WebUIConfig`."""
        return self._config

    @property
    def config_path(self) -> Path:
        """Return the target configuration file path."""
        return self._config_path

    @property
    def bound_host(self) -> str | None:
        """Return the bound socket host once started, or ``None``."""
        return self._bound_host

    @property
    def bound_port(self) -> int | None:
        """Return the bound socket port once started, or ``None``."""
        return self._bound_port

    @property
    def url(self) -> str | None:
        """Return the HTTP URL for the bound listener, or ``None``."""
        if self._bound_host is None or self._bound_port is None:
            return None
        return _format_http_url(self._bound_host, self._bound_port)

    @property
    def is_running(self) -> bool:
        """Return ``True`` while the background Hypercorn server task is active."""
        return self._serve_task is not None and not self._serve_task.done()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _record_bound_sockets(self, sockets: Sockets) -> None:
        for sock in (*sockets.insecure_sockets, *sockets.secure_sockets):
            sockname = sock.getsockname()
            if isinstance(sockname, tuple) and len(sockname) >= 2:
                self._bound_host = str(sockname[0])
                self._bound_port = int(sockname[1])
                return

    def _mark_listening(self) -> None:
        self._ready_event.set()

    async def start(self) -> None:
        """Start serving the Config WebUI in a background task and wait until bound."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("WebUIServer has already been closed.")

        if not _is_loopback_host(self._bind_host) and not self._config.auth_token:
            logger.warning(
                "Config WebUI is binding to non-loopback host %r without an "
                "auth_token! Set webui.auth_token or bind to 127.0.0.1 to protect "
                "API keys and settings.",
                self._bind_host,
            )

        hc_config = _BoundHypercornConfig(
            on_sockets_bound=self._record_bound_sockets,
            on_listening=self._mark_listening,
        )
        hc_config.bind = [_format_bind_address(self._bind_host, self._bind_port)]
        hc_config.accesslog = None
        hc_config.errorlog = None
        hc_config.graceful_timeout = 2.0
        hc_config.shutdown_timeout = 2.0

        self._serve_task = asyncio.create_task(
            hypercorn_serve(
                self._app,
                hc_config,
                shutdown_trigger=self._shutdown_event.wait,
                mode="asgi",
            ),
            name="spectra-webui-server",
        )
        ready_task = asyncio.create_task(
            self._ready_event.wait(),
            name="spectra-webui-ready",
        )
        try:
            done, _ = await asyncio.wait(
                {ready_task, self._serve_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._serve_task in done:
                ready_task.cancel()
                await asyncio.gather(ready_task, return_exceptions=True)
                self._serve_task.result()
                raise RuntimeError("WebUIServer exited before listening.")
            self._started = True
        except BaseException:
            if not ready_task.done():
                ready_task.cancel()
                await asyncio.gather(ready_task, return_exceptions=True)
            await self.close()
            raise

        logger.info(
            "Config WebUI listening at %s (config=%s, auth=%s)",
            self.url,
            self._config_path,
            "enabled" if self._config.auth_token else "disabled",
        )

    def request_shutdown(self) -> None:
        """Signal the Hypercorn server to stop gracefully (signal-handler safe)."""
        self._shutdown_event.set()

    async def wait(self) -> None:
        """Wait until the running Hypercorn server task finishes."""
        if self._serve_task is not None:
            await self._serve_task

    async def close(self) -> None:
        """Idempotently stop the Hypercorn server and release all sockets."""
        if self._closed:
            return
        self._closed = True
        self._started = False
        self._shutdown_event.set()

        task = self._serve_task
        self._serve_task = None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except BaseException:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise
