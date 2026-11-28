"""CLI entrypoint for Spectra-bot (``python -m spectra`` or ``spectra``)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from spectra.bot import SpectraBot
from spectra.config import AppConfig, WebUIConfig, load_config, save_config
from spectra.webui import WebUIServer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the Spectra-bot CLI."""
    parser = argparse.ArgumentParser(
        prog="spectra",
        description=(
            "Spectra-bot — lightweight async QQ bot built directly on NapCat "
            "(OneBot v11)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the JSON configuration file (default: config.json).",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help=(
            "Write a default configuration file to --config if it does not "
            "exist, then exit."
        ),
    )
    parser.add_argument(
        "--webui-only",
        action="store_true",
        help=(
            "Start only the schema-driven Config WebUI without connecting to "
            "NapCat or running the bot."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity level (default: INFO).",
    )
    return parser


def _init_config(path: Path) -> int:
    if path.exists() or path.is_symlink():
        print(
            f"Refusing to overwrite existing configuration file: {path}",
            file=sys.stderr,
        )
        return 1
    save_config(AppConfig(), path)
    print(f"Initialized default configuration at: {path}")
    return 0


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_webui_config_for_standalone(config_path: Path) -> WebUIConfig:
    """Load :class:`WebUIConfig` for ``--webui-only`` with safe recovery fallback.

    If the full ``config.json`` fails validation, attempts to preserve any valid
    ``webui`` section (including ``auth_token``) before falling back to safe
    loopback defaults so the user can repair the file in the WebUI.
    """
    if not config_path.exists():
        return WebUIConfig()
    try:
        return load_config(config_path).webui
    except (ValidationError, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to load full config from %s (%s); checking webui section "
            "for standalone recovery mode.",
            config_path,
            type(exc).__name__,
        )

    try:
        raw = json.loads(
            config_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Invalid JSON constant: {token}")
            ),
        )
        if isinstance(raw, dict) and isinstance(raw.get("webui"), dict):
            return WebUIConfig.model_validate(raw["webui"])
    except (
        ValidationError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ):
        pass

    logger.warning(
        "Falling back to default loopback WebUI settings (127.0.0.1:6200) "
        "for --webui-only recovery."
    )
    return WebUIConfig()


async def _run_webui_only(
    config_path: Path,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_callback: object | None = None,
) -> None:
    webui_cfg = _load_webui_config_for_standalone(config_path)
    server = WebUIServer(
        webui_cfg,
        config_path=config_path,
        bot_running=False,
        host=host,
        port=port,
    )
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, server.request_shutdown)
            installed_signals.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    try:
        await server.start()
        if callable(ready_callback):
            ready_callback(server)
        await server.wait()
    finally:
        await server.close()
        for sig in installed_signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass


async def _run_bot(config_path: Path) -> None:
    config = load_config(config_path)
    bot = SpectraBot(config)
    webui: WebUIServer | None = None
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []

    def _handle_shutdown() -> None:
        bot.request_shutdown()
        if webui is not None:
            webui.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_shutdown)
            installed_signals.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    logger.info(
        "Starting Spectra-bot (config=%s, napcat=%s:%d, db=%s, webui=%s)",
        config_path,
        config.napcat.host,
        config.napcat.port,
        config.db.path,
        f"{config.webui.host}:{config.webui.port}"
        if config.webui.enabled
        else "disabled",
    )
    try:
        if config.webui.enabled:
            await bot.start()
            webui = WebUIServer(
                config.webui,
                config_path=config_path,
                bot_running=True,
                persona_store=bot.persona_store,
            )
            await webui.start()
        if webui is None:
            await bot.run()
        else:
            bot_task = asyncio.create_task(bot.run(), name="spectra-bot-run")
            webui_task = asyncio.create_task(webui.wait(), name="spectra-webui-wait")
            try:
                done, _ = await asyncio.wait(
                    {bot_task, webui_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if bot_task in done:
                    webui.request_shutdown()
                    await asyncio.gather(webui_task, return_exceptions=True)
                    bot_task.result()
                else:
                    bot.request_shutdown()
                    await bot_task
                    webui_task.result()
            except BaseException:
                bot.request_shutdown()
                webui.request_shutdown()
                for task in (bot_task, webui_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(bot_task, webui_task, return_exceptions=True)
                raise
    finally:
        if webui is not None:
            await webui.close()
        await bot.close()
        for sig in installed_signals:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run Spectra-bot."""
    args = build_parser().parse_args(argv)
    if args.init_config:
        return _init_config(args.config)

    _configure_logging(args.log_level)
    try:
        if args.webui_only:
            asyncio.run(_run_webui_only(args.config))
        else:
            asyncio.run(_run_bot(args.config))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
