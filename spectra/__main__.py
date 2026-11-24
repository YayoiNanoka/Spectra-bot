"""CLI entrypoint for Spectra-bot (``python -m spectra`` or ``spectra``)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from spectra.bot import SpectraBot
from spectra.config import AppConfig, load_config, save_config

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
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity level (default: INFO).",
    )
    return parser


def _init_config(path: Path) -> int:
    if path.exists():
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


async def _run_bot(config_path: Path) -> None:
    config = load_config(config_path)
    bot = SpectraBot(config)
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.request_shutdown)
            installed_signals.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    logger.info(
        "Starting Spectra-bot (config=%s, napcat=%s:%d, db=%s)",
        config_path,
        config.napcat.host,
        config.napcat.port,
        config.db.path,
    )
    try:
        await bot.run()
    finally:
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
        asyncio.run(_run_bot(args.config))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
