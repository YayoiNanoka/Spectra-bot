"""Unified Pydantic v2 configuration for Spectra-bot.

Every field declares explicit metadata (``title`` and ``description``) so
:meth:`AppConfig.model_json_schema` can directly drive a future WebUI settings
panel without duplicating schema definitions.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatabaseConfig(BaseModel):
    """SQLite database storage and retention configuration."""

    model_config = ConfigDict(extra="forbid")

    path: Path = Field(
        default=Path("data/bot.db"),
        title="Database Path",
        description="Filesystem path to the local SQLite database file.",
    )
    message_retention_days: int = Field(
        default=7,
        ge=1,
        title="Message Retention Days",
        description="Number of days to keep persisted chat messages before cleanup.",
    )


class AccessConfig(BaseModel):
    """Access control policy for group and private QQ chats."""

    model_config = ConfigDict(extra="forbid")

    admin_users: list[str] = Field(
        default_factory=list,
        title="Admin Users",
        description="QQ IDs with administrator privileges.",
    )
    ignored_users: list[str] = Field(
        default_factory=lambda: ["2854196310"],
        title="Ignored Users",
        description=(
            "QQ IDs ignored unconditionally (defaults to QQ Guanjia official bot)."
        ),
    )
    group_mode: Literal["blacklist", "whitelist"] = Field(
        default="blacklist",
        title="Group Access Mode",
        description="Whether group access filtering uses a blacklist or whitelist.",
    )
    group_blacklist: list[str] = Field(
        default_factory=list,
        title="Group Blacklist",
        description="Group IDs blocked when group_mode is 'blacklist'.",
    )
    group_whitelist: list[str] = Field(
        default_factory=list,
        title="Group Whitelist",
        description="Group IDs allowed when group_mode is 'whitelist'.",
    )
    private_mode: Literal["blacklist", "whitelist"] = Field(
        default="blacklist",
        title="Private Access Mode",
        description="Whether private chat filtering uses a blacklist or whitelist.",
    )
    private_blacklist: list[str] = Field(
        default_factory=list,
        title="Private Blacklist",
        description="User IDs blocked in private chat and non-admin group messages.",
    )
    private_whitelist: list[str] = Field(
        default_factory=list,
        title="Private Whitelist",
        description="User IDs allowed when private_mode is 'whitelist'.",
    )

    def is_allowed(
        self,
        *,
        group_id: str | None,
        user_id: str,
        self_id: str,
    ) -> bool:
        """Return whether an inbound message should be processed.

        Args:
            group_id: Group QQ ID for group messages, or ``None`` for private chat.
            user_id: Sender QQ ID.
            self_id: Bot's own QQ ID.

        Returns:
            ``True`` if the message passes all access rules, ``False`` otherwise.
        """
        uid = str(user_id)
        sid = str(self_id)
        if uid == sid or uid in self.ignored_users:
            return False

        is_admin = uid in self.admin_users

        if group_id is not None:
            gid = str(group_id)
            group_allowed = (
                gid not in self.group_blacklist
                if self.group_mode == "blacklist"
                else gid in self.group_whitelist
            )
            if not group_allowed:
                return False
            return is_admin or uid not in self.private_blacklist

        if is_admin:
            return True
        if self.private_mode == "blacklist":
            return uid not in self.private_blacklist
        return uid in self.private_whitelist


class NapCatConfig(BaseModel):
    """NapCat (OneBot v11 reverse WebSocket) transport and reply settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(
        default="0.0.0.0",
        title="Reverse WS Host",
        description="Host interface for the reverse WebSocket server.",
    )
    port: int = Field(
        default=6199,
        ge=1,
        le=65535,
        title="Reverse WS Port",
        description="TCP port for the reverse WebSocket server.",
    )
    access_token: str = Field(
        default="",
        title="Access Token",
        description="Optional OneBot v11 bearer token for NapCat authentication.",
    )
    bot_name: str = Field(
        default="Spectra",
        title="Bot Name",
        description=(
            "Display name recorded as sender_name for outbound assistant replies."
        ),
    )
    enable_segmented_reply: bool = Field(
        default=True,
        title="Enable Segmented Reply",
        description=(
            "Split short replies into sentence chunks with natural typing pauses."
        ),
    )
    segmented_words_threshold: int = Field(
        default=150,
        ge=1,
        title="Segmented Reply Length Threshold",
        description="Maximum character length eligible for segmented reply splitting.",
    )
    segmented_regex: str = Field(
        default=r".*?[。？！~…]+|.+$",
        title="Segmented Reply Regex",
        description="Regular expression used to split short replies into chunks.",
    )
    segmented_log_base: float = Field(
        default=2.6,
        gt=1.0,
        title="Segmented Delay Log Base",
        description="Logarithm base for computing per-segment typing delay.",
    )
    forward_threshold: int = Field(
        default=1500,
        ge=1,
        title="Forward Message Threshold",
        description=(
            "Character count above which replies are sent as a forward message."
        ),
    )


class VisionConfig(BaseModel):
    """OpenAI-compatible vision captioner configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        title="Enable Vision Captioning",
        description="Whether inbound images and animated stickers are captioned.",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        title="Vision API Base URL",
        description="OpenAI-compatible API endpoint base URL for vision requests.",
    )
    api_key: str = Field(
        default="",
        title="Vision API Key",
        description="API key used to authenticate vision caption requests.",
    )
    model: str = Field(
        default="gpt-4o-mini",
        title="Vision Model",
        description="Model identifier used for image captioning.",
    )
    prompt: str = Field(
        default="请用简短中文（30字以内）描述这张图片或表情包的画面内容、图中文字和情绪。",
        title="Vision Prompt",
        description="Instruction prompt sent alongside each image.",
    )
    timeout_sec: float = Field(
        default=15.0,
        gt=0.0,
        title="Vision Timeout (Seconds)",
        description="HTTP and API timeout in seconds for downloading and captioning.",
    )
    max_input_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1024,
        title="Max Input Image Bytes",
        description="Maximum raw source image size accepted before aborting download.",
    )
    max_image_edge: int = Field(
        default=1280,
        ge=64,
        title="Max Image Edge (Pixels)",
        description=(
            "Maximum longest edge in pixels for prepared still images and montages."
        ),
    )


class AppConfig(BaseModel):
    """Root application configuration for Spectra-bot."""

    model_config = ConfigDict(extra="forbid")

    db: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        title="Database",
        description="SQLite database and message retention settings.",
    )
    access: AccessConfig = Field(
        default_factory=AccessConfig,
        title="Access Control",
        description="Whitelist, blacklist, admin, and ignored user rules.",
    )
    napcat: NapCatConfig = Field(
        default_factory=NapCatConfig,
        title="NapCat Adapter",
        description="OneBot v11 reverse WebSocket and outbound reply settings.",
    )
    vision: VisionConfig = Field(
        default_factory=VisionConfig,
        title="Vision Captioner",
        description="Image download, preprocessing, and OpenAI-compatible captioning.",
    )


def load_config(path: Path | str = Path("config.json")) -> AppConfig:
    """Load and validate application configuration from a UTF-8 JSON file.

    Returns default :class:`AppConfig` values when ``path`` does not exist.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Validated :class:`AppConfig` instance.

    Raises:
        pydantic.ValidationError: If the JSON content fails schema validation.
        OSError: If reading an existing file fails.
    """
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    raw = config_path.read_text(encoding="utf-8")
    return AppConfig.model_validate_json(raw)


def save_config(config: AppConfig, path: Path | str = Path("config.json")) -> None:
    """Atomically write application configuration as formatted UTF-8 JSON.

    Creates parent directories if needed and writes to a temporary file in the
    same directory before replacing the target via :func:`os.replace`.

    Args:
        config: Configuration model to serialize.
        path: Destination JSON file path.

    Raises:
        OSError: If writing or replacing the file fails.
    """
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        config.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(payload)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, config_path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
