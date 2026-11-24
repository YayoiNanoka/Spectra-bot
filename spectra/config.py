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


class SegmentedReplyConfig(BaseModel):
    """AstrBot-aligned configuration for segmented outbound replies."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(
        default=True,
        title="Enable Segmented Reply",
        description=(
            "Split short replies into multiple chunks with natural typing pauses."
        ),
    )
    interval_method: Literal["random", "log"] = Field(
        default="random",
        title="Interval Method",
        description="Delay calculation strategy between segments ('random' or 'log').",
    )
    interval: str = Field(
        default="1.5,3.5",
        title="Random Interval Range",
        description=(
            "Comma-separated min,max delay in seconds when interval_method is 'random'."
        ),
    )
    log_base: float = Field(
        default=2.6,
        gt=1.0,
        allow_inf_nan=False,
        title="Logarithmic Delay Base",
        description=(
            "Logarithm base (> 1) used to compute typing delay when "
            "interval_method is 'log'."
        ),
    )
    words_count_threshold: int = Field(
        default=150,
        ge=1,
        title="Words Count Threshold",
        description="Maximum character length eligible for segmented reply splitting.",
    )
    split_mode: Literal["regex", "words"] = Field(
        default="regex",
        title="Split Mode",
        description="How reply text is segmented ('regex' or 'words').",
    )
    regex: str = Field(
        default=r".*?[。？！~…]+|.+$",
        title="Split Regex",
        description="Regular expression used to split replies in 'regex' mode.",
    )
    split_words: list[str] = Field(
        default_factory=lambda: ["。", "？", "！", "~", "…"],
        title="Split Words",
        description="Delimiter tokens used to split replies in 'words' mode.",
    )
    content_cleanup_rule: str = Field(
        default="",
        title="Content Cleanup Rule",
        description=(
            "Optional regular expression stripped from each segment before sending."
        ),
    )


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
    segmented_reply: SegmentedReplyConfig = Field(
        default_factory=SegmentedReplyConfig,
        title="Segmented Reply",
        description="AstrBot-compatible segmented reply splitting and delay settings.",
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
    max_size: int = Field(
        default=1280,
        ge=3,
        title="Max Image Size (Pixels)",
        description=(
            "Maximum longest edge in pixels for prepared still images and montages."
        ),
    )


class LLMConfig(BaseModel):
    """OpenAI-compatible chat LLM client and agent tool-loop configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(
        default="https://api.openai.com/v1",
        title="LLM API Base URL",
        description="OpenAI-compatible API endpoint base URL for chat completions.",
    )
    api_key: str = Field(
        default="",
        title="LLM API Key",
        description="API key used to authenticate chat completion requests.",
    )
    model: str = Field(
        default="gpt-4o",
        title="Primary Model",
        description="Primary model identifier used for chat completions.",
    )
    fallback_model: str = Field(
        default="",
        title="Fallback Model",
        description=(
            "Optional fallback model identifier used when the primary model fails "
            "after retries."
        ),
    )
    default_effort: str = Field(
        default="",
        title="Default Reasoning Effort",
        description=(
            "Value sent as reasoning_effort when effort is 'default'; empty string "
            "omits reasoning_effort from the API payload."
        ),
    )
    high_effort: str = Field(
        default="high",
        title="High Reasoning Effort",
        description=(
            "Value sent as reasoning_effort when effort is 'high'; empty string "
            "omits reasoning_effort from the API payload."
        ),
    )
    timeout_sec: float = Field(
        default=60.0,
        gt=0.0,
        allow_inf_nan=False,
        title="Request Timeout (Seconds)",
        description="Per-request HTTP and API timeout in seconds.",
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        title="Max Retry Attempts",
        description=(
            "Maximum total attempts per model for retryable network or HTTP errors."
        ),
    )
    max_tool_steps: int = Field(
        default=10,
        ge=1,
        title="Max Tool Loop Steps",
        description=(
            "Maximum number of LLM completion rounds allowed in a single tool loop."
        ),
    )


class JevConfig(BaseModel):
    """System One / Jev decision and effort router configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        title="Enable Jev Decision Router",
        description=(
            "Whether to route proactive reply and reasoning-effort decisions "
            "through the System One / Jev decision API."
        ),
    )
    base_url: str = Field(
        default="https://opencode.ai/inference/go/systemone/v1/systemone",
        title="Jev API Base URL",
        description=(
            "System One / Jev endpoint URL or API base URL. URLs ending with "
            "'/systemone' or '/decisions' are used verbatim; otherwise "
            "'/systemone' is appended."
        ),
    )
    api_key: str = Field(
        default="",
        title="Jev API Key",
        description="API key used to authenticate System One / Jev requests.",
    )
    model: str = Field(
        default="jev-1.14",
        title="Jev Model",
        description="Model identifier used for System One / Jev decision calls.",
    )
    timeout_sec: float = Field(
        default=1.5,
        gt=0.0,
        allow_inf_nan=False,
        title="Jev Timeout (Seconds)",
        description="Hard single-shot HTTP timeout in seconds for Jev requests.",
    )
    reply_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="Proactive Reply Threshold",
        description=(
            "Minimum 'should_reply' noul probability required to trigger a "
            "proactive group reply."
        ),
    )
    high_effort_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="High Effort Threshold",
        description=(
            "Minimum probability/confidence required to route a turn to "
            "'high' reasoning effort."
        ),
    )

    @property
    def is_configured(self) -> bool:
        """Return whether Jev is enabled with non-empty URL, API key, and model."""
        return bool(
            self.enabled
            and self.base_url.strip()
            and self.api_key.strip()
            and self.model.strip()
        )


class ActiveReplyConfig(BaseModel):
    """AstrBot-aligned fallback proactive reply configuration when Jev is unused."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["possibility_reply"] = Field(
        default="possibility_reply",
        title="Active Reply Method",
        description=(
            "Fallback proactive reply strategy used when Jev is not configured."
        ),
    )
    possibility_reply: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="Proactive Reply Probability",
        description=(
            "Probability in [0, 1] of replying to unmentioned group chatter "
            "when Jev is not configured."
        ),
    )
    whitelist: list[str] = Field(
        default_factory=list,
        title="Proactive Reply Group Whitelist",
        description=(
            "Optional list of group IDs eligible for proactive group replies; "
            "empty list allows all accessible groups."
        ),
    )


class GateConfig(BaseModel):
    """Entry gate debounce, local pre-filter, and proactive quota configuration."""

    model_config = ConfigDict(extra="forbid")

    proactive_mode: Literal["off", "shadow", "on"] = Field(
        default="on",
        title="Proactive Reply Mode",
        description=(
            "Controls unmentioned group replies: 'off' disables them, 'shadow' "
            "evaluates and records decisions without sending, and 'on' replies."
        ),
    )
    private_debounce_sec: float = Field(
        default=1.5,
        ge=0.0,
        allow_inf_nan=False,
        title="Private Debounce (Seconds)",
        description="Debounce window in seconds for batching consecutive private DMs.",
    )
    group_debounce_sec: float = Field(
        default=2.5,
        ge=0.0,
        allow_inf_nan=False,
        title="Group Debounce (Seconds)",
        description=(
            "Debounce window in seconds for batching consecutive group messages."
        ),
    )
    max_batch_wait_sec: float = Field(
        default=6.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Max Batch Wait (Seconds)",
        description=(
            "Maximum total seconds a debounce batch may wait from its first "
            "message before flushing."
        ),
    )
    proactive_cooldown_sec: float = Field(
        default=60.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Proactive Cooldown (Seconds)",
        description=(
            "Minimum interval in seconds between proactive group replies "
            "outside the hot-chat window."
        ),
    )
    proactive_max_per_hour: int = Field(
        default=15,
        ge=0,
        title="Proactive Max Per Hour",
        description=(
            "Maximum number of proactive replies allowed per group within a "
            "rolling 1-hour window."
        ),
    )
    hot_window_sec: float = Field(
        default=30.0,
        ge=0.0,
        allow_inf_nan=False,
        title="Hot-Chat Window (Seconds)",
        description=(
            "Duration in seconds after any bot reply in a group during which "
            "proactive cooldown is bypassed."
        ),
    )
    state_message_limit: int = Field(
        default=10,
        ge=1,
        title="Decision State Message Limit",
        description=(
            "Maximum number of prior session messages included in Jev "
            "'recent_messages' context."
        ),
    )
    active_reply: ActiveReplyConfig = Field(
        default_factory=ActiveReplyConfig,
        title="Active Reply Fallback",
        description=(
            "AstrBot-compatible proactive reply whitelist and probability "
            "fallback settings."
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
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        title="LLM Client",
        description=(
            "OpenAI-compatible chat model, retry, fallback, and tool-loop settings."
        ),
    )
    jev: JevConfig = Field(
        default_factory=JevConfig,
        title="Jev Decision Router",
        description=(
            "System One / Jev decision model endpoint, thresholds, and timeout."
        ),
    )
    gate: GateConfig = Field(
        default_factory=GateConfig,
        title="Entry Gate",
        description=(
            "Session debounce, proactive filtering, quota, and fallback settings."
        ),
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
