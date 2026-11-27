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
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatabaseConfig(BaseModel):
    """SQLite database storage and retention configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="数据库配置",
        json_schema_extra={
            "description": "本地 SQLite 数据库存储路径与消息保留策略配置。"
        },
    )

    path: Path = Field(
        default=Path("data/bot.db"),
        title="数据库文件路径",
        description="本地 SQLite 数据库文件的保存路径。",
    )
    message_retention_days: int = Field(
        default=7,
        ge=1,
        title="消息保留天数",
        description="持久化聊天消息在定期清理前保留的天数（至少 1 天）。",
    )


class AccessConfig(BaseModel):
    """Access control policy for group and private QQ chats."""

    model_config = ConfigDict(
        extra="forbid",
        title="访问控制配置",
        json_schema_extra={
            "description": "群聊与私聊的黑白名单、管理员及全局忽略用户策略。"
        },
    )

    admin_users: list[str] = Field(
        default_factory=list,
        title="管理员 QQ 列表",
        description=(
            "拥有管理员权限的 QQ 号列表"
            "（私聊免受黑白名单限制，群聊免受私聊黑名单限制）。"
        ),
    )
    ignored_users: list[str] = Field(
        default_factory=lambda: ["2854196310"],
        title="全局忽略用户列表",
        description="无条件忽略的 QQ 号列表（默认包含 QQ 管家官方机器人 2854196310）。",
    )
    group_mode: Literal["blacklist", "whitelist"] = Field(
        default="blacklist",
        title="群聊访问模式",
        description="群聊消息过滤模式：blacklist（黑名单）或 whitelist（白名单）。",
    )
    group_blacklist: list[str] = Field(
        default_factory=list,
        title="群聊黑名单",
        description="当群聊访问模式为黑名单时，禁止响应的群号列表。",
    )
    group_whitelist: list[str] = Field(
        default_factory=list,
        title="群聊白名单",
        description="当群聊访问模式为白名单时，允许响应的群号列表。",
    )
    private_mode: Literal["blacklist", "whitelist"] = Field(
        default="blacklist",
        title="私聊访问模式",
        description="私聊消息过滤模式：blacklist（黑名单）或 whitelist（白名单）。",
    )
    private_blacklist: list[str] = Field(
        default_factory=list,
        title="私聊黑名单",
        description="私聊以及群聊中非管理员用户的屏蔽 QQ 号列表。",
    )
    private_whitelist: list[str] = Field(
        default_factory=list,
        title="私聊白名单",
        description="当私聊访问模式为白名单时，允许响应的 QQ 号列表。",
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

    model_config = ConfigDict(
        extra="forbid",
        title="分段回复配置",
        json_schema_extra={
            "description": "对齐 AstrBot 的出站分段回复切分与打字间隔策略。"
        },
    )

    enable: bool = Field(
        default=True,
        title="启用分段回复",
        description="开启后将较短回复切分为多条消息，并模拟自然打字停顿依次发送。",
    )
    interval_method: Literal["random", "log"] = Field(
        default="random",
        title="分段间隔计算方式",
        description=(
            "相邻分段之间的发送延迟策略：random（随机区间）或 log（对数延迟）。"
        ),
    )
    interval: str = Field(
        default="1.5,3.5",
        title="随机间隔区间（秒）",
        description=(
            "当间隔方式为 random 时生效，格式为「最小值,最大值」（如 1.5,3.5）。"
        ),
    )
    log_base: float = Field(
        default=2.6,
        gt=1.0,
        allow_inf_nan=False,
        title="对数延迟底数",
        description="当间隔方式为 log 时用于计算打字延迟的对数底数（必须大于 1）。",
    )
    words_count_threshold: int = Field(
        default=150,
        ge=1,
        title="分段字数阈值",
        description="允许进行分段发送的最大回复字数，超过此长度将整段直接发送。",
    )
    split_mode: Literal["regex", "words"] = Field(
        default="regex",
        title="文本切分模式",
        description="回复文本的切分方式：regex（正则表达式）或 words（分隔符列表）。",
    )
    regex: str = Field(
        default=r".*?[。？！~…]+|.+$",
        title="切分正则表达式",
        description="当切分模式为 regex 时用于分段匹配回复文本的正则表达式。",
    )
    split_words: list[str] = Field(
        default_factory=lambda: ["。", "？", "！", "~", "…"],
        title="切分分隔符列表",
        description="当切分模式为 words 时用于切分回复文本的标点或词语列表。",
    )
    content_cleanup_rule: str = Field(
        default="",
        title="分段内容清理规则",
        description="可选正则表达式；发送前会从每个分段中移除匹配到的内容。",
    )


class NapCatConfig(BaseModel):
    """NapCat (OneBot v11 reverse WebSocket) transport and reply settings."""

    model_config = ConfigDict(
        extra="forbid",
        title="NapCat 适配配置",
        json_schema_extra={
            "description": "NapCat（OneBot v11 反向 WebSocket）连接与消息发送设置。"
        },
    )

    host: str = Field(
        default="0.0.0.0",
        title="反向 WebSocket 监听地址",
        description="本地 OneBot v11 反向 WebSocket 服务绑定的监听主机地址。",
    )
    port: int = Field(
        default=6199,
        ge=1,
        le=65535,
        title="反向 WebSocket 端口",
        description="本地 OneBot v11 反向 WebSocket 服务监听的 TCP 端口号（1~65535）。",
    )
    access_token: str = Field(
        default="",
        title="连接鉴权令牌",
        description="可选的 OneBot v11 Bearer 鉴权令牌，需与 NapCat 端配置保持一致。",
    )
    bot_name: str = Field(
        default="Spectra",
        title="机器人显示名称",
        description="写入消息记录与上下文中的机器人昵称（sender_name）。",
    )
    segmented_reply: SegmentedReplyConfig = Field(
        default_factory=SegmentedReplyConfig,
        title="分段回复",
        description="兼容 AstrBot 的分段发送切分与停顿延迟配置。",
    )
    forward_threshold: int = Field(
        default=1500,
        ge=1,
        title="合并转发字数阈值",
        description="回复字符数超过该阈值时，自动转为 QQ 合并转发消息发送。",
    )


class VisionConfig(BaseModel):
    """OpenAI-compatible vision captioner configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="视觉识图配置",
        json_schema_extra={
            "description": "图片下载预处理与 OpenAI 兼容视觉识图模型配置。"
        },
    )

    enabled: bool = Field(
        default=True,
        title="启用图片识图",
        description="是否自动对收到的图片与动态表情包进行视觉描述。",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        title="识图接口地址",
        description="OpenAI 兼容视觉识图接口的 Base URL。",
    )
    api_key: str = Field(
        default="",
        title="识图 API 密钥",
        description="用于视觉识图接口鉴权的 API Key。",
    )
    model: str = Field(
        default="gpt-4o-mini",
        title="识图模型名称",
        description="用于生成图片与表情包描述的模型标识符。",
    )
    prompt: str = Field(
        default="请用简短中文（30字以内）描述这张图片或表情包的画面内容、图中文字和情绪。",
        title="识图提示词",
        description="随每张图片一并发送给视觉模型的描述指令。",
    )
    timeout_sec: float = Field(
        default=15.0,
        gt=0.0,
        allow_inf_nan=False,
        title="识图超时时间（秒）",
        description="图片下载与视觉模型请求的超时时间（秒）。",
    )
    max_size: int = Field(
        default=1280,
        ge=3,
        title="图片最大边长（像素）",
        description="静态图片与动图拼图预处理时允许的最大最长边像素值。",
    )
    reasoning_effort: str = Field(
        default="",
        title="识图推理强度",
        description=(
            "可选，传给多模态推理模型的 reasoning_effort（如 low）；留空则不传该字段。"
        ),
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        title="识图自定义请求头",
        description=(
            "合并到视觉接口请求中的额外 HTTP 请求头（不区分大小写，可覆盖网关默认值）。"
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        title="识图附加请求体参数",
        description="合并到视觉接口 Chat Completions 请求体中的额外 JSON 参数。",
    )


class LLMConfig(BaseModel):
    """OpenAI-compatible chat LLM client and agent tool-loop configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="大语言模型配置",
        json_schema_extra={
            "description": "OpenAI 兼容对话模型、重试降级与工具调用循环配置。"
        },
    )

    base_url: str = Field(
        default="https://api.openai.com/v1",
        title="模型接口地址",
        description="OpenAI 兼容对话补全接口的 Base URL。",
    )
    api_key: str = Field(
        default="",
        title="模型 API 密钥",
        description="用于对话补全接口鉴权的 API Key。",
    )
    model: str = Field(
        default="gpt-4o",
        title="主对话模型",
        description="日常回复与工具调用优先使用的主模型标识符。",
    )
    fallback_model: str = Field(
        default="",
        title="备用降级模型",
        description="可选；当主模型重试耗尽仍失败时自动切换的备用模型标识符。",
    )
    default_effort: str = Field(
        default="",
        title="常规推理强度",
        description=(
            "常规路由（default）时传入的 reasoning_effort 值；留空则省略该字段。"
        ),
    )
    high_effort: str = Field(
        default="high",
        title="深度推理强度",
        description=(
            "复杂任务路由（high）时传入的 reasoning_effort 值；留空则省略该字段。"
        ),
    )
    timeout_sec: float = Field(
        default=60.0,
        gt=0.0,
        allow_inf_nan=False,
        title="单次请求超时（秒）",
        description="单次模型 HTTP/API 请求的超时时间（秒）。",
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        title="最大重试次数",
        description="遇到可重试网络或 HTTP 错误时，单个模型的最大尝试总次数。",
    )
    max_tool_steps: int = Field(
        default=10,
        ge=1,
        title="最大工具调用轮数",
        description="单次 Agent 工具循环中允许执行的最大模型补全轮数。",
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        title="模型自定义请求头",
        description=(
            "合并到对话模型请求中的额外 HTTP 请求头（不区分大小写，可覆盖网关默认值）。"
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        title="模型附加请求体参数",
        description="合并到对话模型 Chat Completions 请求体中的额外 JSON 参数。",
    )


class JevConfig(BaseModel):
    """System One / Jev decision and effort router configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="Jev 决策路由配置",
        json_schema_extra={
            "description": "System One / Jev 群聊读空气与推理强度决策路由配置。"
        },
    )

    enabled: bool = Field(
        default=True,
        title="启用 Jev 决策路由",
        description="是否通过 System One / Jev 决策接口判断群聊主动插话与推理强度。",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/alpha/decisions",
        title="Jev 决策接口地址",
        description=(
            "决策接口完整端点或 Base URL（默认使用 OpenRouter Alpha Decisions，"
            "兼容 System One 与 OpenAI Decisions 网关）。"
        ),
    )
    api_key: str = Field(
        default="",
        title="Jev API 密钥",
        description="用于决策接口鉴权的独立 API Key（不会从主对话模型隐式继承）。",
    )
    model: str = Field(
        default="typesafe/jev-1.14",
        title="Jev 决策模型",
        description="用于 System One / OpenAI 决策请求的模型标识符。",
    )
    protocol: Literal["auto", "system_one", "openai_decision"] = Field(
        default="auto",
        title="决策接口协议",
        description=(
            "决策通信协议：auto 按地址自动识别，system_one 使用 System One 协议，"
            "openai_decision 使用 OpenAI 决策协议。"
        ),
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        title="Jev 自定义请求头",
        description=(
            "合并到决策接口请求中的额外 HTTP 请求头（不区分大小写，可覆盖网关默认值）。"
        ),
    )
    timeout_sec: float = Field(
        default=1.5,
        gt=0.0,
        allow_inf_nan=False,
        title="Jev 请求超时（秒）",
        description="单次决策 HTTP 请求的硬超时时间（秒），超时后安全降级。",
    )
    reply_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="主动回复置信度阈值",
        description=(
            "群聊未 @ 机器人时，触发主动插话所需的 should_reply 最低概率（0~1）。"
        ),
    )
    high_effort_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="深度推理置信度阈值",
        description="将本轮对话路由至高推理强度（high）所需的最低置信度（0~1）。",
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

    model_config = ConfigDict(
        extra="forbid",
        title="主动回复回退策略",
        json_schema_extra={
            "description": (
                "未配置 Jev 时的 AstrBot 兼容群聊主动回复白名单与概率回退配置。"
            )
        },
    )

    method: Literal["possibility_reply"] = Field(
        default="possibility_reply",
        title="主动回复回退方式",
        description="当未配置 Jev 决策服务时使用的群聊主动回复策略。",
    )
    possibility_reply: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        title="主动回复概率",
        description="未配置 Jev 时，对群聊未 @ 消息随机触发主动回复的概率（0~1）。",
    )
    whitelist: list[str] = Field(
        default_factory=list,
        title="主动回复群白名单",
        description="可选，仅允许列表中的群触发主动插话；留空则对所有允许访问的群生效。",
    )


class GateConfig(BaseModel):
    """Entry gate debounce, local pre-filter, and proactive quota configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="进门门控配置",
        json_schema_extra={
            "description": "会话消息防抖合并、本地预筛、主动插话限流与回退策略配置。"
        },
    )

    proactive_mode: Literal["off", "shadow", "on"] = Field(
        default="on",
        title="群聊主动回复模式",
        description=(
            "控制未 @ 机器人时的群聊响应：off 关闭，shadow 仅评估并记录日志不发送，"
            "on 正常主动回复。"
        ),
    )
    private_debounce_sec: float = Field(
        default=1.5,
        ge=0.0,
        allow_inf_nan=False,
        title="私聊防抖窗口（秒）",
        description="合并连续私聊消息的防抖等待时间（秒）。",
    )
    group_debounce_sec: float = Field(
        default=2.5,
        ge=0.0,
        allow_inf_nan=False,
        title="群聊防抖窗口（秒）",
        description="合并连续群聊消息的防抖等待时间（秒）。",
    )
    max_batch_wait_sec: float = Field(
        default=6.0,
        ge=0.0,
        allow_inf_nan=False,
        title="批次最长等待时间（秒）",
        description="从首条消息到达起，单批防抖合并允许等待的最长时间（秒）。",
    )
    proactive_cooldown_sec: float = Field(
        default=60.0,
        ge=0.0,
        allow_inf_nan=False,
        title="主动回复冷却时间（秒）",
        description="在热聊窗口之外，同一群两次主动插话之间的最短间隔（秒）。",
    )
    proactive_max_per_hour: int = Field(
        default=15,
        ge=0,
        title="每小时主动回复上限",
        description="单个群在滚动 1 小时内允许主动插话的最大次数。",
    )
    hot_window_sec: float = Field(
        default=30.0,
        ge=0.0,
        allow_inf_nan=False,
        title="热聊窗口时长（秒）",
        description="机器人在群内成功回复后，跳过主动回复冷却限制的持续时间（秒）。",
    )
    state_message_limit: int = Field(
        default=10,
        ge=1,
        title="决策上下文消息条数",
        description="送入 Jev 决策模型 recent_messages 的最近历史消息最大条数。",
    )
    active_reply: ActiveReplyConfig = Field(
        default_factory=ActiveReplyConfig,
        title="主动回复回退设置",
        description="兼容 AstrBot 的主动回复群白名单与概率回退配置。",
    )


class SessionConfig(BaseModel):
    """Persona prompt directory, rolling context window, and timezone settings."""

    model_config = ConfigDict(
        extra="forbid",
        title="会话与上下文配置",
        json_schema_extra={
            "description": "本地人设提示词目录、滚动上下文窗口大小与显示时区设置。"
        },
    )

    prompts_dir: Path = Field(
        default=Path("prompts"),
        title="人设提示词目录",
        description=(
            "存放本地 Markdown 人设文件的目录"
            "（支持 default.md、group_{群号}.md、private_{QQ号}.md）。"
        ),
    )
    window_max_messages: int = Field(
        default=60,
        ge=2,
        title="上下文窗口最大消息数",
        description=(
            "触发块滚动（推进 window_start_id）前保留的活跃会话消息上限（至少 2 条）。"
        ),
    )
    window_keep_messages: int = Field(
        default=30,
        ge=1,
        title="滚动后保留消息数",
        description="上下文窗口滚动后保留的最近消息条数（必须严格小于最大消息数）。",
    )
    timezone: str = Field(
        default="Asia/Shanghai",
        title="消息显示时区",
        description="格式化群聊上下文时间戳所用的 IANA 时区名称（如 Asia/Shanghai）。",
    )

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid IANA timezone: {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _validate_window_bounds(self) -> SessionConfig:
        if self.window_keep_messages >= self.window_max_messages:
            raise ValueError(
                "window_keep_messages must be strictly less than "
                f"window_max_messages (got {self.window_keep_messages} >= "
                f"{self.window_max_messages})."
            )
        return self


class TraceConfig(BaseModel):
    """Structured JSONL trace logging configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="追踪日志配置",
        json_schema_extra={
            "description": "门控决策与大语言模型执行过程的结构化 JSONL 日志配置。"
        },
    )

    enabled: bool = Field(
        default=True,
        title="启用追踪日志",
        description="是否将每次消息批次评估与模型执行遥测写入结构化 JSONL 日志。",
    )
    path: Path = Field(
        default=Path("data/traces.jsonl"),
        title="追踪日志文件路径",
        description="只追加写入的 JSONL 追踪日志文件路径。",
    )


WebSearchProvider = Literal[
    "tavily",
    "bocha",
    "brave",
    "firecrawl",
    "baidu_ai_search",
    "exa",
    "anysearch",
    "deepseek",
    "serpapi",
]


class WebSearchConfig(BaseModel):
    """AstrBot-aligned web search provider and API key rotation configuration."""

    model_config = ConfigDict(
        extra="forbid",
        title="联网搜索配置",
        json_schema_extra={
            "description": (
                "对齐 AstrBot 的 9 家联网搜索服务商选择、超时与多密钥轮询配置。"
            )
        },
    )

    enabled: bool = Field(
        default=False,
        title="启用联网搜索",
        description="开启后将为对话模型注册当前所选搜索服务商的联网搜索与网页抓取工具。",
    )
    provider: WebSearchProvider = Field(
        default="tavily",
        title="搜索服务商",
        description=(
            "当前启用的联网搜索后端，支持 Tavily、BoCha、Brave、Firecrawl、"
            "百度 AI 搜索、Exa、AnySearch、DeepSeek 与 SerpApi。"
        ),
    )
    timeout_sec: float = Field(
        default=30.0,
        gt=0.0,
        allow_inf_nan=False,
        title="搜索请求超时（秒）",
        description="联网搜索与网页抓取 HTTP 请求的超时时间（秒）。",
    )
    tavily_keys: list[str] = Field(
        default_factory=list,
        title="Tavily API 密钥列表",
        description="支持多密钥轮询与自动故障转移的 Tavily API Key 列表。",
    )
    bocha_keys: list[str] = Field(
        default_factory=list,
        title="BoCha 博查 API 密钥列表",
        description="支持多密钥轮询与自动故障转移的博查（BoCha）API Key 列表。",
    )
    brave_keys: list[str] = Field(
        default_factory=list,
        title="Brave Search API 密钥列表",
        description="支持轮询使用的 Brave Search 订阅令牌列表。",
    )
    firecrawl_keys: list[str] = Field(
        default_factory=list,
        title="Firecrawl API 密钥列表",
        description="支持多密钥轮询与自动故障转移的 Firecrawl API Key 列表。",
    )
    baidu_app_builder_key: str = Field(
        default="",
        title="百度千帆 AppBuilder 密钥",
        description="用于百度 AI 搜索的百度千帆 AppBuilder API Key。",
    )
    exa_keys: list[str] = Field(
        default_factory=list,
        title="Exa API 密钥列表",
        description="支持多密钥轮询与自动故障转移的 Exa API Key 列表。",
    )
    anysearch_keys: list[str] = Field(
        default_factory=list,
        title="AnySearch API 密钥列表",
        description="支持多密钥轮询的 AnySearch API Key 列表；留空则使用匿名模式。",
    )
    deepseek_keys: list[str] = Field(
        default_factory=list,
        title="DeepSeek API 密钥列表",
        description="用于 DeepSeek 搜索与网页抓取的轮询 API Key 列表。",
    )
    serpapi_keys: list[str] = Field(
        default_factory=list,
        title="SerpApi API 密钥列表",
        description="用于 Google 搜索引擎族的轮询 SerpApi API Key 列表。",
    )


class WebUIConfig(BaseModel):
    """Schema-driven configuration WebUI HTTP server settings."""

    model_config = ConfigDict(
        extra="forbid",
        title="配置台 WebUI 配置",
        json_schema_extra={
            "description": "本地 Schema 驱动配置台 HTTP 服务与访问令牌设置。"
        },
    )

    enabled: bool = Field(
        default=True,
        title="启用配置台 WebUI",
        description="启动 Spectra-bot 时是否同时运行本地 WebUI 配置界面。",
    )
    host: str = Field(
        default="127.0.0.1",
        title="WebUI 监听地址",
        description=(
            "配置台 HTTP 服务绑定的监听主机地址"
            "（默认仅监听 127.0.0.1 本地回环以确保安全）。"
        ),
    )
    port: int = Field(
        default=6200,
        ge=1,
        le=65535,
        title="WebUI 监听端口",
        description="配置台 HTTP 服务监听的 TCP 端口号（1~65535）。",
    )
    auth_token: str = Field(
        default="",
        title="WebUI 访问令牌",
        description=(
            "可选的 Bearer 访问令牌，用于保护全部 /api/config* 接口；"
            "监听非本地地址时强烈建议设置。"
        ),
    )


class AppConfig(BaseModel):
    """Root application configuration for Spectra-bot."""

    model_config = ConfigDict(
        extra="forbid",
        title="Spectra-bot 全局配置",
        json_schema_extra={"description": "Spectra-bot 根级应用配置。"},
    )

    db: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        title="数据库",
        description="本地 SQLite 数据库路径与聊天消息保留策略。",
    )
    access: AccessConfig = Field(
        default_factory=AccessConfig,
        title="访问控制",
        description="群聊/私聊黑白名单、管理员与全局忽略用户规则。",
    )
    napcat: NapCatConfig = Field(
        default_factory=NapCatConfig,
        title="NapCat 适配",
        description="OneBot v11 反向 WebSocket 连接与出站回复策略。",
    )
    vision: VisionConfig = Field(
        default_factory=VisionConfig,
        title="视觉识图",
        description="图片下载预处理与 OpenAI 兼容视觉识图设置。",
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        title="大语言模型",
        description="OpenAI 兼容对话模型、重试、降级与工具循环配置。",
    )
    jev: JevConfig = Field(
        default_factory=JevConfig,
        title="Jev 决策路由",
        description="System One / Jev 决策模型端点、阈值与超时配置。",
    )
    gate: GateConfig = Field(
        default_factory=GateConfig,
        title="进门门控",
        description="消息防抖合并、群聊主动回复限流与概率回退配置。",
    )
    session: SessionConfig = Field(
        default_factory=SessionConfig,
        title="会话与上下文",
        description="本地人设目录、滚动上下文窗口与显示时区设置。",
    )
    trace: TraceConfig = Field(
        default_factory=TraceConfig,
        title="追踪日志",
        description="门控决策与模型执行的结构化 JSONL 追踪日志配置。",
    )
    web_search: WebSearchConfig = Field(
        default_factory=WebSearchConfig,
        title="联网搜索",
        description="对齐 AstrBot 的 9 家联网搜索后端选择、超时与密钥轮询配置。",
    )
    webui: WebUIConfig = Field(
        default_factory=WebUIConfig,
        title="配置台 WebUI",
        description="本地 Schema 驱动配置台服务端口与访问鉴权配置。",
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
        allow_nan=False,
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
