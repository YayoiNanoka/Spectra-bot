"""Spectra-bot core package."""

from spectra.bot import SpectraBot
from spectra.config import (
    AccessConfig,
    ActiveReplyConfig,
    AppConfig,
    DatabaseConfig,
    GateConfig,
    JevConfig,
    LLMConfig,
    NapCatConfig,
    SegmentedReplyConfig,
    SessionConfig,
    TraceConfig,
    VisionConfig,
    WebSearchConfig,
    WebUIConfig,
    load_config,
    save_config,
)
from spectra.db import Database, ExecuteResult, Transaction
from spectra.gate import EntryGate, GateDecision
from spectra.jev import EffortLevel, JevClient, JevDecision, resolve_jev_endpoint
from spectra.llm import (
    AgentResult,
    EmptyLLMResponseError,
    LLMClient,
    LLMError,
    LLMResponse,
    MaxToolStepsExceededError,
    TokenUsage,
    ToolCall,
    ToolSpec,
    sanitize_messages,
)
from spectra.messages import MessageRecord, MessageStore
from spectra.napcat import InboundMessage, NapCatAdapter
from spectra.prompts import PromptLoader
from spectra.providers import (
    DEFAULT_USER_AGENT,
    DecisionEndpointProfile,
    DecisionProtocol,
    OpenAICompatProfile,
    build_decision_payload,
    build_provider_headers,
    normalize_decision_answers,
    resolve_decision_endpoint,
    resolve_decision_profile,
    resolve_openai_compat,
    should_omit_empty_tool_parameters,
)
from spectra.sessions import SessionRecord, SessionStore
from spectra.trace import TraceEntry, TraceLogger
from spectra.vision import ImageCaptioner
from spectra.web_search import WebSearchToolkit
from spectra.webui import WebUIServer

__all__ = [
    "AccessConfig",
    "ActiveReplyConfig",
    "AgentResult",
    "AppConfig",
    "DEFAULT_USER_AGENT",
    "Database",
    "DatabaseConfig",
    "DecisionEndpointProfile",
    "DecisionProtocol",
    "EffortLevel",
    "EmptyLLMResponseError",
    "EntryGate",
    "ExecuteResult",
    "GateConfig",
    "GateDecision",
    "ImageCaptioner",
    "InboundMessage",
    "JevClient",
    "JevConfig",
    "JevDecision",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LLMResponse",
    "MaxToolStepsExceededError",
    "MessageRecord",
    "MessageStore",
    "NapCatAdapter",
    "NapCatConfig",
    "OpenAICompatProfile",
    "PromptLoader",
    "SegmentedReplyConfig",
    "SessionConfig",
    "SessionRecord",
    "SessionStore",
    "SpectraBot",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
    "TraceConfig",
    "TraceEntry",
    "TraceLogger",
    "Transaction",
    "VisionConfig",
    "WebSearchConfig",
    "WebSearchToolkit",
    "WebUIConfig",
    "WebUIServer",
    "build_decision_payload",
    "build_provider_headers",
    "load_config",
    "normalize_decision_answers",
    "resolve_decision_endpoint",
    "resolve_decision_profile",
    "resolve_jev_endpoint",
    "resolve_openai_compat",
    "sanitize_messages",
    "save_config",
    "should_omit_empty_tool_parameters",
]
