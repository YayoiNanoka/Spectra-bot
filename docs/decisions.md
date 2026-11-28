# Spectra-bot 设计与流程敲定记录

> 最后更新：2026-11-24
> 背景与调研记录见 [requirements.md](requirements.md)

## 全局已敲定项

- **项目与包命名**：项目/仓库名为 `Spectra-bot`（本地路径 `/Users/yayoinanoka/projects/develop/Spectra-bot`），Python 源码包名为 `spectra`。
- **架构独立（脱离 AstrBot）**：不再基于 AstrBot 插件开发，直接构建对接 **NapCat（OneBot v11）** 的独立轻量 Python 服务，彻底消除第三方框架流水线与多插件冲突包袱。
- **核心原则**：方案与代码严格遵循**第一性原理**与**奥卡姆剃刀原理**——从实际问题出发，砍掉非必要抽象、冗余表结构与超前预留代码。
- **AstrBot 对齐原则**：凡 Spectra-bot 所需功能与 AstrBot 一致时，直接复用 AstrBot 的成熟实现、固定常量与配置结构；仅在需要偏离 AstrBot 时单独讨论与记录。
- **PR 纵切原则（Vertical Slicing）**：所有开发 PR 按纵切拆分，每个 PR 仅交付单一职责的通用底座或一个端到端可独立验证的业务切片；通用底座严禁夹带未来业务表结构或超前预留逻辑。
- **模型协议与配置零隐式继承规则（MVP）**：第一版所有 Chat、视觉识图及 Embedding 模型统一采用 **OpenAI 兼容 API**；Jev 进门决策路由采用独立的 **System One / OpenAI Decisions 决策 API**（默认走 OpenRouter Alpha Decisions `https://openrouter.ai/api/alpha/decisions` + `typesafe/jev-1.14`，支持 `auto` / `system_one` / `openai_decision` 双协议）。每个模型配置段（`vision`、`llm`、`jev`）必须**显式独立声明**各自的 `base_url`、`api_key`、`model`、`timeout_sec`、`custom_headers` 等字段，**严禁跨配置段做隐式继承或 magic fallback**（例如绝不从 `llm.api_key` 隐式推导 `jev.api_key`）。
- **出站分段回复策略**：出站分段逻辑与配置结构直接对齐 AstrBot 的 `segmented_reply`（嵌套于 `napcat.segmented_reply`，含 `enable`（Spectra-bot 默认 `True` 且可配置）、`interval_method`（`random` / `log`）、`interval`（默认 `"1.5,3.5"`）、`log_base`（默认 `2.6`）、`words_count_threshold`（默认 `150`）、`split_mode`（`regex` / `words`）、`regex`、`split_words`、`content_cleanup_rule`），超长回复（默认 `> 1500`）转合并转发；无论出站如何分段或转合并转发，**持久化至 `messages` 表时始终只写 1 条合并后的 `assistant` 记录**（私聊的 `user_id` 固定写对方 QQ 号，确保按 `user_id` 拉取私聊记录时双边消息完整交错）。
- **统一配置与访问控制默认值**：采用单一本地 `config.json`（由 `spectra/config.py` 中的 Pydantic v2 `AppConfig` 管理，字段自带 `title`/`description`，可直接输出 `model_json_schema()` 驱动后续 WebUI 配置页；仓库提交 `config.example.json` 并将 `config.json` 加入 `.gitignore`）；`group_mode` 与 `private_mode` 默认均为 **`blacklist`**（默认屏蔽 QQ 管家 `2854196310` 与 Bot 自身），按需切换为 `whitelist`。
- **图片与动图识图策略**：`spectra/vision.py` 直接沿用 AstrBot `media_utils.py` 的预处理实现与固定常量——输入上限 `MODEL_IMAGE_MAX_INPUT_BYTES = 64 MiB`（流式下载强制截断）、模型提交字节上限 `MODEL_IMAGE_MAX_BYTES = 512 KiB`（循环等比缩放压缩）、可配置最长边 `vision.max_size = 1280`（符合条件的静态 PNG/JPEG 原样直通）；多帧动图（GIF/APNG/动态 WebP）均匀抽取最多 9 帧拼装为白底 **3×3 JPEG 网格图**（跳过 APNG 独立封面帧，不放大），依赖 Pillow 内置默认限制；按原始字节 `sha256` 写入独立 `image_cache` 表缓存并做轻量的进程内并发去重。
- **目标平台**：仅挂载于 **QQ（NapCat / OneBot v11）**，直接以 QQ 号 / 群号作为标识。
- **存储与检索底座**：共用单个本地 SQLite 数据库 `bot.db`（WAL 模式 + `SQLite FTS5` 全文检索 + `sqlite-vec` 向量检索），Trace 日志写入结构化 `traces.jsonl`。
- **离线处理形态**：上下文压缩、记忆提取、自进化等后台任务采用**固定 Workflow（确定性流水线）**编排，不用自由循环的 Agent。
- **自进化边界**：核心基础人设保持固定（由本地 `prompts/*.md` 管理），自进化仅更新“群聊画像”与“用户画像（含对该用户的印象与相处方式）”。
- **第一版范围收敛**：
  - 暂缓 **日记系统**（连带暂不需要跨会话的主 Agent）；
  - 暂缓 **`no_reply` 主动不回复工具**（第一版不实现）；
  - 暂缓 **群聊无语义水屏动态规则库**；
  - **好感度系统**留作后期需求。

---

## 第一版需自行实现的 5 个核心底座

1. **NapCat（OneBot v11）收发与富消息处理**：WebSocket 收发与断线重连；入站解析 `@`、调 `get_msg` 还原引用原话、展开合并转发、提取卡片标题、下载时效图片送识图；出站支持引用回复、分段打字发送、长文转合并转发。
2. **安全门卫**：群白名单 / 私聊白名单、管理员 QQ 识别、单群限流、忽略其他 Bot 账号防死循环对刷。
3. **模型调用与容错封装**：统一封装 Chat / 视觉识图 / Embedding 调用，支持超时控制、429 指数退避重试、主备降级，以及单次请求直传 `reasoning_effort`（`default` / `high`）。
4. **Agent Tool-Loop**：工具 Schema 注册、解析 `tool_calls` 执行本地函数、回填循环及 `max_steps` 上限保护。
5. **本地 SQLite + 检索封装**：`bot.db` 连接管理、`FTS5` 与 `sqlite-vec` 封装。

---

## 已敲定业务模块设计

### 0. 通用 SQLite 底座（`spectra/db.py`）

- **零业务表边界**：`spectra/db.py` 仅作为通用异步 SQLite 底座，内置唯一内部表 `_migrations(module TEXT PRIMARY KEY, version INTEGER NOT NULL)`；严禁在底座中硬编码任何业务表（`messages`、`sessions`、`image_cache` 等均由各业务模块通过 `db.migrate(...)` 独立注册与演进）。
- **连接与并发策略**：基于 `aiosqlite` 且强制 `isolation_level=None`（显式事务控制，避免隐式提交破坏 DDL 事务原子性）、`row_factory = aiosqlite.Row`；连接时统一配置 `PRAGMA journal_mode=WAL`、`busy_timeout=30000`、`synchronous=NORMAL`、`foreign_keys=ON`、`temp_store=MEMORY`。内部采用轻量读写门控（读操作之间无锁并发，写/事务/迁移/关闭操作独占连接），杜绝单连接下的脏读、游标重置竞争与关闭竞态；通过 `ContextVar` 检测并拒绝同上下文（含 `wait_for`/`TaskGroup` 子任务）在事务内重入 `Database` 读写/写锁接口，主动抛出 `RuntimeError` 防死锁。
- **6 个公开接口契约**：
  1. `await connect()` / `await close()`（幂等关闭 + `async with Database(...) as db:` 上下文管理，初始化中途失败自动关闭连接防线程泄漏）
  2. `await fetchone(sql, params=()) -> aiosqlite.Row | None`
  3. `await fetchall(sql, params=()) -> list[aiosqlite.Row]`
  4. `await execute(sql, params=()) -> ExecuteResult`（写锁保护，返回 `ExecuteResult(lastrowid, rowcount)`）
  5. `async with db.transaction() as tx:`（持写锁执行 `BEGIN IMMEDIATE` / `COMMIT` / 任意 `BaseException` 自动 `ROLLBACK`，`tx` 提供 `execute` / `fetchone` / `fetchall`）
  6. `await migrate(module, steps)`（按模块独立版本号增量执行每步 SQL 并与 `_migrations` 更新同事务原子提交；禁止历史 `steps` 回缩，多语句步骤安全切分且绝不调用会隐式提交的 `executescript()`）

### 0.5 通用 LLM 客户端与 Agent Tool-Loop 底座（`spectra/llm.py` & `LLMConfig`）

- **配置项（`LLMConfig` / `VisionConfig`）**：挂于 `AppConfig.llm` 与 `AppConfig.vision`（`extra="forbid"`，全字段带 `title`/`description` 供 WebUI JSON Schema 渲染）；`LLMConfig` 包含 `base_url`、`api_key`、`model`（默认 `"gpt-4o"`）、`fallback_model`（可选主备降级模型）、`default_effort`（默认 `""`，为空时不传 `reasoning_effort` 以兼容各类模型）、`high_effort`（默认 `"high"`）、`timeout_sec`（默认 `60.0`）、`max_retries`（默认 `5`，对齐 AstrBot `REQUEST_RETRY_ATTEMPTS`）、`max_tool_steps`（默认 `10`）、`custom_headers`（可选自定义请求头）与 `extra_body`（可选附加请求体参数）；`VisionConfig` 额外支持 `reasoning_effort`（默认 `""`，如在 `deepseek-v4.1-flash` 上按需传 `"low"`）、`custom_headers` 与 `extra_body`。
- **统一模型与网关兼容层（`spectra/providers.py`）**：遵循第一性原理与奥卡姆剃刀，以单一声明式薄模块统一收口全部网关/协议差异（严禁引入多文件 Provider 子类树）：
  - **请求头与 User-Agent**：默认注入 `User-Agent: spectra-bot/0.1.0`，依次合并网关默认头与用户 `custom_headers`（大小写不敏感覆盖、不产生重复头、忽略空白 `User-Agent` 覆盖，对齐 AstrBot `headers.py`）。
  - **OpenCode Go 兼容**：`opencode.ai` 域名下 `/zen/go/v1` 或 `/zen/go` 自动改写为 `/inference/go/openai/v1`；凡路径含 `/go/` 或 `/go` 段时自动注入会话亲和头 `x-opencode-session`（由 `SpectraBot` 透传 `decision.session_key`，`vision` 默认 `spectra:vision`，`llm` 默认 `spectra:llm`，支持被 `custom_headers` 显式覆盖）。
  - **OpenRouter 兼容**：`openrouter.ai` 域名自动注入 `HTTP-Referer: https://github.com/YayoiNanoka/Spectra-bot` 与 `X-OpenRouter-Title: Spectra-bot`（可被 `custom_headers` 覆盖），并将推理字段名切换为 `reasoning`（对齐 AstrBot `openrouter_source.py`）；`sanitize_messages` 与响应解析双向兼容 `reasoning_content` ↔ `reasoning`。
  - **Gemini 工具 Schema 兼容**：`should_omit_empty_tool_parameters(model)` 统一判断 Gemini 系列模型在工具 `parameters.properties` 为空时省略 `parameters` 字段。
  - **双决策协议兼容（`system_one` / `openai_decision`）**：`resolve_decision_profile(JevConfig)` 按 `protocol`（默认 `"auto"`：`api.openai.com` 或非 OpenRouter 的 `/v1/decisions` 自动走 `openai_decision`，其余走 `system_one`）归一化端点 URL（OpenRouter 下自动落到 `/api/alpha/decisions`）、模型名（OpenRouter 下为 `jev-*` 补全 `typesafe/` 并剥离 `.0` 补丁号如 `typesafe/jev-1.14.0` → `typesafe/jev-1.14`；`api.typesafe.ai` 下剥离 `typesafe/` 前缀）与请求头，并完成 `system_one` 与 `openai_decision`（含 `input_json`、将 `should_reply` 判据并入 `predicate.instructions`、按 `name` 匹配 `answers` 列表与 `refusal` 安全降级）的双向报文翻译。
- **AstrBot 对齐的请求清理、重试与降级**：
  - 禁用 OpenAI SDK 内置重试（`max_retries=0`），统一由 `tenacity.AsyncRetrying` 处理可重试错误（`APIConnectionError`/`APITimeoutError`/`httpx.TimeoutException`/`httpx.TransportError` 及 HTTP 状态码 `{408, 409, 429, 500, 502, 503, 504, 529}` 或 `500..599`），指数退避 `min=0.2s, max=30.0s`，`400`/`401` 等不可重试错误立即失败；主模型重试耗尽或失败且配置了不同的 `fallback_model` 时自动切换备模执行同策略重试。
  - 发送前对消息列表做深拷贝清洗（绝不修改调用方入参）：丢弃无内容/无 `tool_calls`/无推理文本的空 `assistant` 消息；保留仅含推理文本（`reasoning_content` 或 `reasoning`）的 `assistant` 消息并补 `content=""` 且统一按目标网关 `reasoning_key` 输出；含 `tool_calls` 且空内容时置 `content=None`；过滤孤立或重复的 `role="tool"` 消息；针对 Gemini 系列模型在工具 `parameters.properties` 为空时省略 `parameters` 字段。
  - 统一规范 `content`（兼容 `str`/`list[dict]`/`dict`/`None`）、提取推理字段（优先 `reasoning_key` 并回退 `reasoning_content`/`reasoning` 及 `model_extra`）与 `TokenUsage`（含 `prompt_tokens_details.cached_tokens` 拆分并防负值）；`choices` 为空抛出 `EmptyLLMResponseError`。
- **公开接口与 Tool-Loop**：`LLMClient` 支持 `async with` 与幂等 `close()`（仅关闭自有客户端），提供 `complete(messages, *, effort="default"|"high", tools=(), session_key=None) -> LLMResponse` 与 `run_tool_loop(messages, *, tools=(), effort="default"|"high", max_steps=None, session_key=None) -> AgentResult`。工具循环按序执行同步/异步 `ToolSpec.handler`，将未知工具、非 JSON 对象参数及普通 `Exception` 转为可读错误回填模型继续推理，不吞 `BaseException`/`CancelledError`，超出 `max_steps` 抛出 `MaxToolStepsExceededError`。

---

### 1. 事件日志层（`messages` 表）与多 Session 状态（`sessions` 表）

#### `messages` 表（只追加原始消息流，7 列）
- **字段**：
  - `id`：`INTEGER PRIMARY KEY AUTOINCREMENT`（严格递增主键与游标）
  - `group_id`：`TEXT`（群聊填群号，**私聊为 `NULL`**，据此区分群聊与私聊）
  - `user_id`：`TEXT NOT NULL`（QQ 号；群聊为发言人 QQ，私聊固定为对方用户 QQ，由 `role` 区分发言方）
  - `sender_name`：`TEXT NOT NULL`（发言时群名片/昵称快照）
  - `role`：`TEXT NOT NULL`（`'user'` | `'assistant'`）
  - `content`：`TEXT NOT NULL`（压平后的正文，含引用原话及图片视觉描述；图片按哈希缓存描述，重复表情包不重复识图）
  - `created_at`：`INTEGER NOT NULL`（秒级时间戳）
- **索引（3 个）**：
  1. `(group_id, id)`：按群拉取窗口消息
  2. `(user_id, id) WHERE group_id IS NULL`：按私聊拉取窗口消息
  3. `(created_at)`：定期过期清理
- **读接口（2 个）**：
  1. `get_private_messages(user_id, limit, after_id=None)`
  2. `get_group_messages(group_id, limit, after_id=None)`
- **定期清理规则**：每日定时清理超过保留期（如 7 天）**且** `id < min(window_start_id, last_extracted_id)` 的记录，确保未压缩或未提取记忆的消息绝不被误删。

#### `sessions` 表（多 Session 上下文投影游标与摘要，6 列）
- **机制**：参考 Apache Maka 的 Event Log + Projection 思路，单个 Session 内消息只追加不修改（保证前缀稳定命中 Prompt Cache），达到上下文窗口阈值后压缩旧消息更新摘要，并推进 `window_start_id`。
- **字段**：
  - `session_key`：`TEXT PRIMARY KEY`（`group:{群号}` 或 `private:{QQ号}`）
  - `window_start_id`：`INTEGER NOT NULL DEFAULT 0`（当前活跃上下文第一条未压缩消息的 `messages.id`）
  - `summary`：`TEXT NOT NULL DEFAULT ''`（阈值压缩生成的 100~200 字滚动近期前情提要）
  - `last_extracted_id`：`INTEGER NOT NULL DEFAULT 0`（长期记忆提取游标，与压缩游标解耦）
  - `group_impression`：`TEXT NOT NULL DEFAULT ''`（仅群聊用：自进化更新的本群氛围与相处方式；私聊印象直接合并在用户画像中）
  - `updated_at`：`INTEGER NOT NULL`（最后活跃时间，供冷场兜底提取扫描）

---

### 2. 进门门控层（`spectra/jev.py` & `spectra/gate.py`：防抖合并 + 本地预筛 + Jev 读空气与 Effort 路由）

1. **配置模型（`JevConfig`、`ActiveReplyConfig`、`GateConfig`）**：
   - `jev`（`JevConfig`）：独立声明 `enabled=True`、`base_url="https://openrouter.ai/api/alpha/decisions"`（默认直连 OpenRouter Alpha Decisions；自动兼容 OpenRouter / System One `/systemone` / OpenAI Decisions `/decisions`）、`api_key=""`、`model="typesafe/jev-1.14"`、`protocol="auto"`（`"auto"` / `"system_one"` / `"openai_decision"`）、`custom_headers={}`、`timeout_sec=1.5`、`reply_threshold=0.65`、`high_effort_threshold=0.60`；仅当 `enabled` 且 `base_url`/`api_key`/`model` 均非空时 `is_configured` 为 `True`。
   - `gate`（`GateConfig`）：`proactive_mode`（`off` / `shadow` / `on`，默认 `"on"`）、`private_debounce_sec=1.5`、`group_debounce_sec=2.5`、`max_batch_wait_sec=6.0`、`proactive_cooldown_sec=60.0`、`proactive_max_per_hour=15`、`hot_window_sec=30.0`、`state_message_limit=10`，并内嵌对齐 AstrBot 的 `active_reply`（`ActiveReplyConfig`：`method="possibility_reply"`、`possibility_reply=0.1`、`whitelist=[]`）。
2. **会话级防抖合并与忙碌协调**：
   - 消息先由 NapCat 适配器持久化写入 `messages` 表再提交给 `EntryGate.submit(msg)`。
   - 按 `session_key`（`group:{group_id}` / `private:{user_id}`）做防抖合并：首条消息设定 `max_deadline = first_arrival + max_batch_wait_sec`，每条续发更新 `deadline = min(max_deadline, now + debounce_sec)`；批次内任意一条 `@Bot` 或引用回复 Bot 即标记 `batch_mentioned = True`，群聊被 `@` 时 `reply_to_msg_id` 优先取批次内最后一条 `@Bot` 消息。
   - **生成中（`in_flight`）策略**：当前会话正在执行 `on_trigger` 时，群聊未 `@Bot` 且缓冲中无 `@Bot` 消息的普通闲聊**立即丢弃**（`filtered_busy`），绝不排队积压过时闲聊；私聊消息、群聊 `@Bot` 消息及追加到已含 `@Bot` 缓冲的消息正常防抖并在当前生成结束后自动合并触发。
3. **群聊闲聊本地零成本预筛（仅针对未 `@Bot` 的普通群消息，严格按序）**：
   1. `proactive_mode == "off"` → `filtered_proactive_off`；
   2. `active_reply.whitelist` 非空且群号不在白名单内 → `filtered_whitelist`；
   3. 当前会话正在生成回复中 → `filtered_busy`；
   4. 末条消息以 `@其他群友` 开头，或首条以 `@其他群友` 开头且整批均来自同一发言人 → `filtered_at_other`；
   5. 滚动 1 小时内本群主动插话次数达到 `proactive_max_per_hour`（默认 `15`，热聊窗口内同样生效）→ `filtered_hourly_cap`；
   6. 不在 `hot_window_sec`（默认 `30s`，由本群任意成功回复刷新）内且距上次主动插话不足 `proactive_cooldown_sec`（默认 `60s`）→ `filtered_cooldown`。
4. **Jev 单次请求判断、降级与 AstrBot 概率回退**：
   - **输入 `state`**：从 `MessageStore` 拉取该会话严格早于本批 `min(db_id)` 的最近 `state_message_limit`（默认 `10`）条消息拼装 `recent_messages`（`[{bot_name}(Bot)]: ...` / `[{sender_name}]: ...`），连同 `chat_type`、`bot_name` 与 `latest_batch` 一并传入。
   - **已敲定 `Q1` / `Q2` 问题契约**：
     - `should_reply`（`type: "noul"`，仅未 `@Bot` 群聊携带）：结合 `recent_messages` 判断机器人 `bot_name` 自然顺嘴接一句 `latest_batch` 是否合适不突兀；`true` 对应开放话题/求助提问/玩梗吐槽/公共讨论/延续与机器人对话，`false` 对应特定单聊对线/纯表情包刷屏/正经通知或严肃争吵/话未说完。
     - `effort`（`type: "choice"`，固定顺序 `"default"` 在前、`"high"` 在后，所有 Jev 调用必带）：`default` 对应日常闲聊/玩梗接话/情绪回应/简单常识/打招呼，`high` 对应代码编写与报错排查/数学逻辑推理/复杂技术方案分析/**联网搜索查实时信息与事实查证**/多步工具调用分析。
   - **单次直连与安全降级**：`JevClient` 复用持久化 `httpx.AsyncClient`（无重试，硬超时 `timeout_sec=1.5s`）；网络/超时/非 2xx/格式异常绝不抛给上层（仅透传 `asyncio.CancelledError`），降级为 `degraded=True`（群聊闲聊落 `filtered_jev_degraded` 不接话，私聊/被 `@` 回退 `effort="default"`）。
   - **未配置 Jev 时的 AstrBot 回退**：当 `jev.is_configured` 为 `False` 时，私聊/被 `@` 直接使用 `effort="default"`；群聊闲聊按 AstrBot `active_reply.possibility_reply` 概率掷骰决定是否触发（`proactive_astrbot_prob` / `shadow_astrbot_prob` / `filtered_prob_rejected`）。
   - **`shadow` 观测模式与状态更新**：`proactive_mode == "shadow"` 时完整跑完预筛与 Jev/概率评估并通过 `on_decision` 记录结果（`shadow=True`），但不触发 `on_trigger`、不消耗冷却与小时配额；`on_trigger` 返回 `False` 或抛异常时不污染生成锁，亦不刷新热聊/冷却/小时配额计数。
5. **直传 Effort 进入 Agent Loop**：脱离 AstrBot 后，直接把 `GateDecision.effort`（`"default"` / `"high"`）作为单次模型调用的 `reasoning_effort` 参数传入 `LLMClient`。

---

### 3. 基础人设管理、Session 上下文投影、Trace 与 MVP 运行时（`prompts.py` / `sessions.py` / `trace.py` / `bot.py` / `__main__.py`）

- **配置扩展（`SessionConfig` & `TraceConfig`）**：
  - `session`：`prompts_dir`（默认 `prompts`）、`window_max_messages`（默认 `60`，`>= 2`）、`window_keep_messages`（默认 `30`，`>= 1` 且严格小于 `window_max_messages`）、`timezone`（默认 `Asia/Shanghai`，按 `zoneinfo.ZoneInfo` 强校验）。
  - `trace`：`enabled`（默认 `True`）、`path`（默认 `data/traces.jsonl`）。
- **本地 Markdown 人设加载（`spectra/prompts.py` & `prompts/default.md`）**：
  - 按 `group_{group_id}.md`（群聊）或 `private_{user_id}.md`（私聊）→ `default.md` → 内置兜底常量的顺序加载，拒绝包含路径穿越字符或越界符号链接的覆盖文件。
  - 以 `(resolved_path, st_mtime_ns, st_size)` 做 stat 缓存：未修改时不重复读盘，新建/编辑/删除无需重启即时生效。
- **Session 状态与前缀稳定上下文投影（`spectra/sessions.py`）**：
  - 由 `SessionStore` 独立通过 `db.migrate("sessions", ...)` 维护 6 列 `sessions` 表，并提供 `get_or_create` / `touch` / `update_window` / `min_protected_message_id` / `build_context`。
  - **块滚动窗口（Block Rollover）**：`build_context` 使用带索引的 `O(window_max_messages)` 倒序限条查询拉取 `id >= window_start_id [AND id <= up_to_id]`；仅当活跃条数超过 `window_max_messages`（如 60）时一次性回滚保留最近 `window_keep_messages`（如 30）条并跳过开头 `assistant` 行推进 `window_start_id`，保证两次滚动之间前缀完全固定以命中 Prompt Cache。
  - **系统消息与严格交替角色拼装**：
    - `system` 消息按序拼装：`base_prompt` →（仅群聊非空时）`## 当前群聊氛围与印象` →（非空时）`## 近期对话前情摘要`。
    - 丢弃窗口开头孤立的 `assistant` 行，群聊 `user` 格式化为 `[昵称(QQ号) HH:MM]: 内容`（按配置时区格式化），私聊与 `assistant` 保持原文；相邻同角色行以 `\n` 合并，输出 `system` 后严格交替的 `user` / `assistant` 序列（兼容 Claude / Gemini / DeepSeek）。
    - （Post-MVP 扩展位：用户画像与字面梗释义将在双长期记忆库阶段按需挂载于上下文末尾。）
- **结构化 Trace 日志（`spectra/trace.py`）**：
  - 遵循 **1 次评估批次 = 1 行紧凑 JSONL**（`ensure_ascii=False`）：被过滤或 `shadow` 决策在 `on_decision` 立即落盘；触发回复的批次在 `on_trigger` 结束后合并门控/Jev 元数据与 LLM 执行遥测（`llm_model`、`steps`、`tool_calls`、Token 统计、`reply_chars`、`reply_sent`、`llm_latency_ms`、`total_latency_ms`、`error`）写 1 行。
  - 通过 `asyncio.Lock` + `asyncio.to_thread` 原子追加并自动建目录，吞掉 `OSError` 打警告不中断聊天。
- **主编排器与 CLI（`spectra/bot.py` & `spectra/__main__.py`）**：
  - `SpectraBot` 串联 `Database` → `MessageStore` → `SessionStore` → `ImageCaptioner`（按 `vision.enabled`）→ `PromptLoader` / `TraceLogger` → `JevClient` / `LLMClient` → `EntryGate` → `NapCatAdapter`，支持全量依赖注入供离线全链路 E2E 测试，启动中途失败自动回滚释放已打开资源，关闭按严格逆序幂等释放。
  - CLI 入口提供 `spectra` 与 `python -m spectra`，支持 `--config`、`--init-config`（不覆盖已有文件）、`--webui-only`（仅启动配置 WebUI，无需连接 NapCat）、`--log-level` 及 `SIGINT`/`SIGTERM` 优雅停机。

---

### 4. AstrBot 对齐联网搜索工具集（`spectra/web_search.py` & `WebSearchConfig`）

- **配置模型（`WebSearchConfig`）**：挂于 `AppConfig.web_search`（`extra="forbid"`，全字段带 `title`/`description` 供 WebUI JSON Schema 渲染），默认 `enabled=False`、`provider="tavily"`、`timeout_sec=30.0`，包含全部 9 家搜索后端的独立 Key 配置（`tavily_keys`、`bocha_keys`、`brave_keys`、`firecrawl_keys`、`exa_keys`、`anysearch_keys`、`deepseek_keys`、`serpapi_keys` 列表及单 Key `baidu_app_builder_key`）。
- **1:1 后端与工具对齐**：`WebSearchToolkit` 复用单一持久化 `httpx.AsyncClient`（支持外部注入与幂等 `close()` / `async with`，不引入 `aiohttp`），`.build_tools()` 仅在 `enabled=True` 时注册当前所选 `provider` 的工具（共覆盖 9 家后端、17 个工具：`tavily` 2 / `bocha` 1 / `brave` 1 / `firecrawl` 2 / `baidu_ai_search` 1 / `exa` 2 / `anysearch` 1 / `deepseek` 2 / `serpapi` 5）。
- **并发安全 Key 轮询与容错**：内置 `asyncio.Lock` 保护的轮询器 `_KeyRotator`，剥离空白 Key 并在单次请求内维护 `tried_keys` 按 AstrBot 精确状态码故障转移（Tavily `{401,403,429,432,433}`、BoCha `{401,403,429}`、Brave `{401,403,429}` 及 `422 SUBSCRIPTION_TOKEN_INVALID`、Firecrawl `{401,402,403,429}`、Exa/DeepSeek `{401,402,429}`、AnySearch `{401,402,403,429}` 且支持空 Key 匿名模式、SerpApi `{401,403,429}`），对 `5xx` 快速失败；SerpApi 对全部返回错误、日志及 `httpx` 传输异常强制脱敏 `api_key` 并阻断异常链泄漏。
- **结果格式、输入校验与运行时接入**：统一输出 AstrBot JSON 结构 `{"results": [{"title", "url", "snippet", "index"}]}`（4 位 `ref_uuid.idx`），**不注入 WebChat `<ref>` 系统提示词**（QQ 无引用渲染器）；所有网页抓取/提取工具强校验 `http`/`https` URL（非空主机、无内嵌凭据、无空白字符），畸形工具参数返回清晰错误文本不抛未处理异常。`SpectraBot` 支持注入外部 `web_search` 或按 `config.web_search.enabled` 自动创建并纳入逆序停机/启动回滚释放，合并工具时调用方显式传入的 `tools` 优先去重。

---

### 5. Schema 驱动配置 WebUI v1（`spectra/webui.py`、`spectra/static/` 与 `WebUIConfig`）

- **配置模型（`WebUIConfig`）**：挂于 `AppConfig.webui`（`extra="forbid"`，全字段带 `title`/`description`）：`enabled=True`、`host="127.0.0.1"`（默认仅绑定本地回环接口确保安全）、`port=6200`（`1..65535`）、`auth_token=""`（非空时保护全部 `/api/config*` 接口；若绑定非回环地址且未设 `auth_token`，启动时打印高优先级安全警告）。
- **后端服务（`spectra/webui.py`）**：直接复用已有依赖栈 `quart` + `hypercorn`（零额外重依赖），通过 `WebUIServer` 封装 Quart ASGI 应用与 `hypercorn.asyncio.serve` 生命周期（支持优雅停机、失败回滚与 `port=0` 临时端口测试自省）；静态资源经白名单 + `importlib.resources` 安全加载，杜绝路径穿越。
- **接口与安全契约**：
  - `GET /` 与 `GET /static/<asset>`：提供零构建原生 HTML/CSS/ES-Module Config Studio 前端。
  - `GET /api/meta`：公开元信息 `{version, config_path, auth_required, bot_running}`，严禁泄露任何配置值或密钥。
  - `GET /api/config/schema`、`GET /api/config/defaults`、`GET /api/config`、`POST /api/config/validate`、`PUT /api/config`：当配置了 `auth_token` 时使用 `hmac.compare_digest` 常量时间校验 `Authorization: Bearer <token>` 或 `X-Spectra-Token`；全部 `/api/*` 响应强制 `Cache-Control: no-store`、`X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer`；`POST`/`PUT` 强制 `application/json` 与 1 MiB 载荷上限，校验错误不回显原始敏感输入；`PUT /api/config` 经 `AppConfig.model_validate` 校验后原子落盘（`save_config`）并返回 `restart_required: true`；当磁盘 `config.json` 损坏或校验失败时，`GET /api/config` 返回结构化 `422` 错误与原始文本，支持用户在 Raw JSON 视图中直接修复或一键载入默认配置。
- **Schema 驱动零构建前端（`spectra/static/webui.{html,css,js}`）**：完全根据 `AppConfig.model_json_schema()` 动态生成侧边栏分区与表单控件（未来新增 `web_search` / `memory` 等配置段无需改动前端），支持 `$defs`/`$ref`、`anyOf` 可空类型、`enum`/`const`、数值区间提示、敏感字段密码遮罩与显隐切换（含 `*_keys` 列表）、字符串数组与键值/JSON 字典编辑、嵌套子模型分组、全局搜索过滤、Pydantic `loc` 字段级错误映射、与表单双向同步的 Raw JSON 编辑器、暗亮主题切换、`Ctrl/Cmd+S` 保存快捷键及 `sessionStorage` Token 登录弹窗。
- **运行时与 CLI 集成**：`spectra/__main__.py` 在正常启动时按 `config.webui.enabled` 与 `SpectraBot` 并行托管 `WebUIServer`（共享 `SIGINT`/`SIGTERM` 停机与异常回滚，避免污染 `SpectraBot` 离线单测端口）；新增 `--webui-only` 独立模式，支持在无 `config.json` 或配置损坏时安全启动修复（尽量保留已有有效 `webui` 配置段及 `auth_token`）。

---

### 6. 向量嵌入客户端与纯标准库向量编解码底座（PR #10：`spectra/embedding.py`、`EmbeddingConfig` 与 `spectra/providers.py`）

- **配置模型（`EmbeddingConfig`）**：独立挂于 `AppConfig.embedding`（零跨段隐式继承、`extra="forbid"`、全字段 100% 中文 Schema 元数据），默认 `base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"`、`api_key=""`、`model=""`（留空默认关闭）、`dimensions=None`、`dimensions_mode="auto"`（`auto` / `always` / `never`）、`timeout_sec=15.0`、`max_retries=2`（`ge=0`，`0` 表示不重试共 1 次尝试）、`batch_size=10`（对齐阿里云百炼 `text-embedding-v3`/`v4` 单批上限）、`custom_headers={}`、`extra_body={}`；`is_configured` 仅要求 `base_url` 与 `model` 非空，支持本地无鉴权 Ollama / vLLM。
- **网关归一化与维度策略（`spectra/providers.py`）**：
  - `resolve_embedding_endpoint(base_url)` 幂等剥离尾部 `/` 与 `/embeddings` 后统一处理网关改写：OpenCode Go `/zen/go` / `/zen/go/v1` → `/inference/go/openai/v1/embeddings`、OpenRouter 裸域名 → `/api/v1/embeddings`、Ollama 原生 `.../api/embed` → `.../v1/embeddings`；对合法 HTTP(S) 路径按正则 `^v\d+(?:[a-z]+\d*|\.\d+)*$` 检测版本段，存在版本段（如 `/compatible-mode/v1`、`/v1beta/openai`）时直接追加 `/embeddings`，否则补全 `/v1/embeddings`。
  - `resolve_embedding_profile(config)` 统一组装网关头（OpenRouter `HTTP-Referer`/`X-OpenRouter-Title`、OpenCode Go `x-opencode-session: spectra:embedding`）、仅在 `api_key` 非空时注入 `Authorization: Bearer ...`、合并 `custom_headers` 并深拷贝 `extra_body`。
  - `should_send_embedding_dimensions(model, mode)` 在 `"auto"` 模式下剥离厂商前缀后按已知支持自定义维度的模型前缀（`text-embedding-3`、`text-embedding-v3`、`text-embedding-v4`、`qwen3-embedding`、`jina-embeddings-v3`、`embed-v4`）判定。
- **零新增依赖纯标准库向量编解码与异步客户端（`spectra/embedding.py`）**：
  - `pack_normalized_vector` / `unpack_vector` / `cosine_similarity` / `format_embedding_model_tag` 完全基于 Python 标准库 `struct`（按维度 LRU 缓存）与 `math.hypot` / `math.sumprod` 实现，L2 归一化后以小端 `float32` 紧凑存储，严格拒绝空向量、`bool`、非数值、`NaN`/`Inf`、近零范数与错位长度。
  - `EmbeddingClient` 支持惰性自有 `httpx.AsyncClient`、外部客户端注入、幂等 `close()` 与 `async with`；通过 `tenacity.AsyncRetrying` 对 `{408, 409, 429}`、`>=500` 及传输/超时错误执行带抖动指数退避重试（`max_retries + 1` 次尝试），对其他 `4xx` 快速失败，并全程脱敏 `api_key`（内部异常使用 `from None` 阻断链式泄漏）。
  - `embed_texts` 保持 1:1 输入索引对齐（空白文本不发请求直接映射 `None`）、按 `batch_size` 分片、按响应 `data[].index` 排序并强校验索引完整唯一与批内维度一致、单批失败安全降级不影响其他批次且始终透传 `asyncio.CancelledError`。

---

### 7. 长期记忆库、物理隐私隔离与 Jev 记忆精排（PR #11：`spectra/memory.py`、`MemoryConfig` 与 `spectra/jev.py`）

- **配置模型（`MemoryConfig`）**：挂于 `AppConfig.memory`（`extra="forbid"`，全字段 100% 简体中文 Schema 元数据），包含 `enabled=True`、`vector_top_k=12`、`min_vector_similarity=0.25`、`rerank_threshold=0.50`、`max_recall_items=5`、`jargon_min_confidence=0.60` 与 `jargon_max_matches=8`。
- **Jev 单次批量记忆精排（`spectra/jev.py`）**：抽离共用决策请求与 `noul`/`predicate` 概率解析内部辅助函数（保持 `evaluate()` 100% 向后兼容），新增 `JevRerankResult` 与 `JevClient.rerank_memories(query, candidates, threshold=0.50)`——单次请求发送 `state={"query", "candidates"}` 与 `m0..m{N-1}` 布尔相关度问题，同时支持 `system_one` 与 `openai_decision` 双协议；空查询/空候选零 HTTP 调用直接返回非降级空结果，异常或任一 `m{i}` 缺失/畸形均安全降级为 `degraded=True`，且始终透传 `asyncio.CancelledError`。
- **4 张精简 SQLite 表与零事务内网络 I/O（`spectra/memory.py`）**：通过 `db.migrate("memory", ...)` 注册 4 张 v1 业务表（无 FTS5、无 `pinned`）：
  1. `group_profiles`（群聊画像：相处风格、常见话题、禁忌边界、成员备注）与局部补丁/中文 Markdown 渲染；
  2. `user_profiles`（用户画像：JSON 去重 `aliases` 数组、相处风格、话题偏好、关系定位、近期关注、私聊专属 `private_notes`），通过物理不含 `private_notes` 字段的 `PublicUserProfile` 与显式排除 `private_notes` 列的 SQL `SELECT` 保证 `render_public_user_hint` 物理无法接触私聊备注；
  3. `memory_items`（`fact`/`preference`/`event`/`promise` 与 `public`/`group_only`/`private` 可见性，带完整 `CHECK` 约束与活跃局部索引），提供 `add_memory`、`reinforce_memory`、原子事务 `supersede_memory` 与软删除 `delete_memory`，所有嵌入计算与 Jev 精排均严格在 SQLite 事务外执行；
  4. `group_jargon`（分群大小写不敏感 `COLLATE NOCASE` 唯一约束的黑话/梗表），`match_jargon_in_text` 采用 ASCII 边界前后瞻（不使用 `\b`），确保 `"今晚go吗"` 准确命中 `"go"` 而 `"good"` 与 `"C++"`（针对 `"C"`）不误判。
- **SQL 级隐私锁与三阶段检索（`MemoryStore.search_memories`）**：
  - 群聊仅拉取本群 `group_only`/`public` 与活跃发言人 `public` 活跃记忆，私聊仅拉取该用户 `private`/`public` 活跃记忆，物理杜绝私聊记忆泄露进群聊、跨群 `group_only` 越权或他人私聊记忆串线；
  - 第一阶段对模型标签与维度匹配的向量计算余弦相似度，对无向量/模型失配或未配置嵌入的场景自动回退至大小写不敏感词法匹配（ASCII 词元 + 2 字 CJK 二元组 + 完整子串）；第二阶段调用 `JevClient.rerank_memories` 精排并在降级时无缝回退第一阶段排序；最终截断并回写 `last_accessed_at`。

---

## 待敲定模块进度

- [x] **项目架构独立（脱离 AstrBot，直连 NapCat）**
- [x] **通用 SQLite 底座（`spectra/db.py` 6 接口契约，零业务表）**
- [x] **统一配置 + 消息存储 + 视觉识图缓存 + NapCat 收发适配（`config.py` / `messages.py` / `vision.py` / `napcat.py`）**
- [x] **通用 OpenAI 兼容 LLM 客户端与 Agent Tool-Loop 底座（`spectra/llm.py` & `LLMConfig`）**
- [x] **事件日志层（`messages`）+ 多 Session 状态表（`sessions`）**
- [x] **进门门控层（防抖合并 + 本地预筛 + Jev 读空气 + Effort 路由）**
- [x] **最小可运行 MVP 运行时（`prompts` + `sessions` 上下文投影 + `trace` + `SpectraBot` + CLI）**
- [x] **统一模型与网关兼容层（`spectra/providers.py`：OpenCode Go / OpenRouter / Gemini / System One & OpenAI Decisions 双协议）**
- [x] **AstrBot 对齐联网搜索工具集（`spectra/web_search.py` & `WebSearchConfig`：9 后端 17 工具 + Key 轮询故障转移 + SerpApi 脱敏）**
- [x] **Schema 驱动配置 WebUI v1（`spectra/webui.py` + `spectra/static/` + `WebUIConfig` + `--webui-only`）**
- [x] **向量嵌入底座（PR #10：`EmbeddingConfig` + `providers.py` 端点归一化 + `spectra/embedding.py` 纯标准库向量编解码与异步客户端）**
- [x] **长期记忆存储底座与 Jev 精排（PR #11：`MemoryConfig` + `spectra/memory.py` 4 表/SQL 隐私锁/群黑话 + `JevClient.rerank_memories`）**
- [ ] **记忆上下文注入与按需检索工具接入（`sessions.py` / `bot.py`）**
- [ ] **后台压缩、记忆提取归属与自进化 Workflow**
