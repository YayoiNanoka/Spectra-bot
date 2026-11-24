# Spectra-bot 设计与流程敲定记录

> 最后更新：2026-11-23
> 背景与调研记录见 [requirements.md](requirements.md)

## 全局已敲定项

- **项目与包命名**：项目/仓库名为 `Spectra-bot`（本地路径 `/Users/yayoinanoka/projects/develop/Spectra-bot`），Python 源码包名为 `spectra`。
- **架构独立（脱离 AstrBot）**：不再基于 AstrBot 插件开发，直接构建对接 **NapCat（OneBot v11）** 的独立轻量 Python 服务，彻底消除第三方框架流水线与多插件冲突包袱。
- **核心原则**：方案与代码严格遵循**第一性原理**与**奥卡姆剃刀原理**——从实际问题出发，砍掉非必要抽象、冗余表结构与超前预留代码。
- **AstrBot 对齐原则**：凡 Spectra-bot 所需功能与 AstrBot 一致时，直接复用 AstrBot 的成熟实现、固定常量与配置结构；仅在需要偏离 AstrBot 时单独讨论与记录。
- **PR 纵切原则（Vertical Slicing）**：所有开发 PR 按纵切拆分，每个 PR 仅交付单一职责的通用底座或一个端到端可独立验证的业务切片；通用底座严禁夹带未来业务表结构或超前预留逻辑。
- **模型协议（MVP）**：第一版所有 Chat、视觉识图及 Embedding 模型统一采用 **OpenAI 兼容 API**。
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

### 2. 进门门控层（防抖合并 + 本地预筛 + Jev 读空气与 Effort 路由）

1. **安全白名单与防刷拦截**：非白名单群/私聊或来自其他 Bot 账号的消息直接忽略。
2. **会话级防抖合并**：
   - 消息先写入 `messages` 表（图片在防抖等待窗口内并行完成识图与哈希缓存）。
   - 私聊防抖约 `1.5s`、群聊约 `2.5s`（单批累计最大等待上限 `6s` 防饿死），合并同一批次连发消息后再统一进入判断；批次内任意一条 `@Bot` 或引用回复 Bot 即标记 `batch_mentioned = True`。
3. **群聊闲聊本地预筛（仅针对未 `@Bot` 的普通群消息）**：
   - 命中以下 **3 条**任一即直接结束：
     1. 当前群正在生成回复中（防并发抢话）；
     2. 处于主动插话冷却期或超出频率上限（*例外：Bot 刚在群内回复后的 `30s` 热聊窗口内放行判断*）；
     3. 消息以 `@其他群友` 开头。
4. **Jev 单次请求判断与降级**：
   - 输入 `state`：仅截取该会话最近 `8~10` 条消息。
   - **群聊普通闲聊**：单次 HTTP 请求同时传 `Q1`（`noul`：是否适合自然接话）+ `Q2`（`choice`：`default` vs `high` effort）；`Q1` 未过阈值或超时则直接结束，通过则直接采用同一响应中的 `Q2` 结果。
   - **私聊 / 群内被 `@` 或引用回复**：跳过 `Q1`，直接调 Jev 判断 `Q2`（`default` vs `high`）。
   - **超时降级**：设 `800ms~1s` 硬超时；超时/异常时，群聊闲聊默认**不接话**，私聊/被 `@` 默认回退 **`default` effort**。
   - **⚠️ 实现前提醒事项**：**落代码实现本模块前，必须先和用户一起敲定 Jev `Q1`、`Q2` 的具体 Prompt / 选项描述写法。**
5. **直传 Effort 进入 Agent Loop**：脱离 AstrBot 后，直接把判出的 `default` / `high` 作为单次模型调用的 `reasoning_effort` 参数传入。

---

### 3. 基础人设管理与 Session 上下文投影拼装（待最终确认收尾）

- **基础 `sysprompt` 管理**：采用本地 Markdown 文件管理（`prompts/default.md` 为全局默认，按需放置 `prompts/group_{群号}.md` 或 `prompts/private_{QQ号}.md` 覆盖）。
- **上下文分段顺序（自上而下，兼顾前缀固定与 Prompt Cache）**：
  - **私聊**：
    1. `sysprompt & 人设 & tool list`
    2. `对方用户画像`（含对该用户的印象与相处方式）
    3. `压缩 summary`（非空时，100~200 字近期前情）
    4. `当前上下文消息`（`id >= window_start_id`，只追加）
  - **群聊**：
    1. `sysprompt & 人设 & tool list`
    2. `群聊画像（group_impression）`
    3. `压缩 summary`（非空时）
    4. `当前上下文消息`（Bot 回复为 `assistant`，群友发言按 `[昵称(QQ号) HH:MM]: 内容` 归并为 `user`，只追加）
    5. `本轮末尾临时补充`（发完即弃不入库，避免换人说话打碎前 4 层 Cache）：本轮发言群友的 `用户画像` + 本句字面命中的梗释义。

---

## 待敲定模块进度

- [x] **项目架构独立（脱离 AstrBot，直连 NapCat）**
- [x] **通用 SQLite 底座（`spectra/db.py` 6 接口契约，零业务表）**
- [x] **统一配置 + 消息存储 + 视觉识图缓存 + NapCat 收发适配（`config.py` / `messages.py` / `vision.py` / `napcat.py`）**
- [x] **事件日志层（`messages`）+ 多 Session 状态表（`sessions`）**
- [x] **进门门控层（防抖合并 + 本地预筛 + Jev 读空气 + Effort 路由）**
- [ ] **Session 上下文投影分层确认**（已给出草案，待确认锁死）
- [ ] **双长期记忆库存储结构（用户画像/事件记忆 + 分群/全局知识库）与按需工具 / Subagent**
- [ ] **后台压缩、记忆提取归属与自进化 Workflow**
