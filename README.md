# Spectra-bot

基于第一性原理设计的轻量级 QQ 群聊 / 私聊 Agent（直连 NapCat OneBot v11）。

核心目标：构建具备**活人感**、能够读空气、拥有前缀稳定上下文与结构化可观测能力的群友型 Bot。

## 快速开始

### 1. 安装依赖

需要 Python `>= 3.12` 与 [`uv`](https://docs.astral.sh/uv/)：

```bash
uv sync
```

### 2. 初始化配置文件

```bash
uv run spectra --init-config
```

该命令会在当前目录生成默认 `config.json`（若已存在则拒绝覆盖，完整示例见 [`config.example.json`](config.example.json)）。

至少需要配置以下字段：
- `llm.base_url` / `llm.api_key` / `llm.model`：对话模型（OpenAI 兼容 API）。
- `vision`：如需识图/表情包描述，填写 `vision.base_url` / `vision.api_key` / `vision.model`；否则可将 `vision.enabled` 设为 `false`。
- `jev`：如需启用 Jev 读空气与 Effort 路由，填写 `jev.api_key`；未填写时自动回退为 AstrBot 风格的 `gate.active_reply.possibility_reply` 概率插话。
- `napcat`：在 NapCat 中添加反向 WebSocket 客户端指向 `ws://<host>:6199/ws`（如有 `access_token` 请两端保持一致）。

### 3. 自定义人设（支持热重载）

人设文件位于 `prompts/` 目录（由 `session.prompts_dir` 指定），编辑或增删即时生效，无需重启：
- `prompts/default.md`：全局默认人设
- `prompts/group_{群号}.md`：指定群聊覆盖人设
- `prompts/private_{QQ号}.md`：指定私聊覆盖人设

### 4. 启动服务

```bash
uv run spectra --config config.json
# 或通过模块方式启动
uv run python -m spectra --config config.json --log-level INFO
```

每个经过评估的消息批次都会在 `data/traces.jsonl`（由 `trace.path` 配置）记录 1 行结构化 JSONL Trace，包含门控原因、Jev 概率、模型耗时、Token 用量与工具调用信息。

## 开发与测试

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -v -W error::ResourceWarning
```

## 核心文档

- [模块设计与流程敲定记录](docs/decisions.md)
- [需求背景与调研记录](docs/requirements.md)
