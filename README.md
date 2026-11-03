# Multi-expert investment council

[中文版](#中文说明) | [English](#english)

---

## English

### Overview

Multi-expert investment council discussion system where multiple AI financial experts debate a given investment idea through structured rounds, guided by a Committee Chair. The system simulates a real expert panel meeting:

1. **User** submits a topic
2. **Committee Chair** opens the meeting, states rules and objectives
3. **Experts** (3-6 AI agents) independently present initial views (Round 1)
4. Experts challenge each other's points and introduce new ideas (Round 2)
5. Experts defend, revise, and converge (Round 3)
6. Optional Round 4 for unresolved key issues
7. **Committee Chair** produces a final report with consensus, disagreements, recommendations, and scores

### Features

- **Multi-provider LLM support** — configure multiple OpenAI-compatible providers with weighted selection
- **Separate Chair LLM** — the Committee Chair can use a different model from the experts
- **Web UI** — WeChat-style group chat interface with real-time WebSocket streaming (port 3088)
- **CLI mode** — run meetings from the command line, suitable for automation and agent-to-agent interaction
- **Web search** — optional Tavily search integration for background research before the meeting
- **Post-meeting chat** — after the meeting ends, continue chatting with the Committee Chair about the results
- **Custom markdown agents** — define your own council of experts as markdown files in `agents/`
- **Language-agnostic meetings** — the council answers in whatever language you ask in (the UI is English)
- **Session persistence** — chat history saved in browser localStorage

### Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**

   Unix/macOS:

   ```bash
   cp .env.example .env
   ```

   PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   Then edit `.env` with your API keys.

   Required configuration:
   - `MODERATOR_LLM_*` — at least one Committee Chair LLM (API key, base URL, model)
   - `LLM_PROVIDER_1_*` — at least one expert LLM provider
   - `SEARCH_API_URL` — (optional) local Tavily-compatible news service for recent structured news
   - `TAVILY_API_KEY` — (optional) open-web search for Eastmoney/Xueqiu/Sina/announcements/research
   - `WEB_SEARCH_API_URL` — (optional) custom Tavily-compatible open-web search backend

   If `.env` is missing on first run, the CLI/server will exit with a clear error message.
   Standard local workflow is: copy `.env.example` to `.env`, fill in keys, then restart.

3. **Run web server**

   ```bash
   python web_server.py
   ```

   Or with Uvicorn:

   ```bash
   uvicorn web_server:app --host 0.0.0.0 --port 3088
   ```

   Open <http://localhost:3088> in your browser.

4. **Run via CLI**

   ```bash
   python main.py "How to design a multi-agent stock research system?"
   python main.py "Topic" --goal "Find 3 actionable solutions" --background "context info"
   python main.py "Topic" --search  # Enable web search before meeting
   python main.py "Topic" --constraint "Budget under $10k" --constraint "Must use Python"
   ```

### Configuration (.env)

| Variable | Description |
| --- | --- |
| `MODERATOR_LLM_NAME` | Committee Chair provider name |
| `MODERATOR_LLM_BASE_URL` | OpenAI-compatible API base URL |
| `MODERATOR_LLM_API_KEY` | API key |
| `MODERATOR_LLM_MODEL` | Model name |
| `MODERATOR_LLM_TIMEOUT` | Request timeout in seconds (default: 300) |
| `LLM_PROVIDER_N_NAME` | Expert provider N name |
| `LLM_PROVIDER_N_BASE_URL` | Expert provider N base URL |
| `LLM_PROVIDER_N_API_KEY` | Expert provider N API key |
| `LLM_PROVIDER_N_MODEL` | Expert provider N model |
| `LLM_PROVIDER_N_WEIGHT` | Selection weight (higher = more likely, default: 1) |
| `LLM_PROVIDER_N_TIMEOUT` | Request timeout in seconds (default: 300) |
| `SEARCH_API_URL` | Tavily-compatible local news API URL |
| `TAVILY_API_KEY` | Tavily API key for open-web search |
| `WEB_SEARCH_API_URL` | Optional Tavily-compatible open-web search API URL |
| `AGENTS_DIR` | Optional directory of markdown agent definitions (default: `./agents`) |

### Custom Agents (Markdown)

The default council (Portfolio Manager, Earnings Reviewer, Business Analyst, Data Researcher) is defined as markdown files in the `agents/` directory — one file per agent. Each file has YAML frontmatter plus a markdown body that becomes the agent's full system prompt:

```markdown
---
name: Portfolio Manager
description: One-line role description shown to the Committee Chair
temperature: 0.45          # optional, default 0.7
---
You are the Head Portfolio Manager on the investment council...
(full persona — constraints, workflow, output expectations)
```

- `name` — display name (defaults to the file name if omitted)
- `description` — shown to the Committee Chair in the opening so it knows each expert's remit
- `temperature` — per-agent sampling temperature (optional)
- Other frontmatter fields (e.g. opencode's `mode`, `tools`, `steps`, `model`) are ignored

Markdown-defined agents keep their **fixed identities**: the Committee Chair introduces them by name and does not re-title them per topic. If the agents directory is missing or empty, the built-in generic experts are used instead. To use a different set, set `AGENTS_DIR` to any directory of markdown agent files.

### Project Structure

```text
├── main.py                 # CLI entry point
├── web_server.py           # FastAPI web server (port 3088)
├── config.py               # .env configuration loader
├── agent_defs.py           # Markdown agent definition loader
├── agents/                 # Council agent definitions (markdown, one per expert)
├── models.py               # Data structures
├── agents.py               # Expert and Moderator agent classes
├── meeting.py              # Meeting orchestrator (CLI)
├── meeting_streaming.py    # Streaming meeting orchestrator (WebSocket)
├── prompts.py              # Prompt templates
├── parser.py               # JSON/Markdown response parser
├── search.py               # Tavily search integration
├── static/index.html       # Web UI
├── tests/                  # Minimal release smoke tests
├── .env.example            # Example configuration
└── requirements.txt        # Python dependencies
```

---

## 中文说明

### 概述

这是一种多专家投资决策讨论系统，由多位人工智能金融专家在委员会主席的引导下，通过结构化的多轮讨论，对特定的投资构想进行辩论。该系统模拟了真实的专家小组会议：

1. **用户** 提出议题
2. **委员会主席** 开场，声明规则和目标
3. **专家** (3-6个AI Agent) 独立陈述初始观点（第一轮）
4. 专家互相质疑、引入新观点（第二轮）
5. 专家辩护、修正、收敛（第三轮）
6. 可选第四轮补充关键未解问题
7. **委员会主席** 产出最终报告，包含共识、分歧、推荐方案和评分

### 功能特性

- **多LLM提供商** — 支持多个OpenAI兼容的LLM服务商，按权重随机选择
- **独立主席LLM** — 委员会主席可以使用与专家不同的模型
- **Web界面** — 微信风格群聊界面，WebSocket实时推送（端口3088）
- **命令行模式** — 通过CLI运行会议，适合自动化和Agent间交互
- **联网搜索** — 可选的Tavily搜索，会议前自动获取背景信息
- **会后对话** — 会议结束后可继续与委员会主席对话讨论结果
- **自定义Markdown专家** — 在 `agents/` 目录中用markdown文件定义自己的专家团
- **语言自适应** — 会议自动以用户提问的语言作答（界面为英文）
- **会话持久化** — 聊天记录保存在浏览器localStorage

### 快速开始

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**

   macOS / Linux:

   ```bash
   cp .env.example .env
   ```

   PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   然后编辑 `.env` 填入你的 API 密钥。

   必须配置：
   - `MODERATOR_LLM_*` — 至少一个委员会主席LLM（API密钥、地址、模型）
   - `LLM_PROVIDER_1_*` — 至少一个专家LLM提供商
   - `TAVILY_API_KEY` — （可选）用于联网搜索功能
   - `SEARCH_API_URL` — （可选）使用本地 Tavily 兼容搜索服务替代 Tavily

   如果首次启动时缺少 `.env`，CLI/服务端会直接退出并提示错误。
   标准本地流程就是：复制 `.env.example` 为 `.env`，填写密钥后重新启动。

3. **启动Web服务**

   ```bash
   python web_server.py
   ```

   或使用 Uvicorn：

   ```bash
   uvicorn web_server:app --host 0.0.0.0 --port 3088
   ```

   浏览器打开 <http://localhost:3088>

4. **命令行运行**

   ```bash
   python main.py "如何设计一个多agent协作做股票研究的系统？"
   python main.py "议题" --goal "找到3个可执行方案" --background "背景信息"
   python main.py "议题" --search  # 启用联网搜索
   python main.py "议题" --constraint "预算10万以内" --constraint "必须使用Python"
   ```

### 配置说明 (.env)

| 变量名 | 说明 |
| --- | --- |
| `MODERATOR_LLM_NAME` | 委员会主席提供商名称 |
| `MODERATOR_LLM_BASE_URL` | OpenAI兼容API地址 |
| `MODERATOR_LLM_API_KEY` | API密钥 |
| `MODERATOR_LLM_MODEL` | 模型名称 |
| `MODERATOR_LLM_TIMEOUT` | 请求超时（秒，默认300） |
| `LLM_PROVIDER_N_NAME` | 专家提供商N名称 |
| `LLM_PROVIDER_N_BASE_URL` | 专家提供商N地址 |
| `LLM_PROVIDER_N_API_KEY` | 专家提供商N API密钥 |
| `LLM_PROVIDER_N_MODEL` | 专家提供商N模型 |
| `LLM_PROVIDER_N_WEIGHT` | 选择权重（越大越可能被选中，默认1） |
| `LLM_PROVIDER_N_TIMEOUT` | 请求超时（秒，默认300） |
| `SEARCH_API_URL` | Tavily兼容的搜索API地址（如本地新闻服务） |
| `TAVILY_API_KEY` | Tavily搜索API密钥（免费注册: <https://tavily.com>） |
| `AGENTS_DIR` | 可选的markdown专家定义目录（默认 `./agents`） |

### 自定义专家（Markdown）

默认专家团（Portfolio Manager、Earnings Reviewer、Business Analyst、Data Researcher）以markdown文件形式定义在 `agents/` 目录中，一个文件一个专家。文件由YAML frontmatter和markdown正文组成，正文即该专家的完整系统提示词：

- `name` — 显示名称（缺省时使用文件名）
- `description` — 开场时展示给委员会主席，说明该专家的职责
- `temperature` — 该专家的采样温度（可选，默认0.7）
- 其他frontmatter字段（如opencode的 `mode`、`tools`、`steps`、`model`）会被忽略

markdown定义的专家保持**固定身份**：委员会主席按其原名介绍，不会按议题重新命名。若 `agents/` 目录不存在或为空，则使用内置的通用专家。可通过 `AGENTS_DIR` 指向其他目录使用不同的专家组合。

### 项目结构

```text
├── main.py                 # 命令行入口
├── web_server.py           # FastAPI Web服务（端口3088）
├── config.py               # .env 配置加载器
├── agent_defs.py           # Markdown专家定义加载器
├── agents/                 # 专家团定义（markdown，每个专家一个文件）
├── models.py               # 数据结构
├── agents.py               # 专家和仲裁者Agent类
├── meeting.py              # 会议编排器（CLI用）
├── meeting_streaming.py    # 流式会议编排器（WebSocket用）
├── prompts.py              # Prompt模板
├── parser.py               # JSON/Markdown响应解析器
├── search.py               # Tavily搜索集成
├── static/index.html       # Web界面
├── tests/                  # 最小发布冒烟测试
├── .env.example            # 配置示例
└── requirements.txt        # Python依赖
```
