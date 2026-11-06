# Multi-expert investment council

## English

### Overview

Multi-expert investment council discussion system where multiple AI financial experts debate a given investment idea through structured rounds, guided by a Committee Chair. The system simulates a real expert panel meeting:

1. **User** submits a topic
2. **Committee Chair** opens the meeting, states rules and objectives
3. **Experts** (4 by default, configurable via `agents/`) independently present initial views (Round 1)
4. Experts challenge each other's points and introduce new ideas (Round 2)
5. Experts defend, revise, and converge (Round 3)
6. Optional Round 4 for unresolved key issues
7. **Committee Chair** produces a final report with consensus, disagreements, recommendations, and scores

### Features

- **Multi-provider LLM support** — configure multiple OpenAI-compatible providers and pin each expert to a specific provider (`LLM_PROVIDER_N_EXPERT`), with weighted random selection as fallback
- **Separate Chair LLM** — the Committee Chair can use a different model from the experts
- **Web UI** — council-room interface: Committee Chair panel, one panel per expert, and a final-report section with a live round-by-round digest; real-time WebSocket streaming (port 3088)
- **Report export** — download the final report as a PDF or Markdown from the Web UI
- **CLI mode** — run meetings from the command line, suitable for automation and agent-to-agent interaction
- **Web search** — optional pre-meeting background research via Tavily, a local news API, or a custom Tavily-compatible backend
- **MCP tools** — connect MCP servers via .env so experts can call their tools during the meeting
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
   - `TAVILY_API_KEY` — (optional) open-web and news search via Tavily
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
   python main.py "Topic" --env /path/to/.env --no-chat  # custom config file, skip post-meeting chat
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
| `LLM_PROVIDER_N_EXPERT` | Pin provider N to an expert by agents/ file name, e.g. `tax_expert.md` (optional) |
| `LLM_PROVIDER_N_TIMEOUT` | Request timeout in seconds (default: 300) |
| `SEARCH_API_URL` | Tavily-compatible local news API URL |
| `TAVILY_API_KEY` | Tavily API key for open-web search |
| `WEB_SEARCH_API_URL` | Optional Tavily-compatible open-web search API URL |
| `AGENTS_DIR` | Optional directory of markdown agent definitions (default: `./agents`) |
| `MCP_SERVER_N_NAME` | Name for MCP server N (shown to experts) |
| `MCP_SERVER_N_URL` | Streamable-HTTP URL of MCP server N (e.g. `http://127.0.0.1:8081/mcp`) |
| `MCP_SERVER_N_API_KEY` | Optional bearer token for MCP server N |

### Custom Agents (Markdown)

The default council (Tax Expert, Quantitative Analyst, Business Analyst, Data Researcher) is defined as markdown files in the `agents/` directory — one file per agent. Each file has YAML frontmatter plus a markdown body that becomes the agent's full system prompt:

```markdown
---
name: Tax Expert
description: One-line role description shown to the Committee Chair
temperature: 0.45          # optional, default 0.7
---
You are a tax specialist on the investment council...
(full persona — constraints, workflow, output expectations)
```

- `name` — display name (defaults to the file name if omitted)
- `description` — shown to the Committee Chair in the opening so it knows each expert's remit
- `temperature` — per-agent sampling temperature (optional)
- Other frontmatter fields (e.g. opencode's `mode`, `tools`, `steps`, `model`) are ignored

Experts can be pinned to specific LLM providers in `.env` with `LLM_PROVIDER_N_EXPERT=<agent file>.md` (each expert may be assigned to at most one provider). When at least one provider has an assignment, **only assigned experts join the meeting** — the roster tracks your configured models; with no assignments at all, every expert in the directory joins and is picked by weighted random selection from all providers. Assigning a file that does not exist in `agents/` is a configuration error (the server refuses to start).

Markdown-defined agents keep their **fixed identities**: the Committee Chair introduces them by name and does not re-title them per topic. If the agents directory is missing or empty, the built-in generic experts are used instead. To use a different set, set `AGENTS_DIR` to any directory of markdown agent files.

### MCP Servers

Experts can call tools exposed by remote [MCP](https://modelcontextprotocol.io) servers. Configure one or more servers in `.env` with the `MCP_SERVER_N_*` scheme (numbered `1`, `2`, …); a server is active when its `_URL` is set:

```env
MCP_SERVER_1_NAME=financial_data
MCP_SERVER_1_URL=http://127.0.0.1:8081/mcp
MCP_SERVER_1_API_KEY=...        # optional
```

- Connections use the Streamable HTTP transport via the official `mcp` Python SDK.
- Each meeting opens one connection per server, lists its tools, and closes them when the meeting ends.
- A server's tools are added to the experts' prompts; an expert calls one by emitting a `tool_call` field, e.g. `{"server": "financial_data", "tool": "get_quote", "arguments": {"symbol": "AAPL"}}`.
- MCP tools are independent of the "🔍 Web Search" toggle (which controls only the search backends and the Chair's pre-meeting research) and are available to experts only, not the Committee Chair.
- A server that fails to connect is skipped with a warning and does not affect the meeting.

### Project Structure

```text
├── main.py                 # CLI entry point
├── web_server.py           # FastAPI web server (port 3088)
├── config.py               # .env configuration loader
├── agent_defs.py           # Markdown agent definition loader
├── i18n.py                 # Display strings and built-in default experts
├── skills/                 # Agent skills (web search)
├── agents/                 # Council agent definitions (markdown, one per expert)
├── models.py               # Data structures
├── agents.py               # Expert and Moderator agent classes
├── meeting.py              # Meeting orchestrator (CLI)
├── meeting_streaming.py    # Streaming meeting orchestrator (WebSocket)
├── prompts.py              # Prompt templates
├── parser.py               # JSON/Markdown response parser
├── search.py               # Tavily / local news / custom backend search integration
├── mcp_gateway.py          # MCP server client gateway
├── static/index.html       # Web UI
├── tests/                  # Minimal release smoke tests
├── .env.example            # Example configuration
└── requirements.txt        # Python dependencies
```
