"""FastAPI web server with WebSocket for the Investment Council."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import MeetingConfig, load_config
from models import MeetingInput
from agents import ModeratorAgent
from meeting_streaming import StreamingMeeting
from i18n import LANG_FOLLOW_INSTRUCTION, MODERATOR_DISPLAY_NAME
from agent_defs import build_expert_agents
from search import (
    build_research_prompt,
    fetch_webpage_content,
    get_search_api_url,
    get_web_search_api_url,
    is_local_news_api_configured,
    parse_research_command,
    tavily_search,
    web_search,
)
from skills.web_search import execute as skill_web_search, is_available as skill_search_available
from mcp_gateway import McpGateway, load_mcp_servers

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Active meetings: session_id -> StreamingMeeting
_active_meetings: dict[str, StreamingMeeting] = {}
# Completed meeting history: session_id -> full discussion text
_meeting_histories: dict[str, str] = {}


def _validate_startup_configuration(target_app: FastAPI) -> MeetingConfig:
    """Validate app configuration and cache it on the FastAPI app state."""
    config = load_config()
    target_app.state.meeting_config = config

    if getattr(target_app.state, "config_validated", False):
        return config

    logger.info("✓ Moderator LLM: %s (%s)", config.moderator_llm.name, config.moderator_llm.model)
    logger.info("✓ Expert providers: %d configured", len(config.expert_providers))
    for i, provider in enumerate(config.expert_providers, 1):
        logger.info("  Provider %d: %s (%s) weight=%d", i, provider.name, provider.model, provider.weight)
    local_search_url = get_search_api_url()
    web_search_url = get_web_search_api_url()
    if local_search_url:
        logger.info("✓ Local search backend: %s", local_search_url)
    if web_search_url:
        logger.info("✓ Web search backend: %s", web_search_url)
    if not local_search_url and not web_search_url:
        logger.warning("✗ No search backend configured — web search disabled")

    target_app.state.config_validated = True
    return config


def _get_runtime_config() -> MeetingConfig:
    config = getattr(app.state, "meeting_config", None)
    if config is None:
        config = _validate_startup_configuration(app)
    return config


@asynccontextmanager
async def lifespan(target_app: FastAPI):
    try:
        _validate_startup_configuration(target_app)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Configuration error: %s", e)
        logger.error("First run: copy .env.example to .env and fill in the required keys.")
        raise RuntimeError("Invalid application configuration.") from e
    yield


app = FastAPI(title="Investment Council", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("WebSocket connected: session=%s", session_id)

    async def send_event(event: dict[str, Any]):
        try:
            await websocket.send_json(event)
        except Exception:
            logger.warning("Failed to send event to session %s", session_id)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "start_meeting":
                topic = data.get("topic", "").strip()
                if not topic:
                    await send_event({"type": "error", "content": "Please enter a discussion topic."})
                    continue

                goal = data.get("goal", "").strip() or None
                background = data.get("background", "").strip() or None
                constraints_raw = data.get("constraints", "").strip()
                constraints = [c.strip() for c in constraints_raw.split("\n") if c.strip()] if constraints_raw else []
                search_enabled = data.get("search_enabled", False)

                # Cancel any existing meeting for this session
                if session_id in _active_meetings:
                    _active_meetings[session_id].cancel()

                # Start meeting in background
                asyncio.create_task(
                    _run_meeting_task(session_id, topic, goal, constraints, background, search_enabled, send_event)
                )

            elif action == "stop_meeting":
                if session_id in _active_meetings:
                    _active_meetings[session_id].cancel()
                    await send_event({"type": "system", "content": "Meeting stopped by user."})
                    await send_event({"type": "meeting_end", "content": "Meeting terminated."})

            elif action == "chat":
                user_msg = data.get("message", "").strip()
                if not user_msg:
                    await send_event({"type": "error", "content": "Please enter a message."})
                    continue
                history = _meeting_histories.get(session_id, "")
                if not history:
                    await send_event({"type": "error", "content": "No meeting history available."})
                    continue
                asyncio.create_task(
                    _chat_with_moderator(session_id, user_msg, history, send_event)
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
        if session_id in _active_meetings:
            _active_meetings[session_id].cancel()
            del _active_meetings[session_id]
        # Keep _meeting_histories — it persists for reconnections


MAX_SEARCH_ROUNDS = 20
RESEARCH_MAX_RESULTS = 20


async def _research_phase(
    topic: str,
    existing_background: str | None,
    send_event,
) -> str:
    """Moderator researches the topic using Tavily, up to MAX_SEARCH_ROUNDS searches."""
    config = _get_runtime_config()
    moderator_client = AsyncOpenAI(
        api_key=config.moderator_llm.api_key,
        base_url=config.moderator_llm.base_url or None,
    )
    model = config.moderator_llm.model

    collected_info: list[str] = []
    searched_queries: list[str] = []
    seen_commands: set[str] = set()
    seen_urls: set[str] = set()
    no_progress_rounds = 0
    last_feedback: str | None = None
    keyword_searches = 0

    await send_event({"type": "system", "content": f"🔍 The {MODERATOR_DISPLAY_NAME} is searching for background information..."})

    system_prompt, instructions = build_research_prompt(is_local_news_api_configured())

    for i in range(MAX_SEARCH_ROUNDS):
        # Ask the chair to generate a search query
        prompt = (
            f"You are the {MODERATOR_DISPLAY_NAME} of a council meeting. You need to gather background information on the upcoming topic using the available search tools.\n\n"
            f"Topic: {topic}\n"
        )
        if existing_background:
            prompt += f"User-provided background: {existing_background}\n"
        if collected_info:
            prompt += f"\nCollected information so far:\n{''.join(collected_info[-3:])}\n"
        if searched_queries:
            prompt += f"\nPreviously executed search/browse commands: {', '.join(searched_queries)}\n"
        if last_feedback:
            prompt += f"\nLast search feedback: {last_feedback}\n"

        prompt += instructions

        try:
            resp = await moderator_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            query = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("Research LLM call failed: %s", e)
            break

        if not query or query.upper() == "DONE":
            await send_event({"type": "system", "content": f"🔍 The {MODERATOR_DISPLAY_NAME} determined information is sufficient."})
            break

        command = parse_research_command(query)
        backend = command["backend"]
        actual_query = command["query"] or ""
        actual_category = command["category"]
        actual_url = command["url"]
        command_key = command["command_key"] or query

        if backend == "local" and not actual_query and keyword_searches == 0:
            last_feedback = "Please try one keyword search first; do not start with ALL or RECENT."
            await send_event({"type": "system", "content": f"🔍 {last_feedback}"})
            continue
        if command_key in seen_commands:
            await send_event({"type": "system", "content": f"🔍 The {MODERATOR_DISPLAY_NAME} repeated the same command: {query}. Stopping to avoid fetching the same news repeatedly."})
            break

        searched_queries.append(query)
        seen_commands.add(command_key)
        await send_event({"type": "search_start", "content": f"🔍 Search #{i + 1}: {query}"})

        try:
            if backend == "local":
                result = await tavily_search(
                    query=actual_query,
                    category=actual_category,
                    max_results=RESEARCH_MAX_RESULTS,
                    topic="news",
                )
                if actual_query:
                    keyword_searches += 1
            elif backend == "web":
                result = await web_search(
                    query=actual_query,
                    max_results=8,
                    search_depth="advanced",
                    topic="general",
                )
            else:
                result = await fetch_webpage_content(actual_url or "")

            result_urls = {r.get("url", "") for r in result.get("results", []) if r.get("url")}
            new_urls = result_urls - seen_urls

            if backend == "local" and actual_query and result.get("result_count", 0) == 0:
                scope = "the local database" if is_local_news_api_configured() else "recent news"
                last_feedback = (
                    f"Keyword {query} returned no results. Try a different short keyword, or use WEB: for announcements, fundamentals, and research beyond {scope}."
                )
                await send_event({
                    "type": "search_result",
                    "content": result["summary"],
                    "result_count": result["result_count"],
                })
                await send_event({"type": "system", "content": f"🔍 {last_feedback}"})
                continue

            if new_urls:
                seen_urls.update(new_urls)
                no_progress_rounds = 0
                last_feedback = None
                collected_info.append(result["summary"])
                await send_event({
                    "type": "search_result",
                    "content": result["summary"],
                    "result_count": result["result_count"],
                })
            else:
                no_progress_rounds += 1
                last_feedback = (
                    f"Command {query} produced no new news items. Do not repeat it; use a different category or reply DONE."
                )
                await send_event({"type": "system", "content": f"🔍 {last_feedback}"})
                if no_progress_rounds >= 2:
                    await send_event({"type": "system", "content": "🔍 Two consecutive searches produced no new information. Stopping."})
                    break
        except Exception as e:
            logger.warning("Search failed for query '%s': %s", query, e)
            message = f"Search failed: {e}"
            error_text = str(e).lower()
            if any(marker in error_text for marker in ("401", "403", "unauthorized", "api key")):
                message += " The search backend rejected the request — check TAVILY_API_KEY (or your search API configuration) in .env."
            await send_event({"type": "system", "content": message})
            break

    if collected_info:
        await send_event({"type": "system", "content": f"🔍 Search complete ({len(searched_queries)} searches). Starting council meeting..."})
    else:
        await send_event({"type": "system", "content": "🔍 No useful information found, starting council meeting directly..."})

    return "\n\n".join(collected_info)


async def _run_meeting_task(
    session_id: str,
    topic: str,
    goal: str | None,
    constraints: list[str],
    background: str | None,
    search_enabled: bool,
    send_event,
):
    mcp_gateway = None
    try:
        config = _get_runtime_config()

        # === Research phase: moderator auto-searches if search skill is enabled ===
        search_background = ""
        if search_enabled:
            if skill_search_available():
                search_background = await _research_phase(topic, background, send_event)
            else:
                await send_event({"type": "system", "content": "🔍 No search backend configured (set TAVILY_API_KEY or SEARCH_API_URL). Skipping pre-meeting research."})

        # Combine user-provided background with search results
        combined_background = ""
        if background:
            combined_background = background
        if search_background:
            combined_background = (combined_background + "\n\n" if combined_background else "") + search_background

        meeting_input = MeetingInput(
            topic=topic, goal=goal, constraints=constraints,
            background=combined_background or None,
        )

        moderator = ModeratorAgent(provider=config.moderator_llm)

        # Experts come from markdown definitions in agents/ (or built-in defaults)
        experts = build_expert_agents(config)

        # Enable search skill on experts if a search backend is configured
        if search_enabled and skill_search_available():
            for expert in experts:
                expert.search_fn = skill_web_search

        # Connect MCP servers (independent of the search toggle); experts may
        # call their tools during the meeting.
        mcp_servers = load_mcp_servers()
        if mcp_servers:
            mcp_gateway = McpGateway(mcp_servers)
            await mcp_gateway.start()
            if mcp_gateway.has_tools():
                for expert in experts:
                    expert.tool_gateway = mcp_gateway

        meeting = StreamingMeeting(
            moderator=moderator,
            experts=experts,
            max_rounds=4,
            on_event=send_event,
        )
        _active_meetings[session_id] = meeting

        final_report = await meeting.run(meeting_input)

        # Save the full discussion history for post-meeting chat
        if final_report:
            _meeting_histories[session_id] = meeting._build_full_discussion(
                meeting.opening_text or ""
            )

    except asyncio.CancelledError:
        await send_event({"type": "system", "content": "Meeting cancelled."})
        await send_event({"type": "meeting_end", "content": "Meeting terminated."})
    except Exception as e:
        logger.exception("Meeting failed for session %s", session_id)
        await send_event({"type": "error", "content": f"Meeting error: {e}"})
        await send_event({"type": "meeting_end", "content": "Meeting ended abnormally."})
    finally:
        _active_meetings.pop(session_id, None)
        if mcp_gateway is not None:
            await mcp_gateway.close()


# Per-session chat history for post-meeting follow-up
_chat_histories: dict[str, list[dict[str, str]]] = {}


async def _chat_with_moderator(
    session_id: str,
    user_message: str,
    meeting_history: str,
    send_event,
):
    """After a meeting ends, let the moderator chat with the user based on meeting history."""
    try:
        config = _get_runtime_config()
        client = AsyncOpenAI(
            api_key=config.moderator_llm.api_key,
            base_url=config.moderator_llm.base_url or None,
        )

        system_prompt = (
            f"You are the {MODERATOR_DISPLAY_NAME} of a council meeting that just ended. Below is the full meeting transcript.\n"
            "The user has follow-up questions about the meeting. Answer based on the transcript.\n"
            "Stay neutral, accurate, and concise. If the user asks about something not discussed, say so.\n\n"
            + f"=== Meeting Record ===\n{meeting_history}\n=== End of Record ==="
            + LANG_FOLLOW_INSTRUCTION
        )

        # Initialize or retrieve session chat history
        if session_id not in _chat_histories:
            _chat_histories[session_id] = []

        _chat_histories[session_id].append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system_prompt}] + _chat_histories[session_id]

        await send_event({"type": "agent_start", "agent": MODERATOR_DISPLAY_NAME, "role": "Arbiter", "round": "chat"})

        resp = await client.chat.completions.create(
            model=config.moderator_llm.model,
            messages=messages,
            temperature=0.5,
        )
        reply = resp.choices[0].message.content or ""

        _chat_histories[session_id].append({"role": "assistant", "content": reply})

        await send_event({
            "type": "agent_message",
            "agent": MODERATOR_DISPLAY_NAME,
            "role": "Arbiter",
            "round": "chat",
            "content": reply,
        })

    except Exception as e:
        logger.exception("Chat with moderator failed for session %s", session_id)
        await send_event({"type": "error", "content": f"Chat error: {e}"})


if __name__ == "__main__":
    import uvicorn

    try:
        _validate_startup_configuration(app)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Configuration error: %s", e)
        logger.error("First run: copy .env.example to .env and fill in the required keys.")
        import sys
        sys.exit(1)

    uvicorn.run(app, host="0.0.0.0", port=3088)

