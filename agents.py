"""Agent interfaces for the Investment Council."""

import json
import logging
from typing import Any, Callable, Coroutine, Dict, Optional

import httpx
from openai import AsyncOpenAI

from config import LLMProviderConfig
from models import AgentMessage, RoundScore, RoundSummary, FinalReport, FinalScore
from parser import parse_json_response, safe_parse_agent_output
from i18n import LANG_FOLLOW_INSTRUCTION, MODERATOR_DISPLAY_NAME
from skills.web_search import get_search_prompt, get_skill_prompt
from mcp_gateway import build_mcp_tools_prompt
from prompts import get_prompt, EXPERT_INTRODUCTION_RETITLE, EXPERT_INTRODUCTION_FIXED

logger = logging.getLogger(__name__)

MAX_RETRIES = 1
MAX_AGENT_SEARCHES = 5
DEFAULT_AGENT_SEARCH_RESULTS = 20
DEFAULT_AGENT_WEB_RESULTS = 8
DEFAULT_AGENT_FETCH_CHARS = 12000


def _build_search_request(search_query: Any) -> tuple[dict[str, Any], str, str]:
    if isinstance(search_query, dict):
        params = dict(search_query)
    else:
        params = {"query": str(search_query)}

    backend = str(params.get("backend", "local") or "local").strip().lower()
    params["backend"] = backend

    query_value = str(params.get("query", "") or "").strip()
    url_value = str(params.get("url", "") or "").strip()

    if backend == "fetch":
        target_url = url_value or query_value
        if not target_url:
            raise ValueError("Fetch requests require a URL")
        params["url"] = target_url
        params["query"] = ""
        if not params.get("max_chars"):
            params["max_chars"] = DEFAULT_AGENT_FETCH_CHARS
        display = f"FETCH:{target_url}"
    elif backend == "web":
        if not query_value:
            raise ValueError("Web search requests require a query")
        params["query"] = query_value
        if not params.get("max_results"):
            params["max_results"] = DEFAULT_AGENT_WEB_RESULTS
        display = f"WEB:{query_value}"
    else:
        if query_value.upper() == "ALL":
            params["query"] = ""
            query_value = ""
        if not params.get("max_results"):
            params["max_results"] = DEFAULT_AGENT_SEARCH_RESULTS
        display = query_value or "ALL"
        if not params.get("query") and params.get("category"):
            display = f"RECENT:{params['category']}"
        if params.get("source"):
            display = f"{display} @ {params['source']}"

    request_key = json.dumps(
        {
            "backend": backend,
            "query": params.get("query", ""),
            "url": params.get("url"),
            "category": params.get("category"),
            "source": params.get("source"),
            "days": params.get("days"),
            "max_results": params.get("max_results"),
            "max_chars": params.get("max_chars"),
            "include_domains": params.get("include_domains"),
            "exclude_domains": params.get("exclude_domains"),
            "search_depth": params.get("search_depth"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return params, display, request_key


def _build_tool_call_request(tool_call: Any) -> tuple[str, str, dict, str, str]:
    """Normalize a model-emitted ``tool_call`` object into dispatchable parts.

    Returns ``(server_name, tool_name, arguments, display, request_key)``.
    Raises ``ValueError`` when the call is malformed.
    """
    if not isinstance(tool_call, dict):
        raise ValueError("tool_call must be an object with 'server', 'tool', and 'arguments'")
    server_name = str(tool_call.get("server", "") or "").strip()
    tool_name = str(tool_call.get("tool", "") or "").strip()
    arguments = tool_call.get("arguments", {}) or {}
    if not isinstance(arguments, dict):
        raise ValueError("tool_call.arguments must be an object")
    if not server_name:
        raise ValueError("tool_call requires a 'server' name")
    if not tool_name:
        raise ValueError("tool_call requires a 'tool' name")
    display = f"MCP:{server_name}.{tool_name}"
    request_key = "MCP:" + json.dumps(
        {"server": server_name, "tool": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
    )
    return server_name, tool_name, arguments, display, request_key


def create_client(provider: LLMProviderConfig) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from a provider config."""
    kwargs: dict = {"api_key": provider.api_key}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    if provider.timeout:
        kwargs["timeout"] = httpx.Timeout(provider.timeout, connect=30.0)
    return AsyncOpenAI(**kwargs)


class BaseMeetingAgent:
    """Base class for all meeting agents."""

    def __init__(
        self,
        name: str,
        role_label: str,
        system_prompt: str,
        provider: LLMProviderConfig,
        temperature: float = 0.7,
    ):
        self.name = name
        self.role_label = role_label
        self.system_prompt = system_prompt
        self.provider = provider
        self.temperature = temperature
        self.client = create_client(provider)
        self.model = provider.model

    async def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM and return the raw text response."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(
                    "[%s] LLM call attempt %d failed (%s/%s): %s",
                    self.name, attempt + 1, self.provider.name, self.model, e,
                )
                if attempt == MAX_RETRIES:
                    raise
        return ""

    async def speak(
        self, round_index: int, task_prompt: str, context: dict
    ) -> AgentMessage:
        """Generate a speech for the given round."""
        raw_text = await self._call_llm(task_prompt)
        content = safe_parse_agent_output(raw_text, list(context.get("expected_keys", [])))
        return AgentMessage(
            round_index=round_index,
            agent_name=self.name,
            content=content,
            raw_text=raw_text,
        )


class ExpertAgent(BaseMeetingAgent):
    """An expert agent participating in the council meeting."""

    # Async research callback: async (**params) -> dict with 'summary', 'result_count'
    search_fn: Optional[Callable[..., Coroutine]] = None

    # Optional MCP gateway exposing connected MCP server tools (mcp_gateway.McpGateway)
    tool_gateway: Optional[Any] = None

    def __init__(
        self,
        name: str,
        role_label: str,
        provider: LLMProviderConfig,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        allow_retitle: bool = True,
    ):
        self.base_name = name
        self.style_label = role_label
        self.base_system_prompt = system_prompt
        self.allow_retitle = allow_retitle
        super().__init__(name, role_label, "", provider, temperature=temperature or 0.7)
        self._refresh_system_prompt()

    def _refresh_system_prompt(self):
        if self.base_system_prompt:
            # Full persona from a markdown agent definition; keep it verbatim.
            self.system_prompt = self.base_system_prompt + LANG_FOLLOW_INSTRUCTION
            return
        self.system_prompt = (
            f"You are an expert on a council meeting. Your current public title for this topic "
            f"is [{self.name}], and your base thinking style is [{self.style_label}]. "
            "Always participate in the discussion in this style, offering constructive, substantive "
            "points, and identify yourself by your current public title."
            + LANG_FOLLOW_INSTRUCTION
        )

    def _skill_prompt(self) -> str:
        """Return skill prompt text for the available tools, empty string otherwise."""
        parts: list[str] = []
        if self.search_fn:
            parts.append(get_skill_prompt())
        if self.tool_gateway is not None:
            parts.append(build_mcp_tools_prompt(self.tool_gateway.catalog()))
        return "".join(parts)

    async def _speak_with_search(
        self, prompt: str, expected_keys: list[str], round_index: int,
    ) -> AgentMessage:
        """Multi-step speak: LLM may search up to MAX_AGENT_SEARCHES times, then answer."""
        raw = await self._call_llm(prompt)
        content = safe_parse_agent_output(raw, expected_keys + ["search_query"])

        search_info: list[dict[str, Any]] = []
        search_history: list[str] = []
        seen_requests: set[str] = set()
        executed_searches = 0
        keyword_searches = 0
        decision_rounds = 0

        if self.search_fn or self.tool_gateway is not None:
            while executed_searches < MAX_AGENT_SEARCHES and decision_rounds < MAX_AGENT_SEARCHES * 3:
                decision_rounds += 1
                search_query = content.pop("search_query", None) if isinstance(content, dict) else None
                tool_call = content.pop("tool_call", None) if isinstance(content, dict) else None

                if tool_call is not None:
                    if self.tool_gateway is None:
                        search_history.append(
                            get_search_prompt("ERROR").format(
                                query=str(tool_call),
                                error="No MCP tools are available in this meeting.",
                                remaining_searches=MAX_AGENT_SEARCHES - executed_searches,
                            )
                        )
                        raw = await self._call_llm(prompt + "".join(search_history))
                        content = safe_parse_agent_output(raw, expected_keys + ["search_query", "tool_call"])
                        continue
                    remaining_searches = MAX_AGENT_SEARCHES - executed_searches
                    try:
                        server_name, tool_name, arguments, query_display, request_key = _build_tool_call_request(tool_call)
                    except Exception as e:
                        search_history.append(
                            get_search_prompt("ERROR").format(
                                query=str(tool_call),
                                error=e,
                                remaining_searches=remaining_searches,
                            )
                        )
                        raw = await self._call_llm(prompt + "".join(search_history))
                        content = safe_parse_agent_output(raw, expected_keys + ["search_query", "tool_call"])
                        continue

                    if request_key in seen_requests:
                        logger.info("[%s] Round %d duplicate MCP call skipped: '%s'", self.name, round_index, query_display)
                        search_history.append(
                            get_search_prompt("DUPLICATE").format(
                                query=query_display,
                                remaining_searches=remaining_searches,
                            )
                        )
                        if remaining_searches == 0:
                            search_history.append(get_search_prompt("FINAL_ONLY"))
                            raw = await self._call_llm(prompt + "".join(search_history))
                            content = safe_parse_agent_output(raw, expected_keys)
                            break
                        raw = await self._call_llm(prompt + "".join(search_history))
                        content = safe_parse_agent_output(raw, expected_keys + ["search_query", "tool_call"])
                        continue

                    seen_requests.add(request_key)
                    executed_searches += 1
                    remaining_searches = MAX_AGENT_SEARCHES - executed_searches
                    try:
                        tool_result = await self.tool_gateway.call_tool(server_name, tool_name, arguments)
                    except Exception as e:
                        logger.warning("[%s] MCP call failed for '%s': %s", self.name, query_display, e)
                        tool_result = {"summary": str(e), "result_count": 0, "is_error": True}

                    if tool_result.get("is_error"):
                        search_history.append(
                            get_search_prompt("ERROR").format(
                                query=query_display,
                                error=tool_result.get("summary", ""),
                                remaining_searches=remaining_searches,
                            )
                        )
                    else:
                        search_info.append(
                            {
                                "query": query_display,
                                "result_count": tool_result.get("result_count", 0),
                                "search_index": executed_searches,
                                "max_searches": MAX_AGENT_SEARCHES,
                            }
                        )
                        logger.info(
                            "[%s] Round %d MCP tool %d/%d: '%s'",
                            self.name, round_index, executed_searches, MAX_AGENT_SEARCHES, query_display,
                        )
                        search_history.append(
                            get_search_prompt("REFINE").format(
                                query=query_display,
                                summary=tool_result.get("summary", ""),
                                search_index=executed_searches,
                                max_searches=MAX_AGENT_SEARCHES,
                                remaining_searches=remaining_searches,
                            )
                        )

                    if remaining_searches == 0:
                        search_history.append(get_search_prompt("FINAL_ONLY"))
                        raw = await self._call_llm(prompt + "".join(search_history))
                        content = safe_parse_agent_output(raw, expected_keys)
                        break

                    raw = await self._call_llm(prompt + "".join(search_history))
                    content = safe_parse_agent_output(raw, expected_keys + ["search_query", "tool_call"])
                    continue

                if not search_query:
                    break
                if self.search_fn is None:
                    search_history.append(
                        get_search_prompt("ERROR").format(
                            query=str(search_query),
                            error="No search tools are available in this meeting.",
                            remaining_searches=MAX_AGENT_SEARCHES - executed_searches,
                        )
                    )
                    if self.tool_gateway is not None and self.tool_gateway.has_tools():
                        search_history.append(
                            "\nNote: Search tools are unavailable. You can still gather information using MCP tools listed above in your skill prompt. Each MCP tool call counts toward your tool budget.\n"
                        )
                    raw = await self._call_llm(prompt + "".join(search_history))
                    content = safe_parse_agent_output(raw, expected_keys + ["search_query", "tool_call"])
                    continue

                try:
                    params, query_display, request_key = _build_search_request(search_query)
                except Exception as e:
                    remaining_searches = MAX_AGENT_SEARCHES - executed_searches
                    search_history.append(
                        get_search_prompt("ERROR").format(
                            query=str(search_query),
                            error=e,
                            remaining_searches=remaining_searches,
                        )
                    )
                    raw = await self._call_llm(prompt + "".join(search_history))
                    content = safe_parse_agent_output(raw, expected_keys + ["search_query"])
                    continue

                remaining_searches = MAX_AGENT_SEARCHES - executed_searches - 1
                backend = str(params.get("backend", "local"))
                is_browse_request = backend == "local" and not str(params.get("query", "") or "").strip()

                if is_browse_request and keyword_searches == 0:
                    search_history.append(
                        get_search_prompt("KEYWORD_FIRST").format(
                            remaining_searches=MAX_AGENT_SEARCHES - executed_searches,
                        )
                    )
                    raw = await self._call_llm(prompt + "".join(search_history))
                    content = safe_parse_agent_output(raw, expected_keys + ["search_query"])
                    continue

                if request_key in seen_requests:
                    logger.info("[%s] Round %d duplicate search skipped: '%s'", self.name, round_index, query_display)
                    search_history.append(
                        get_search_prompt("DUPLICATE").format(
                            query=query_display,
                            remaining_searches=remaining_searches,
                        )
                    )
                    if remaining_searches == 0:
                        search_history.append(get_search_prompt("FINAL_ONLY"))
                        raw = await self._call_llm(prompt + "".join(search_history))
                        content = safe_parse_agent_output(raw, expected_keys)
                        break
                    raw = await self._call_llm(prompt + "".join(search_history))
                    content = safe_parse_agent_output(raw, expected_keys + ["search_query"])
                    continue

                seen_requests.add(request_key)
                executed_searches += 1
                if not is_browse_request:
                    keyword_searches += 1

                try:
                    search_result = await self.search_fn(**params)
                    search_info.append(
                        {
                            "query": query_display,
                            "result_count": search_result.get("result_count", 0),
                            "search_index": executed_searches,
                            "max_searches": MAX_AGENT_SEARCHES,
                        }
                    )
                    logger.info(
                        "[%s] Round %d search %d/%d: '%s' → %d results",
                        self.name,
                        round_index,
                        executed_searches,
                        MAX_AGENT_SEARCHES,
                        query_display,
                        search_info[-1]["result_count"],
                    )
                    search_history.append(
                        get_search_prompt("REFINE").format(
                            query=query_display,
                            summary=search_result.get("summary", ""),
                            search_index=executed_searches,
                            max_searches=MAX_AGENT_SEARCHES,
                            remaining_searches=remaining_searches,
                        )
                    )
                except Exception as e:
                    logger.warning("[%s] Search failed for '%s': %s", self.name, query_display, e)
                    search_history.append(
                        get_search_prompt("ERROR").format(
                            query=query_display,
                            error=e,
                            remaining_searches=remaining_searches,
                        )
                    )
                    if self.tool_gateway is not None and self.tool_gateway.has_tools():
                        search_history.append(
                            "\nNote: Search tools are currently unavailable. You can still gather information using MCP tools listed above in your skill prompt. Each MCP tool call counts toward your tool budget.\n"
                        )

                if remaining_searches == 0:
                    search_history.append(get_search_prompt("FINAL_ONLY"))
                    raw = await self._call_llm(prompt + "".join(search_history))
                    content = safe_parse_agent_output(raw, expected_keys)
                    break

                raw = await self._call_llm(prompt + "".join(search_history))
                content = safe_parse_agent_output(raw, expected_keys + ["search_query"])

        if isinstance(content, dict):
            content.pop("search_query", None)
            content.pop("tool_call", None)

        return AgentMessage(
            round_index=round_index,
            agent_name=self.name,
            content=content,
            raw_text=raw,
            search_info=search_info or None,
        )

    def retitle(self, title: str):
        title = title.strip()
        if not title or title == self.name:
            return
        self.name = title
        self._refresh_system_prompt()

    def opening_assignment_label(self) -> str:
        """Base name + style label, formatted for the moderator opening prompt."""
        return f"{self.base_name} ({self.style_label})"

    async def speak_round1(self, topic: str, opening: str) -> AgentMessage:
        self._refresh_system_prompt()
        prompt = get_prompt("EXPERT_ROUND1").format(
            role_label=self.role_label, topic=topic, opening=opening,
            skill_prompt=self._skill_prompt(),
        )
        return await self._speak_with_search(
            prompt, ["core_position", "key_points", "main_risks", "initial_suggestion"], 1,
        )

    async def speak_round2(
        self, topic: str, round1_summary: str, moderator_summary: str
    ) -> AgentMessage:
        self._refresh_system_prompt()
        prompt = get_prompt("EXPERT_ROUND2").format(
            role_label=self.role_label,
            topic=topic,
            round1_summary=round1_summary,
            moderator_summary=moderator_summary,
            skill_prompt=self._skill_prompt(),
        )
        return await self._speak_with_search(
            prompt, ["new_points", "attacks", "preserved_points"], 2,
        )

    async def speak_round3(
        self,
        topic: str,
        round2_summary: str,
        moderator_summary: str,
        attacks_on_me: str,
    ) -> AgentMessage:
        self._refresh_system_prompt()
        prompt = get_prompt("EXPERT_ROUND3").format(
            role_label=self.role_label,
            topic=topic,
            round2_summary=round2_summary,
            moderator_summary=moderator_summary,
            agent_name=self.name,
            attacks_on_me=attacks_on_me,
            skill_prompt=self._skill_prompt(),
        )
        return await self._speak_with_search(
            prompt,
            ["strongest_attack_on_me", "accepted_criticisms", "revisions",
             "final_position", "preferred_solution"],
            3,
        )

    async def speak_round4(
        self, topic: str, focused_issues: str, compressed_summary: str
    ) -> AgentMessage:
        self._refresh_system_prompt()
        prompt = get_prompt("EXPERT_ROUND4").format(
            role_label=self.role_label,
            topic=topic,
            focused_issues=focused_issues,
            compressed_summary=compressed_summary,
            skill_prompt=self._skill_prompt(),
        )
        return await self._speak_with_search(
            prompt,
            ["focused_issue", "final_addition", "last_attack_or_defense", "closing_view"],
            4,
        )


class ModeratorAgent(BaseMeetingAgent):
    """The Committee Chair of the council meeting."""

    # Optional MCP gateway exposing connected MCP server tools (mcp_gateway.McpGateway)
    tool_gateway: Optional[Any] = None

    def __init__(
        self,
        provider: LLMProviderConfig,
        name: str = MODERATOR_DISPLAY_NAME,
    ):
        super().__init__(name, "Arbiter", self._build_system_prompt(), provider)

    def _build_system_prompt(self) -> str:
        base = (
            "You are the Committee Chair of a council meeting. Your duties are to guide the "
            "discussion, summarize each round, score every expert, and output a recommended "
            "solution at the end. Stay neutral and do not dive deep into domain analysis yourself."
        )
        if self.tool_gateway is not None and self.tool_gateway.has_tools():
            base += "\n\nYou also have access to MCP tools that you can use to look up information "
            base += "during the meeting. To call one, add a \"tool_call\" field to your JSON output: "
            base += '{"tool_call": {"server": "<server name>", "tool": "<tool name>", "arguments": { ... }}}'
        base += LANG_FOLLOW_INSTRUCTION
        return base

    async def opening(
        self,
        topic: str,
        expert_specs: str,
        goal: Optional[str],
        constraints: list,
        background: Optional[str],
        retitle: bool = True,
    ) -> str:
        goal_section = f"Goal: {goal}" if goal else ""
        constraints_section = (
            ("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
            if constraints
            else ""
        )
        background_section = f"Background: {background}" if background else ""

        prompt = get_prompt("MODERATOR_OPENING").format(
            topic=topic,
            expert_specs=expert_specs,
            expert_introduction=EXPERT_INTRODUCTION_RETITLE if retitle else EXPERT_INTRODUCTION_FIXED,
            goal_section=goal_section,
            constraints_section=constraints_section,
            background_section=background_section,
        )
        return await self._speak_with_tools(prompt, expected_keys=["opening_statement"])

    async def _speak_with_tools(self, prompt: str, expected_keys: list[str]) -> str:
        """Speak with optional MCP tool calls, returning the final LLM text."""
        max_tool_calls = 3
        tool_calls_done = 0
        while tool_calls_done < max_tool_calls:
            raw = await self._call_llm(prompt)
            # Try to extract a tool_call from the response
            try:
                parsed = safe_parse_agent_output(raw, expected_keys + ["tool_call"])
            except Exception:
                return raw
            if not isinstance(parsed, dict):
                return raw
            tool_call = parsed.pop("tool_call", None)
            if tool_call is None:
                return raw
            # Dispatch the MCP tool call
            if self.tool_gateway is None:
                return raw
            display = ""
            try:
                server_name, tool_name, arguments, display, _ = _build_tool_call_request(tool_call)
                result = await self.tool_gateway.call_tool(server_name, tool_name, arguments)
                tool_calls_done += 1
                prompt = prompt + (
                    f"\n\nYou called tool {display}. Result:\n{result.get('summary', '')}"
                    + (" [error]" if result.get("is_error") else "")
                    + "\nNow continue with your response."
                )
            except Exception as e:
                logger.warning("[%s] Moderator MCP call failed: %s", self.name, e)
                tool_calls_done += 1
                prompt = prompt + (
                    f"\n\nYou called tool {display or 'tool_call'}. Error: {e}"
                    + "\nNow continue with your response."
                )
        return raw

    async def summarize_and_score(
        self,
        round_index: int,
        topic: str,
        round_messages: str,
        previous_context: str = "",
    ) -> RoundSummary:
        prompt = get_prompt("MODERATOR_SUMMARY").format(
            round_index=round_index,
            topic=topic,
            round_messages=round_messages,
            previous_context=(
                f"\nContext from previous rounds:\n{previous_context}"
                if previous_context
                else ""
            ),
        )
        raw = await self._call_llm(prompt)
        parsed = parse_json_response(raw)

        if parsed is None:
            logger.warning("Failed to parse moderator summary, using raw text")
            return RoundSummary(
                round_index=round_index,
                new_valuable_ideas=[],
                strong_critiques=[],
                points_worth_preserving=[],
                scores=[],
                next_step="",
                should_continue=round_index < 3,
                raw_text=raw,
            )

        scores = []
        for s in parsed.get("scores", []):
            scores.append(
                RoundScore(
                    round_index=round_index,
                    agent_name=s.get("agent_name", ""),
                    novelty_score=int(s.get("novelty_score", 0)),
                    critique_score=int(s.get("critique_score", 0)),
                    comment=s.get("comment", ""),
                )
            )

        return RoundSummary(
            round_index=round_index,
            new_valuable_ideas=parsed.get("new_valuable_ideas", []),
            strong_critiques=parsed.get("strong_critiques", []),
            points_worth_preserving=parsed.get("points_worth_preserving", []),
            scores=scores,
            next_step=parsed.get("next_step", ""),
            should_continue=parsed.get("should_continue", round_index < 3),
            raw_text=raw,
        )

    async def finalize(self, topic: str, all_discussion: str) -> FinalReport:
        prompt = get_prompt("MODERATOR_FINAL").format(
            topic=topic, all_discussion=all_discussion
        )
        raw = await self._call_llm(prompt)
        parsed = parse_json_response(raw)

        if parsed is None:
            logger.warning("Failed to parse final report, using raw text")
            return FinalReport(
                problem_definition=topic,
                main_consensus=[],
                main_disagreements=[],
                recommended_solution="",
                why_this_solution="",
                preserved_minority_opinions=[],
                agent_contributions={},
                final_scores=[],
                raw_markdown=raw,
            )

        final_scores = []
        for s in parsed.get("final_scores", []):
            final_scores.append(
                FinalScore(
                    agent_name=s.get("agent_name", ""),
                    contribution_score=int(s.get("contribution_score", 0)),
                    summary=s.get("summary", ""),
                )
            )

        return FinalReport(
            problem_definition=parsed.get("problem_definition", ""),
            main_consensus=parsed.get("main_consensus", []),
            main_disagreements=parsed.get("main_disagreements", []),
            recommended_solution=parsed.get("recommended_solution", ""),
            why_this_solution=parsed.get("why_this_solution", ""),
            preserved_minority_opinions=parsed.get("preserved_minority_opinions", []),
            agent_contributions=parsed.get("agent_contributions", {}),
            final_scores=final_scores,
            raw_markdown=raw,
        )

