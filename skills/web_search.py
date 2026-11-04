"""Research skill — allows agents to use local news, web search, and page fetch.

Each agent may use this skill up to five times per round to:
- Support or verify their own arguments with facts
- Find evidence to critique others' arguments
- Gather recent local-news context and open-web fundamentals
"""

import logging
from typing import Optional

from search import (
    fetch_webpage_content,
    get_search_api_url,
    get_web_search_api_url,
    is_local_news_api_configured,
    tavily_search,
    web_search,
)

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if any research backend is configured."""
    return get_search_api_url() is not None or get_web_search_api_url() is not None


async def execute(
    backend: str = "local",
    query: str = "",
    url: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
    days: int = 3,
    max_results: int = 20,
    search_depth: str = "advanced",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    max_chars: int = 12000,
) -> dict:
    """Run a research operation and return results.

    Returns:
        Dict with 'results', 'answer', 'summary', and 'result_count'.
    """
    normalized_backend = (backend or "local").strip().lower()

    if normalized_backend == "local":
        actual_query = "" if query.upper() == "ALL" else query
        return await tavily_search(
            query=actual_query,
            category=category,
            source=source,
            days=days,
            max_results=max_results,
            topic="news",
        )

    if normalized_backend == "web":
        return await web_search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic="general",
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            days=days,
        )

    if normalized_backend == "fetch":
        target_url = (url or query).strip()
        return await fetch_webpage_content(target_url, max_chars=max_chars)

    raise ValueError(f"Unknown research backend: {backend}")


# English prompt instruction for the research skill.
# Keys: REFINE / DUPLICATE / KEYWORD_FIRST / ERROR / FINAL_ONLY
# (search-loop feedback in ExpertAgent._speak_with_search).
# The SKILL prompt is _SKILL_BASE plus a local-browse addendum when the local
# news database is configured.

_SKILL_BASE = """


## Available skill: research tools
You have up to 5 research tool calls available. The research tools come in three types:

1. News search (backend="local")
   Best for: recent news and market moves from the last few days.

2. Open-web search (backend="web")
   Best for: company announcements, financial statements, analyst ratings, fundamentals, and any fact beyond recent news.

3. Webpage fetch (backend="fetch")
   Best for: when you have already found a high-value URL via web search and need to read the page body.

If you need to look up the latest material to support an argument, verify a fact, or rebut someone's point, add an extra "search_query" field to your JSON output.
Write all search queries in the same language as the topic being discussed.
You can return a string or a JSON object:

1. News keyword search (string, default backend=local):
"search_query": "central bank rate decision"

2. Open-web search (object):
"search_query": {
    "backend": "web",
    "query": "Acme Corp Q3 earnings and analyst ratings",
    "include_domains": ["sec.gov", "reuters.com"],
    "max_results": 8
}

3. Webpage fetch (object):
"search_query": {
    "backend": "fetch",
    "url": "https://example.com/report"
}

Rules:
- At most 5 research tool calls per round
- The news query should be a short keyword or phrase; the open-web query can be a natural-language phrase
- Prefer backend="web" or backend="fetch" for company announcements, financial metrics, analyst ratings, or deep research
- If you already have a high-value webpage URL, do not keep broad-searching — fetch the page body directly with backend="fetch"
- If a search or fetch yields no new information, do not repeat the same parameters
- News search returns 20 items by default; open-web search is suggested at 8
- If the information is sufficient, output your final answer directly and do not add search_query
- If no search is needed, do NOT add the field
"""

_SKILL_LOCAL_BROWSE = """

The local news database also supports browsing recent news:

4. Local news browse (object):
"search_query": {
    "backend": "local",
    "query": "",
    "category": "finance",
    "days": 3,
    "max_results": 20
}

Available categories include: china_finance, finance, international, tech, defense, asia, europe, research, middle_east.
- The local-news keyword query must be a single word
- Use browse mode ("ALL" or query="") only after keyword misses or for a broad market overview
"""

_PROMPTS: dict[str, str] = {
    "REFINE": """
--- Search results {search_index}/{max_searches} ---
You previously asked to search for "{query}". Here is what was found:

{summary}

Based on the search results above, combined with your professional judgment and prior analysis, continue your answer.
- If the information is already sufficient, output your complete final answer directly and do not add a search_query field.
- If key information is still missing, you may continue adding new search_query values.
- Remaining searches: {remaining_searches}
- Do not repeat query/category/source combinations you have already searched.
""",
    "DUPLICATE": """
--- Search reminder ---
You just requested "{query}" again. The system will not run the same search a second time.
If you still need to fill gaps, use a new query/category/source; otherwise output your final answer directly.
Remaining searches: {remaining_searches}
""",
    "KEYWORD_FIRST": """
--- Search reminder ---
In this round, please do at least one keyword search first; do not start with ALL or an empty query to browse.
Give a single-word keyword first; if the keyword returns nothing, switch to browse mode.
Remaining searches: {remaining_searches}
""",
    "ERROR": """
--- Search failed ---
Your search for "{query}" failed: {error}
If you still need to fill gaps, use a new query/category/source; otherwise output your final answer directly.
Remaining searches: {remaining_searches}
""",
    "FINAL_ONLY": """
You have used up all search calls for this round. You must now output your final answer directly based on the results you already have, and do not add a search_query field.
""",
}


def get_prompt(name: str) -> str:
    """Return the research-skill prompt for *name*."""
    return _PROMPTS[name]


def get_skill_prompt() -> str:
    """Return the skill prompt text, including local-browse docs when available."""
    return _SKILL_BASE + (_SKILL_LOCAL_BROWSE if is_local_news_api_configured() else "")


def get_search_prompt(name: str) -> str:
    """Return a search-loop feedback prompt."""
    return get_prompt(name)
