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
# Keys: SKILL (appended to expert prompts), REFINE / DUPLICATE / KEYWORD_FIRST /
# ERROR / FINAL_ONLY (search-loop feedback in ExpertAgent._speak_with_search).

_PROMPTS: dict[str, str] = {
    "SKILL": """


## Available skill: research tools
You have up to 5 research tool calls available. The research tools come in three types:

1. Local news database (backend="local")
   Best for: finance news from the last 1-3 days, market moves, policy news, industry briefs.

2. Open-web search (backend="web")
   Best for: Eastmoney, Xueqiu, Sina Finance, company announcements, broker research reports, fundamentals, supply-chain analysis.

3. Webpage fetch (backend="fetch")
   Best for: when you have already found a high-value URL via web search and need to read the page body.

If you need to look up the latest material to support an argument, verify a fact, or rebut someone's point, add an extra "search_query" field to your JSON output.
You can return a string or a JSON object:

1. Local news keyword (string, default backend=local):
"search_query": "tariff"

2. Local news browse (object):
"search_query": {
    "backend": "local",
    "query": "",
    "category": "china_finance",
    "days": 3,
    "max_results": 20
}

3. Open-web search (object):
"search_query": {
    "backend": "web",
    "query": "Pengding Holdings Shiyun Circuit Eastmoney March 2026",
    "include_domains": ["eastmoney.com", "sina.com.cn", "xueqiu.com"],
    "max_results": 8
}

4. Webpage fetch (object):
"search_query": {
    "backend": "fetch",
    "url": "https://caifuhao.eastmoney.com/news/..."
}

Rules:
- At most 5 research tool calls per round
- Do keyword search first by default; do not start with ALL or an empty query to browse local news
- The local-news query must be a single word; the open-web query can be a natural-language phrase
- Prefer backend="web" or backend="fetch" for stock investment value, company announcements, financial metrics, broker ratings, or Eastmoney/Xueqiu pages
- Use local browse mode (ALL or query="") only when local keywords return nothing or you need the full market picture
- If you already have a high-value webpage URL, do not keep broad-searching — fetch the page body directly with backend="fetch"
- If browsing a category or fetching a URL yields no new information, do not repeat the same parameters
- Local news returns 20 items by default; open-web search is suggested at 8
- If the information is sufficient, output your final answer directly and do not add search_query
- If no search is needed, do NOT add the field
""",
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
    """Return the skill prompt text."""
    return get_prompt("SKILL")


def get_search_prompt(name: str) -> str:
    """Return a search-loop feedback prompt."""
    return get_prompt(name)
