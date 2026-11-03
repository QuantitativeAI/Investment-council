"""Search and fetch integration for local news and open-web research.

Supports:
- local Tavily-compatible news APIs via SEARCH_API_URL
- generic web search via Tavily (or WEB_SEARCH_API_URL)
- direct webpage fetching for deeper page-level research
"""

import logging
import os
import re
from html import unescape
from html.parser import HTMLParser
from tavily import AsyncTavilyClient
from typing import Any, Optional


import httpx

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"
DEFAULT_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)


def get_search_api_url() -> Optional[str]:
    """Return the configured search API URL, or None if no search backend is available."""
    url = os.environ.get("SEARCH_API_URL", "").strip()
    if url:
        return url
    # Fall back to Tavily if API key is set
    if get_tavily_api_key():
        return TAVILY_API_URL
    return None


def get_tavily_api_key() -> Optional[str]:
    return os.environ.get("TAVILY_API_KEY", "").strip() or None


def get_web_search_api_url() -> Optional[str]:
    """Return the configured open-web search API URL."""
    url = os.environ.get("WEB_SEARCH_API_URL", "").strip()
    if url:
        return url
    if get_tavily_api_key():
        return TAVILY_API_URL
    return None


def is_local_news_api_configured() -> bool:
    """True when SEARCH_API_URL points at the local news database."""
    url = os.environ.get("SEARCH_API_URL", "").strip()
    return bool(url) and _is_local_news_api(url)


def _tavily_base_url_from(url: Optional[str]) -> Optional[str]:
    """Derive a Tavily-client base URL from a configured search URL.

    The Tavily client appends ``/search`` to ``api_base_url`` itself, so a
    trailing ``/search`` on the configured URL is stripped. Returns ``None``
    (client default, api.tavily.com) when nothing is configured or the URL
    points at the local News API, which authenticates via the request body
    rather than the client's ``Authorization: Bearer`` header.
    """
    if not url or _is_local_news_api(url):
        return None
    base = url.rstrip("/")
    if base.endswith("/search"):
        base = base[: -len("/search")]
    return base or None


def parse_local_news_command(command: str) -> tuple[str, Optional[str], str]:
    """Parse a moderator browse/search command for the local news API.

    Supported commands:
    - one-word keyword, e.g. "A股"
    - ALL or RECENT
    - RECENT:<category>, e.g. RECENT:china_finance
    """
    normalized = command.strip()
    upper = normalized.upper()

    if upper in {"ALL", "RECENT"}:
        return "", None, "ALL"

    if upper.startswith("RECENT:"):
        category = normalized.split(":", 1)[1].strip().lower()
        return "", category or None, f"RECENT:{category}" if category else "ALL"

    return normalized, None, normalized


def parse_research_command(command: str) -> dict[str, Optional[str]]:
    """Parse moderator research commands across local news and open web.

    Supported commands:
    - local keyword: tariff (any short keyword in the topic's language)
    - local browse all: ALL / RECENT
    - local browse category: RECENT:china_finance
    - open web search: WEB:Acme Corp Q3 earnings and analyst ratings
    - webpage fetch: FETCH:https://example.com/report
    """
    normalized = command.strip()
    upper = normalized.upper()

    if upper.startswith("WEB:"):
        query = normalized.split(":", 1)[1].strip()
        return {
            "backend": "web",
            "query": query,
            "category": None,
            "url": None,
            "command_key": f"WEB:{query}",
            "display": f"WEB:{query}",
        }

    if upper.startswith("FETCH:"):
        url = normalized.split(":", 1)[1].strip()
        return {
            "backend": "fetch",
            "query": None,
            "category": None,
            "url": url,
            "command_key": f"FETCH:{url}",
            "display": f"FETCH:{url}",
        }

    query, category, command_key = parse_local_news_command(normalized)
    return {
        "backend": "local",
        "query": query,
        "category": category,
        "url": None,
        "command_key": command_key,
        "display": command_key,
    }


def _is_local_news_api(url: str) -> bool:
    return "192.168.3.89:4001" in url


def _get_recent_api_url(url: str) -> str:
    if url.endswith("/v1/search"):
        return url[:-len("/v1/search")] + "/recent"
    if url.endswith("/search"):
        return url[:-len("/search")] + "/recent"
    return url.rstrip("/") + "/recent"


def _normalize_recent_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": item.get("title", "Untitled"),
        "url": item.get("link", ""),
        "content": item.get("summary", ""),
        "score": item.get("score", 0.0),
        "raw_content": item.get("content"),
        "published": item.get("published"),
        "source": item.get("source"),
        "category": item.get("category"),
    }


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor for fetched pages."""

    _BLOCK_TAGS = {
        "article", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "footer", "li", "main", "p", "section", "table", "td", "th", "tr",
    }
    _IGNORE_TAGS = {"script", "style", "noscript", "svg", "iframe"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._ignore_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs):
        if tag in self._IGNORE_TAGS:
            self._ignore_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in self._IGNORE_TAGS and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._ignore_depth:
            return
        text = re.sub(r"\s+", " ", unescape(data)).strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self._parts.append(text)

    def get_text(self) -> str:
        text = " ".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def _extract_html_text(html: str) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    title = parser.title.strip()
    text = parser.get_text()
    return title, text


def _format_search_response(results: list[dict[str, Any]], answer: str = "") -> dict:
    lines = []
    if answer:
        lines.append(f"## Search Summary\n{answer}\n")
    lines.append(f"## Search Results ({len(results)} total)\n")
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        result_url = result.get("url", "")
        content = result.get("content", "")
        lines.append(f"### {i}. {title}")
        if result_url:
            lines.append(f"Source: {result_url}")
        if content:
            lines.append(content)
        lines.append("")

    return {
        "results": results,
        "answer": answer,
        "summary": "\n".join(lines),
        "result_count": len(results),
    }


def _format_fetch_response(url: str, title: str, content: str) -> dict:
    display_title = title or url
    summary = "\n".join([
        "## Webpage Fetch Result",
        f"Title: {display_title}",
        f"Source: {url}",
        "",
        content,
    ])
    return {
        "results": [{"title": display_title, "url": url, "content": content}],
        "answer": "",
        "summary": summary,
        "result_count": 1,
    }


async def browse_recent_news(
    max_results: int = 20,
    category: Optional[str] = None,
    source: Optional[str] = None,
    days: int = 3,
) -> dict:
    """Browse latest news from the local News API via /recent."""
    url = get_search_api_url()
    if not url:
        raise ValueError("No search backend configured. Set SEARCH_API_URL or TAVILY_API_KEY in .env")
    if not _is_local_news_api(url):
        raise ValueError("Recent-news browse mode is only supported by the local News API")

    params: dict[str, Any] = {"limit": max_results}
    if category:
        params["category"] = category
    if source:
        params["source"] = source
    if days > 0:
        params["hours"] = days * 24

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(_get_recent_api_url(url), params=params)
        resp.raise_for_status()
        data = resp.json()

    results = [_normalize_recent_result(item) for item in data.get("results", [])]
    return _format_search_response(results)


async def web_search(
    query: str,
    max_results: int = 8,
    search_depth: str = "advanced",
    topic: str = "general",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    days: int = 7,
) -> dict:
    """Search the open web via Tavily or another generic web-search backend."""
    url = get_web_search_api_url()
    if not url:
        raise ValueError("No open-web search backend configured. Set TAVILY_API_KEY or WEB_SEARCH_API_URL in .env")

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Web search requires a non-empty query")

    if _tavily_base_url_from(url) is not None:
        # Tavily (or Tavily-compatible) endpoint: use the official client,
        # which sends the API key as a proper credential.
        client = AsyncTavilyClient(
            api_key=api_key or get_tavily_api_key(),
            api_base_url=_tavily_base_url_from(url),
        )
        kwargs: dict[str, Any] = {
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": True,
        }
        if topic == "news" and days > 0:
            kwargs["days"] = days
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        try:
            response = await client.search(query=normalized_query, **kwargs)
        except Exception as e:
            logger.error("Tavily web search failed: %s", e)
            raise
        return _format_search_response(response.get("results", []), response.get("answer", ""))

    # Custom Tavily-compatible backend: raw request.
    payload: dict[str, Any] = {
        "query": normalized_query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "include_answer": True,
        "include_raw_content": True,
    }
    if topic == "news" and days > 0:
        payload["days"] = days
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    key = api_key or get_tavily_api_key()
    if key:
        payload["api_key"] = key

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return _format_search_response(data.get("results", []), data.get("answer", ""))


async def fetch_webpage_content(url: str, max_chars: int = 12000) -> dict:
    """Fetch a webpage directly and extract readable text content."""
    target_url = url.strip()
    if not target_url:
        raise ValueError("Webpage fetch requires a URL")

    headers = {"User-Agent": DEFAULT_FETCH_USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
        resp = await client.get(target_url)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        text = resp.text

    if "html" in content_type or "<html" in text[:500].lower():
        title, extracted = _extract_html_text(text)
    else:
        title = target_url
        extracted = text

    cleaned = re.sub(r"\n{3,}", "\n\n", extracted).strip()
    if max_chars > 0:
        cleaned = cleaned[:max_chars]
    return _format_fetch_response(target_url, title, cleaned)


async def tavily_search(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
    topic: str = "general",
    days: int = 3,
    api_key: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """Search using Tavily-compatible API.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 10).
        search_depth: "basic" or "advanced" (default "advanced").
        topic: "general" or "news" (default "general").
        days: Search within the last N days (default 3).
        api_key: Tavily API key. Falls back to TAVILY_API_KEY env var.
        category: News category filter (e.g. china_finance, tech).
        source: News source filter.

    Returns:
        Dict with 'results' list and formatted 'summary' string.
    """
    url = get_search_api_url()
    if not url:
        raise ValueError("No search backend configured. Set SEARCH_API_URL or TAVILY_API_KEY in .env")

    normalized_query = query.strip()
    if _is_local_news_api(url) and not normalized_query:
        return await browse_recent_news(
            max_results=max_results,
            category=category,
            source=source,
            days=days,
        )

    if _tavily_base_url_from(url) is not None:
        # Tavily (or Tavily-compatible) endpoint: use the official client,
        # which sends the API key as a proper credential.
        client = AsyncTavilyClient(
            api_key=api_key or get_tavily_api_key(),
            api_base_url=_tavily_base_url_from(url),
        )
        kwargs: dict[str, Any] = {
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "include_answer": True,
        }
        if topic == "news" and days > 0:
            kwargs["days"] = days
        try:
            response = await client.search(query=normalized_query, **kwargs)
        except Exception as e:
            logger.error("Tavily search failed: %s", e)
            raise
        return _format_search_response(response.get("results", []), response.get("answer", ""))

    # Local news API: raw request with body-based auth and local filters.
    payload = {
        "query": normalized_query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "days": days,
        "include_answer": True,
    }
    if category:
        payload["category"] = category
    if source:
        payload["source"] = source

    # Include api_key only when explicitly provided (body-based auth).
    key = api_key or get_tavily_api_key()
    if key:
        payload["api_key"] = key

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return _format_search_response(data.get("results", []), data.get("answer", ""))

async def search_for_topic(topic: str, api_key: Optional[str] = None) -> str:
    """Convenience function: search for a topic and return a formatted background string."""
    if not get_search_api_url():
        raise ValueError("No search backend configured. Set SEARCH_API_URL or TAVILY_API_KEY in .env")

    result = await tavily_search(query=topic, api_key=api_key, max_results=10)
    return result.get("summary", "")


def build_research_prompt(has_local_news: bool) -> tuple[str, str]:
    """Build ``(system_prompt, instructions)`` for the pre-meeting research phase.

    The prompt adapts to the configured backends: with the local news database
    available the chair gets keyword/browse/category modes; otherwise it only
    gets news-keyword, ``WEB:`` and ``FETCH:`` modes.
    """
    language_rule = "Write all search queries in the same language as the user's topic.\n"
    if has_local_news:
        system = (
            "You are a research assistant. You have five modes: 1) one-word local news keyword search; "
            "2) browse all recent news with ALL; 3) browse recent news by category with RECENT:category; "
            "4) open-web search with WEB:query; 5) webpage fetch with FETCH:url. "
            "ALL / RECENT:* go to the local /recent API, WEB: goes to open-web search, and FETCH: reads the page body. "
            "Do not repeat commands. Use WEB: or FETCH: when you need announcements or fundamentals beyond the local database."
        )
        instructions = (
            "\nPlease decide whether more searching is needed. Local search is unlimited — search as much as you need.\n"
            "If information is sufficient, reply: DONE\n"
            "If more search is needed, reply in exactly one of these forms: 1) one local-news keyword; 2) ALL; 3) RECENT:category; 4) WEB:search terms; 5) FETCH:url. Do not reply with sentences.\n"
            "Available categories include: china_finance, finance, international, tech, defense, asia, europe, research, middle_east.\n"
            + language_rule
            + "Important: do not repeat any command that has already been executed. If ALL or RECENT:china_finance has already been used, switch to another category or reply DONE.\n"
            "Strategy: use local-news keywords for recent market news; use WEB: for fundamentals, announcements, and research; when you already have a valuable URL, use FETCH:url to read the page.\n"
            "Start with 1-2 keyword searches by default. Do not use ALL or RECENT first. Local browse mode should be used only after local keyword misses or for broad market overview; WEB: may be used directly when external information is clearly needed."
        )
    else:
        system = (
            "You are a research assistant. You have three modes: 1) news search with a short keyword or phrase; "
            "2) open-web search with WEB:query; 3) webpage fetch with FETCH:url. "
            "News mode searches recent news, WEB: searches the open web, and FETCH: reads the page body of a URL. "
            "Do not repeat commands. Use WEB: or FETCH: when you need facts beyond recent news."
        )
        instructions = (
            "\nPlease decide whether more searching is needed. Search as much as you need.\n"
            "If information is sufficient, reply: DONE\n"
            "If more search is needed, reply in exactly one of these forms: 1) a news keyword or short phrase; 2) WEB:search terms; 3) FETCH:url. Do not reply with sentences.\n"
            + language_rule
            + "Important: do not repeat any command that has already been executed.\n"
            "Strategy: use news keywords for recent market and company news; use WEB: for fundamentals, announcements, financial data, and research reports; when you already have a valuable URL, use FETCH:url to read the page.\n"
            "Start with 1-2 keyword searches by default."
        )
    return system, instructions

