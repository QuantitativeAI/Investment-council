"""MCP (Model Context Protocol) client gateway.

Connects to one or more remote MCP servers over the Streamable HTTP
transport and exposes their tools to expert agents.

Configuration (read from the environment after ``load_dotenv``), mirroring
the ``LLM_PROVIDER_N_*`` numbering scheme::

    MCP_SERVER_1_NAME=financial_data
    MCP_SERVER_1_URL=http://127.0.0.1:8081/mcp
    MCP_SERVER_1_API_KEY=...            # optional -> "Authorization: Bearer"
    MCP_SERVER_1_TIMEOUT=60             # optional, seconds (default 60)

A server is recognized as configured when its ``_URL`` is set. Servers are
numbered ``MCP_SERVER_1``, ``MCP_SERVER_2``, ...

The :class:`McpGateway` is meant to live for the duration of one meeting:
call :meth:`McpGateway.start` before the meeting and :meth:`McpGateway.close`
afterwards (a ``finally`` block). A server that fails to connect is skipped
with a warning so one bad server cannot take down the others or the meeting.
"""

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MCP_TIMEOUT = 60


@dataclass
class McpServerConfig:
    """Configuration for a single MCP server."""

    name: str
    url: str
    api_key: Optional[str] = None
    timeout: int = DEFAULT_MCP_TIMEOUT


def load_mcp_servers() -> List[McpServerConfig]:
    """Load MCP server configs from ``MCP_SERVER_N_*`` env vars.

    Returns an empty list when no server is configured.
    """
    servers: List[McpServerConfig] = []
    for i in range(1, 100):
        prefix = f"MCP_SERVER_{i}"
        url = os.environ.get(f"{prefix}_URL", "").strip()
        if not url:
            break
        name = os.environ.get(f"{prefix}_NAME", "").strip() or f"mcp_{i}"
        try:
            timeout = int(os.environ.get(f"{prefix}_TIMEOUT", "") or DEFAULT_MCP_TIMEOUT)
        except ValueError:
            timeout = DEFAULT_MCP_TIMEOUT
        servers.append(
            McpServerConfig(
                name=name,
                url=url,
                api_key=os.environ.get(f"{prefix}_API_KEY", "").strip() or None,
                timeout=timeout,
            )
        )
    return servers


def is_mcp_available() -> bool:
    """True when at least one MCP server is configured in the environment."""
    return bool(load_mcp_servers())


def _format_content(result: Any) -> Tuple[str, bool]:
    """Extract a text summary and error flag from a ``CallToolResult``."""
    texts: List[str] = []
    for block in getattr(result, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            texts.append(getattr(block, "text", "") or "")
        else:
            texts.append(f"[{btype or 'non-text'} content]")
    summary = "\n".join(t for t in texts if t).strip()
    is_error = bool(getattr(result, "is_error", False))
    return summary, is_error


class McpGateway:
    """Manages connections to MCP servers for the lifetime of one meeting."""

    def __init__(self, servers: List[McpServerConfig]):
        self.servers = servers
        # name -> {"http_client", "client_cm", "session_cm", "session"}
        self._live: dict[str, dict[str, Any]] = {}
        # name -> list[Tool]
        self._tools: dict[str, list] = {}

    @property
    def connected(self) -> List[str]:
        """Names of servers currently connected."""
        return list(self._live.keys())

    def has_tools(self) -> bool:
        """True when at least one connected server exposes a tool."""
        return any(len(self._tools.get(name, [])) for name in self.connected)

    async def start(self) -> None:
        """Connect to every configured server and list its tools."""
        for server in self.servers:
            try:
                await self._connect_one(server)
            except Exception as e:  # noqa: BLE001 - one bad server must not kill the rest
                logger.warning("MCP server '%s' (%s) unavailable: %s", server.name, server.url, e)

    async def _connect_one(self, server: McpServerConfig) -> None:
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {}
        if server.api_key:
            headers["Authorization"] = f"Bearer {server.api_key}"

        # We create (and therefore own) the HTTP client so we can pass auth
        # headers; the transport will not close a client we supply.
        http_client = httpx2.AsyncClient(headers=headers or None, timeout=server.timeout)
        client_cm = streamable_http_client(server.url, http_client=http_client)
        read_stream, write_stream = await client_cm.__aenter__()
        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()
        try:
            await session.initialize()
            tools_result = await session.list_tools()
        except Exception:
            with contextlib.suppress(Exception):
                await session_cm.__aexit__(None, None, None)
            with contextlib.suppress(Exception):
                await client_cm.__aexit__(None, None, None)
            with contextlib.suppress(Exception):
                await http_client.aclose()
            raise

        self._live[server.name] = {
            "http_client": http_client,
            "client_cm": client_cm,
            "session_cm": session_cm,
            "session": session,
        }
        self._tools[server.name] = list(getattr(tools_result, "tools", None) or [])
        logger.info("MCP server '%s': %d tool(s)", server.name, len(self._tools[server.name]))

    def catalog(self) -> List[Tuple[str, list]]:
        """Return ``(server_name, [Tool, ...])`` for each connected server."""
        return [(name, self._tools.get(name, [])) for name in self.connected]

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        """Call a tool on a connected server.

        Returns a dict compatible with the research-skill result shape
        (``summary`` / ``result_count``) plus an ``is_error`` flag.
        """
        entry = self._live.get(server_name)
        if entry is None:
            raise ValueError(f"MCP server is not connected: {server_name}")
        session = entry["session"]
        result = await session.call_tool(tool_name, arguments or {})
        summary, is_error = _format_content(result)
        return {
            "summary": summary or "(no text returned)",
            "result_count": 0 if is_error else 1,
            "is_error": is_error,
        }

    async def close(self) -> None:
        """Tear down all connections. Safe to call multiple times."""
        for entry in list(self._live.values()):
            for key in ("session_cm", "client_cm"):
                cm = entry.get(key)
                if cm is not None:
                    with contextlib.suppress(Exception, RuntimeError):
                        await cm.__aexit__(None, None, None)
            http_client = entry.get("http_client")
            if http_client is not None:
                with contextlib.suppress(Exception):
                    await http_client.aclose()
        self._live.clear()
        # Yield control briefly to let any MCP SDK background tasks settle.
        # The streamable_http client spawns background tasks that can raise
        # "cancel scope in a different task" errors if the event loop is not
        # given a chance to process them after the context manager exits.
        await asyncio.sleep(0)
        await asyncio.sleep(0)


def _describe_arguments(input_schema: Any) -> Optional[str]:
    """Render a compact one-line description of a tool's argument schema."""
    if not isinstance(input_schema, dict):
        return None
    props = input_schema.get("properties", {})
    if not isinstance(props, dict) or not props:
        return None
    parts: List[str] = []
    for arg_name, arg_schema in props.items():
        if not isinstance(arg_schema, dict):
            parts.append(str(arg_name))
            continue
        atype = arg_schema.get("type", "any")
        if isinstance(atype, list):
            atype = "/".join(str(x) for x in atype)
        adesc = arg_schema.get("description", "")
        parts.append(f"{arg_name} ({atype})" + (f": {adesc}" if adesc else ""))
    return ", ".join(parts)


def build_mcp_tools_prompt(tools_by_server: List[Tuple[str, list]]) -> str:
    """Render the "Available tools: MCP servers" section of the skill prompt.

    Returns an empty string when no connected server exposes any tool.
    """
    entries = [(server, tools) for server, tools in tools_by_server if tools]
    if not entries:
        return ""

    lines = [
        "",
        "",
        "## Available tools: MCP servers",
        'You also have access to tools from connected MCP servers. To call one, add a "tool_call" field to your JSON output:',
        '  "tool_call": { "server": "<server name>", "tool": "<tool name>", "arguments": { ... } }',
        'Match "arguments" exactly to the argument schema listed below. MCP tool calls count toward the same per-round tool budget as research searches.',
        "",
    ]
    for server_name, tools in entries:
        lines.append(f"### Server: {server_name}")
        for tool in tools:
            desc = " ".join((getattr(tool, "description", "") or "").split())
            lines.append(f"- **{tool.name}**: {desc}")
            args = _describe_arguments(getattr(tool, "input_schema", None))
            if args:
                lines.append(f"  Arguments: {args}")
        lines.append("")
    return "\n".join(lines)
