"""Tests for the MCP gateway (mcp_gateway.py) and the tool_call dispatch in agents.py.

The pure-logic tests run offline. One integration test exercises a live MCP
server (default http://127.0.0.1:8081/mcp, override with MCP_LIVE_TEST_URL)
and skips itself when the server is unreachable.
"""

import os
from types import SimpleNamespace

import pytest

from mcp_gateway import (
    McpGateway,
    McpServerConfig,
    _format_content,
    build_mcp_tools_prompt,
    is_mcp_available,
    load_mcp_servers,
)
from agents import _build_tool_call_request


def _tool(name, description="", schema=None):
    return SimpleNamespace(name=name, description=description, input_schema=schema)


def test_load_mcp_servers_parses_env(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_1_NAME", "financial")
    monkeypatch.setenv("MCP_SERVER_1_URL", "http://127.0.0.1:8081/mcp")
    monkeypatch.setenv("MCP_SERVER_1_API_KEY", "secret")
    monkeypatch.setenv("MCP_SERVER_1_TIMEOUT", "30")
    monkeypatch.setenv("MCP_SERVER_2_URL", "http://127.0.0.1:9000/mcp")

    servers = load_mcp_servers()

    assert len(servers) == 2
    assert servers[0].name == "financial"
    assert servers[0].url == "http://127.0.0.1:8081/mcp"
    assert servers[0].api_key == "secret"
    assert servers[0].timeout == 30
    assert servers[1].name == "mcp_2"  # falls back to a generated name
    assert servers[1].api_key is None
    assert servers[1].timeout == 60  # default


def test_load_mcp_servers_empty_when_unset(monkeypatch):
    for i in range(1, 10):
        monkeypatch.delenv(f"MCP_SERVER_{i}_URL", raising=False)
    assert load_mcp_servers() == []


def test_is_mcp_available_follows_env(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_1_URL", raising=False)
    assert is_mcp_available() is False
    monkeypatch.setenv("MCP_SERVER_1_URL", "http://x/mcp")
    assert is_mcp_available() is True


def test_build_mcp_tools_prompt_empty_when_no_tools():
    assert build_mcp_tools_prompt([]) == ""
    assert build_mcp_tools_prompt([("financial", [])]) == ""


def test_build_mcp_tools_prompt_renders_tools():
    tools = [
        _tool("get_current_time", "Returns the current date and time.",
              {"type": "object", "properties": {}}),
        _tool("get_quote", "Get a stock quote.",
              {"type": "object", "properties": {"symbol": {"type": "string", "description": "Ticker"}}}),
    ]
    prompt = build_mcp_tools_prompt([("financial", tools)])

    assert "### Server: financial" in prompt
    assert "get_current_time" in prompt
    assert "get_quote" in prompt
    assert '"tool_call"' in prompt
    assert "symbol (string): Ticker" in prompt
    # A tool with no arguments gets no "Arguments:" line.
    assert prompt.count("Arguments:") == 1


def test_format_content_extracts_text_and_error_flag():
    ok = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="2026-11-03T13:31:14")],
        is_error=False,
    )
    summary, is_error = _format_content(ok)
    assert is_error is False
    assert "2026-11-03T13:31:14" in summary

    err = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="boom")],
        is_error=True,
    )
    summary, is_error = _format_content(err)
    assert is_error is True
    assert summary == "boom"


def test_build_tool_call_request_valid():
    server, tool, args, display, key = _build_tool_call_request(
        {"server": "financial", "tool": "get_quote", "arguments": {"symbol": "AAPL"}}
    )
    assert server == "financial"
    assert tool == "get_quote"
    assert args == {"symbol": "AAPL"}
    assert display == "MCP:financial.get_quote"
    assert "financial" in key and "get_quote" in key


def test_build_tool_call_request_defaults_empty_arguments():
    _, _, args, _, _ = _build_tool_call_request({"server": "s", "tool": "t"})
    assert args == {}


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-dict",
        {"tool": "t"},                              # missing server
        {"server": "s"},                            # missing tool
        {"server": "s", "tool": "t", "arguments": [1, 2]},  # arguments not a dict
    ],
)
def test_build_tool_call_request_rejects_malformed(bad):
    with pytest.raises(ValueError):
        _build_tool_call_request(bad)


@pytest.mark.anyio
async def test_expert_can_call_mcp_tool(monkeypatch):
    """An expert routes a tool_call to the gateway and folds the result in."""
    from agents import ExpertAgent, MAX_AGENT_SEARCHES
    from config import LLMProviderConfig

    provider = LLMProviderConfig(name="t", base_url="", api_key="k", model="m")
    expert = ExpertAgent("Data Researcher", "Evidence", provider)

    calls: list[tuple] = []

    class FakeGateway:
        def catalog(self):
            return [("financial", [_tool("get_current_time", "Returns the current date and time.",
                                         {"type": "object", "properties": {}})])]

        async def call_tool(self, server, tool, arguments):
            calls.append((server, tool, arguments))
            return {"summary": "2026-11-03T13:31:14", "result_count": 1, "is_error": False}

    expert.tool_gateway = FakeGateway()

    # The skill prompt must advertise the MCP tool so the model knows it exists.
    assert "get_current_time" in expert._skill_prompt()

    responses = [
        '{"tool_call":{"server":"financial","tool":"get_current_time","arguments":{}},'
        '"core_position":"p1","key_points":["k1"],"main_risks":["r1"],"initial_suggestion":"s1"}',
        '{"core_position":"final","key_points":["kf"],"main_risks":["rf"],"initial_suggestion":"sf"}',
    ]

    async def fake_call_llm(prompt: str) -> str:
        return responses.pop(0)

    expert._call_llm = fake_call_llm  # type: ignore[method-assign]

    result = await expert.speak_round1(topic="What time is it?", opening="Opening")

    assert calls == [("financial", "get_current_time", {})]
    assert result.content["core_position"] == "final"
    assert result.search_info is not None
    assert result.search_info[0]["query"] == "MCP:financial.get_current_time"
    # The tool_call field must not leak into the final content.
    assert "tool_call" not in result.content


@pytest.mark.anyio
async def test_live_mcp_server_roundtrip():
    """Connect to a real MCP server, list tools, and call one. Skips if down."""
    url = os.environ.get("MCP_LIVE_TEST_URL", "http://127.0.0.1:8081/mcp")
    gateway = McpGateway([McpServerConfig(name="livetest", url=url, timeout=15)])
    await gateway.start()
    if not gateway.connected:
        pytest.skip(f"MCP live server not reachable at {url}")
    try:
        assert gateway.has_tools()
        catalog = dict(gateway.catalog())
        tool_names = [t.name for t in catalog["livetest"]]
        assert "get_current_time" in tool_names

        result = await gateway.call_tool("livetest", "get_current_time", {})
        assert result["is_error"] is False
        assert result["summary"]  # a non-empty timestamp string
    finally:
        await gateway.close()
