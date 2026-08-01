from __future__ import annotations
import asyncio
from types import SimpleNamespace
from contextlib import asynccontextmanager

import pytest


pytest.importorskip("mcp.types")

from mcp.types import Tool, TextContent, CallToolResult

from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    RemoteToolCallResponse,
    RevocableMcpToolArguments,
    McpToolArgumentsRevokedError,
    validate_http_mcp_url,
    build_namespaced_tool_name,
)


def test_validate_http_mcp_url_rejects_non_http_scheme() -> None:
    """Only HTTP(S) MCP endpoints are supported."""
    with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
        validate_http_mcp_url("stdio://local-server")


def test_validate_http_mcp_url_rejects_non_local_plain_http() -> None:
    """Remote servers must use HTTPS unless they are local development endpoints."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_http_mcp_url("http://example.com/mcp")


def test_build_namespaced_tool_name_normalizes_tool_segment() -> None:
    """Remote tool names are normalized into app-safe tool IDs."""
    assert build_namespaced_tool_name("gradio_docs", "search-docs") == "gradio_docs__search_docs"


def test_remote_tool_spec_translates_to_function_spec() -> None:
    """Discovered MCP tools should translate into app function specs."""
    tool = Tool(
        name="search-docs",
        description="Search the docs",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    spec = RemoteToolSpec.from_mcp_tool("gradio_docs", tool)

    assert spec.remote_name == "search-docs"
    assert spec.namespaced_name == "gradio_docs__search_docs"
    assert spec.to_function_spec() == {
        "type": "function",
        "name": "gradio_docs__search_docs",
        "description": "Search the docs",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def test_remote_tool_error_result_maps_to_app_payload() -> None:
    """Remote tool errors should remain visible after response mapping."""
    result = CallToolResult(
        content=[TextContent(type="text", text="Search backend unavailable")],
        structuredContent=None,
        isError=True,
    )

    payload = RemoteToolCallResponse.from_call_tool_result(
        server_alias="gradio_docs",
        remote_tool_name="search-docs",
        result=result,
    ).to_tool_result()

    assert payload["status"] == "error"
    assert payload["namespaced_tool_name"] == "gradio_docs__search_docs"
    assert payload["text"] == "Search backend unavailable"


@pytest.mark.asyncio
async def test_revoked_private_arguments_clear_cancellation_resistant_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late transport retains only the shared empty map and cannot return a result."""
    namespaced_name = "official_search__search_web"
    spec = RemoteToolSpec(
        server_alias="official_search",
        remote_name="search_web",
        namespaced_name=namespaced_name,
        description="Search",
        parameters_schema={"type": "object"},
    )
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(alias="official_search", url="https://example.com/mcp/"),
        known_tools=[spec],
    )
    started = asyncio.Event()
    release = asyncio.Event()
    captured: dict[str, object] = {}

    async def call_tool(_name: str, *, arguments: dict[str, object], **_kwargs: object) -> CallToolResult:
        captured["arguments"] = arguments
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return CallToolResult(content=[], structuredContent={"query": "private query", "results": []})

    session = SimpleNamespace(call_tool=call_tool)

    @asynccontextmanager
    async def fake_session():
        yield session

    monkeypatch.setattr(client, "_session", fake_session)
    owned_arguments: dict[str, object] = {"query": "private query", "nested": ["canary"]}
    private_arguments = RevocableMcpToolArguments(owned_arguments)
    task = asyncio.create_task(client.call_tool(namespaced_name, private_arguments))
    await started.wait()

    task.cancel()
    private_arguments.revoke()

    assert captured["arguments"] is owned_arguments
    assert owned_arguments == {}
    release.set()
    with pytest.raises(McpToolArgumentsRevokedError):
        await task


@pytest.mark.asyncio
async def test_private_tool_result_does_not_copy_unbounded_content_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The private path forwards only the structured result reference for bounded validation."""
    namespaced_name = "official_search__search_web"
    spec = RemoteToolSpec(
        server_alias="official_search",
        remote_name="search_web",
        namespaced_name=namespaced_name,
        description="Search",
        parameters_schema={"type": "object"},
    )
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(alias="official_search", url="https://example.com/mcp/"),
        known_tools=[spec],
    )

    class ContentCopyCanary:
        def model_dump(self, **_kwargs: object) -> dict[str, object]:
            raise AssertionError("private content block was copied")

    structured_content = {"query": "private query", "results": []}

    async def call_tool(_name: str, **_kwargs: object) -> object:
        return SimpleNamespace(
            content=[ContentCopyCanary()],
            structuredContent=structured_content,
            isError=False,
        )

    @asynccontextmanager
    async def fake_session():
        yield SimpleNamespace(call_tool=call_tool)

    monkeypatch.setattr(client, "_session", fake_session)
    payload = await client.call_tool(
        namespaced_name,
        RevocableMcpToolArguments({"query": "private query"}),
    )

    assert payload == {
        "server_alias": "official_search",
        "remote_tool_name": "search_web",
        "namespaced_tool_name": namespaced_name,
        "status": "ok",
        "structured_content": structured_content,
    }
