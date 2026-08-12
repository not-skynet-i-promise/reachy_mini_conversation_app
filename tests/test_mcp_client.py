from __future__ import annotations
import time
import asyncio
from types import SimpleNamespace
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("mcp.types")

from mcp.types import Tool, Prompt, TextContent, PromptMessage, CallToolResult

from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    McpToolTimeoutError,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    RemoteToolCallResponse,
    RevocableMcpToolResult,
    RevocableMcpToolArguments,
    validate_http_mcp_url,
    build_namespaced_tool_name,
)


def test_revocable_private_result_scrubs_shared_map_and_rejects_late_capture() -> None:
    """Every holder of the shared result map sees revocation before later work can retain it."""
    result_lease = RevocableMcpToolResult()
    raw_result = {"query": "private query", "results": [{"snippet": "private result"}]}

    assert result_lease.capture(raw_result) is raw_result
    result_lease.revoke()

    assert result_lease.revoked
    assert raw_result == {}
    late_result = {"query": "late private query"}
    assert result_lease.capture(late_result) == {"error": "Remote tool unavailable"}
    assert late_result == {}


def test_validate_http_mcp_url_rejects_non_http_scheme() -> None:
    """Only HTTP(S) MCP endpoints are supported."""
    with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
        validate_http_mcp_url("stdio://local-server")


def test_validate_http_mcp_url_rejects_non_local_plain_http() -> None:
    """Remote servers must use HTTPS unless they are local development endpoints."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_http_mcp_url("http://example.com/mcp")


@pytest.mark.parametrize(
    "url",
    (
        "http://user:secret@localhost:9123/mcp",
        "http://localhost:9123/mcp?access_token=secret",
        "http://localhost:9123/mcp#secret",
        " http://localhost:9123/mcp",
    ),
)
def test_validate_http_mcp_url_rejects_credential_bearing_or_ambiguous_urls(url: str) -> None:
    """A generic MCP URL cannot persist credentials or alternate request targets."""
    with pytest.raises(ValueError):
        validate_http_mcp_url(url)


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


@pytest.mark.parametrize(
    "schema",
    (
        {"type": "string"},
        {"type": "object", "properties": []},
        {"type": "object", "properties": {}, "required": ["missing"]},
        {"type": "object", "properties": {}, "minimum": float("nan")},
        {"type": "object", "properties": {"target": {"type": "unknown"}}},
        {"type": "object", "properties": {"target": {"enum": ["kitchen"]}}},
        {"type": "object", "properties": {"target": {"type": "object"}}},
        {"type": "object", "properties": {"targets": {"type": "array"}}},
        {
            "type": "object",
            "properties": {"targets": {"type": "array", "items": {"type": "object"}}},
        },
    ),
)
def test_remote_tool_spec_rejects_non_object_or_malformed_schemas(schema: dict[str, object]) -> None:
    """Only bounded JSON object arguments may enter the function-tool registry."""
    with pytest.raises(ValueError):
        RemoteToolSpec(
            server_alias="home_assistant",
            remote_name="HassTurnOn",
            namespaced_name="home_assistant__HassTurnOn",
            description="Turn on one target",
            parameters_schema=schema,
        )


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
async def test_discover_catalog_requires_exact_no_argument_text_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic provisioning should cache one exact prompt and the complete tool catalog atomically."""
    client = RemoteMcpToolClient(RemoteMcpServerConfig(alias="home_assistant", url="http://127.0.0.1:9123/mcp"))
    tool = Tool(name="HassTurnOff", description="Turn off exposed devices", inputSchema={"type": "object"})
    session = SimpleNamespace(
        list_prompts=AsyncMock(
            return_value=SimpleNamespace(
                prompts=[Prompt(name="assist", description="Assist prompt")],
                nextCursor=None,
            )
        ),
        get_prompt=AsyncMock(
            return_value=SimpleNamespace(
                messages=[
                    PromptMessage(
                        role="assistant",
                        content=TextContent(type="text", text="Control only exposed Home Assistant devices."),
                    )
                ]
            )
        ),
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[tool], nextCursor=None)),
    )

    @asynccontextmanager
    async def fake_session():
        yield session

    monkeypatch.setattr(client, "_session", fake_session)
    catalog = await client.discover_catalog("assist")

    assert catalog.prompt_text == "Control only exposed Home Assistant devices."
    assert [spec.namespaced_name for spec in catalog.tools] == ["home_assistant__HassTurnOff"]
    session.get_prompt.assert_awaited_once_with("assist")


@pytest.mark.asyncio
async def test_catalog_pagination_rejects_cursor_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cyclic catalog cannot hold provisioning in an unbounded discovery loop."""
    client = RemoteMcpToolClient(RemoteMcpServerConfig(alias="home_assistant", url="http://127.0.0.1:9123/mcp"))
    session = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=[], nextCursor="same"),
                SimpleNamespace(tools=[], nextCursor="same"),
            ]
        )
    )

    with pytest.raises(ValueError, match="pagination"):
        await client._list_all_tools(session)
    assert session.list_tools.await_count == 2


@pytest.mark.asyncio
async def test_mcp_transport_never_inherits_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed loopback private endpoint cannot be diverted through HTTP proxy environment."""
    import reachy_mini_conversation_app.mcp_client as mcp_client_mod

    captured: dict[str, object] = {}

    class HttpClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> HttpClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    class Transport:
        async def __aenter__(self) -> tuple[object, object, None]:
            return object(), object(), None

        async def __aexit__(self, *_args: object) -> None:
            return None

    class Session:
        def __init__(self, *_args: object) -> None:
            pass

        async def __aenter__(self) -> Session:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def initialize(self) -> None:
            return None

    monkeypatch.setattr(
        mcp_client_mod,
        "_load_httpx",
        lambda: SimpleNamespace(AsyncClient=HttpClient),
    )
    monkeypatch.setattr(
        mcp_client_mod,
        "_load_mcp_sdk",
        lambda: (Session, lambda *_args, **_kwargs: Transport()),
    )
    client = RemoteMcpToolClient(RemoteMcpServerConfig(alias="home_assistant", url="http://127.0.0.1:9123/mcp"))

    async with client._session():
        pass

    assert captured["follow_redirects"] is False
    assert captured["trust_env"] is False


@pytest.mark.asyncio
async def test_tool_timeout_covers_connection_initialization_and_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The action deadline starts before session connection/initialization work."""
    spec = RemoteToolSpec(
        server_alias="home_assistant",
        remote_name="HassTurnOn",
        namespaced_name="home_assistant__HassTurnOn",
        description="Turn on one target",
        parameters_schema={"type": "object"},
    )
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="home_assistant",
            url="http://127.0.0.1:9123/mcp",
            request_timeout_s=1,
            tool_timeout_s=0.02,
        ),
        known_tools=[spec],
    )
    tool_called = False

    async def call_tool(*_args: object, **_kwargs: object) -> CallToolResult:
        nonlocal tool_called
        tool_called = True
        return CallToolResult(content=[])

    @asynccontextmanager
    async def slow_session():
        await asyncio.sleep(0.2)
        yield SimpleNamespace(call_tool=call_tool)

    monkeypatch.setattr(client, "_session", slow_session)
    started = time.monotonic()
    with pytest.raises(McpToolTimeoutError):
        await client.call_tool("home_assistant__HassTurnOn", {})

    assert time.monotonic() - started < 0.1
    assert tool_called is False


@pytest.mark.asyncio
async def test_tool_timeout_revokes_private_arguments_before_stubborn_initialization_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deadline returns promptly and removes authority from a cancellation-resistant setup."""
    name = "home_assistant__HassTurnOn"
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias="home_assistant",
            url="http://127.0.0.1:9123/mcp",
            tool_timeout_s=0.02,
        ),
        known_tools=[
            RemoteToolSpec(
                server_alias="home_assistant",
                remote_name="HassTurnOn",
                namespaced_name=name,
                description="Turn on one target",
                parameters_schema={"type": "object"},
            )
        ],
    )
    release = asyncio.Event()
    finished = asyncio.Event()
    tool_called = False

    async def call_tool(*_args: object, **_kwargs: object) -> CallToolResult:
        nonlocal tool_called
        tool_called = True
        return CallToolResult(content=[])

    @asynccontextmanager
    async def stubborn_session():
        try:
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            yield SimpleNamespace(call_tool=call_tool)
        finally:
            finished.set()

    monkeypatch.setattr(client, "_session", stubborn_session)
    raw_arguments: dict[str, object] = {"name": "private room canary"}
    private_arguments = RevocableMcpToolArguments(raw_arguments)
    started = time.monotonic()

    with pytest.raises(McpToolTimeoutError):
        await client.call_tool(name, private_arguments)

    assert time.monotonic() - started < 0.1
    assert private_arguments.revoked is True
    assert raw_arguments == {}
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=0.1)
    assert tool_called is False


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
    with pytest.raises(asyncio.CancelledError):
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    (
        [],
        [TextContent(type="text", text="first"), TextContent(type="text", text="second")],
        [SimpleNamespace(type="image")],
    ),
)
async def test_private_tool_result_rejects_unsupported_success_shapes(
    monkeypatch: pytest.MonkeyPatch,
    content: list[object],
) -> None:
    """No usable bounded private result can ever retain a success status."""
    name = "home_assistant__HassTurnOn"
    client = RemoteMcpToolClient(
        RemoteMcpServerConfig(alias="home_assistant", url="http://127.0.0.1:9123/mcp"),
        known_tools=[
            RemoteToolSpec(
                server_alias="home_assistant",
                remote_name="HassTurnOn",
                namespaced_name=name,
                description="Turn on one target",
                parameters_schema={"type": "object"},
            )
        ],
    )

    @asynccontextmanager
    async def fake_session():
        yield SimpleNamespace(
            call_tool=AsyncMock(return_value=SimpleNamespace(content=content, structuredContent=None, isError=False))
        )

    monkeypatch.setattr(client, "_session", fake_session)
    payload = await client.call_tool(name, RevocableMcpToolArguments({"name": "kitchen"}))

    assert payload["status"] == "error"
    assert "text" not in payload
    assert "structured_content" not in payload


@pytest.mark.asyncio
async def test_private_tool_result_forwards_one_text_block_without_copying_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private path retains the official text-only result for bounded validation."""
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
    text = "{'query': 'private query', 'results': []}"

    async def call_tool(_name: str, **_kwargs: object) -> object:
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            structuredContent=None,
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
        "text": text,
    }
