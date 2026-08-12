from __future__ import annotations
import sys
import json
import importlib
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.tool_spaces as tool_spaces_mod
from reachy_mini_conversation_app.mcp_client import (
    McpToolTimeoutError,
    McpToolInvocationError,
    RevocableMcpToolArguments,
    McpToolArgumentsRevokedError,
)
from reachy_mini_conversation_app.tool_spaces import (
    InstalledToolSpace,
    InstalledToolSpaceTool,
    InstalledToolSpacesManifest,
    write_installed_tool_spaces,
)


SEARCH_SPACE_SLUG = "example/search-tool"
SEARCH_ALIAS = "example_search_tool"
SEARCH_TOOL_ID = f"{SEARCH_ALIAS}__search_web"
SEARCH_CLIENT_TOOL_ID = f"{SEARCH_ALIAS}__search_tool_search_web"
SEARCH_MCP_URL = "https://example-search-tool.hf.space/gradio_api/mcp/"


def _reload_core_tools() -> ModuleType:
    for module_name in list(sys.modules):
        if module_name.startswith("reachy_mini_conversation_app.tools."):
            sys.modules.pop(module_name, None)

    sys.modules.pop("reachy_mini_conversation_app.tools.core_tools", None)
    return importlib.import_module("reachy_mini_conversation_app.tools.core_tools")


def _installed_search_space() -> InstalledToolSpace:
    return InstalledToolSpace(
        slug=SEARCH_SPACE_SLUG,
        alias=SEARCH_ALIAS,
        mcp_url=SEARCH_MCP_URL,
        private=False,
        tools=[
            InstalledToolSpaceTool(
                local_name=SEARCH_TOOL_ID,
                client_tool_name=SEARCH_CLIENT_TOOL_ID,
                remote_name="search_tool_search_web",
                description="Search the web",
                parameters_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
    )


@pytest.mark.parametrize(
    ("private", "slug", "alias", "mcp_url", "headers", "client_tool_name", "expected"),
    [
        (False, SEARCH_SPACE_SLUG, SEARCH_ALIAS, SEARCH_MCP_URL, {}, SEARCH_CLIENT_TOOL_ID, True),
        (True, SEARCH_SPACE_SLUG, SEARCH_ALIAS, SEARCH_MCP_URL, {}, SEARCH_CLIENT_TOOL_ID, False),
        (False, "other/search", SEARCH_ALIAS, SEARCH_MCP_URL, {}, SEARCH_CLIENT_TOOL_ID, False),
        (False, SEARCH_SPACE_SLUG, "other_alias", SEARCH_MCP_URL, {}, SEARCH_CLIENT_TOOL_ID, False),
        (
            False,
            SEARCH_SPACE_SLUG,
            SEARCH_ALIAS,
            "https://alternate.example/mcp/",
            {},
            SEARCH_CLIENT_TOOL_ID,
            False,
        ),
        (
            False,
            SEARCH_SPACE_SLUG,
            SEARCH_ALIAS,
            SEARCH_MCP_URL,
            {"Authorization": "Bearer cached-token"},
            SEARCH_CLIENT_TOOL_ID,
            False,
        ),
        (False, SEARCH_SPACE_SLUG, SEARCH_ALIAS, SEARCH_MCP_URL, {}, "other_tool", False),
    ],
)
def test_expected_remote_tool_resolution_binds_actual_client_source(
    monkeypatch: pytest.MonkeyPatch,
    private: bool,
    slug: str,
    alias: str,
    mcp_url: str,
    headers: dict[str, str],
    client_tool_name: str,
    expected: bool,
) -> None:
    """Only the exact public MCP client checked by Stage 4 yields a dispatch capability."""
    core_tools_mod = _reload_core_tools()
    client = SimpleNamespace(server=SimpleNamespace(alias=alias, url=mcp_url, headers=headers))
    tool = core_tools_mod.RemoteMcpTool(
        slug=slug,
        private=private,
        name=SEARCH_TOOL_ID,
        description="Search the web",
        parameters_schema={"type": "object"},
        client_tool_name=client_tool_name,
        remote_name="search_tool_search_web",
        client=client,
    )
    monkeypatch.setattr(core_tools_mod, "initialize_tools", lambda: None)
    core_tools_mod.ALL_TOOLS = {SEARCH_TOOL_ID: tool}

    resolved = core_tools_mod.resolve_expected_remote_mcp_tool(
        SEARCH_TOOL_ID,
        slug=SEARCH_SPACE_SLUG,
        alias=SEARCH_ALIAS,
        mcp_url=SEARCH_MCP_URL,
        client_tool_name=SEARCH_CLIENT_TOOL_ID,
        remote_name="search_tool_search_web",
    )

    assert (resolved is tool) is expected


@pytest.mark.asyncio
async def test_initialize_tools_loads_enabled_installed_remote_tools_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled public Space tools should join the registry and dispatch through the normal path."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "mcp_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text(f"{SEARCH_TOOL_ID}\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mcp_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    client = AsyncMock()
    client.server = SimpleNamespace(alias=SEARCH_ALIAS, url=SEARCH_MCP_URL, headers={})
    client.call_tool.return_value = {
        "status": "ok",
        "server_alias": SEARCH_ALIAS,
        "remote_tool_name": "reachy_mini_search_tool_search_web",
        "namespaced_tool_name": SEARCH_CLIENT_TOOL_ID,
        "content_blocks": [],
        "text": "hello",
    }
    captured_cached_tools: list[InstalledToolSpaceTool] | None = None

    def _build_remote_client(
        alias: str,
        mcp_url: str,
        *,
        private: bool,
        cached_tools: list[InstalledToolSpaceTool],
    ) -> AsyncMock:
        nonlocal captured_cached_tools
        assert alias == SEARCH_ALIAS
        assert mcp_url == SEARCH_MCP_URL
        assert private is False
        captured_cached_tools = cached_tools
        return client

    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", _build_remote_client)

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(spaces=[_installed_search_space()]),
    )

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert SEARCH_TOOL_ID in core_tools_mod.ALL_TOOLS
    assert captured_cached_tools == _installed_search_space().tools
    tool_specs = core_tools_mod.get_tool_specs()
    assert any(spec["name"] == SEARCH_TOOL_ID for spec in tool_specs)
    assert (
        core_tools_mod.resolve_expected_remote_mcp_tool(
            SEARCH_TOOL_ID,
            slug=SEARCH_SPACE_SLUG,
            alias=SEARCH_ALIAS,
            mcp_url=SEARCH_MCP_URL,
            client_tool_name=SEARCH_CLIENT_TOOL_ID,
            remote_name="search_tool_search_web",
        )
        is core_tools_mod.ALL_TOOLS[SEARCH_TOOL_ID]
    )

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(
            reachy_mini=object(),
            movement_manager=object(),
        ),
    )

    assert result["namespaced_tool_name"] == SEARCH_TOOL_ID
    assert result["tool_space_slug"] == SEARCH_SPACE_SLUG
    client.call_tool.assert_awaited_once_with(SEARCH_CLIENT_TOOL_ID, {"query": "hello"})


def test_initialize_tools_warns_when_enabled_tool_missing_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool enabled in the profile but absent from the cached manifest is skipped with a warning."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "remote_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text(f"{SEARCH_TOOL_ID}\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "remote_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            spaces=[
                InstalledToolSpace(
                    slug=SEARCH_SPACE_SLUG,
                    alias=SEARCH_ALIAS,
                    mcp_url=SEARCH_MCP_URL,
                    private=False,
                ),
            ]
        ),
    )

    core_tools_mod = _reload_core_tools()
    with caplog.at_level("WARNING"):
        core_tools_mod.initialize_tools()

    assert any(SEARCH_SPACE_SLUG in record.message for record in caplog.records)
    assert SEARCH_TOOL_ID not in core_tools_mod.ALL_TOOLS


def test_initialize_tools_inherits_default_tools_txt_for_profile_without_local_tool_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiles without a local tools.txt should inherit the built-in default tool set."""
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "inherit_default"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "inherit_default")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    assert "dance" in core_tools_mod.ALL_TOOLS


def _mcp_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "mcp_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text(f"{SEARCH_TOOL_ID}\n", encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "mcp_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)


@pytest.mark.asyncio
async def test_remote_tool_retries_once_after_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient remote transport failures should get one fast retry."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    client = AsyncMock()
    client.call_tool.side_effect = [
        McpToolInvocationError("connection reset"),
        {
            "status": "ok",
            "server_alias": SEARCH_ALIAS,
            "remote_tool_name": "reachy_mini_search_tool_search_web",
            "namespaced_tool_name": SEARCH_CLIENT_TOOL_ID,
            "content_blocks": [],
            "text": "hello",
        },
    ]
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))

    core_tools_mod = _reload_core_tools()
    monkeypatch.setattr(core_tools_mod, "_REMOTE_TOOL_RETRY_DELAY_S", 0.0)
    core_tools_mod.initialize_tools()

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=object()),
    )

    assert result["status"] == "ok"
    assert client.call_tool.await_count == 2


@pytest.mark.asyncio
async def test_remote_tool_does_not_retry_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote timeouts should fail once instead of doubling the user wait."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)

    client = AsyncMock()
    client.call_tool.side_effect = McpToolTimeoutError("slow tool")
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))

    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()

    result = await core_tools_mod.dispatch_tool_call(
        SEARCH_TOOL_ID,
        json.dumps({"query": "hello"}),
        core_tools_mod.ToolDependencies(reachy_mini=object(), movement_manager=object()),
    )

    assert "error" in result
    assert client.call_tool.await_count == 1


@pytest.mark.asyncio
async def test_generic_mcp_tool_is_isolated_and_never_retries_transport_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action-bearing generic MCP sources must use the private coordinator with one attempt."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "home_profile"
    profile_dir.mkdir(parents=True)
    tool_name = "home_assistant__HassTurnOff"
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text(f"{tool_name}\n", encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "home_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    installed = InstalledToolSpace(
        slug="mcp/home_assistant",
        alias="home_assistant",
        mcp_url="http://127.0.0.1:9123/mcp",
        private=False,
        tools=[
            InstalledToolSpaceTool(
                local_name=tool_name,
                client_tool_name=tool_name,
                remote_name="HassTurnOff",
                description="Turn off exposed devices",
                parameters_schema={"type": "object"},
            )
        ],
        source_kind="generic_mcp",
        prompt_name="assist",
        prompt_text="Control exposed devices.",
        retry_tool_failures=False,
        isolated_response=True,
    )
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(version=3, spaces=[installed]))
    client = AsyncMock()
    client.server = SimpleNamespace(alias="home_assistant", url=installed.mcp_url, headers={})
    client.call_tool.side_effect = McpToolInvocationError("private transport canary")
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *args, **kwargs: client)

    core_tools_mod = _reload_core_tools()
    monkeypatch.setattr(
        core_tools_mod.config_module,
        "has_private_mcp_local_realtime_boundary",
        lambda: True,
    )
    core_tools_mod.initialize_tools()
    tool = core_tools_mod.ALL_TOOLS[tool_name]
    result = await core_tools_mod.dispatch_bound_remote_tool_call(
        tool,
        tool_name=tool_name,
        arguments=RevocableMcpToolArguments({"name": "private room canary"}),
    )

    assert tool.uses_isolated_response is True
    assert result == {"error": "Remote tool unavailable"}
    assert client.call_tool.await_count == 1


def test_generic_mcp_tool_is_not_registered_for_deployed_realtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private generic MCP tools must not be exposed to a deployed realtime service."""
    monkeypatch.chdir(tmp_path)
    external_profiles_root = tmp_path / "external_profiles"
    profile_dir = external_profiles_root / "home_profile"
    profile_dir.mkdir(parents=True)
    tool_name = "home_assistant__HassTurnOff"
    (profile_dir / "instructions.txt").write_text("hello\n", encoding="utf-8")
    (profile_dir / "tools.txt").write_text(f"{tool_name}\n", encoding="utf-8")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_CUSTOM_PROFILE", "home_profile")
    monkeypatch.setattr(config_mod.config, "PROFILES_DIRECTORY", external_profiles_root)
    monkeypatch.setattr(config_mod.config, "TOOLS_DIRECTORY", None)
    monkeypatch.setattr(config_mod.config, "AUTOLOAD_EXTERNAL_TOOLS", False)
    write_installed_tool_spaces(
        None,
        InstalledToolSpacesManifest(
            version=3,
            spaces=[
                InstalledToolSpace(
                    slug="mcp/home_assistant",
                    alias="home_assistant",
                    mcp_url="http://127.0.0.1:9123/mcp",
                    private=False,
                    tools=[
                        InstalledToolSpaceTool(
                            local_name=tool_name,
                            client_tool_name=tool_name,
                            remote_name="HassTurnOff",
                            description="Turn off exposed devices",
                            parameters_schema={"type": "object"},
                        )
                    ],
                    source_kind="generic_mcp",
                    prompt_name="assist",
                    prompt_text="PRIVATE CACHED HOME GUIDANCE CANARY",
                    retry_tool_failures=False,
                    isolated_response=True,
                )
            ],
        ),
    )

    core_tools_mod = _reload_core_tools()
    monkeypatch.setattr(
        core_tools_mod.config_module,
        "has_private_mcp_local_realtime_boundary",
        lambda: False,
    )
    core_tools_mod.initialize_tools()

    assert tool_name not in core_tools_mod.ALL_TOOLS


@pytest.mark.asyncio
async def test_remote_tool_redaction_preserves_one_retry_without_leaking_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The private search path retries identically once and emits only a fixed local error."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)
    first_error = "private-first-error"
    second_error = "private-second-error"
    query = "private-query"

    client = AsyncMock()
    client.call_tool.side_effect = [
        McpToolInvocationError(first_error),
        McpToolInvocationError(second_error),
    ]
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))

    core_tools_mod = _reload_core_tools()
    monkeypatch.setattr(core_tools_mod, "_REMOTE_TOOL_RETRY_DELAY_S", 0.0)
    core_tools_mod.initialize_tools()

    bound_tool = core_tools_mod.ALL_TOOLS[SEARCH_TOOL_ID]
    core_tools_mod.ALL_TOOLS[SEARCH_TOOL_ID] = object()
    result = await core_tools_mod.dispatch_bound_remote_tool_call(
        bound_tool,
        tool_name=SEARCH_TOOL_ID,
        arguments=RevocableMcpToolArguments({"query": query}),
    )

    assert result == {"error": "Remote tool unavailable"}
    assert client.call_tool.await_count == 2
    assert client.call_tool.await_args_list[0] == client.call_tool.await_args_list[1]
    assert first_error not in caplog.text
    assert second_error not in caplog.text
    assert query not in caplog.text


@pytest.mark.asyncio
async def test_revoked_bound_remote_tool_does_not_retry_or_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A revoked private argument lease fails once with fixed local details."""
    monkeypatch.chdir(tmp_path)
    _mcp_profile(tmp_path, monkeypatch)
    private_canary = "revoked-private-query-canary"
    client = AsyncMock()
    client.call_tool.side_effect = McpToolArgumentsRevokedError(private_canary)
    monkeypatch.setattr(tool_spaces_mod, "build_remote_client", lambda *a, **k: client)
    write_installed_tool_spaces(None, InstalledToolSpacesManifest(spaces=[_installed_search_space()]))
    core_tools_mod = _reload_core_tools()
    core_tools_mod.initialize_tools()
    bound_tool = core_tools_mod.ALL_TOOLS[SEARCH_TOOL_ID]
    private_arguments = RevocableMcpToolArguments({"query": private_canary})

    result = await core_tools_mod.dispatch_bound_remote_tool_call(
        bound_tool,
        tool_name=SEARCH_TOOL_ID,
        arguments=private_arguments,
    )

    assert result == {"error": "Remote tool unavailable"}
    assert client.call_tool.await_count == 1
    assert private_canary not in caplog.text
