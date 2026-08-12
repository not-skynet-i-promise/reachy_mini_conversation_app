"""Manage installed Hugging Face Space tool sources for the conversation app."""

from __future__ import annotations
import re
import json
import asyncio
import logging
import argparse
from typing import Any, Literal
from pathlib import Path
from collections import Counter
from dataclasses import field, asdict, dataclass
from collections.abc import Sequence

from huggingface_hub import HfApi, SpaceInfo, get_token
from huggingface_hub.errors import RepositoryNotFoundError

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.mcp_client import (
    McpClientError,
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    validate_http_mcp_url,
    validate_catalog_cache,
    apply_name_normalization,
    build_namespaced_tool_name,
)


logger = logging.getLogger(__name__)

INSTALLED_TOOL_SPACES_FILENAME = "installed_tool_spaces.json"
INSTALLED_TOOL_SPACES_VERSION = 3
INSTALLED_TOOL_SPACES_MAX_BYTES = 1024 * 1024
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
# Bundled Pollen Spaces seeded when no manifest exists, so startup needs no Hugging Face discovery.
PREINSTALLED_TOOL_SPACE_SPECS = {
    "pollen-robotics/reachy-mini-search-tool": (
        RemoteToolSpec(
            server_alias="pollen_robotics_reachy_mini_search_tool",
            remote_name="reachy_mini_search_tool_search_web",
            namespaced_name=build_namespaced_tool_name(
                "pollen_robotics_reachy_mini_search_tool", "reachy_mini_search_tool_search_web"
            ),
            description=(
                "Search the web for current information and return a short list of results (title, snippet, url). "
                "Call this directly whenever the user asks to search, check the web, look something up, "
                "find today's events, or learn what is happening now. Do not just say you'll look it up."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["query"],
            },
        ),
    ),
    "pollen-robotics/reachy-mini-time-tool": (
        RemoteToolSpec(
            server_alias="pollen_robotics_reachy_mini_time_tool",
            remote_name="reachy_mini_time_tool_get_time",
            namespaced_name=build_namespaced_tool_name(
                "pollen_robotics_reachy_mini_time_tool", "reachy_mini_time_tool_get_time"
            ),
            description=(
                "Get the current date and time for an IANA timezone, and optionally the difference to a second timezone. "
                "Call this directly whenever the user asks what time it is or the time somewhere. Pass an IANA name like "
                "'Europe/Paris' or 'Asia/Tokyo' for a named place (derive it from the place), or leave the timezone empty "
                "for the user's own local time. Do not ask for their city and do not just say you'll check."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "default": "",
                        "description": "IANA timezone like 'Europe/Paris'. Empty resolves the user's local time.",
                    },
                    "compare_timezone": {
                        "type": "string",
                        "default": "",
                        "description": "Optional second IANA timezone to compare against.",
                    },
                },
                "required": [],
            },
        ),
    ),
    "pollen-robotics/reachy-mini-weather-tool": (
        RemoteToolSpec(
            server_alias="pollen_robotics_reachy_mini_weather_tool",
            remote_name="reachy_mini_weather_tool_get_weather",
            namespaced_name=build_namespaced_tool_name(
                "pollen_robotics_reachy_mini_weather_tool", "reachy_mini_weather_tool_get_weather"
            ),
            description=(
                "Get today's weather for a place: current conditions, high and low temperature, and rain chance. "
                "Call this directly whenever the user asks about the weather, forecast, or temperature for "
                "somewhere. Do not just say you'll check."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Place name for the weather lookup."},
                },
                "required": ["location"],
            },
        ),
    ),
}
_SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class InstalledToolSpaceTool:
    """App-facing metadata for one remote tool exposed by an installed Space."""

    local_name: str
    client_tool_name: str
    remote_name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass(frozen=True)
class InstalledToolSpace:
    """Persisted record for one installed Space and the tools discovered at install time."""

    slug: str
    alias: str
    mcp_url: str
    private: bool
    tools: list[InstalledToolSpaceTool] = field(default_factory=list)
    source_kind: Literal["huggingface_space", "generic_mcp"] = "huggingface_space"
    prompt_name: str | None = None
    prompt_text: str | None = None
    retry_tool_failures: bool = True
    isolated_response: bool = False


@dataclass(frozen=True)
class InstalledToolSpacesManifest:
    """Persisted manifest of installed Space tool sources."""

    version: int = INSTALLED_TOOL_SPACES_VERSION
    spaces: list[InstalledToolSpace] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedInstalledToolSpace:
    """Runtime description of an installed Space."""

    slug: str
    alias: str
    mcp_url: str
    private: bool
    tags: list[str]
    tools: list[InstalledToolSpaceTool]
    client: RemoteMcpToolClient
    source_kind: Literal["huggingface_space", "generic_mcp"] = "huggingface_space"
    prompt_name: str | None = None
    prompt_text: str | None = None
    retry_tool_failures: bool = True
    isolated_response: bool = False


def get_installed_tool_spaces_path(instance_path: str | Path | None) -> Path:
    """Return the installed tool-spaces manifest path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / INSTALLED_TOOL_SPACES_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / INSTALLED_TOOL_SPACES_FILENAME


def _preinstalled_installed_spaces() -> list[InstalledToolSpace]:
    """Build the bundled Pollen Spaces as manifest entries with their tools cached from static specs."""
    spaces: list[InstalledToolSpace] = []
    for slug, remote_specs in PREINSTALLED_TOOL_SPACE_SPECS.items():
        alias = normalize_space_alias(slug)
        spaces.append(
            InstalledToolSpace(
                slug=slug,
                alias=alias,
                mcp_url=f"https://{slug.replace('/', '-')}.hf.space/gradio_api/mcp/",
                private=False,
                tools=_build_installed_tool_space_tools(slug=slug, alias=alias, remote_specs=list(remote_specs)),
            )
        )
    return spaces


def read_installed_tool_spaces(instance_path: str | Path | None) -> InstalledToolSpacesManifest:
    """Read the installed tool-spaces manifest, or seed the bundled Pollen Spaces when none exists."""
    manifest_path = get_installed_tool_spaces_path(instance_path)
    if not manifest_path.exists():
        return InstalledToolSpacesManifest(spaces=_preinstalled_installed_spaces())

    try:
        if manifest_path.stat().st_size > INSTALLED_TOOL_SPACES_MAX_BYTES:
            raise RuntimeError("installed tool spaces manifest exceeds its byte limit")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read installed tool spaces from {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: expected a JSON object.")

    raw_spaces = payload.get("spaces", [])
    if not isinstance(raw_spaces, list):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: 'spaces' must be a list.")
    version = payload.get("version", 1)
    if not isinstance(version, int):
        raise RuntimeError(f"Invalid installed tool spaces payload in {manifest_path}: 'version' must be an int.")

    spaces: list[InstalledToolSpace] = []
    seen_slugs: set[str] = set()
    seen_aliases: set[str] = set()
    for raw_space in raw_spaces:
        if not isinstance(raw_space, dict):
            raise RuntimeError(f"Invalid installed tool spaces entry in {manifest_path}: expected an object.")

        source_kind_value = str(raw_space.get("source_kind", "huggingface_space"))
        if source_kind_value not in {"huggingface_space", "generic_mcp"}:
            raise RuntimeError(f"Invalid installed tool source kind '{source_kind_value}' in {manifest_path}.")
        source_kind: Literal["huggingface_space", "generic_mcp"] = (
            "generic_mcp" if source_kind_value == "generic_mcp" else "huggingface_space"
        )
        slug = validate_space_slug(str(raw_space.get("slug", "")))
        alias = normalize_space_alias(slug) if source_kind == "huggingface_space" else str(raw_space.get("alias", ""))
        try:
            build_namespaced_tool_name(alias, "tool")
        except ValueError as exc:
            raise RuntimeError(f"Invalid installed MCP server alias '{alias}' in {manifest_path}.") from exc
        if slug in seen_slugs:
            raise RuntimeError(f"Duplicate installed tool space '{slug}' found in {manifest_path}.")
        if alias in seen_aliases:
            raise RuntimeError(
                f"Installed tool spaces manifest contains alias collision '{alias}' in {manifest_path}. "
                "Remove one of the conflicting spaces with 'tool-spaces remove'."
            )
        raw_mcp_url = raw_space.get("mcp_url", "")
        if source_kind == "generic_mcp" and not isinstance(raw_mcp_url, str):
            raise RuntimeError(f"Invalid generic MCP endpoint for '{alias}' in {manifest_path}.")
        mcp_url = raw_mcp_url if source_kind == "generic_mcp" else str(raw_mcp_url).strip()
        if not mcp_url:
            logger.warning(
                "Installed Space '%s' predates cached tool metadata and will be skipped. Re-run 'tool-spaces add %s'.",
                slug,
                slug,
            )
            continue
        raw_tools = raw_space.get("tools", [])
        if not isinstance(raw_tools, list):
            raise RuntimeError(f"Invalid tool catalog for '{alias}' in {manifest_path}.")
        cached_tools: list[InstalledToolSpaceTool] = []
        for tool in raw_tools:
            if not isinstance(tool, dict) or not tool.get("local_name") or not tool.get("client_tool_name"):
                if source_kind == "generic_mcp":
                    raise RuntimeError(f"Invalid generic MCP tool catalog for '{alias}' in {manifest_path}.")
                continue
            if source_kind == "generic_mcp":
                if (
                    not isinstance(tool.get("local_name"), str)
                    or not isinstance(tool.get("client_tool_name"), str)
                    or not isinstance(tool.get("remote_name"), str)
                    or not isinstance(tool.get("description", ""), str)
                    or not isinstance(tool.get("parameters_schema"), dict)
                ):
                    raise RuntimeError(f"Invalid generic MCP tool catalog for '{alias}' in {manifest_path}.")
                schema = tool["parameters_schema"]
            else:
                try:
                    schema = dict(tool.get("parameters_schema") or {})
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Invalid tool schema for '{alias}' in {manifest_path}.") from exc
            cached_tools.append(
                InstalledToolSpaceTool(
                    local_name=str(tool["local_name"]),
                    client_tool_name=str(tool["client_tool_name"]),
                    remote_name=str(tool.get("remote_name", "")),
                    description=str(tool.get("description", "")),
                    parameters_schema=schema,
                )
            )
        raw_prompt_name = raw_space.get("prompt_name")
        raw_prompt_text = raw_space.get("prompt_text")
        if source_kind == "generic_mcp":
            prompt_name = raw_prompt_name if isinstance(raw_prompt_name, str) else None
            prompt_text = raw_prompt_text if isinstance(raw_prompt_text, str) else None
        else:
            prompt_name = str(raw_prompt_name) if raw_prompt_name is not None else None
            prompt_text = str(raw_prompt_text) if raw_prompt_text is not None else None
        retry_tool_failures = bool(raw_space.get("retry_tool_failures", True))
        isolated_response = bool(raw_space.get("isolated_response", False))
        if source_kind == "generic_mcp":
            try:
                validated_url = validate_http_mcp_url(mcp_url)
                specs = [
                    RemoteToolSpec(
                        server_alias=alias,
                        remote_name=tool.remote_name,
                        namespaced_name=tool.client_tool_name,
                        description=tool.description,
                        parameters_schema=tool.parameters_schema,
                    )
                    for tool in cached_tools
                ]
                normalized_prompt = validate_catalog_cache(prompt_text, specs)
                if (
                    version != INSTALLED_TOOL_SPACES_VERSION
                    or slug != f"mcp/{alias}"
                    or bool(raw_space.get("private", False))
                    or not isinstance(prompt_name, str)
                    or not prompt_name.strip()
                    or len(prompt_name) > 128
                    or retry_tool_failures
                    or not isolated_response
                    or any(
                        tool.local_name != spec.namespaced_name
                        or tool.client_tool_name != spec.namespaced_name
                        or spec.namespaced_name != build_namespaced_tool_name(alias, spec.remote_name)
                        for tool, spec in zip(cached_tools, specs, strict=True)
                    )
                ):
                    raise ValueError("invalid generic MCP source")
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid generic MCP source '{alias}' in {manifest_path}: "
                    "a bounded prompt, bounded object-schema catalog, credential-free endpoint, "
                    "isolated responses, and no-retry mode are required."
                ) from exc
            mcp_url = validated_url
            prompt_name = prompt_name.strip()
            prompt_text = normalized_prompt
        seen_slugs.add(slug)
        seen_aliases.add(alias)
        spaces.append(
            InstalledToolSpace(
                slug=slug,
                alias=alias,
                mcp_url=mcp_url,
                private=bool(raw_space.get("private", False)),
                tools=cached_tools,
                source_kind=source_kind,
                prompt_name=prompt_name,
                prompt_text=prompt_text,
                retry_tool_failures=retry_tool_failures,
                isolated_response=isolated_response,
            )
        )

    return InstalledToolSpacesManifest(version=version, spaces=spaces)


def write_installed_tool_spaces(
    instance_path: str | Path | None,
    manifest: InstalledToolSpacesManifest,
) -> Path:
    """Persist the installed tool-spaces manifest."""
    manifest_path = get_installed_tool_spaces_path(instance_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "spaces": [asdict(space) for space in manifest.spaces],
    }
    manifest_path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")
    return manifest_path


def _append_tools_to_profile(profile: str, tool_ids: list[str]) -> list[str]:
    """Append tool IDs to a profile's tools.txt. Returns the IDs that were added."""
    tools_txt = _require_profile_tools_path(profile)
    existing_content = tools_txt.read_text(encoding="utf-8") if tools_txt.exists() else ""
    existing: set[str] = set()
    for line in existing_content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            existing.add(stripped)

    to_add = [tid for tid in tool_ids if tid not in existing]
    if to_add:
        with tools_txt.open("a", encoding="utf-8") as f:
            if existing_content and not existing_content.endswith("\n"):
                f.write("\n")
            for tid in to_add:
                f.write(f"{tid}\n")
    return to_add


def _require_profile_tools_path(profile: str) -> Path:
    """Resolve an existing profile before any generic source is persisted."""
    tools_txt = config.resolve_profile_dir(profile) / "tools.txt"
    if not tools_txt.parent.is_dir():
        raise RuntimeError(
            f"Profile '{profile}' not found at {tools_txt.parent}. Use --install-only to skip profile wiring."
        )
    return tools_txt


def _restore_file_snapshot(path: Path, snapshot: bytes | None) -> None:
    """Restore one exact pre-provisioning file snapshot."""
    if snapshot is None:
        path.unlink(missing_ok=True)
        return
    path.write_bytes(snapshot)


def _disable_space_tools_in_profiles(alias: str) -> list[tuple[str, list[str]]]:
    """Strip a Space's tool IDs from every profile's tools.txt. Returns (profile, removed IDs) per profile touched."""
    prefix = f"{alias}__"
    removed_by_profile: list[tuple[str, list[str]]] = []
    seen: set[Path] = set()
    for root in (config.PROFILES_DIRECTORY, config.user_personalities_root()):
        for tools_txt in sorted(root.glob("*/tools.txt")):
            resolved = tools_txt.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            lines = tools_txt.read_text(encoding="utf-8").splitlines()
            removed = [line.strip() for line in lines if line.strip().startswith(prefix)]
            if not removed:
                continue
            kept = [line for line in lines if not line.strip().startswith(prefix)]
            tools_txt.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
            removed_by_profile.append((tools_txt.parent.name, removed))
    return removed_by_profile


def validate_space_slug(slug: str) -> str:
    """Validate a public HF Space slug."""
    candidate = slug.strip()
    if _SLUG_PATTERN.fullmatch(candidate) is None:
        raise ValueError(
            f"Invalid Space slug '{slug}'. Expected the form 'owner/space-name' with alnum, '.', '_' or '-'."
        )
    return candidate


def normalize_space_alias(slug: str) -> str:
    """Derive a local alias from a Space slug."""
    normalized = apply_name_normalization(slug)
    if not normalized:
        raise ValueError(f"Space slug '{slug}' cannot be normalized into a local alias.")
    if normalized[0].isdigit():
        normalized = f"space_{normalized}"
    return normalized


def _normalize_segment(value: str) -> str:
    normalized = apply_name_normalization(value)
    if not normalized:
        return "tool"
    if normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return normalized


def _clean_space_tool_name(slug: str, alias: str, remote_name: str) -> str:
    normalized_remote_name = _normalize_segment(remote_name)
    space_name = slug.split("/", maxsplit=1)[1]
    normalized_space_name = _normalize_segment(space_name)
    redundant_prefix = f"{normalized_space_name}_"

    if normalized_remote_name.startswith(redundant_prefix):
        cleaned_name = normalized_remote_name[len(redundant_prefix) :]
        if cleaned_name:
            return f"{alias}__{cleaned_name}"
    return f"{alias}__{normalized_remote_name}"


def _build_installed_tool_space_tools(
    *,
    slug: str,
    alias: str,
    remote_specs: Sequence[RemoteToolSpec],
) -> list[InstalledToolSpaceTool]:
    cleaned_names = [_clean_space_tool_name(slug, alias, spec.remote_name) for spec in remote_specs]
    collisions = {name for name, count in Counter(cleaned_names).items() if count > 1}

    tools: list[InstalledToolSpaceTool] = []
    for remote_spec, cleaned_name in zip(remote_specs, cleaned_names, strict=True):
        local_name = remote_spec.namespaced_name if cleaned_name in collisions else cleaned_name
        tools.append(
            InstalledToolSpaceTool(
                local_name=local_name,
                client_tool_name=remote_spec.namespaced_name,
                remote_name=remote_spec.remote_name,
                description=remote_spec.description,
                parameters_schema=dict(remote_spec.parameters_schema),
            )
        )
    return tools


def _build_installed_generic_mcp_tools(
    remote_specs: Sequence[RemoteToolSpec],
) -> list[InstalledToolSpaceTool]:
    """Preserve generic MCP names exactly; HF redundant-prefix cleanup does not apply."""
    return [
        InstalledToolSpaceTool(
            local_name=spec.namespaced_name,
            client_tool_name=spec.namespaced_name,
            remote_name=spec.remote_name,
            description=spec.description,
            parameters_schema=dict(spec.parameters_schema),
        )
        for spec in remote_specs
    ]


def _build_space_mcp_url(space_info: SpaceInfo, slug: str) -> str:
    host = (space_info.host or "").strip()
    if host:
        if host.startswith("http://") or host.startswith("https://"):
            return f"{host.rstrip('/')}/gradio_api/mcp/"
        return f"https://{host.rstrip('/')}/gradio_api/mcp/"

    subdomain = (space_info.subdomain or "").strip()
    if subdomain:
        return f"https://{subdomain}.hf.space/gradio_api/mcp/"

    slug_host = slug.replace("/", "-")
    return f"https://{slug_host}.hf.space/gradio_api/mcp/"


def _validate_space_info(slug: str, space_info: SpaceInfo) -> None:
    if bool(space_info.disabled):
        raise RuntimeError(f"Space '{slug}' is disabled and cannot be installed.")
    if (space_info.sdk or "").strip().lower() != "gradio":
        raise RuntimeError(f"Space '{slug}' is not a Gradio Space and cannot expose the standard MCP endpoint.")


def build_remote_client(
    alias: str,
    mcp_url: str,
    *,
    private: bool,
    cached_tools: Sequence[InstalledToolSpaceTool] = (),
    use_huggingface_auth: bool = True,
    tool_timeout_s: float = 30.0,
) -> RemoteMcpToolClient:
    """Build an MCP client for an installed Space, sending the HF token only to private Spaces."""
    token = (config.HF_TOKEN or get_token()) if use_huggingface_auth else None
    headers = {"Authorization": f"Bearer {token}"} if private and token else {}
    return RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=alias,
            url=mcp_url,
            headers=headers,
            request_timeout_s=10.0,
            tool_timeout_s=tool_timeout_s,
        ),
        known_tools=[
            RemoteToolSpec(
                server_alias=alias,
                remote_name=tool.remote_name,
                namespaced_name=tool.client_tool_name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
            for tool in cached_tools
            if tool.remote_name
        ],
    )


async def resolve_tool_space(slug: str) -> ResolvedInstalledToolSpace:
    """Validate and discover tools from one HF Space, authenticating private Spaces with the HF token."""
    validated_slug = validate_space_slug(slug)
    alias = normalize_space_alias(validated_slug)
    token = config.HF_TOKEN or get_token()
    try:
        space_info = HfApi().space_info(validated_slug, timeout=10.0, token=token or False)
    except RepositoryNotFoundError as exc:
        if token is None:
            raise RuntimeError(
                f"Space '{validated_slug}' was not found. If it is private, set HF_TOKEN "
                "or run 'hf auth login' for an account that can access it."
            ) from exc
        raise RuntimeError(
            f"Space '{validated_slug}' was not found, or the current Hugging Face token cannot access it."
        ) from exc
    _validate_space_info(validated_slug, space_info)

    mcp_url = _build_space_mcp_url(space_info, validated_slug)
    private = bool(space_info.private)
    client = build_remote_client(alias, mcp_url, private=private)
    try:
        remote_specs = await client.list_tool_specs()
    except McpClientError as exc:
        raise RuntimeError(f"Failed to discover MCP tools for '{validated_slug}': {exc}") from exc

    return ResolvedInstalledToolSpace(
        slug=validated_slug,
        alias=alias,
        mcp_url=mcp_url,
        private=private,
        tags=sorted(space_info.tags or []),
        tools=_build_installed_tool_space_tools(slug=validated_slug, alias=alias, remote_specs=remote_specs),
        client=client,
    )


async def resolve_generic_mcp_server(alias: str, mcp_url: str, prompt_name: str) -> ResolvedInstalledToolSpace:
    """Discover one generic local MCP server with an exact prompt and isolated no-retry tools."""
    validated_alias = alias.strip()
    build_namespaced_tool_name(validated_alias, "tool")
    client = build_remote_client(
        validated_alias,
        mcp_url,
        private=False,
        use_huggingface_auth=False,
        tool_timeout_s=5.0,
    )
    try:
        catalog = await client.discover_catalog(prompt_name)
    except (McpClientError, ValueError) as exc:
        raise RuntimeError(f"Failed to discover generic MCP server '{validated_alias}': {exc}") from exc
    slug = f"mcp/{validated_alias}"
    return ResolvedInstalledToolSpace(
        slug=slug,
        alias=validated_alias,
        mcp_url=client.server.url,
        private=False,
        tags=[],
        tools=_build_installed_generic_mcp_tools(catalog.tools),
        client=client,
        source_kind="generic_mcp",
        prompt_name=catalog.prompt_name,
        prompt_text=catalog.prompt_text,
        retry_tool_failures=False,
        isolated_response=True,
    )


def resolve_generic_mcp_server_sync(alias: str, mcp_url: str, prompt_name: str) -> ResolvedInstalledToolSpace:
    """Resolve one generic MCP server synchronously."""
    return asyncio.run(resolve_generic_mcp_server(alias, mcp_url, prompt_name))


def resolve_tool_space_sync(slug: str) -> ResolvedInstalledToolSpace:
    """Resolve one Space synchronously."""
    return asyncio.run(resolve_tool_space(slug))


def format_space_tool_listing(space: ResolvedInstalledToolSpace) -> str:
    """Format one resolved Space for terminal output."""
    lines = [
        f"{space.slug} ({space.alias})",
        f"  MCP endpoint: {space.mcp_url}",
    ]
    if space.tools:
        lines.append("  Tools:")
        lines.extend([f"    - {tool.local_name}" for tool in space.tools])
    else:
        lines.append("  Tools: none discovered")
    return "\n".join(lines)


def handle_tool_spaces_command(args: argparse.Namespace, *, instance_path: str | Path | None = None) -> int:
    """Handle tool-spaces subcommands from the main CLI."""
    command = getattr(args, "tool_spaces_command", None)
    if command == "add-server":
        try:
            validated_alias = args.alias.strip()
            build_namespaced_tool_name(validated_alias, "tool")
            validated_url = validate_http_mcp_url(args.mcp_url)
            manifest = read_installed_tool_spaces(instance_path)
            previous = next((space for space in manifest.spaces if space.alias == validated_alias), None)
            if previous is not None and (previous.source_kind != "generic_mcp" or previous.mcp_url != validated_url):
                logger.error("Tool source alias already installed for a different endpoint: %s", validated_alias)
                return 1
            resolved_space = resolve_generic_mcp_server_sync(validated_alias, validated_url, args.prompt)
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        installed = InstalledToolSpace(
            slug=resolved_space.slug,
            alias=resolved_space.alias,
            mcp_url=resolved_space.mcp_url,
            private=False,
            tools=resolved_space.tools,
            source_kind="generic_mcp",
            prompt_name=resolved_space.prompt_name,
            prompt_text=resolved_space.prompt_text,
            retry_tool_failures=False,
            isolated_response=True,
        )
        updated_spaces = sorted(
            [space for space in manifest.spaces if space is not previous] + [installed],
            key=lambda space: space.slug,
        )
        target_profile = None if args.install_only else args.profile or config.REACHY_MINI_CUSTOM_PROFILE or "default"
        profile_path: Path | None = None
        profile_snapshot: bytes | None = None
        try:
            if target_profile is not None:
                profile_path = _require_profile_tools_path(target_profile)
                profile_snapshot = profile_path.read_bytes() if profile_path.exists() else None
        except (OSError, RuntimeError) as exc:
            logger.error("Cannot enable tools: %s", exc)
            return 1
        manifest_path = get_installed_tool_spaces_path(instance_path)
        try:
            manifest_snapshot = manifest_path.read_bytes() if manifest_path.exists() else None
        except OSError as exc:
            logger.error("Cannot snapshot the existing MCP manifest: %s", exc)
            return 1
        try:
            write_installed_tool_spaces(
                instance_path,
                InstalledToolSpacesManifest(version=INSTALLED_TOOL_SPACES_VERSION, spaces=updated_spaces),
            )
        except OSError as exc:
            try:
                _restore_file_snapshot(manifest_path, manifest_snapshot)
            except OSError as rollback_exc:
                logger.error("Cannot restore the prior MCP manifest after persistence failure: %s", rollback_exc)
            logger.error("Cannot persist generic MCP source: %s", exc)
            return 1
        logger.info(
            "%s generic MCP source '%s' in %s",
            "Refreshed" if previous is not None else "Installed",
            resolved_space.alias,
            manifest_path,
        )
        if args.install_only:
            return 0
        assert target_profile is not None
        assert profile_path is not None
        try:
            added = _append_tools_to_profile(target_profile, [tool.local_name for tool in resolved_space.tools])
        except (OSError, RuntimeError) as exc:
            for path, snapshot in ((profile_path, profile_snapshot), (manifest_path, manifest_snapshot)):
                try:
                    _restore_file_snapshot(path, snapshot)
                except OSError as rollback_exc:
                    logger.error("Cannot restore provisioning file %s after profile failure: %s", path, rollback_exc)
            logger.error("Cannot enable tools: %s", exc)
            return 1
        logger.info("Enabled in profile '%s': %s", target_profile, added or "already enabled")
        return 0

    if command == "remove-server":
        alias = args.alias.strip()
        manifest = read_installed_tool_spaces(instance_path)
        generic_installed = next(
            (space for space in manifest.spaces if space.source_kind == "generic_mcp" and space.alias == alias),
            None,
        )
        if generic_installed is None:
            logger.warning("Generic MCP source not installed: %s", alias)
            return 1
        write_installed_tool_spaces(
            instance_path,
            InstalledToolSpacesManifest(
                version=manifest.version,
                spaces=[space for space in manifest.spaces if space is not generic_installed],
            ),
        )
        for profile_name, disabled_tool_ids in _disable_space_tools_in_profiles(alias):
            logger.info("Disabled in profile '%s': %s", profile_name, disabled_tool_ids)
        return 0
    if command == "add":
        try:
            resolved_space = resolve_tool_space_sync(args.space_slug)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        manifest = read_installed_tool_spaces(instance_path)
        already_installed = any(space.slug == resolved_space.slug for space in manifest.spaces)
        if already_installed:
            logger.info("Space already installed: %s", resolved_space.slug)
            logger.info("%s", format_space_tool_listing(resolved_space))
        else:
            alias_conflict = next((s for s in manifest.spaces if s.alias == resolved_space.alias), None)
            if alias_conflict:
                logger.error(
                    "Cannot install '%s': its local alias '%s' conflicts with already-installed '%s'. "
                    "Rename one Space on Hugging Face to get a distinct alias.",
                    resolved_space.slug,
                    resolved_space.alias,
                    alias_conflict.slug,
                )
                return 1

            installed = InstalledToolSpace(
                slug=resolved_space.slug,
                alias=resolved_space.alias,
                mcp_url=resolved_space.mcp_url,
                private=resolved_space.private,
                tools=resolved_space.tools,
            )
            updated_spaces = sorted(
                [*manifest.spaces, installed],
                key=lambda space: space.slug,
            )
            manifest_path = write_installed_tool_spaces(
                instance_path,
                InstalledToolSpacesManifest(version=INSTALLED_TOOL_SPACES_VERSION, spaces=updated_spaces),
            )
            logger.info("Installed Space tool source: %s", resolved_space.slug)
            logger.info("Manifest: %s", manifest_path)
            logger.info("%s", format_space_tool_listing(resolved_space))

        if args.install_only:
            logger.info("Tools installed. Add tool IDs to a profile's tools.txt to enable them.")
            return 0

        target_profile = args.profile
        if target_profile is None:
            target_profile = config.REACHY_MINI_CUSTOM_PROFILE or "default"

        tool_ids = [tool.local_name for tool in resolved_space.tools]
        try:
            added = _append_tools_to_profile(target_profile, tool_ids)
        except RuntimeError as exc:
            logger.error("Cannot enable tools: %s", exc)
            return 1
        if added:
            logger.info("Enabled in profile '%s': %s", target_profile, added)
        else:
            logger.info("All tool IDs already present in profile '%s'.", target_profile)
        return 0

    if command == "remove":
        validated_slug = validate_space_slug(args.space_slug)
        manifest = read_installed_tool_spaces(instance_path)
        remaining_spaces = [space for space in manifest.spaces if space.slug != validated_slug]
        if len(remaining_spaces) == len(manifest.spaces):
            logger.warning("Space not installed: %s", validated_slug)
            return 1

        write_installed_tool_spaces(
            instance_path,
            InstalledToolSpacesManifest(version=manifest.version, spaces=remaining_spaces),
        )
        logger.info("Removed Space tool source: %s", validated_slug)
        for profile_name, disabled_tool_ids in _disable_space_tools_in_profiles(normalize_space_alias(validated_slug)):
            logger.info("Disabled in profile '%s': %s", profile_name, disabled_tool_ids)
        return 0

    if command == "list":
        manifest = read_installed_tool_spaces(instance_path)
        manifest_path = get_installed_tool_spaces_path(instance_path)
        logger.info("Manifest: %s", manifest_path)
        if not manifest.spaces:
            logger.info("No installed Space tool sources.")
            return 0

        for installed_space in manifest.spaces:
            if installed_space.source_kind == "generic_mcp":
                logger.info(
                    "%s (%s)\n  MCP endpoint: %s\n  Prompt: %s\n  Tools: %s",
                    installed_space.slug,
                    installed_space.alias,
                    installed_space.mcp_url,
                    installed_space.prompt_name,
                    ", ".join(tool.local_name for tool in installed_space.tools) or "none",
                )
                continue
            try:
                resolved_space = resolve_tool_space_sync(installed_space.slug)
            except Exception as exc:
                logger.warning("Space '%s' (%s) is unavailable: %s", installed_space.slug, installed_space.alias, exc)
                continue
            logger.info("%s", format_space_tool_listing(resolved_space))
        return 0

    raise RuntimeError(f"Unknown tool-spaces command: {command}")
