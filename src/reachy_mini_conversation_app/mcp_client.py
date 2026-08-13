"""Helpers for consuming remote MCP tools over HTTP(S).

This module validates remote endpoints, discovers tools, and maps calls/results
into the app's tool interface without mutating the local project environment or
downloading third-party Python code.
"""

from __future__ import annotations
import re
import json
import asyncio
from typing import TYPE_CHECKING, Any, Mapping, Sequence, AsyncIterator, cast
from datetime import timedelta
from contextlib import asynccontextmanager
from dataclasses import field, dataclass
from urllib.parse import urlparse

from jsonschema import SchemaError
from jsonschema.validators import validator_for


if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import Tool as McpTool
    from mcp.types import Prompt as McpPrompt
    from mcp.types import CallToolResult as McpCallToolResult


_NAME_SEGMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NAME_NORMALIZER_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
_LOCAL_HTTP_HOSTS = {"127.0.0.1", "localhost", "::1"}
_NAMESPACE_SEPARATOR = "__"
_PROMPT_MAX_BYTES = 32 * 1024
_PRIVATE_RESULT_MAX_BYTES = 16 * 1024
_CATALOG_MAX_TOOLS = 128
_CATALOG_MAX_BYTES = 256 * 1024
_CATALOG_MAX_PROMPTS = 128
_CATALOG_MAX_PAGES = 32
_CATALOG_CURSOR_MAX_CHARS = 256
_PRIVATE_OBJECT_SCHEMA_KEYS = frozenset(
    {"type", "title", "description", "properties", "required", "additionalProperties"}
)
_PRIVATE_SCALAR_SCHEMA_KEYS = frozenset(
    {
        "type",
        "title",
        "description",
        "enum",
        "const",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
    }
)
_PRIVATE_ARRAY_SCHEMA_KEYS = frozenset(
    {
        "type",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)
_PRIVATE_UNION_SCHEMA_KEYS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "anyOf",
    }
)


class McpClientError(RuntimeError):
    """Base error for the MCP client."""


class McpDependencyError(McpClientError):
    """Raised when a required MCP client dependency is not installed."""


class McpTransportError(McpClientError):
    """Raised when discovery fails before a remote tool runs."""


class McpToolInvocationError(McpClientError):
    """Raised when a remote tool call fails at the transport layer."""


class McpToolTimeoutError(McpToolInvocationError):
    """Raised when a remote tool call exceeds the configured timeout."""


class McpToolArgumentsRevokedError(McpClientError):
    """Raised when private tool arguments lose local lifecycle ownership."""


def _consume_late_task_result(task: asyncio.Task[dict[str, Any]]) -> None:
    """Own one cancellation-resistant transport task without leaking its exception."""
    if not task.cancelled():
        task.exception()


def scrub_private_mutable(value: Any) -> None:
    """Recursively clear shared mutable containers without copying them."""
    if isinstance(value, dict):
        while value:
            _, nested = value.popitem()
            scrub_private_mutable(nested)
    elif isinstance(value, list):
        while value:
            scrub_private_mutable(value.pop())
    elif isinstance(value, bytearray):
        value.clear()


class RevocableMcpToolArguments:
    """Own one mutable private argument map that teardown can synchronously revoke."""

    def __init__(self, arguments: dict[str, Any]) -> None:
        """Take ownership of one mutable argument map."""
        self._arguments = arguments
        self._revoked = False

    @property
    def revoked(self) -> bool:
        """Return whether the argument lease has been permanently revoked."""
        return self._revoked

    def borrow(self) -> dict[str, Any]:
        """Return the shared map only while this lease is live."""
        if self._revoked:
            raise McpToolArgumentsRevokedError("Private MCP tool arguments were revoked")
        return self._arguments

    def revoke(self) -> None:
        """Latch closed and recursively discard every app-owned mutable value."""
        self._revoked = True
        scrub_private_mutable(self._arguments)


class RevocableMcpToolResult:
    """Own one mutable private result map that teardown can synchronously revoke."""

    def __init__(self) -> None:
        """Create one empty live result lease."""
        self._result: dict[str, Any] | None = None
        self._revoked = False

    @property
    def revoked(self) -> bool:
        """Return whether this result lease has been permanently revoked."""
        return self._revoked

    def capture(self, result: dict[str, Any]) -> dict[str, Any]:
        """Retain and return the shared result map only while this lease is live."""
        if self._revoked or self._result is not None:
            scrub_private_mutable(result)
            return {"error": "Remote tool unavailable"}
        self._result = result
        return result

    def borrow(self) -> dict[str, Any] | None:
        """Return the shared result only while its private lease remains live."""
        return None if self._revoked else self._result

    def revoke(self) -> None:
        """Latch closed and recursively discard the shared result map, if present."""
        self._revoked = True
        if self._result is not None:
            scrub_private_mutable(self._result)
            self._result = None


def _require_name_segment(label: str, value: str) -> str:
    candidate = value.strip()
    if _NAME_SEGMENT_PATTERN.fullmatch(candidate) is None:
        raise ValueError(f"Invalid {label} '{value}'. Expected pattern '[A-Za-z_][A-Za-z0-9_]*'.")
    return candidate


def apply_name_normalization(value: str) -> str:
    """Replace non-identifier characters with underscores and collapse runs."""
    normalized = _NAME_NORMALIZER_PATTERN.sub("_", value).strip("_")
    return re.sub(r"_+", "_", normalized)


def _normalize_name_segment(label: str, value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError(f"{label.capitalize()} cannot be empty.")

    normalized = apply_name_normalization(raw)
    if not normalized:
        raise ValueError(f"{label.capitalize()} '{value}' cannot be normalized into a valid tool identifier.")
    if normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return _require_name_segment(label, normalized)


def validate_http_mcp_url(url: str) -> str:
    """Validate one credential-free HTTP(S) MCP endpoint."""
    if not isinstance(url, str) or url != url.strip() or any(character.isspace() for character in url):
        raise ValueError("MCP URL must be one plain credential-free HTTP(S) endpoint.")
    try:
        parsed = urlparse(url)
        parsed.port
    except ValueError as exc:
        raise ValueError("MCP URL must be one plain credential-free HTTP(S) endpoint.") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported MCP URL scheme '{parsed.scheme}'. Use http:// or https://.")
    if not parsed.netloc:
        raise ValueError("Invalid MCP URL. Missing host.")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("MCP URL must not contain credentials, query parameters, or a fragment.")

    host = (parsed.hostname or "").lower()
    if parsed.scheme == "http" and host not in _LOCAL_HTTP_HOSTS:
        raise ValueError("Remote MCP servers must use HTTPS. Plain HTTP is only allowed for localhost.")
    return url


def build_namespaced_tool_name(server_alias: str, tool_name: str) -> str:
    """Build a local tool name for a remote MCP tool."""
    alias = _require_name_segment("server alias", server_alias)
    tool_segment = _normalize_name_segment("tool name", tool_name)
    return f"{alias}{_NAMESPACE_SEPARATOR}{tool_segment}"


def _validate_private_property_schema(schema: object, *, allow_union: bool = True) -> dict[str, Any]:
    """Require a supported scalar, homogeneous scalar-array, or bounded union."""
    if not isinstance(schema, dict):
        raise ValueError("Remote MCP tool properties must contain schemas.")
    property_type = schema.get("type")
    if property_type in {"string", "integer", "number", "boolean"}:
        if not set(schema).issubset(_PRIVATE_SCALAR_SCHEMA_KEYS):
            raise ValueError("Remote MCP scalar schemas contain unsupported keywords.")
        return schema
    if property_type == "array" and set(schema).issubset(_PRIVATE_ARRAY_SCHEMA_KEYS):
        items = schema.get("items")
        if not isinstance(items, dict) or items.get("type") not in {
            "string",
            "integer",
            "number",
            "boolean",
        }:
            raise ValueError("Remote MCP tool arrays must contain one supported scalar schema.")
        if not set(items).issubset(_PRIVATE_SCALAR_SCHEMA_KEYS):
            raise ValueError("Remote MCP array item schemas contain unsupported keywords.")
        return schema

    branches = schema.get("anyOf")
    if (
        not allow_union
        or property_type is not None
        or not set(schema).issubset(_PRIVATE_UNION_SCHEMA_KEYS)
        or not isinstance(branches, list)
        or not 2 <= len(branches) <= 4
    ):
        raise ValueError("Remote MCP tool properties must use supported scalar, scalar-array, or union schemas.")
    array_item_types = {
        items.get("type")
        for branch in branches
        if isinstance(branch, dict)
        and branch.get("type") == "array"
        and isinstance((items := branch.get("items")), dict)
        and items.get("type") in {"string", "integer", "number", "boolean"}
    }
    normalized_branches: list[dict[str, Any]] = []
    for branch in branches:
        if branch == {}:
            if len(array_item_types) != 1:
                raise ValueError("Remote MCP union has an unconstrained branch.")
            branch = {"type": next(iter(array_item_types))}
        normalized_branches.append(_validate_private_property_schema(branch, allow_union=False))
    normalized = dict(schema)
    normalized["anyOf"] = normalized_branches
    return normalized


def _validate_object_schema(schema: object) -> dict[str, Any]:
    """Require one valid flat schema supported by the private authority gate."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Remote MCP tool input schema must be an object schema.")
    try:
        serialized = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        normalized = cast(dict[str, Any], json.loads(serialized))
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Remote MCP tool input schema must be valid JSON.") from exc
    normalized.setdefault("properties", {})
    properties = normalized["properties"]
    required = normalized.get("required", [])
    if (
        not set(normalized).issubset(_PRIVATE_OBJECT_SCHEMA_KEYS)
        or ("additionalProperties" in normalized and normalized["additionalProperties"] is not False)
        or not isinstance(properties, dict)
        or not isinstance(required, list)
        or not all(isinstance(name, str) and name in properties for name in required)
    ):
        raise ValueError("Remote MCP tool input schema has invalid properties or required fields.")
    for property_name, property_schema in properties.items():
        properties[property_name] = _validate_private_property_schema(property_schema)
    try:
        validator_type = validator_for(normalized)
        validator_type.check_schema(normalized)
    except (SchemaError, TypeError, ValueError) as exc:
        raise ValueError("Remote MCP tool input schema must be valid JSON Schema.") from exc
    return normalized


def validate_catalog_cache(prompt_text: object, tools: Sequence[RemoteToolSpec]) -> str:
    """Apply discovery's prompt/catalog bounds to persisted generic MCP metadata."""
    if not isinstance(prompt_text, str):
        raise ValueError("MCP prompt must be text.")
    normalized_prompt = prompt_text.strip()
    try:
        prompt_bytes = normalized_prompt.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("MCP prompt is not valid UTF-8 text.") from exc
    if not normalized_prompt or len(prompt_bytes) > _PROMPT_MAX_BYTES:
        raise ValueError(f"MCP prompt must be between 1 and {_PROMPT_MAX_BYTES} UTF-8 bytes.")
    if not tools or len(tools) > _CATALOG_MAX_TOOLS:
        raise ValueError(f"MCP tool catalog must contain 1-{_CATALOG_MAX_TOOLS} tools.")
    try:
        catalog_bytes = json.dumps(
            [spec.to_function_spec() for spec in tools],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("MCP tool catalog must contain valid bounded JSON schemas.") from exc
    if len(catalog_bytes) > _CATALOG_MAX_BYTES:
        raise ValueError(f"MCP tool catalog must be at most {_CATALOG_MAX_BYTES} UTF-8 bytes.")
    return normalized_prompt


def _dump_content_block(block: object) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        dumped = block.model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {"type": getattr(block, "type", "unknown")}


def _join_text_content(content_blocks: list[dict[str, Any]]) -> str | None:
    text_parts = [
        block["text"] for block in content_blocks if block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if not text_parts:
        return None
    return "\n\n".join(text_parts)


def _exception_contains_timeout(exc: BaseException) -> bool:
    timeout_exception = _httpx_timeout_exception_type()
    if isinstance(exc, timeout_exception):
        return True

    if "timed out" in str(exc).lower() or "deadline exceeded" in str(exc).lower():
        return True

    nested: list[BaseException] = []
    grouped_exceptions = getattr(exc, "exceptions", None)
    if isinstance(grouped_exceptions, tuple):
        nested.extend(grouped_exceptions)
    if exc.__cause__ is not None:
        nested.append(exc.__cause__)
    if exc.__context__ is not None:
        nested.append(exc.__context__)

    return any(_exception_contains_timeout(item) for item in nested)


def _load_mcp_sdk() -> tuple[type["ClientSession"], Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise McpDependencyError(
            "Remote MCP tools require the app's MCP client dependencies. Reinstall or update the app environment."
        ) from exc
    return ClientSession, streamable_http_client


def _load_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise McpDependencyError(
            "Remote MCP tools require the app's HTTP client dependencies. Reinstall or update the app environment."
        ) from exc
    return httpx


def _httpx_timeout_exception_type() -> tuple[type[BaseException], ...]:
    try:
        timeout_exception = _load_httpx().TimeoutException
    except McpDependencyError:
        return (TimeoutError,)
    return (TimeoutError, timeout_exception)


@dataclass(frozen=True)
class RemoteMcpServerConfig:
    """Allowlisted MCP server configuration."""

    alias: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    request_timeout_s: float = 10.0
    tool_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration once the dataclass has been created."""
        object.__setattr__(self, "alias", _require_name_segment("server alias", self.alias))
        object.__setattr__(self, "url", validate_http_mcp_url(self.url))
        object.__setattr__(self, "headers", {str(k): str(v) for k, v in self.headers.items()})
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be greater than zero.")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be greater than zero.")


@dataclass(frozen=True)
class RemoteToolSpec:
    """App-facing representation of a remote MCP tool."""

    server_alias: str
    remote_name: str
    namespaced_name: str
    description: str
    parameters_schema: dict[str, Any]

    def __post_init__(self) -> None:
        """Require the app's bounded object-argument function-tool contract."""
        _require_name_segment("server alias", self.server_alias)
        if not self.remote_name or len(self.remote_name) > 128:
            raise ValueError("Remote MCP tool name must contain 1-128 characters.")
        _require_name_segment("namespaced tool name", self.namespaced_name)
        if not isinstance(self.description, str):
            raise ValueError("Remote MCP tool description must be text.")
        object.__setattr__(self, "parameters_schema", _validate_object_schema(self.parameters_schema))

    @classmethod
    def from_mcp_tool(cls, server_alias: str, tool: "McpTool") -> "RemoteToolSpec":
        """Build an app-facing spec from an MCP SDK tool descriptor."""
        description = (getattr(tool, "description", None) or "").strip()
        parameters_schema = getattr(tool, "inputSchema", None)
        if not isinstance(parameters_schema, dict):
            raise ValueError("Remote MCP tool input schema must be an object schema.")

        remote_name = str(getattr(tool, "name", "")).strip()
        if not remote_name:
            raise ValueError("Remote MCP tool is missing a name.")

        return cls(
            server_alias=server_alias,
            remote_name=remote_name,
            namespaced_name=build_namespaced_tool_name(server_alias, remote_name),
            description=description or f"Remote MCP tool '{remote_name}' from server '{server_alias}'.",
            parameters_schema=dict(parameters_schema),
        )

    def to_function_spec(self) -> dict[str, Any]:
        """Translate to the app's function-calling shape."""
        return {
            "type": "function",
            "name": self.namespaced_name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


@dataclass(frozen=True)
class RemoteMcpCatalog:
    """One install-time MCP prompt and its complete cached tool catalog."""

    prompt_name: str
    prompt_text: str
    tools: list[RemoteToolSpec]


@dataclass(frozen=True)
class RemoteToolCallResponse:
    """Mapped result for a remote MCP tool call."""

    server_alias: str
    remote_tool_name: str
    namespaced_tool_name: str
    status: str
    content_blocks: list[dict[str, Any]]
    text: str | None
    structured_content: Any | None

    @classmethod
    def from_call_tool_result(
        cls,
        *,
        server_alias: str,
        remote_tool_name: str,
        result: "McpCallToolResult",
    ) -> "RemoteToolCallResponse":
        """Convert an MCP SDK tool result into the app's result envelope."""
        content_blocks = [_dump_content_block(block) for block in getattr(result, "content", [])]
        return cls(
            server_alias=server_alias,
            remote_tool_name=remote_tool_name,
            namespaced_tool_name=build_namespaced_tool_name(server_alias, remote_tool_name),
            status="error" if bool(getattr(result, "isError", False)) else "ok",
            content_blocks=content_blocks,
            text=_join_text_content(content_blocks),
            structured_content=getattr(result, "structuredContent", None),
        )

    def to_tool_result(self) -> dict[str, Any]:
        """Return a dict shaped like the app's tool results."""
        payload: dict[str, Any] = {
            "status": self.status,
            "server_alias": self.server_alias,
            "remote_tool_name": self.remote_tool_name,
            "namespaced_tool_name": self.namespaced_tool_name,
            "content_blocks": self.content_blocks,
        }
        if self.text is not None:
            payload["text"] = self.text
        if self.structured_content is not None:
            payload["structured_content"] = self.structured_content
        return payload


class RemoteMcpToolClient:
    """Minimal async client for allowlisted remote MCP tool servers."""

    def __init__(self, server: RemoteMcpServerConfig, known_tools: Sequence[RemoteToolSpec] = ()) -> None:
        """Store one allowlisted server configuration and an in-memory tool cache."""
        self.server = server
        self._tool_index = _index_remote_tools(list(known_tools))

    async def list_tool_specs(self) -> list[RemoteToolSpec]:
        """Discover remote tools and translate them into app-facing specs."""
        try:
            async with self._session() as session:
                discovered = await self._list_all_tools(session)
        except McpDependencyError:
            raise
        except Exception as exc:
            raise McpTransportError(
                f"Failed to discover MCP tools from '{self.server.alias}' at {self.server.url}: {exc}"
            ) from exc

        specs = [RemoteToolSpec.from_mcp_tool(self.server.alias, tool) for tool in discovered]
        self._tool_index = _index_remote_tools(specs)
        return specs

    async def list_function_specs(self) -> list[dict[str, Any]]:
        """Discover tools and translate them into function-calling specs."""
        return [spec.to_function_spec() for spec in await self.list_tool_specs()]

    async def discover_catalog(self, prompt_name: str) -> RemoteMcpCatalog:
        """Discover one exact no-argument text prompt and the complete tool catalog."""
        requested_prompt = prompt_name.strip()
        if not requested_prompt:
            raise ValueError("MCP prompt name cannot be empty.")
        try:
            async with self._session() as session:
                prompts = await self._list_all_prompts(session)
                matches = [prompt for prompt in prompts if getattr(prompt, "name", None) == requested_prompt]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected exactly one MCP prompt named '{requested_prompt}', found {len(matches)}."
                    )
                if getattr(matches[0], "arguments", None):
                    raise ValueError(f"MCP prompt '{requested_prompt}' must not require arguments.")
                prompt_result = await session.get_prompt(requested_prompt)
                discovered = await self._list_all_tools(session)
        except (McpDependencyError, ValueError):
            raise
        except Exception as exc:
            raise McpTransportError(
                f"Failed to discover MCP catalog from '{self.server.alias}' at {self.server.url}: {exc}"
            ) from exc

        messages = getattr(prompt_result, "messages", None)
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(f"MCP prompt '{requested_prompt}' must contain exactly one text message.")
        message = messages[0]
        content = getattr(message, "content", None)
        prompt_text = getattr(content, "text", None)
        if not isinstance(prompt_text, str):
            raise ValueError(f"MCP prompt '{requested_prompt}' must contain one text message.")
        specs = [RemoteToolSpec.from_mcp_tool(self.server.alias, tool) for tool in discovered]
        self._tool_index = _index_remote_tools(specs)
        normalized_prompt = validate_catalog_cache(prompt_text, specs)
        return RemoteMcpCatalog(
            prompt_name=requested_prompt,
            prompt_text=normalized_prompt,
            tools=specs,
        )

    async def call_tool(
        self,
        namespaced_tool_name: str,
        arguments: Mapping[str, Any] | RevocableMcpToolArguments | None = None,
    ) -> dict[str, Any]:
        """Invoke one tool inside a true end-to-end wall-clock deadline."""
        task = asyncio.create_task(self._call_tool_once(namespaced_tool_name, arguments))
        try:
            done, _ = await asyncio.wait({task}, timeout=self.server.tool_timeout_s)
        except BaseException:
            if isinstance(arguments, RevocableMcpToolArguments):
                arguments.revoke()
            task.cancel()
            task.add_done_callback(_consume_late_task_result)
            raise
        if not done:
            if isinstance(arguments, RevocableMcpToolArguments):
                arguments.revoke()
            task.cancel()
            task.add_done_callback(_consume_late_task_result)
            raise McpToolTimeoutError(
                f"Timed out calling MCP tool '{namespaced_tool_name}' from '{self.server.alias}'."
            )
        return task.result()

    async def _call_tool_once(
        self,
        namespaced_tool_name: str,
        arguments: Mapping[str, Any] | RevocableMcpToolArguments | None,
    ) -> dict[str, Any]:
        """Resolve, connect, initialize, call, and tear down one remote MCP tool."""
        if isinstance(arguments, RevocableMcpToolArguments):
            private_arguments: RevocableMcpToolArguments | None = arguments
            ordinary_arguments: Mapping[str, Any] | None = None
            arguments.borrow()
        else:
            private_arguments = None
            ordinary_arguments = arguments
        spec = await self._resolve_tool_spec(namespaced_tool_name)
        if private_arguments is not None:
            call_arguments = private_arguments.borrow()
        else:
            call_arguments = dict(ordinary_arguments or {})
        timeout_exception = _httpx_timeout_exception_type()

        try:
            async with self._session() as session:
                if private_arguments is not None:
                    call_arguments = private_arguments.borrow()
                result = await session.call_tool(
                    spec.remote_name,
                    arguments=call_arguments,
                    read_timeout_seconds=timedelta(seconds=self.server.tool_timeout_s),
                )
            if private_arguments is not None:
                private_arguments.borrow()
        except McpToolArgumentsRevokedError:
            raise
        except McpDependencyError:
            raise
        except timeout_exception as exc:
            if private_arguments is not None:
                private_arguments.borrow()
            raise McpToolTimeoutError(
                f"Timed out calling MCP tool '{namespaced_tool_name}' from '{self.server.alias}'."
            ) from exc
        except Exception as exc:
            if private_arguments is not None:
                private_arguments.borrow()
            if _exception_contains_timeout(exc):
                raise McpToolTimeoutError(
                    f"Timed out calling MCP tool '{namespaced_tool_name}' from '{self.server.alias}'."
                ) from exc
            raise McpToolInvocationError(
                f"Failed to call MCP tool '{namespaced_tool_name}' from '{self.server.alias}': {exc}"
            ) from exc

        if private_arguments is not None:
            payload: dict[str, Any] = {
                "server_alias": self.server.alias,
                "remote_tool_name": spec.remote_name,
                "namespaced_tool_name": namespaced_tool_name,
                "status": "error" if result.isError else "ok",
            }
            structured_content = result.structuredContent
            if structured_content is not None:
                try:
                    encoded_result = json.dumps(
                        structured_content,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                except (TypeError, ValueError, UnicodeEncodeError):
                    encoded_result = b""
                if not encoded_result or len(encoded_result) > _PRIVATE_RESULT_MAX_BYTES:
                    payload["status"] = "error"
                else:
                    payload["structured_content"] = structured_content
            else:
                content = getattr(result, "content", [])
                if len(content) == 1 and getattr(content[0], "type", None) == "text":
                    text = getattr(content[0], "text", None)
                    try:
                        text_bytes = text.encode("utf-8") if isinstance(text, str) else b""
                    except UnicodeEncodeError:
                        text_bytes = b""
                    if isinstance(text, str) and text_bytes and len(text_bytes) <= _PRIVATE_RESULT_MAX_BYTES:
                        payload["text"] = text
                    else:
                        payload["status"] = "error"
                else:
                    payload["status"] = "error"
            return payload

        return RemoteToolCallResponse.from_call_tool_result(
            server_alias=self.server.alias,
            remote_tool_name=spec.remote_name,
            result=result,
        ).to_tool_result()

    async def _resolve_tool_spec(self, namespaced_tool_name: str) -> RemoteToolSpec:
        spec = self._tool_index.get(namespaced_tool_name)
        if spec is not None:
            return spec

        await self.list_tool_specs()
        spec = self._tool_index.get(namespaced_tool_name)
        if spec is None:
            raise ValueError(f"Unknown remote MCP tool '{namespaced_tool_name}' for server '{self.server.alias}'.")
        return spec

    async def _list_all_tools(self, session: "ClientSession") -> list["McpTool"]:
        tools: list[McpTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_count = 0
        while True:
            page = await session.list_tools(cursor=cursor)
            page_count += 1
            page_tools = getattr(page, "tools", None)
            if not isinstance(page_tools, list):
                raise ValueError("MCP tool catalog page is malformed.")
            tools.extend(page_tools)
            if len(tools) > _CATALOG_MAX_TOOLS:
                raise ValueError(f"MCP tool catalog exceeds {_CATALOG_MAX_TOOLS} tools.")
            cursor = self._next_catalog_cursor(page, seen_cursors, page_count)
            if cursor is None:
                return tools

    async def _list_all_prompts(self, session: "ClientSession") -> list["McpPrompt"]:
        prompts: list[McpPrompt] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_count = 0
        while True:
            page = await session.list_prompts(cursor=cursor)
            page_count += 1
            page_prompts = getattr(page, "prompts", None)
            if not isinstance(page_prompts, list):
                raise ValueError("MCP prompt catalog page is malformed.")
            prompts.extend(page_prompts)
            if len(prompts) > _CATALOG_MAX_PROMPTS:
                raise ValueError(f"MCP prompt catalog exceeds {_CATALOG_MAX_PROMPTS} prompts.")
            cursor = self._next_catalog_cursor(page, seen_cursors, page_count)
            if cursor is None:
                return prompts

    @staticmethod
    def _next_catalog_cursor(page: object, seen: set[str], page_count: int) -> str | None:
        """Reject malformed, cyclic, or excessive pagination before another request."""
        cursor = getattr(page, "nextCursor", None)
        if cursor is None:
            return None
        if (
            not isinstance(cursor, str)
            or not cursor
            or len(cursor) > _CATALOG_CURSOR_MAX_CHARS
            or cursor in seen
            or page_count >= _CATALOG_MAX_PAGES
        ):
            raise ValueError("MCP catalog pagination is invalid or exceeds its bound.")
        seen.add(cursor)
        return cursor

    @asynccontextmanager
    async def _session(self) -> AsyncIterator["ClientSession"]:
        client_session_cls, streamable_http_client = _load_mcp_sdk()
        httpx = _load_httpx()
        client_timeout = max(self.server.request_timeout_s, self.server.tool_timeout_s)

        async with httpx.AsyncClient(
            headers=self.server.headers,
            follow_redirects=False,
            trust_env=False,
            timeout=client_timeout,
        ) as http_client:
            async with streamable_http_client(self.server.url, http_client=http_client) as transport:
                read_stream, write_stream, _ = transport
                async with client_session_cls(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session


def _index_remote_tools(specs: list[RemoteToolSpec]) -> dict[str, RemoteToolSpec]:
    index: dict[str, RemoteToolSpec] = {}
    collisions: dict[str, list[str]] = {}

    for spec in specs:
        existing = index.get(spec.namespaced_name)
        if existing is None:
            index[spec.namespaced_name] = spec
            continue

        collisions.setdefault(spec.namespaced_name, [existing.remote_name]).append(spec.remote_name)

    if collisions:
        details = "; ".join(
            f"{tool_name}: {sorted(remote_names)}" for tool_name, remote_names in sorted(collisions.items())
        )
        raise ValueError(f"Remote MCP tool names collide after local namespacing/normalization. Conflicts: {details}")

    return index
