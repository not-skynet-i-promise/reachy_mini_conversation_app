"""Provider-, media-, and robot-free tests for private transcript routing."""

import uuid
import asyncio
import logging
import secrets
import traceback
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app import console as console_mod
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.conversation_handler import PrivateTranscriptRoute
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler


class _Event:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.type = payload["type"]
        self._payload = payload

    def model_dump(self, *, exclude_unset: bool = False) -> dict[str, Any]:
        assert exclude_unset
        return dict(self._payload)


class _Events:
    def __init__(self, events: list[_Event], *, stop_when_empty: bool = False) -> None:
        self._events: asyncio.Queue[_Event] = asyncio.Queue()
        self._stop_when_empty = stop_when_empty
        for event in events:
            self._events.put_nowait(event)

    def __aiter__(self) -> "_Events":
        return self

    async def __anext__(self) -> _Event:
        if self._stop_when_empty and self._events.empty():
            raise StopAsyncIteration
        return await self._events.get()


class _Resource:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]], name: str) -> None:
        self._calls = calls
        self._name = name

    async def update(self, **kwargs: Any) -> None:
        self._calls.append((f"{self._name}.update", kwargs))

    async def append(self, **kwargs: Any) -> None:
        self._calls.append((f"{self._name}.append", kwargs))

    async def create(self, **kwargs: Any) -> None:
        self._calls.append((f"{self._name}.create", kwargs))

    async def cancel(self, **kwargs: Any) -> None:
        self._calls.append((f"{self._name}.cancel", kwargs))


class _Connection:
    def __init__(self, events: list[_Event]) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sent: list[dict[str, Any]] = []
        self.session = _Resource(self.calls, "session")
        self.input_audio_buffer = _Resource(self.calls, "audio")
        self.response = _Resource(self.calls, "response")
        self.conversation = SimpleNamespace(item=_Resource(self.calls, "item"))
        self._events = _Events(events, stop_when_empty=True)

    async def __aenter__(self) -> "_Connection":
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False

    def __aiter__(self) -> _Events:
        return self._events

    async def send(self, payload: dict[str, Any]) -> None:
        # Model a websocket serialization boundary: the client does not retain
        # the caller-owned mutable payload after ``send`` returns.
        self.sent.append(deepcopy(payload))

    async def close(self) -> None:
        return None


def _handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


def _home_assistant_tool() -> hf_mod.core_tools.RemoteMcpTool:
    client = SimpleNamespace(
        server=SimpleNamespace(alias="home_assistant", url=hf_mod._HOME_ASSISTANT_MCP_URL, headers={})
    )
    return hf_mod.core_tools.RemoteMcpTool(
        slug="mcp/home_assistant",
        private=False,
        name="home_assistant__GetLiveContext",
        description="Read exposed state.",
        parameters_schema={"type": "object", "properties": {"area": {"type": "string"}}},
        client_tool_name="home_assistant__GetLiveContext",
        remote_name="GetLiveContext",
        client=client,
        retry_transport_failures=False,
        isolated_response=True,
    )


def _home_assistant_session_config() -> dict[str, Any]:
    return {
        "instructions": "Use exact tools.",
        "tools": [
            {
                "type": "function",
                "name": "home_assistant__GetLiveContext",
                "description": "Read exposed state.",
                "parameters": {"type": "object", "properties": {"area": {"type": "string"}}},
            }
        ],
    }


async def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Timed out waiting for condition")


async def _prime_accepted_response(
    handler: HuggingFaceRealtimeHandler,
    connection: Any,
) -> None:
    """Mirror one accepted barrier turn immediately before response.create."""
    key = ("ab" * 32, 1, "item-accepted")
    handler.connection = connection
    await handler._outbound_arbiter.bind(connection, negotiate=False)
    await handler._outbound_arbiter.begin_private_turn(connection, key)
    await handler._outbound_arbiter.send(connection, "barrier_resolve", AsyncMock())
    await handler._outbound_arbiter.complete_resolution(connection, key, accepted=True)


def _private_event(event_type: str, nonce: str, **fields: Any) -> _Event:
    return _Event(
        {
            "type": event_type,
            "event_id": f"event-{event_type}",
            "version": 1,
            "nonce": nonce,
            **fields,
        }
    )


def _replacement_created(*, item_id: str, transcript: str, previous_item_id: object) -> _Event:
    """Mirror the merged speech-to-speech conversation-item wire event."""
    return _Event(
        {
            "type": "conversation.item.created",
            "event_id": "created-1",
            "previous_item_id": previous_item_id,
            "item": {
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": transcript}],
            },
        }
    )


def test_private_router_rejects_voice_observer_composition_and_bad_timeout() -> None:
    """Identity routing cannot quietly inherit speaker evidence or an unbounded wait."""
    handler = _handler()

    async def route(_transcript: str) -> PrivateTranscriptRoute:
        return "accept_ordinary"

    async def observe(_utterance: Any) -> None:
        return None

    with pytest.raises(ValueError, match="timeout"):
        handler.set_private_transcript_router(route, timeout_seconds=0.0)
    handler.set_private_transcript_router(route)
    with pytest.raises(ValueError, match="cannot be combined"):
        handler.set_completed_utterance_observer(observe)


@pytest.mark.asyncio
async def test_connection_arbiter_drains_then_holds_every_ordinary_mutation() -> None:
    """A completed turn drains admitted work and then closes every ordinary lane."""
    arbiter = hf_mod._ConnectionOutboundArbiter()
    connection = object()
    await arbiter.bind(connection, negotiate=False)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_send() -> object:
        entered.set()
        await release.wait()
        return None

    first_send = asyncio.create_task(arbiter.send(connection, "audio_append", slow_send))
    await entered.wait()
    pending = asyncio.create_task(arbiter.begin_private_turn(connection, ("ab" * 32, 1, "item-1")))
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    await first_send
    await pending

    async def sent() -> object:
        return None

    for mutation in ("session_update", "audio_append", "item_create", "response_create", "response_cancel"):
        with pytest.raises(hf_mod._OutboundMutationBlocked):
            await arbiter.send(connection, mutation, sent)

    await arbiter.send(connection, "barrier_resolve", sent)
    await arbiter.complete_resolution(connection, ("ab" * 32, 1, "item-1"), accepted=True)
    await arbiter.send(connection, "response_create", sent)
    assert arbiter.state == "accepted_response_active"
    with pytest.raises(hf_mod._OutboundMutationBlocked):
        await arbiter.send(connection, "audio_append", sent)
    await arbiter.send(connection, "response_cancel", sent)
    await arbiter.finish_accepted_response(connection)
    assert str(arbiter.state) == "normal"


@pytest.mark.asyncio
async def test_home_assistant_guard_uses_exact_first_update_before_opening_arbiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The required guard must bind before any ordinary outbound mutation."""
    nonce = "19" * 32
    digest = "31998ba3b2f469ec18599044ba0cef9959859cac65c27dec9b77410d1e83e7b6"
    connection = _Connection(
        [
            _Event({"type": "session.created"}),
            _private_event(
                "reachy.home_assistant_guard.ready",
                nonce,
                session_contract_sha256=digest,
                tool_count=1,
            ),
        ]
    )
    handler = _handler()
    handler.set_require_home_assistant_guard(True)
    handler._session_tools_by_name = {"home_assistant__GetLiveContext": _home_assistant_tool()}
    monkeypatch.setattr(secrets, "token_hex", lambda _size: nonce)
    await handler._outbound_arbiter.bind(connection, negotiate=True)

    await handler._activate_private_extensions(connection, connection.__aiter__(), _home_assistant_session_config())

    assert handler._outbound_arbiter.state == "normal"
    assert handler._home_assistant_guard_nonce == nonce
    assert handler._home_assistant_guard_contract_sha256 == digest
    assert handler._home_assistant_guard_tool_count == 1
    assert connection.calls == [
        (
            "session.update",
            {
                "session": {
                    **_home_assistant_session_config(),
                    "reachy_home_assistant_guard": {
                        "version": 1,
                        "nonce": nonce,
                        "session_contract_sha256": digest,
                        "tool_count": 1,
                    },
                }
            },
        )
    ]


@pytest.mark.asyncio
async def test_combined_private_extensions_share_one_update_and_exact_ready_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both private extensions negotiate atomically in one first update."""
    transcript_nonce = "ab" * 32
    home_assistant_nonce = "cd" * 32
    digest = "31998ba3b2f469ec18599044ba0cef9959859cac65c27dec9b77410d1e83e7b6"
    connection = _Connection(
        [
            _Event({"type": "session.created"}),
            _private_event("reachy.transcript_barrier.ready", transcript_nonce),
            _private_event(
                "reachy.home_assistant_guard.ready",
                home_assistant_nonce,
                session_contract_sha256=digest,
                tool_count=1,
            ),
        ]
    )
    handler = _handler()

    async def route(_transcript: str) -> PrivateTranscriptRoute:
        return "accept_ordinary"

    handler.set_private_transcript_router(route)
    handler.set_require_home_assistant_guard(True)
    handler._session_tools_by_name = {"home_assistant__GetLiveContext": _home_assistant_tool()}
    nonces = iter((transcript_nonce, home_assistant_nonce))
    monkeypatch.setattr(secrets, "token_hex", lambda _size: next(nonces))
    await handler._outbound_arbiter.bind(connection, negotiate=True)

    await handler._activate_private_extensions(connection, connection.__aiter__(), _home_assistant_session_config())

    activation = connection.calls[0][1]["session"]
    assert activation["reachy_private_transcript_barrier"] == {"version": 1, "nonce": transcript_nonce}
    assert activation["reachy_home_assistant_guard"] == {
        "version": 1,
        "nonce": home_assistant_nonce,
        "session_contract_sha256": digest,
        "tool_count": 1,
    }
    assert len(connection.calls) == 1
    assert handler._outbound_arbiter.state == "normal"


@pytest.mark.asyncio
async def test_home_assistant_ready_mismatch_keeps_every_outbound_lane_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mismatched guard acknowledgement cannot open an outbound lane."""
    nonce = "19" * 32
    connection = _Connection(
        [
            _Event({"type": "session.created"}),
            _private_event(
                "reachy.home_assistant_guard.ready",
                nonce,
                session_contract_sha256="00" * 32,
                tool_count=1,
            ),
        ]
    )
    handler = _handler()
    handler.set_require_home_assistant_guard(True)
    handler._session_tools_by_name = {"home_assistant__GetLiveContext": _home_assistant_tool()}
    monkeypatch.setattr(secrets, "token_hex", lambda _size: nonce)
    await handler._outbound_arbiter.bind(connection, negotiate=True)

    with pytest.raises(hf_mod._HomeAssistantGuardProtocolError):
        await handler._activate_private_extensions(
            connection, connection.__aiter__(), _home_assistant_session_config()
        )

    assert handler._outbound_arbiter.state == "negotiating"
    assert not any(name in {"audio.append", "item.create", "response.create"} for name, _ in connection.calls)


@pytest.mark.asyncio
async def test_exact_rejection_reopens_only_accepted_response_create_lane() -> None:
    """An exact backend rejection permits one retry without reopening other lanes."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    await handler._outbound_arbiter.send(connection, "response_create", AsyncMock())

    await handler._outbound_arbiter.reject_accepted_response(connection)

    assert handler._outbound_arbiter.state == "accepted_response"
    for mutation in (
        "barrier_activate",
        "barrier_resolve",
        "session_update",
        "audio_append",
        "item_create",
        "response_cancel",
    ):
        with pytest.raises(hf_mod._OutboundMutationBlocked):
            await handler._outbound_arbiter.send(connection, mutation, AsyncMock())
    await handler._outbound_arbiter.send(connection, "response_create", AsyncMock())
    assert handler._outbound_arbiter.state == "accepted_response_active"


@pytest.mark.asyncio
async def test_exact_rejection_tombstones_late_attempt_then_retry_done_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late success from a rejected attempt is suppressed before one retry."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    monkeypatch.setattr(hf_mod, "_RESPONSE_REJECTION_RETRY_DELAY", 0.05)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await _wait_until(lambda: connection.response.create.await_count == 1)
    first_request = connection.response.create.await_args_list[0].kwargs
    first_marker = first_request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    assert handler._outbound_arbiter.state == "accepted_response_active"
    await handler._handle_realtime_error(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                event_id=first_request["event_id"],
                code="conversation_already_has_active_response",
                message="response rejected",
            ),
        )
    )

    assert handler._outbound_arbiter.state == "accepted_response"
    assert first_marker in handler._rejected_response_markers
    assert handler._active_response_marker is None
    late_response = SimpleNamespace(
        id="response-rejected-late",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: first_marker},
        status="completed",
    )
    late_created = SimpleNamespace(type="response.created", response=late_response)
    assert not handler._observe_response_created(late_created)
    assert handler._response_event_is_suppressed(late_created)
    late_done = SimpleNamespace(type="response.done", response=late_response)
    assert not handler._handle_response_done(late_done)
    assert handler._response_event_is_suppressed(late_done)
    late_tool = SimpleNamespace(
        type="response.function_call_arguments.done",
        response_id=late_response.id,
    )
    assert handler._response_event_is_suppressed(late_tool)
    assert handler._outbound_arbiter.state == "accepted_response"

    await _wait_until(lambda: connection.response.create.await_count == 2)
    second_request = connection.response.create.await_args_list[1].kwargs
    marker = second_request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="response-accepted",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="completed",
    )
    assert handler._observe_response_created(SimpleNamespace(type="response.created", response=response))
    handler._response_done_event.clear()
    done = SimpleNamespace(type="response.done", response=response)
    assert handler._handle_response_done(done)
    await handler._outbound_arbiter.finish_accepted_response(connection)

    await _wait_until(lambda: handler._active_response_marker is None)
    assert handler._outbound_arbiter.state == "normal"
    assert connection.response.create.await_count == 2
    sender.cancel()
    await sender


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_id",
    ["event-stale", None, "", 7],
)
async def test_untrusted_rejection_cannot_reopen_accepted_response_lane(event_id: object) -> None:
    """Unrelated, eventless, and malformed errors leave the accepted gate active."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    await handler._outbound_arbiter.send(connection, "response_create", AsyncMock())
    handler._active_response_event_id = "event-current"

    await handler._handle_realtime_error(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                event_id=event_id,
                code="conversation_already_has_active_response",
                message="untrusted rejection",
            ),
        )
    )

    assert handler._outbound_arbiter.state == "accepted_response_active"
    assert not handler._rejected_response_markers
    assert not handler._last_response_rejected
    assert not handler._response_started_or_rejected_event.is_set()
    with pytest.raises(hf_mod._OutboundMutationBlocked):
        await handler._outbound_arbiter.send(connection, "response_create", AsyncMock())
    with pytest.raises(hf_mod._OutboundMutationBlocked):
        await handler._outbound_arbiter.send(connection, "audio_append", AsyncMock())


@pytest.mark.asyncio
async def test_eventless_rejection_cannot_authorize_accepted_response_retry() -> None:
    """An uncorrelated error may not duplicate a later successful response."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await _wait_until(lambda: connection.response.create.await_count == 1)
    request = connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    await handler._handle_realtime_error(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                event_id=None,
                code="conversation_already_has_active_response",
                message="uncorrelated rejection",
            ),
        )
    )

    await asyncio.sleep(0.01)
    assert connection.response.create.await_count == 1
    assert handler._outbound_arbiter.state == "accepted_response_active"
    response = SimpleNamespace(
        id="response-after-eventless-error",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="completed",
    )
    assert handler._observe_response_created(SimpleNamespace(type="response.created", response=response))
    handler._response_done_event.clear()
    done = SimpleNamespace(type="response.done", response=response)
    assert handler._handle_response_done(done)
    await handler._outbound_arbiter.finish_accepted_response(connection)

    await _wait_until(lambda: handler._active_response_marker is None)
    assert connection.response.create.await_count == 1
    assert handler._outbound_arbiter.state == "normal"
    sender.cancel()
    await sender


@pytest.mark.asyncio
async def test_created_then_exact_rejection_terminates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contradictory rejection after created fails closed instead of duplicating."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await _wait_until(lambda: connection.response.create.await_count == 1)
    request = connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="response-created-before-error",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="completed",
    )
    assert handler._observe_response_created(SimpleNamespace(type="response.created", response=response))
    handler._response_done_event.clear()

    await handler._handle_realtime_error(
        SimpleNamespace(
            type="error",
            error=SimpleNamespace(
                event_id=request["event_id"],
                code="conversation_already_has_active_response",
                message="contradictory rejection",
            ),
        )
    )

    await _wait_until(lambda: restart.await_count == 1)
    assert connection.response.create.await_count == 1
    assert handler._outbound_arbiter.state == "closed"
    assert marker not in handler._rejected_response_markers
    assert response.id in handler._suppressed_response_ids
    late_done = SimpleNamespace(type="response.done", response=response)
    assert not handler._handle_response_done(late_done)
    assert handler._response_event_is_suppressed(late_done)
    assert handler._response_event_is_suppressed(
        SimpleNamespace(type="response.function_call_arguments.done", response_id=response.id)
    )
    sender.cancel()
    await sender
    assert restart.await_count == 1


@pytest.mark.asyncio
async def test_accepted_response_retry_exhaustion_closes_gate_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five exact rejections terminate the session instead of wedging its gate."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)
    monkeypatch.setattr(hf_mod, "_RESPONSE_REJECTION_RETRY_DELAY", 0.0)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    for attempt in range(5):
        await _wait_until(lambda: connection.response.create.await_count > attempt)
        request = connection.response.create.await_args_list[attempt].kwargs
        await handler._handle_realtime_error(
            SimpleNamespace(
                type="error",
                error=SimpleNamespace(
                    event_id=request["event_id"],
                    code="conversation_already_has_active_response",
                    message="response rejected",
                ),
            )
        )

    await _wait_until(lambda: restart.await_count == 1)
    assert handler._outbound_arbiter.state == "closed"
    assert connection.response.create.await_count == 5
    assert len(handler._rejected_response_markers) == 5
    with pytest.raises(hf_mod._OutboundMutationBlocked):
        await handler._outbound_arbiter.send(connection, "audio_append", AsyncMock())
    sender.cancel()
    await sender


@pytest.mark.asyncio
async def test_accepted_response_without_created_event_closes_gate_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sent request with no created event has a bounded fail-closed terminal."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await _wait_until(lambda: restart.await_count == 1)
    assert connection.response.create.await_count == 1
    assert handler._outbound_arbiter.state == "closed"
    sender.cancel()
    await sender


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_state", "expected_restarts"),
    [
        ("completed", "normal", 0),
        ("failed", "closed", 1),
        ("incomplete", "closed", 1),
        ("cancelled", "closed", 1),
        (None, "closed", 1),
        (7, "closed", 1),
    ],
)
async def test_matched_done_has_one_sender_owned_accepted_terminal(
    monkeypatch: pytest.MonkeyPatch,
    status: object,
    expected_state: str,
    expected_restarts: int,
) -> None:
    """Only an exact completed status reopens; every other status recovers once."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await _wait_until(lambda: connection.response.create.await_count == 1)
    request = connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="response-terminal-status",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="in_progress",
    )
    assert handler._observe_response_created(SimpleNamespace(type="response.created", response=response))
    handler._response_done_event.clear()
    response.status = status
    done = SimpleNamespace(type="response.done", response=response)

    assert handler._handle_response_done(done)
    # The receiver only classifies and signals. The sender owns settlement.
    assert handler._outbound_arbiter.state == "accepted_response_active"
    await _wait_until(lambda: handler._outbound_arbiter.state == expected_state)
    if expected_restarts:
        await _wait_until(lambda: restart.await_count == expected_restarts)
    else:
        restart.assert_not_awaited()

    assert not handler._handle_response_done(done)
    await handler._settle_accepted_response(connection, terminal="failed")
    await asyncio.sleep(0)
    assert restart.await_count == expected_restarts
    sender.cancel()
    await sender


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ["waiting_previous", "awaiting_created", "awaiting_done"])
async def test_cancelled_accepted_sender_closes_once_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    cancel_phase: str,
) -> None:
    """Sender cancellation is a failed terminal, never a wedged accepted gate."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)
    if cancel_phase == "waiting_previous":
        handler._response_done_event.clear()
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    if cancel_phase == "waiting_previous":
        await _wait_until(handler._pending_responses.empty)
    else:
        await _wait_until(lambda: connection.response.create.await_count == 1)
    if cancel_phase == "awaiting_done":
        request = connection.response.create.await_args.kwargs
        marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        response = SimpleNamespace(
            id="response-before-sender-cancel",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
            status="in_progress",
        )
        assert handler._observe_response_created(SimpleNamespace(type="response.created", response=response))
        handler._response_done_event.clear()

    sender.cancel()
    await sender

    assert handler._outbound_arbiter.state == "closed"
    await _wait_until(lambda: restart.await_count == 1)
    await handler._settle_accepted_response(connection, terminal="failed")
    await asyncio.sleep(0)
    assert restart.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_done", ["missing_response", "missing_marker"])
async def test_unmatched_done_times_out_to_one_accepted_recovery(
    monkeypatch: pytest.MonkeyPatch,
    malformed_done: str,
) -> None:
    """A missing or malformed terminal cannot reopen an accepted turn."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)
    monkeypatch.setattr(hf_mod, "_RESPONSE_STALL_TIMEOUT", 0.01)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await _wait_until(lambda: connection.response.create.await_count == 1)
    request = connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="response-before-malformed-done",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="in_progress",
    )
    assert handler._observe_response_created(SimpleNamespace(type="response.created", response=response))
    handler._response_done_event.clear()
    if malformed_done == "missing_response":
        done = SimpleNamespace(type="response.done")
    else:
        done = SimpleNamespace(
            type="response.done",
            response=SimpleNamespace(id=response.id, metadata={}, status="completed"),
        )
    assert not handler._handle_response_done(done)

    await _wait_until(lambda: restart.await_count == 1)
    assert handler._outbound_arbiter.state == "closed"
    await handler._settle_accepted_response(connection, terminal="failed")
    await asyncio.sleep(0)
    assert restart.await_count == 1
    sender.cancel()
    await sender


@pytest.mark.asyncio
async def test_accepted_response_send_timeout_terminates_without_replacement_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation-resistant SDK send is retained and blocks replacement."""
    handler = _handler()
    connection = AsyncMock()
    await _prime_accepted_response(handler, connection)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def resistant_create(**_kwargs: Any) -> None:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    connection.response.create.side_effect = resistant_create
    build_replacement = AsyncMock()
    start_replacement = MagicMock()
    monkeypatch.setattr(handler, "_build_realtime_client", build_replacement)
    monkeypatch.setattr(handler, "_start_realtime_restart_task", start_replacement)
    monkeypatch.setattr(hf_mod, "_RESPONSE_CREATE_TIMEOUT", 0.01)
    monkeypatch.setattr(hf_mod, "_OBSERVER_SESSION_STOP_TIMEOUT", 0.01)
    sender = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create()

    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    await _wait_until(lambda: handler._outbound_arbiter.state == "closed")
    await _wait_until(lambda: handler.connection is None)

    assert any(not task.done() for task in handler._late_response_create_tasks)
    assert not handler.shutdown_complete()
    build_replacement.assert_not_awaited()
    start_replacement.assert_not_called()

    release.set()
    await _wait_until(lambda: not handler._late_response_create_tasks)
    sender.cancel()
    await sender
    await _wait_until(lambda: not handler._realtime_restart_tasks)
    await _wait_until(lambda: not handler._shutdown_pending_tasks)


@pytest.mark.asyncio
async def test_failed_accepted_response_terminal_respects_supersession_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale terminal cannot close a replacement, and shutdown never restarts."""
    handler = _handler()
    old_connection = AsyncMock()
    await _prime_accepted_response(handler, old_connection)
    await handler._outbound_arbiter.send(old_connection, "response_create", AsyncMock())
    replacement = AsyncMock()
    handler.connection = replacement
    await handler._outbound_arbiter.bind(replacement, negotiate=False)
    restart = AsyncMock()
    monkeypatch.setattr(handler, "_restart_session", restart)

    await handler._settle_accepted_response(old_connection, terminal="failed")

    assert handler._outbound_arbiter.state == "normal"
    restart.assert_not_awaited()

    await _prime_accepted_response(handler, replacement)
    await handler._outbound_arbiter.send(replacement, "response_create", AsyncMock())
    handler._shutdown_requested = True
    await handler._settle_accepted_response(replacement, terminal="failed")

    assert handler._outbound_arbiter.state == "closed"
    restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_consume_identity_negotiates_and_stays_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The consume skeleton discards one exact held turn without any output authority."""
    nonce = "ab" * 32
    transcript = "My name is Private"
    events = [
        _Event({"type": "session.created"}),
        _private_event("reachy.transcript_barrier.ready", nonce),
        _private_event(
            "reachy.transcript_barrier.completed",
            nonce,
            sequence=1,
            item_id="input-1",
            transcript=transcript,
            language_code=None,
        ),
        _private_event(
            "reachy.transcript_barrier.resolved",
            nonce,
            sequence=1,
            input_item_id="input-1",
            replacement_item_id=None,
            action="discarded",
        ),
    ]
    connection = _Connection(events)
    handler = _handler()

    async def consume(normalized: str) -> PrivateTranscriptRoute:
        assert normalized == transcript
        return "consume_identity"

    handler.set_private_transcript_router(consume)
    handler.client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **_kwargs: connection))
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown = AsyncMock()
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf_mod, "has_private_mcp_local_realtime_boundary", lambda: True)
    monkeypatch.setattr(secrets, "token_hex", lambda _size: nonce)

    await handler._run_realtime_session()

    session_config = connection.calls[0][1]["session"]
    assert session_config["reachy_private_transcript_barrier"] == {"version": 1, "nonce": nonce}
    assert session_config["audio"]["input"]["turn_detection"]["create_response"] is False
    assert connection.sent == [
        {
            "type": "reachy.transcript_barrier.resolve",
            "version": 1,
            "nonce": nonce,
            "sequence": 1,
            "input_item_id": "input-1",
            "action": "discard",
        }
    ]
    assert handler.output_queue.empty()
    assert not any(name in {"item.create", "response.create", "audio.append"} for name, _ in connection.calls)


@pytest.mark.asyncio
async def test_accept_is_provisional_until_exact_resolved_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post-greeting replacement becomes ordinary only after its exact ack."""
    nonce = "cd" * 32
    transcript = "  ordinary   turn  "
    replacement_id = "msg_replacement"
    connection = _Connection([])
    handler = _handler()
    routed: list[str] = []

    async def accept(normalized: str) -> PrivateTranscriptRoute:
        routed.append(normalized)
        return "accept_ordinary"

    handler.set_private_transcript_router(accept)
    handler.connection = connection
    handler._private_transcript_nonce = nonce
    handler._safe_response_create = AsyncMock()
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="replacement"))
    await handler._outbound_arbiter.bind(connection, negotiate=False)

    completed = _private_event(
        "reachy.transcript_barrier.completed",
        nonce,
        sequence=1,
        item_id="input-1",
        transcript=transcript,
        language_code=None,
    )
    event_stream = _Events(
        [
            _replacement_created(
                item_id=replacement_id,
                transcript=transcript,
                previous_item_id="msg_startup",
            )
        ]
    )
    task = asyncio.create_task(handler._handle_private_completed(completed, event_stream, connection))
    while not connection.sent:
        await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert handler.output_queue.empty()
    handler._safe_response_create.assert_not_awaited()
    await event_stream._events.put(
        _private_event(
            "reachy.transcript_barrier.resolved",
            nonce,
            sequence=1,
            input_item_id="input-1",
            replacement_item_id=replacement_id,
            action="accepted",
        )
    )
    await task

    assert routed == ["ordinary turn"]
    assert connection.sent[0]["item"]["content"][0]["text"] == transcript
    output = handler.output_queue.get_nowait()
    assert isinstance(output, AdditionalOutputs)
    assert output.args == ({"role": "user", "content": "ordinary   turn"},)
    handler._safe_response_create.assert_awaited_once_with()
    assert handler._accepted_transcript_item_id == replacement_id
    assert handler._outbound_arbiter.state == "accepted_response"


@pytest.mark.parametrize("previous_item_id", [None, "msg_startup", "item_prior_history"])
def test_provisional_replacement_accepts_backend_owned_history(previous_item_id: str | None) -> None:
    """The merged backend may report no predecessor, the greeting, or later history."""
    transcript = "ordinary turn"
    replacement_id = "msg_replacement"

    HuggingFaceRealtimeHandler._validate_provisional_replacement(
        _replacement_created(
            item_id=replacement_id,
            transcript=transcript,
            previous_item_id=previous_item_id,
        ),
        item_id=replacement_id,
        transcript=transcript,
    )


@pytest.mark.parametrize("previous_item_id", ["", "x" * 257, 7, False])
def test_provisional_replacement_rejects_malformed_backend_history(previous_item_id: object) -> None:
    """Backend-owned lineage is accepted only as a bounded item identifier."""
    with pytest.raises(hf_mod._PrivateTranscriptProtocolError):
        HuggingFaceRealtimeHandler._validate_provisional_replacement(
            _replacement_created(
                item_id="msg_replacement",
                transcript="ordinary turn",
                previous_item_id=previous_item_id,
            ),
            item_id="msg_replacement",
            transcript="ordinary turn",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_completed_boundary_scrubs_full_router_failure_and_debug_trace(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    synchronous: bool,
) -> None:
    """The production session boundary leaves only a content-free sentinel."""
    handler = _handler()
    private_canary = "router-exception-private-transcript-canary"
    nonce = "f1" * 32
    completed = _private_event(
        "reachy.transcript_barrier.completed",
        nonce,
        sequence=1,
        item_id="input-private",
        transcript=private_canary,
        language_code=None,
    )
    connection = _Connection(
        [
            _Event({"type": "session.created"}),
            _private_event("reachy.transcript_barrier.ready", nonce),
            completed,
        ]
    )
    retained_failures: list[RuntimeError] = []
    retained_callback_tasks: list[asyncio.Task[Any]] = []

    if synchronous:

        def failing_router_sync(transcript: str) -> Any:
            failure = RuntimeError(f"callback failed with {transcript}")
            retained_failures.append(failure)
            raise failure

        failing_router: Any = failing_router_sync

    else:

        async def failing_router_async(transcript: str) -> PrivateTranscriptRoute:
            task = asyncio.current_task()
            assert task is not None
            retained_callback_tasks.append(task)
            failure = RuntimeError(f"callback failed with {transcript}")
            retained_failures.append(failure)
            raise failure

        failing_router = failing_router_async

    handler.set_private_transcript_router(failing_router)
    handler.client = SimpleNamespace(  # type: ignore[assignment]
        realtime=SimpleNamespace(connect=lambda **_kwargs: connection)
    )
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown = AsyncMock()
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf_mod, "has_private_mcp_local_realtime_boundary", lambda: True)
    monkeypatch.setattr(secrets, "token_hex", lambda _size: nonce)

    with pytest.raises(hf_mod._PrivateTranscriptProtocolError) as captured:
        await handler._run_realtime_session()

    failure = captured.value
    assert failure.__cause__ is None
    assert failure.__context__ is None
    rendered = "".join(traceback.format_exception(type(failure), failure, failure.__traceback__))
    assert private_canary not in rendered
    current = failure.__traceback__
    while current is not None:
        if current.tb_frame.f_code.co_filename.endswith("huggingface_realtime.py"):
            assert private_canary not in repr(current.tb_frame.f_locals)
        current = current.tb_next

    assert completed.__dict__ == {}
    assert len(retained_failures) == 1
    assert retained_failures[0].args == ()
    assert retained_failures[0].__traceback__ is None
    assert retained_failures[0].__cause__ is None
    assert retained_failures[0].__context__ is None
    assert private_canary not in repr(retained_failures[0].__dict__)
    for task in retained_callback_tasks:
        assert task.done()
        assert private_canary not in repr(task)
        task_failure = task.exception()
        assert task_failure is retained_failures[0]
        assert task_failure.args == ()
        assert task_failure.__traceback__ is None

    console_logger = logging.getLogger("reachy_mini_conversation_app.console")
    caplog.set_level(logging.DEBUG, logger=console_logger.name)
    console_logger.error(
        "Sanitized private router failure",
        exc_info=(type(failure), failure, failure.__traceback__),
    )
    assert private_canary not in caplog.text
    for record in caplog.records:
        assert private_canary not in record.getMessage()
        assert private_canary not in repr(record.__dict__)
    assert private_canary not in repr(handler.__dict__)
    assert not handler._private_transcript_router_tasks


@pytest.mark.asyncio
async def test_outer_cancellation_retains_cancellation_resistant_router() -> None:
    """Cancelling a turn cannot orphan a callback that suppresses cancellation."""
    handler = _handler()
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown_complete.return_value = True
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_router(_transcript: str) -> PrivateTranscriptRoute:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
            raise
        raise AssertionError("router unexpectedly resumed")

    handler.set_private_transcript_router(resistant_router)
    route = asyncio.create_task(handler._route_private_transcript("private turn"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    child = next(iter(handler._private_transcript_router_tasks))

    route.cancel()
    with pytest.raises(asyncio.CancelledError):
        await route
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    assert child in handler._owned_shutdown_tasks()
    assert not child.done()
    assert not handler.shutdown_complete()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await child
    await asyncio.sleep(0)
    assert handler.shutdown_complete()


@pytest.mark.asyncio
async def test_shutdown_retains_cancellation_resistant_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown completion remains false until a resistant router really exits."""
    handler = _handler()
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_router(_transcript: str) -> PrivateTranscriptRoute:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
            raise
        raise AssertionError("router unexpectedly resumed")

    handler.set_private_transcript_router(resistant_router)
    route = asyncio.create_task(handler._route_private_transcript("private turn"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    child = next(iter(handler._private_transcript_router_tasks))
    monkeypatch.setattr(hf_mod, "_HANDLER_SHUTDOWN_TASK_TIMEOUT", 0.01)

    await handler.shutdown()
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    assert child in handler._owned_shutdown_tasks()
    assert not child.done()
    assert not handler.shutdown_complete()

    release.set()
    with pytest.raises(hf_mod._PrivateTranscriptProtocolError):
        await route
    await asyncio.sleep(0)
    assert handler.shutdown_complete()


@pytest.mark.asyncio
async def test_reconnect_aborts_with_stopped_observer_while_resistant_router_is_live(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stopped observer is insufficient while its private child remains live."""
    handler = _handler()
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown_complete.return_value = True
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_router(_transcript: str) -> PrivateTranscriptRoute:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
            raise
        raise AssertionError("router unexpectedly resumed")

    handler.set_private_transcript_router(resistant_router)
    connection = SimpleNamespace(close=AsyncMock())
    handler.connection = connection
    handler.client = MagicMock()
    assert handler._observer_session_stopped.is_set()
    private_transcript = "private reconnect transcript"
    route = asyncio.create_task(handler._route_private_transcript(private_transcript))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    child = next(iter(handler._private_transcript_router_tasks))
    build_replacement = AsyncMock()
    start_replacement = MagicMock()
    monkeypatch.setattr(handler, "_build_realtime_client", build_replacement)
    monkeypatch.setattr(handler, "_start_realtime_restart_task", start_replacement)
    monkeypatch.setattr(hf_mod, "_OBSERVER_SESSION_STOP_TIMEOUT", 0.01)
    await handler._restart_session()
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    connection.close.assert_awaited_once_with()
    build_replacement.assert_not_awaited()
    start_replacement.assert_not_called()
    assert handler.connection is None
    assert child in handler._owned_shutdown_tasks()
    assert child in handler._shutdown_pending_tasks
    assert not child.done()
    assert not handler.shutdown_complete()
    assert private_transcript not in caplog.text

    release.set()
    with pytest.raises(hf_mod._PrivateTranscriptProtocolError):
        await route
    await asyncio.sleep(0)
    assert child not in handler._private_transcript_router_tasks
    assert child not in handler._shutdown_pending_tasks
    assert handler.shutdown_complete()
    assert private_transcript not in caplog.text


@pytest.mark.asyncio
async def test_reconnect_waits_for_superseded_router_exit_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement can start only after the superseded callback really exits."""
    handler = _handler()
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown_complete.return_value = True
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def resistant_router(_transcript: str) -> PrivateTranscriptRoute:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
            raise
        finally:
            order.append("router_done")
        raise AssertionError("router unexpectedly resumed")

    handler.set_private_transcript_router(resistant_router)
    connection = SimpleNamespace(close=AsyncMock())
    handler.connection = connection
    handler.client = MagicMock()
    route = asyncio.create_task(handler._route_private_transcript("private turn"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    child = next(iter(handler._private_transcript_router_tasks))

    async def build_replacement() -> MagicMock:
        assert child.done()
        order.append("replacement_built")
        return MagicMock()

    def start_replacement() -> asyncio.Task[None]:
        assert child.done()
        order.append("replacement_started")
        handler._connected_event.set()
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(handler, "_build_realtime_client", build_replacement)
    monkeypatch.setattr(handler, "_start_realtime_restart_task", start_replacement)
    restart = asyncio.create_task(handler._restart_session())
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert order == []
    assert not restart.done()

    release.set()
    with pytest.raises(hf_mod._PrivateTranscriptProtocolError):
        await route
    await restart

    assert order == ["router_done", "replacement_built", "replacement_started"]
    connection.close.assert_awaited_once_with()
    assert handler.connection is None
    assert child.done()
    assert child not in handler._private_transcript_router_tasks


@pytest.mark.asyncio
async def test_router_timeout_blocks_reconnect_until_resistant_child_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callback timeout cannot leave a child overlapping a replacement session."""
    handler = _handler()
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown_complete.return_value = True
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_router(_transcript: str) -> PrivateTranscriptRoute:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
            raise
        raise AssertionError("router unexpectedly resumed")

    handler.set_private_transcript_router(resistant_router, timeout_seconds=0.01)
    route = asyncio.create_task(handler._route_private_transcript("private turn"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    child = next(iter(handler._private_transcript_router_tasks))
    with pytest.raises(hf_mod._PrivateTranscriptProtocolError):
        await route
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    handler.client = MagicMock()
    build_replacement = AsyncMock()
    start_replacement = MagicMock()
    monkeypatch.setattr(handler, "_build_realtime_client", build_replacement)
    monkeypatch.setattr(handler, "_start_realtime_restart_task", start_replacement)
    monkeypatch.setattr(hf_mod, "_OBSERVER_SESSION_STOP_TIMEOUT", 0.01)
    await handler._restart_session()

    build_replacement.assert_not_awaited()
    start_replacement.assert_not_called()
    assert child in handler._shutdown_pending_tasks
    assert not handler.shutdown_complete()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await child
    await asyncio.sleep(0)
    assert child not in handler._private_transcript_router_tasks
    assert child not in handler._shutdown_pending_tasks
    assert handler.shutdown_complete()


@pytest.mark.asyncio
async def test_console_protocol_retry_refuses_live_resistant_router_generation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The generic console retry cannot build across a timed-out private child."""
    nonce = "f2" * 32
    private_canary = "console-retry-private-transcript-canary"
    completed = _private_event(
        "reachy.transcript_barrier.completed",
        nonce,
        sequence=1,
        item_id="input-private",
        transcript=private_canary,
        language_code=None,
    )
    connection = _Connection(
        [
            _Event({"type": "session.created"}),
            _private_event("reachy.transcript_barrier.ready", nonce),
            completed,
        ]
    )
    handler = _handler()
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def resistant_router(_transcript: str) -> PrivateTranscriptRoute:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
            raise
        raise AssertionError("router unexpectedly resumed")

    handler.set_private_transcript_router(resistant_router, timeout_seconds=0.01)
    client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **_kwargs: connection))
    build_client = AsyncMock(return_value=client)
    handler._build_realtime_client = build_client  # type: ignore[method-assign]
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown = AsyncMock()
    handler.tool_manager.shutdown_complete.return_value = True
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(clear_player=MagicMock())))
    stream = LocalStream(handler, robot)
    sleeps = 0

    async def stop_after_retry(_delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            stream._stop_event.set()

    monkeypatch.setattr(stream, "_sleep_or_restart_requested", stop_after_retry)
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: True)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf_mod, "has_private_mcp_local_realtime_boundary", lambda: True)
    monkeypatch.setattr(secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(hf_mod, "_OBSERVER_SESSION_STOP_TIMEOUT", 0.01)
    caplog.set_level(logging.DEBUG, logger=console_mod.logger.name)

    await stream._run_handler_startup_loop()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    child = next(iter(handler._private_transcript_router_tasks))
    assert build_client.await_count == 1
    assert completed.__dict__ == {}
    assert not child.done()
    assert child in handler._shutdown_pending_tasks
    assert not handler.shutdown_complete()
    assert private_canary not in caplog.text
    assert private_canary not in repr(stream.__dict__)
    for record in caplog.records:
        assert private_canary not in record.getMessage()
        assert private_canary not in repr(record.__dict__)

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await child
    await asyncio.sleep(0)
    assert handler.shutdown_complete()


@pytest.mark.asyncio
async def test_bad_ready_never_starts_tools_or_exposes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mismatched activation acknowledgement poisons startup before app work."""
    nonce = "ef" * 32
    connection = _Connection(
        [
            _Event({"type": "session.created"}),
            _private_event("reachy.transcript_barrier.ready", "00" * 32),
        ]
    )
    handler = _handler()

    async def accept(_normalized: str) -> PrivateTranscriptRoute:
        return "accept_ordinary"

    handler.set_private_transcript_router(accept)
    handler.client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **_kwargs: connection))
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown = AsyncMock()
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "has_private_mcp_local_realtime_boundary", lambda: True)
    monkeypatch.setattr(secrets, "token_hex", lambda _size: nonce)

    with pytest.raises(hf_mod._PrivateTranscriptProtocolError):
        await handler._run_realtime_session()

    handler.tool_manager.start_up.assert_not_called()
    assert handler.connection is None
    assert handler.output_queue.empty()
