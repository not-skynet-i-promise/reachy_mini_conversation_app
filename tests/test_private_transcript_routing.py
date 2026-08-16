"""Provider-, media-, and robot-free tests for private transcript routing."""

import uuid
import asyncio
import secrets
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.huggingface_realtime as hf_mod
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
        self.sent.append(payload)

    async def close(self) -> None:
        return None


def _handler() -> HuggingFaceRealtimeHandler:
    return HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))


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
            await release.wait()
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
