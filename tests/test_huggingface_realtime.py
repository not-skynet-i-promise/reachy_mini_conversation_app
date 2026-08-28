import time
import base64
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, RealtimeToolResult
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


class _FakeEvent:
    """A minimal realtime event: a `type` plus arbitrary attributes."""

    def __init__(self, event_type: str, **fields: Any) -> None:
        """Store the event type and any extra attributes."""
        self.type = event_type
        self.__dict__.update(fields)


def _make_fake_realtime_client(
    *,
    events: tuple[_FakeEvent, ...] = (),
    captured_update: dict[str, Any] | None = None,
    captured_connect: dict[str, Any] | None = None,
    captured_items: list[dict[str, Any]] | None = None,
    deleted_items: list[str] | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if captured_update is not None:
                captured_update.update(kwargs)

    class FakeNoop:
        async def append(self, **_kw: Any) -> None:
            pass

        async def create(self, **kwargs: Any) -> None:
            if captured_items is not None and isinstance(kwargs.get("item"), dict):
                captured_items.append(kwargs["item"])

        async def cancel(self, **_kw: Any) -> None:
            pass

        async def delete(self, *, item_id: str, event_id: str | None = None) -> None:
            if deleted_items is not None:
                deleted_items.append(item_id)

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

        def __init__(self) -> None:
            self._events = iter(events)

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> _FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **kwargs: Any) -> FakeConn:
            if captured_connect is not None:
                captured_connect.update(kwargs)
            return FakeConn()

    class FakeClient:
        realtime = FakeRealtime()

    return FakeClient()


def _fake_openai_client(captured_kwargs: dict[str, Any]) -> type:
    """Return a fake AsyncOpenAI class that records its constructor kwargs."""

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    return FakeClient


def _fake_allocator(
    connect_url: str,
    posts: list[tuple[str, dict[str, str] | None, dict[str, str] | None]],
) -> type:
    """Return a fake httpx.AsyncClient that records allocator requests."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"session_id": "session-123", "connect_url": connect_url}

    class FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, str] | None = None,
        ) -> FakeResponse:
            posts.append((url, headers, json))
            return FakeResponse()

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_partial_transcription_uses_latest_snapshot(monkeypatch: Any) -> None:
    """Partial transcription snapshots should replace older snapshots for the same item."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey"),
            _FakeEvent(
                "conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey, how are you?"
            ),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert handler.input_transcript_chunks_by_item.item_id == "item-1"
    assert handler.input_transcript_chunks_by_item.deltas == ["Hey, how are you?"]


@pytest.mark.asyncio
async def test_emit_skips_idle_signal_while_response_active(monkeypatch: Any) -> None:
    """Idle tools should not trigger while a response is still active."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = HuggingFaceRealtimeHandler(deps)
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler._response_done_event.clear()

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_sender_releases_isolated_input_after_acceptance(monkeypatch: Any) -> None:
    """The sender should drop raw explicit input before waiting for response completion."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._response_done_event.set()

    async def accept_response(**kwargs: Any) -> None:
        handler._response_done_event.clear()
        handler._accept_correlated_response_create(
            _FakeEvent(
                "response.created",
                response=SimpleNamespace(metadata={hf_mod._RESPONSE_CREATE_ID_METADATA_KEY: kwargs["event_id"]}),
            )
        )

    handler.connection = SimpleNamespace(response=SimpleNamespace(create=accept_response))
    request = {"response": {"input": "RAW_RESULT_SENTINEL"}}
    response_payload = request["response"]
    await handler._pending_responses.put(request)
    sender = asyncio.create_task(handler._response_sender_loop())

    for _ in range(10):
        if not request:
            break
        await asyncio.sleep(0)

    assert request == {}
    assert response_payload == {}
    handler.connection = None
    handler._response_done_event.set()
    await sender


@pytest.mark.asyncio
async def test_response_sender_failure_does_not_log_isolated_input(caplog: Any) -> None:
    """A provider exception must not reflect private request content into logs."""
    sentinel = "RAW_ISOLATED_RESULT_SENTINEL"
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    async def fail_response_create(**_kwargs: Any) -> None:
        raise RuntimeError(sentinel)

    handler.connection = SimpleNamespace(response=SimpleNamespace(create=fail_response_create))
    request = {"response": {"input": sentinel, "conversation": "none"}}
    await handler._pending_responses.put(request)

    with caplog.at_level("DEBUG"):
        sender = asyncio.create_task(handler._response_sender_loop())
        for _ in range(20):
            if request == {}:
                break
            await asyncio.sleep(0)
        handler.connection = None
        handler._response_done_event.set()
        await handler._stop_response_sender(sender)

    assert request == {}
    assert sentinel not in caplog.text
    assert "response.create send failed (RuntimeError)" in caplog.text


@pytest.mark.asyncio
async def test_response_sender_preserves_isolated_request_behind_empty_request() -> None:
    """Ordinary response deduplication must never consume an isolated request."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._response_done_event.set()
    sent: list[dict[str, Any]] = []

    async def accept_response(**kwargs: Any) -> None:
        response = dict(kwargs.get("response", {}))
        metadata = dict(response.get("metadata", {}))
        metadata.pop(hf_mod._RESPONSE_CREATE_ID_METADATA_KEY, None)
        if metadata:
            response["metadata"] = metadata
        else:
            response.pop("metadata", None)
        sent.append({"response": response} if response else {})
        handler._accept_correlated_response_create(
            _FakeEvent(
                "response.created",
                response=SimpleNamespace(metadata={hf_mod._RESPONSE_CREATE_ID_METADATA_KEY: kwargs["event_id"]}),
            )
        )
        handler._response_done_event.set()
        if len(sent) == 2:
            handler.connection = None

    handler.connection = SimpleNamespace(response=SimpleNamespace(create=accept_response))
    ordinary: dict[str, Any] = {}
    isolated = {"response": {"input": "RAW_RESULT_SENTINEL", "conversation": "none"}}
    await handler._pending_responses.put(ordinary)
    await handler._pending_responses.put(isolated)

    await handler._response_sender_loop()

    assert sent == [{}, {"response": {"input": "RAW_RESULT_SENTINEL", "conversation": "none"}}]
    assert isolated == {}


@pytest.mark.asyncio
async def test_response_sender_ignores_unrelated_response_created() -> None:
    """An implicit response must not consume a queued isolated request."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._response_done_event.set()
    request = {"response": {"input": "RAW_RESULT_SENTINEL", "conversation": "none"}}
    saw_unrelated = asyncio.Event()

    async def accept_after_unrelated(**kwargs: Any) -> None:
        assert not handler._accept_correlated_response_create(
            _FakeEvent(
                "response.created",
                response=SimpleNamespace(metadata={hf_mod._RESPONSE_CREATE_ID_METADATA_KEY: "implicit_response"}),
            )
        )
        assert request["response"]["input"] == "RAW_RESULT_SENTINEL"
        saw_unrelated.set()
        assert handler._accept_correlated_response_create(
            _FakeEvent(
                "response.created",
                response=SimpleNamespace(metadata={hf_mod._RESPONSE_CREATE_ID_METADATA_KEY: kwargs["event_id"]}),
            )
        )
        handler._response_done_event.set()
        handler.connection = None

    handler.connection = SimpleNamespace(response=SimpleNamespace(create=accept_after_unrelated))
    await handler._pending_responses.put(request)

    await handler._response_sender_loop()

    assert saw_unrelated.is_set()
    assert request == {}


@pytest.mark.asyncio
async def test_response_sender_waits_for_implicit_response_after_correlated_rejection() -> None:
    """An active-response rejection must wait for that implicit response to finish."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._response_done_event.set()
    request = {"response": {"input": "RAW_RESULT_SENTINEL", "conversation": "none"}}
    attempts = 0

    async def reject_then_accept(**kwargs: Any) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            assert handler._reject_correlated_response_create(kwargs["event_id"], active_response=True)
            return
        handler._accept_correlated_response_create(
            _FakeEvent(
                "response.created",
                response=SimpleNamespace(metadata={hf_mod._RESPONSE_CREATE_ID_METADATA_KEY: kwargs["event_id"]}),
            )
        )
        handler._response_done_event.set()
        handler.connection = None

    handler.connection = SimpleNamespace(response=SimpleNamespace(create=reject_then_accept))
    await handler._pending_responses.put(request)
    sender = asyncio.create_task(handler._response_sender_loop())

    for _ in range(20):
        if attempts == 1:
            break
        await asyncio.sleep(0)
    assert attempts == 1
    await asyncio.sleep(hf_mod._RESPONSE_REJECTION_RETRY_DELAY * 2)
    assert attempts == 1

    handler._response_done_event.set()
    await sender

    assert attempts == 2
    assert request == {}


@pytest.mark.asyncio
async def test_parallel_tool_calls_trigger_single_response(monkeypatch: Any) -> None:
    """Parallel tool calls in one turn should yield one response, not one per completed tool."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    handler._in_flight_tool_calls = {"call_a", "call_b"}

    def _completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="test__parallel_probe",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    await handler._handle_tool_result(_completed("call_a"))
    assert create.await_count == 0

    await handler._handle_tool_result(_completed("call_b"))
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_failed_final_tool_result_terminates_partial_batch(monkeypatch: Any) -> None:
    """A partially delivered tool batch must close instead of stranding its follow-up."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    create_item = AsyncMock(side_effect=[None, RuntimeError("simulated delivery failure")])
    connection = SimpleNamespace(
        close=AsyncMock(),
        conversation=SimpleNamespace(item=SimpleNamespace(create=create_item)),
    )
    handler.connection = connection
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create_response)

    def completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="test__callback_recovery",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    handler._in_flight_tool_calls = {"call_first", "call_failed"}
    await handler._handle_tool_result(completed("call_first"))
    assert handler._tool_batch_needs_response

    with pytest.raises(RuntimeError, match="simulated delivery failure"):
        await handler._handle_tool_result(completed("call_failed"))

    assert handler._in_flight_tool_calls == set()
    assert not handler._tool_batch_needs_response
    assert handler.connection is None
    assert handler._private_tool_delete_terminal
    connection.close.assert_awaited_once()
    create_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_realtime_tool_result_uses_actual_handler_isolation(monkeypatch: Any, caplog: Any) -> None:
    """The actual handler should expose only a marker and queue an isolated response."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._in_flight_tool_calls.add("call_public")
    handler._redacted_tool_calls.add("call_public")
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    queue_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", queue_response)
    monkeypatch.setattr(
        hf_mod.core_tools,
        "get_tools",
        lambda: {"public_information": SimpleNamespace(expose_arguments=False, needs_response=True)},
    )
    raw_result = "RAW_PUBLIC_RESULT"
    isolated = RealtimeToolResult(
        model_status="handled_out_of_band",
        isolated_input=raw_result,
        isolated_instructions="Use only the supplied untrusted public result.",
    )

    with caplog.at_level("DEBUG"):
        await handler._handle_tool_result(
            ToolNotification(
                id="call_public",
                tool_name="public_information",
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result=isolated,
            )
        )

    assert handler.connection.conversation.item.create.await_args_list == [
        call(
            item={
                "type": "function_call",
                "call_id": "call_public",
                "name": "public_information",
                "arguments": "{}",
            }
        ),
        call(
            item={
                "type": "function_call_output",
                "call_id": "call_public",
                "output": '{"status": "handled_out_of_band"}',
            }
        ),
    ]
    assert handler.output_queue.empty()
    queued_response = queue_response.await_args.kwargs["response"]
    assert queued_response["conversation"] == queued_response["tool_choice"] == "none"
    assert "metadata" not in queued_response
    assert raw_result in str(queued_response["input"])
    assert raw_result not in caplog.text


@pytest.mark.parametrize(
    ("status", "input_text", "instructions"),
    (("raw result", "result", "narrate"), ("ok", "", "narrate"), ("ok", "result", "")),
)
def test_realtime_tool_result_rejects_unbounded_or_missing_fields(
    status: str, input_text: str, instructions: str
) -> None:
    """The result contract should accept only bounded markers and explicit inline text."""
    with pytest.raises(ValueError):
        RealtimeToolResult(
            model_status=status,
            isolated_input=input_text,
            isolated_instructions=instructions,
        )


@pytest.mark.asyncio
async def test_isolated_speech_uses_normal_audio_without_text_reentry(monkeypatch: Any) -> None:
    """An out-of-band answer should play audio without entering text sinks or history."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    answer = "The synthetic forecast is mild."
    pcm = b"\x01\x00\x02\x00"
    response = SimpleNamespace(
        id="response_public",
        conversation_id=None,
        metadata=None,
    )
    captured_items: list[dict[str, Any]] = []
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("response.created", response=response),
            _FakeEvent("response.done", response=response),
            _FakeEvent(
                "response.output_audio_transcript.done",
                response_id="response_public",
                transcript=answer,
            ),
            _FakeEvent("response.output_audio.delta", delta=base64.b64encode(pcm).decode("ascii")),
        ),
        captured_items=captured_items,
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    transcript_observer = MagicMock()
    handler.set_transcript_observer(transcript_observer)

    await handler._run_realtime_session()

    assert captured_items == []
    queued = [handler.output_queue.get_nowait()]
    assert queued[0][1].tobytes() == pcm
    assert not any(answer in str(item) for item in queued)
    transcript_observer.assert_not_called()
    assert not handler._isolated_response_ids


@pytest.mark.asyncio
async def test_private_tool_arguments_are_redacted_from_logs_and_status(monkeypatch: Any, caplog: Any) -> None:
    """A tool may keep its arguments out of logs and UI status while preserving dispatch."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    query = "private sentinel query"
    deleted_items: list[str] = []
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent(
                "response.function_call_arguments.done",
                name="public_information",
                call_id="call_public",
                item_id="item_private",
                arguments=f'{{"query":"{query}"}}',
            ),
            _FakeEvent("conversation.item.deleted", item_id="item_private"),
        ),
        deleted_items=deleted_items,
    )
    monkeypatch.setattr(
        hf_mod.core_tools, "get_tools", lambda: {"public_information": SimpleNamespace(expose_arguments=False)}
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="public_information-call_public"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)

    with caplog.at_level("DEBUG"):
        await handler._run_realtime_session()

    start_tool.assert_awaited_once()
    status = handler.output_queue.get_nowait()
    assert status.args[0]["content"].count("[redacted]") == 1
    assert query not in status.args[0]["content"]
    assert query not in caplog.text
    assert deleted_items == ["item_private"]


@pytest.mark.asyncio
async def test_private_tool_waits_for_delete_acknowledgement(monkeypatch: Any) -> None:
    """Private arguments must not reach a tool until the server confirms deletion."""
    query = "PRIVATE_ARGUMENT_SENTINEL"
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent(
                "response.function_call_arguments.done",
                name="private",
                call_id="call_private",
                item_id="fc_private",
                arguments=f'{{"query":"{query}"}}',
            ),
            _FakeEvent("conversation.item.deleted", item_id="fc_private"),
        )
    )
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"private": SimpleNamespace(expose_arguments=False)})
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    started = AsyncMock(return_value=SimpleNamespace(tool_id="private-call_private"))
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", started)

    await handler._run_realtime_session()

    started.assert_awaited_once()
    assert started.await_args.kwargs["tool_call_routine"].args_json_str == f'{{"query":"{query}"}}'
    assert not handler._pending_private_tool_calls


@pytest.mark.asyncio
async def test_private_tool_delete_rejection_fails_closed(monkeypatch: Any) -> None:
    """A backend without delete support must not execute the private tool."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent(
                "response.function_call_arguments.done",
                name="private",
                call_id="call_private",
                item_id="fc_private",
                arguments='{"query":"PRIVATE_ARGUMENT_SENTINEL"}',
            ),
            _FakeEvent(
                "error",
                error=SimpleNamespace(
                    event_id="event_fixed",
                    code="unknown_or_invalid_event",
                    type="invalid_request_error",
                    message="unsupported",
                ),
            ),
        )
    )
    monkeypatch.setattr(hf_mod.uuid, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"private": SimpleNamespace(expose_arguments=False)})
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    started = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", started)

    with pytest.raises(RuntimeError, match="deletion was rejected"):
        await handler._run_realtime_session()

    started.assert_not_awaited()
    assert not handler._pending_private_tool_calls


@pytest.mark.asyncio
async def test_private_tool_delete_timeout_keeps_guard_when_close_fails(monkeypatch: Any) -> None:
    """A failed transport close must not remove the private-history fence."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    async def fail_close() -> None:
        raise RuntimeError("transport close failed")

    connection = SimpleNamespace(close=fail_close)
    handler.connection = connection
    pending = hf_mod._PendingPrivateToolCall(
        event_id="event_private",
        item_id="item_private",
        tool_name="private",
        arguments='{"query":"PRIVATE_ARGUMENT_SENTINEL"}',
        call_id="call_private",
    )
    handler._pending_private_tool_calls[pending.item_id] = pending
    monkeypatch.setattr(hf_mod, "_PRIVATE_TOOL_DELETE_TIMEOUT", 0.0)

    await handler._expire_private_tool_delete(pending.item_id, pending.event_id)

    assert handler.connection is None
    assert handler._pending_private_tool_calls == {pending.item_id: pending}
    assert handler._private_tool_delete_terminal

    started = AsyncMock()
    monkeypatch.setattr(handler, "_start_realtime_tool_call", started)
    await handler._acknowledge_private_tool_delete(pending.item_id)

    started.assert_not_awaited()
    assert handler._pending_private_tool_calls == {pending.item_id: pending}


@pytest.mark.asyncio
async def test_private_delete_terminal_rejects_later_public_tool(monkeypatch: Any) -> None:
    """A failed-close iterator must not admit another tool after the privacy fence trips."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._private_tool_delete_terminal = True
    started = AsyncMock()
    monkeypatch.setattr(handler, "_start_realtime_tool_call", started)

    class TerminalThenToolConnection:
        session = SimpleNamespace(update=AsyncMock())

        def __init__(self) -> None:
            self._events = iter(
                [
                    _FakeEvent(
                        "response.function_call_arguments.done",
                        name="public",
                        call_id="call_public",
                        item_id="item_public",
                        arguments="{}",
                    )
                ]
            )

        async def __aenter__(self) -> "TerminalThenToolConnection":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        def __aiter__(self) -> "TerminalThenToolConnection":
            return self

        async def __anext__(self) -> _FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration from None

    connection = TerminalThenToolConnection()
    handler.client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **_kwargs: connection))
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    with pytest.raises(RuntimeError, match="before another tool call"):
        await handler._run_realtime_session()

    started.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_preserves_failed_close_terminal_until_receive_loop_stops(monkeypatch: Any) -> None:
    """Public shutdown must not reopen admission while a failed-close iterator may live."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    async def fail_close() -> None:
        raise RuntimeError("transport close failed")

    handler.connection = SimpleNamespace(close=fail_close)
    handler._pending_private_tool_calls["item_private"] = hf_mod._PendingPrivateToolCall(
        event_id="event_private",
        item_id="item_private",
        tool_name="private",
        arguments='{"query":"PRIVATE_ARGUMENT_SENTINEL"}',
        call_id="call_private",
    )
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler.shutdown()

    assert handler._private_tool_delete_terminal
    assert not handler._pending_private_tool_calls

    await handler._clear_private_tool_deletes(reset_terminal=True)
    assert not handler._private_tool_delete_terminal


@pytest.mark.asyncio
async def test_shutdown_scrubs_queued_isolated_response(monkeypatch: Any) -> None:
    """Public shutdown must release queued raw result data before reuse."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    connection = AsyncMock()
    handler.connection = connection
    request = {"response": {"input": "RAW_RESULT_SENTINEL", "conversation": "none"}}
    await handler._pending_responses.put(request)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler.shutdown()

    assert request == {}
    assert handler._pending_responses.empty()
    connection.response.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_scrubs_sender_owned_isolated_response(monkeypatch: Any) -> None:
    """Shutdown must cancel a stalled sender after it dequeues private input."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    send_started = asyncio.Event()

    async def stalled_create(**_kwargs: Any) -> None:
        send_started.set()
        await asyncio.Event().wait()

    async def failed_close() -> None:
        raise RuntimeError("transport close failed")

    handler.connection = SimpleNamespace(
        close=failed_close,
        response=SimpleNamespace(create=stalled_create),
    )
    request = {"response": {"input": "RAW_RESULT_SENTINEL", "conversation": "none"}}
    await handler._pending_responses.put(request)
    sender = asyncio.create_task(handler._response_sender_loop())
    handler._response_sender_task = sender
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await asyncio.wait_for(send_started.wait(), timeout=1.0)
    await handler.shutdown()

    assert sender.done()
    assert request == {}
    assert handler._response_sender_task is None


@pytest.mark.asyncio
async def test_shutdown_drops_late_events_from_failed_close_iterator(monkeypatch: Any) -> None:
    """A detached receive iterator must not emit speech-derived output."""

    class FailedCloseConnection:
        def __init__(self) -> None:
            self.session = SimpleNamespace(update=AsyncMock())
            self.events: asyncio.Queue[_FakeEvent] = asyncio.Queue()

        async def __aenter__(self) -> "FailedCloseConnection":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        def __aiter__(self) -> "FailedCloseConnection":
            return self

        async def __anext__(self) -> _FakeEvent:
            return await self.events.get()

        async def close(self) -> None:
            raise RuntimeError("transport close failed")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    connection = FailedCloseConnection()
    handler.client = SimpleNamespace(realtime=SimpleNamespace(connect=lambda **_kwargs: connection))
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    session = asyncio.create_task(handler._run_realtime_session())
    await asyncio.wait_for(handler._connected_event.wait(), timeout=1.0)
    await handler.shutdown()
    await connection.events.put(
        _FakeEvent(
            "conversation.item.input_audio_transcription.completed",
            transcript="LATE_PRIVATE_TRANSCRIPT",
        )
    )
    await asyncio.wait_for(session, timeout=1.0)

    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_realtime_sessions_serialize_cleanup_before_replacement(monkeypatch: Any) -> None:
    """A replacement cannot activate until the detached session has fully cleaned up."""

    class BlockingConnection:
        def __init__(self) -> None:
            self.session = SimpleNamespace(update=AsyncMock())
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def __aenter__(self) -> "BlockingConnection":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        def __aiter__(self) -> "BlockingConnection":
            return self

        async def __anext__(self) -> _FakeEvent:
            self.entered.set()
            await self.release.wait()
            raise StopAsyncIteration

        async def close(self) -> None:
            self.release.set()

    first = BlockingConnection()
    second = BlockingConnection()
    connections = iter((first, second))
    connect_count = 0

    def connect(**_kwargs: Any) -> BlockingConnection:
        nonlocal connect_count
        connect_count += 1
        return next(connections)

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = SimpleNamespace(realtime=SimpleNamespace(connect=connect))
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    shutdown = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", shutdown)

    old_session = asyncio.create_task(handler._run_realtime_session())
    await asyncio.wait_for(first.entered.wait(), timeout=1.0)
    handler._pending_private_tool_calls["old"] = MagicMock()

    replacement = asyncio.create_task(handler._run_realtime_session())
    await asyncio.sleep(0)
    assert connect_count == 1

    first.release.set()
    await asyncio.wait_for(old_session, timeout=1.0)
    await asyncio.wait_for(second.entered.wait(), timeout=1.0)
    assert connect_count == 2
    assert shutdown.await_count == 1
    assert not handler._pending_private_tool_calls

    handler._pending_private_tool_calls["new"] = MagicMock()
    await asyncio.sleep(0)
    assert "new" in handler._pending_private_tool_calls

    second.release.set()
    await asyncio.wait_for(replacement, timeout=1.0)


@pytest.mark.asyncio
async def test_correlated_realtime_error_is_content_free(monkeypatch: Any, caplog: Any) -> None:
    """An isolated response rejection must not expose reflected input in logs or UI."""
    sentinel = "RAW_ISOLATED_RESULT_SENTINEL"
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent(
                "error",
                error=SimpleNamespace(
                    event_id="sensitive_create",
                    code="invalid_request_error",
                    type="invalid_request_error",
                    message=sentinel,
                ),
            ),
        )
    )
    handler._pending_response_create_id = "sensitive_create"
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    with caplog.at_level("DEBUG"):
        await handler._run_realtime_session()

    queued = handler.output_queue.get_nowait()
    assert queued.args[0]["content"] == "[error] Realtime request failed"
    assert sentinel not in caplog.text
    assert sentinel not in str(queued.args)


@pytest.mark.asyncio
async def test_private_tool_failure_is_content_free(monkeypatch: Any, caplog: Any) -> None:
    """Opted-out tool exceptions should not expose their text in any returned channel."""

    class PrivateFailingTool:
        expose_arguments = False

        async def __call__(self, _deps: ToolDependencies, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("PRIVATE_FAILURE_SENTINEL")

    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"private": PrivateFailingTool()})
    with caplog.at_level("DEBUG"):
        result = await hf_mod.core_tools.dispatch_tool_call(
            "private",
            '{"query":"PRIVATE_ARGUMENT_SENTINEL"}',
            ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        )

    assert result == {"error": "Tool failed"}
    assert "PRIVATE_FAILURE_SENTINEL" not in caplog.text
    assert "PRIVATE_ARGUMENT_SENTINEL" not in caplog.text

    class PrivateErrorTool:
        expose_arguments = False

        async def __call__(self, _deps: ToolDependencies, **_kwargs: Any) -> dict[str, Any]:
            return {"error": "PRIVATE_RETURNED_ERROR_SENTINEL"}

    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"private": PrivateErrorTool()})
    returned_error = await hf_mod.core_tools.dispatch_tool_call(
        "private",
        "{}",
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
    )
    assert returned_error == {"error": "Tool failed"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._in_flight_tool_calls.add("call_private")
    handler._redacted_tool_calls.add("call_private")
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    queue_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", queue_response)

    with caplog.at_level("DEBUG"):
        await handler._handle_tool_result(
            ToolNotification(
                id="call_private",
                tool_name="private",
                is_idle_tool_call=False,
                status=ToolState.FAILED,
                error="PRIVATE_FAILURE_SENTINEL",
            )
        )

    submitted = handler.connection.conversation.item.create.await_args_list
    assert "PRIVATE_FAILURE_SENTINEL" not in str(submitted)
    assert "PRIVATE_FAILURE_SENTINEL" not in str(handler.output_queue.get_nowait())
    assert "PRIVATE_FAILURE_SENTINEL" not in caplog.text
    assert queue_response.await_count == 1


def test_handler_uses_hf_startup_voice_at_startup(monkeypatch: Any) -> None:
    """Hugging Face startup should restore persisted HF voices."""
    handler = HuggingFaceRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Aiden",
    )

    assert handler.get_current_voice() == "Aiden"


def test_handler_ignores_unsupported_hf_profile_voice(monkeypatch: Any) -> None:
    """Unsupported profile voices should not be sent to the Hugging Face backend."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "cedar")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == HF_DEFAULT_VOICE
    session = handler._get_session_config([])
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_handler_normalizes_hf_voice_case(monkeypatch: Any) -> None:
    """Lowercase Hugging Face speaker names should resolve to the curated UI value."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "serena")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_run_realtime_session_uses_default_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Use the backend default speaker when no profile voice is selected for the hf LB."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_update=captured_update)

    await handler._run_realtime_session()

    session = captured_update["session"]
    # HF at 16 kHz passes None so the backend uses its optimal default (16 kHz).
    assert session["audio"]["input"]["format"]["rate"] is None
    assert session["audio"]["output"]["format"]["rate"] is None
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_huggingface_session_uses_configured_transcription_language(monkeypatch: Any) -> None:
    """Hugging Face realtime sessions should forward the configured transcription language."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "zh")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    session = handler._get_session_config([])

    assert session["audio"]["input"]["transcription"]["language"] == "zh"


@pytest.mark.asyncio
async def test_run_realtime_session_passes_allocated_session_query(monkeypatch: Any) -> None:
    """Hugging Face sessions must forward the allocated session token to the websocket connect call."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    captured_connect: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_connect=captured_connect)
    handler._realtime_connect_query = {"session_token": "abc123"}

    await handler._run_realtime_session()

    assert "model" not in captured_connect
    assert captured_connect["extra_query"] == {"session_token": "abc123"}


@pytest.mark.parametrize(("hf_token", "expected_api_key"), [(None, "DUMMY"), ("hf-secret", "hf-secret")])
@pytest.mark.asyncio
async def test_build_realtime_client_local_uses_explicit_hf_token_only(
    monkeypatch: Any,
    hf_token: str | None,
    expected_api_key: str,
) -> None:
    """Local websocket mode must never forward cached Hugging Face credentials."""
    client_kwargs: dict[str, Any] = {}

    def _no_allocator(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session allocator should not be called in direct websocket mode")

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _no_allocator)
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)
    monkeypatch.setattr(hf_mod, "get_token", lambda: "hf-cached")
    monkeypatch.setattr(
        config,
        "HF_REALTIME_WS_URL",
        "ws://127.0.0.1:8765/v1/realtime?session_token=abc123&model=ignored-by-sdk",
    )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
    assert client_kwargs["websocket_base_url"] == "ws://127.0.0.1:8765/v1"
    assert handler._realtime_connect_query == {"session_token": "abc123"}


@pytest.mark.parametrize(
    (
        "hf_token",
        "cached_token",
        "hardware_id",
        "status_error",
        "expected_header",
        "expected_api_key",
        "expected_payload",
    ),
    [
        (
            "hf-secret",
            "hf-cached",
            "0123456789abcdef",
            None,
            {
                "User-Agent": "reachy-mini-conversation-app",
                "X-Reachy-Mini-Authorization": "Bearer hf-secret",
            },
            "hf-secret",
            {"hardware_id": "0123456789abcdef"},
        ),
        (
            None,
            "hf-cached",
            None,
            None,
            {
                "User-Agent": "reachy-mini-conversation-app",
                "X-Reachy-Mini-Authorization": "Bearer hf-cached",
            },
            "hf-cached",
            {},
        ),
        (None, None, None, None, {"User-Agent": "reachy-mini-conversation-app"}, "DUMMY", {}),
        (
            None,
            None,
            None,
            TimeoutError("status unavailable"),
            {"User-Agent": "reachy-mini-conversation-app"},
            "DUMMY",
            {},
        ),
    ],
)
@pytest.mark.asyncio
async def test_build_realtime_client_deployed_resolves_hf_token(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
    hf_token: str | None,
    cached_token: str | None,
    hardware_id: str | None,
    status_error: Exception | None,
    expected_header: dict[str, str],
    expected_api_key: str,
    expected_payload: dict[str, str],
) -> None:
    """Deployed allocation reports available credentials and robot identity."""
    client_kwargs: dict[str, Any] = {}
    posts: list[tuple[str, dict[str, str] | None, dict[str, str] | None]] = []
    connect_url = "wss://hf.example.test/v1/realtime?session_token=allocated"
    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _fake_allocator(connect_url, posts))
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    # A stale local URL must be ignored in deployed mode.
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)
    monkeypatch.setattr(hf_mod, "get_token", lambda: cached_token)

    reachy_mini = MagicMock()
    reachy_mini.client.get_status.return_value.hardware_id = hardware_id
    if status_error:
        reachy_mini.client.get_status.side_effect = status_error
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert posts == [("https://lb.example.test/session", expected_header, expected_payload)]
    reachy_mini.client.get_status.assert_called_once_with(wait=False)
    if status_error:
        assert "Daemon status unavailable for realtime session allocation" in caplog.text
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "https://hf.example.test/v1"
    assert client_kwargs["websocket_base_url"] == "wss://hf.example.test/v1"
    assert handler._realtime_connect_query == {"session_token": "allocated"}


@pytest.mark.asyncio
async def test_apply_personality_uses_selected_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Live personality updates should honor the selected Qwen CustomVoice speaker."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Serena")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    monkeypatch.setattr(handler, "_restart_session", AsyncMock(return_value=None))

    result = await handler.apply_personality("mars_rover")

    assert "restarted realtime session" in result.lower()
    session = captured_update["session"]
    assert session["instructions"] == "new instructions"
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_apply_personality_restores_profile_when_tools_fail(monkeypatch: Any) -> None:
    """A failed tool reload should leave the previous profile selected."""
    selected_profiles: list[str | None] = []

    def select_profile(profile: str | None) -> None:
        selected_profiles.append(profile)
        config.REACHY_MINI_CUSTOM_PROFILE = profile

    def fail_tool_reload(*, force: bool = False) -> None:
        raise RuntimeError("tool reload failed")

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(hf_mod, "set_custom_profile", select_profile)
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod.core_tools, "initialize_tools", fail_tool_reload)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    result = await handler.apply_personality("broken")

    assert result == "Failed to apply personality: tool reload failed"
    assert config.REACHY_MINI_CUSTOM_PROFILE == "default"
    assert selected_profiles == ["broken", "default"]


@pytest.mark.asyncio
async def test_change_voice_updates_live_hf_session_without_restart(monkeypatch: Any) -> None:
    """Changing Hugging Face voice should update the active session in place."""
    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    restart = AsyncMock(return_value=None)
    monkeypatch.setattr(handler, "_restart_session", restart)

    result = await handler.change_voice("Serena")

    assert result == "Voice changed to Serena."
    assert handler.get_current_voice() == "Serena"
    restart.assert_not_awaited()
    session = captured_update["session"]
    assert session["audio"]["output"]["voice"] == "Serena"
