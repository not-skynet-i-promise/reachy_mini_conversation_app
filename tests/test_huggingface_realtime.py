import json
import time
import base64
import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from collections.abc import Callable

import numpy as np
import pytest

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


@pytest.fixture(autouse=True)
def _bind_reviewed_search_source(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Give handler-only tests one already-attested official remote tool."""
    tool = MagicMock(spec=hf_mod.core_tools.RemoteMcpTool)
    tool.name = hf_mod._OFFICIAL_SEARCH_TOOL_NAME
    monkeypatch.setattr(
        hf_mod.core_tools,
        "resolve_expected_remote_mcp_tool",
        MagicMock(return_value=tool),
    )
    return tool


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
    session_update_error: BaseException | None = None,
    session_update_callback: Callable[[], None] | None = None,
    hold_open_until_close: bool = False,
    close_unblocks: bool = True,
    connection_exit_callback: Callable[[], None] | None = None,
    connection_exit_started: asyncio.Event | None = None,
    connection_exit_gate: asyncio.Event | None = None,
    connection_exit_error: BaseException | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if session_update_error is not None:
                raise session_update_error
            if session_update_callback is not None:
                session_update_callback()
            if captured_update is not None:
                captured_update.update(kwargs)

    class FakeNoop:
        async def append(self, **_kw: Any) -> None:
            pass

        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            pass

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

        def __init__(self) -> None:
            self._events = iter(events)
            self._closed = asyncio.Event()

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            if connection_exit_started is not None:
                connection_exit_started.set()
            if connection_exit_gate is not None:
                await connection_exit_gate.wait()
            if connection_exit_callback is not None:
                connection_exit_callback()
            if connection_exit_error is not None:
                raise connection_exit_error
            return False

        async def close(self) -> None:
            if close_unblocks:
                self._closed.set()

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> _FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                if hold_open_until_close and not self._closed.is_set():
                    await self._closed.wait()
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


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    """Wait for an asynchronous test condition."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Timed out waiting for condition")


async def _allow_search_space_gate() -> bool:
    """Approve the fixed metadata preflight in handler-only offline tests."""
    return True


async def _accept_response(
    handler: HuggingFaceRealtimeHandler,
    request_index: int = 0,
    response_id: str | None = None,
) -> dict[str, Any]:
    """Acknowledge one metadata-tagged response request from the fake connection."""
    await _wait_until(lambda: handler.connection.response.create.await_count > request_index)
    request = handler.connection.response.create.await_args_list[request_index].kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id=response_id or f"resp-{request_index}",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="completed",
    )
    handler._response_done_event.clear()
    assert handler._observe_response_created(_FakeEvent("response.created", response=response))
    done = _FakeEvent("response.done", response=response)
    assert handler._observe_response_done(done)
    handler._finish_response_suppression(done)
    return request


def _official_search_result(query: str, *, snippet: str = "A bounded result.") -> dict[str, Any]:
    """Return one exact synthetic official-search result envelope."""
    return {
        "status": "ok",
        "server_alias": hf_mod._OFFICIAL_SEARCH_SERVER_ALIAS,
        "remote_tool_name": hf_mod._OFFICIAL_SEARCH_REMOTE_NAME,
        "namespaced_tool_name": hf_mod._OFFICIAL_SEARCH_TOOL_NAME,
        "tool_space_slug": hf_mod._OFFICIAL_SEARCH_SPACE_SLUG,
        "content_blocks": [],
        "structured_content": {
            "query": query,
            "results": [
                {
                    "title": "Current result",
                    "snippet": snippet,
                    "url": "https://example.com/current",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_completed_utterance_observer_is_opt_in_and_retains_only_successful_audio() -> None:
    """Disabled sessions remain unchanged and failed sends never enter the ring."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    frame = np.arange(160, dtype=np.int16)

    default_turn_detection = handler._get_session_config([])["audio"]["input"]["turn_detection"]
    assert default_turn_detection == {"type": "server_vad", "interrupt_response": True}
    assert "create_response" not in default_turn_detection
    await handler.receive((handler.SAMPLE_RATE, frame))
    assert handler._audio_ring == bytearray()
    assert handler._audio_ring_end_sample == 0

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    connection = handler.connection
    handler.connection = None
    handler.set_completed_utterance_observer(observer)
    handler.connection = connection
    observed_turn_detection = handler._get_session_config([])["audio"]["input"]["turn_detection"]
    assert observed_turn_detection["create_response"] is False
    assert observed_turn_detection["interrupt_response"] is True

    handler.connection.input_audio_buffer.append.side_effect = RuntimeError("send failed")
    await handler.receive((handler.SAMPLE_RATE, frame))
    assert handler._audio_ring == bytearray()
    assert handler._audio_ring_end_sample == 0

    handler.connection.input_audio_buffer.append.side_effect = None
    await handler.receive((handler.SAMPLE_RATE, frame))
    assert handler._audio_ring == bytearray(frame.tobytes())
    assert handler._audio_ring_end_sample == frame.size
    with pytest.raises(RuntimeError, match="cannot change during a realtime session"):
        handler.set_completed_utterance_observer(None)


@pytest.mark.asyncio
async def test_observer_slices_context_and_discards_only_completed_audio() -> None:
    """One observer result precedes one response while later PCM remains buffered."""
    observed: list[conv_mod.CompletedUserUtterance] = []

    async def observer(utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        observed.append(utterance)
        return {"status": "matched", "display_name": " Test Person ", "score": "private"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(480, dtype=np.int16)
    for frame in np.split(samples, 3):
        await handler.receive((handler.SAMPLE_RATE, frame))

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=5)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=25)
    )
    observer_task = handler._utterance_observer_task
    assert observer_task is not None
    await observer_task
    assert len(observed) == 1

    later_samples = np.arange(480, 640, dtype=np.int16)
    await handler.receive((handler.SAMPLE_RATE, later_samples))
    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None

    request = await _accept_response(handler)
    await completion_task
    await _wait_until(lambda: handler._active_response_marker is None)
    sender_task.cancel()
    await sender_task

    context_input = request["response"]["input"]
    assert [item["type"] for item in context_input] == ["function_call", "function_call_output"]
    assert context_input[0]["name"] == hf_mod._UTTERANCE_CONTEXT_FUNCTION_NAME
    assert context_input[0]["call_id"] == context_input[1]["call_id"]
    assert json.loads(context_input[1]["output"]) == {
        "status": "matched",
        "display_name": "Test Person",
    }
    assert observed[0].item_id == "item-1"
    assert observed[0].sample_rate == handler.SAMPLE_RATE
    assert observed[0].pcm16 == samples[80:400].tobytes()
    expected_tail = np.concatenate((samples[400:], later_samples))
    assert handler._audio_ring_start_sample == 400
    assert handler._audio_ring == bytearray(expected_tail.tobytes())


@pytest.mark.asyncio
async def test_observer_work_overlaps_transcript_delay() -> None:
    """A ready observer result does not add its runtime after transcription."""
    observer_started = asyncio.Event()
    observer_release = asyncio.Event()

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        observer_started.set()
        await observer_release.wait()
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(160, dtype=np.int16)
    await handler.receive((handler.SAMPLE_RATE, samples))

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    observer_task = handler._utterance_observer_task
    assert observer_task is not None
    await observer_started.wait()
    observer_release.set()
    assert await observer_task == {"status": "unknown"}

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    request = await _accept_response(handler)
    await completion_task
    sender_task.cancel()
    await sender_task

    assert json.loads(request["response"]["input"][1]["output"]) == {"status": "unknown"}


@pytest.mark.asyncio
async def test_soft_stop_reopen_concatenates_exact_segments() -> None:
    """A reopened backend item is assessed from all of its ordered segments."""
    observed: list[conv_mod.CompletedUserUtterance] = []

    async def observer(utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        observed.append(utterance)
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(640, dtype=np.int16)
    await handler.receive((handler.SAMPLE_RATE, samples))

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    assert handler._utterance_observer_task is not None
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=20)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=30)
    )
    assert handler._utterance_observer_task is not None

    expected = np.concatenate((samples[:160], samples[320:480]))
    assert observed == []
    assert handler._utterance_spans == [(0, 160), (320, 480)]

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    request = await _accept_response(handler)
    await completion_task
    sender_task.cancel()
    await sender_task

    assert len(observed) == 1
    assert observed[0].pcm16 == expected.tobytes()
    assert json.loads(request["response"]["input"][1]["output"]) == {"status": "unknown"}
    assert handler._audio_ring_start_sample == 480
    assert handler._audio_ring == bytearray(samples[480:].tobytes())


@pytest.mark.asyncio
async def test_completed_revision_reopen_retains_prior_audio_until_response_done() -> None:
    """A response-created/reopen race reassesses the combined same-item audio."""
    observed: list[conv_mod.CompletedUserUtterance] = []

    async def observer(utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        observed.append(utterance)
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(640, dtype=np.int16)
    await handler.receive((handler.SAMPLE_RATE, samples))

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    first_completion = handler._utterance_completion_task
    assert first_completion is not None

    await _wait_until(lambda: handler.connection.response.create.await_count == 1)
    first_request = handler.connection.response.create.await_args_list[0].kwargs
    first_marker = first_request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    first_response = SimpleNamespace(
        id="resp-0",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: first_marker},
    )
    handler._response_done_event.clear()
    assert handler._observe_response_created(_FakeEvent("response.created", response=first_response))
    await first_completion
    assert handler._utterance_span_pcm == [samples[:160].tobytes()]
    assert observed[0].pcm16 == samples[:160].tobytes()

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=20)
    )
    handler.connection.response.cancel.assert_awaited_once()
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=30)
    )
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello again",
    )
    second_completion = handler._utterance_completion_task
    assert second_completion is not None

    cancelled_response = SimpleNamespace(
        id="resp-0",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: first_marker},
        status="cancelled",
    )
    cancelled_done = _FakeEvent("response.done", response=cancelled_response)
    assert handler._observe_response_done(cancelled_done)
    handler._finish_response_suppression(cancelled_done)

    second_request = await _accept_response(handler, request_index=1)
    await second_completion
    sender_task.cancel()
    await sender_task

    expected = np.concatenate((samples[:160], samples[320:480]))
    assert len(observed) == 2
    assert observed[1].item_id == "item-1"
    assert observed[1].pcm16 == expected.tobytes()
    assert json.loads(second_request["response"]["input"][1]["output"]) == {"status": "unknown"}
    assert handler._audio_ring_start_sample == 480
    assert handler._audio_ring == bytearray(samples[480:].tobytes())


@pytest.mark.parametrize("status", ["failed", "incomplete"])
@pytest.mark.asyncio
async def test_terminal_response_failure_releases_current_utterance_audio(status: str) -> None:
    """Every terminal current response releases its bounded PCM reservation."""

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(160, dtype=np.int16)
    await handler.receive((handler.SAMPLE_RATE, samples))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    await _wait_until(lambda: handler.connection.response.create.await_count == 1)
    request = handler.connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="resp-terminal",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status=status,
    )
    assert handler._observe_response_created(_FakeEvent("response.created", response=response))
    await completion_task
    assert handler._utterance_span_pcm_bytes == len(samples.tobytes())

    done = _FakeEvent("response.done", response=response)
    assert handler._observe_response_done(done)
    handler._finish_response_suppression(done)
    assert handler._utterance_span_pcm == []
    assert handler._utterance_span_pcm_bytes == 0

    await _wait_until(lambda: handler._active_utterance_token is None)
    sender_task.cancel()
    await sender_task


@pytest.mark.asyncio
async def test_late_response_done_releases_audio_after_sender_timeout(monkeypatch: Any) -> None:
    """Response-ID ownership outlives sender bookkeeping long enough to release PCM."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(160, dtype=np.int16)
    await handler.receive((handler.SAMPLE_RATE, samples))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    await _wait_until(lambda: handler.connection.response.create.await_count == 1)
    request = handler.connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="resp-late",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="completed",
    )
    assert handler._observe_response_created(_FakeEvent("response.created", response=response))
    await completion_task
    await _wait_until(lambda: handler._active_response_marker is None)
    assert handler._utterance_span_pcm_bytes == len(samples.tobytes())

    done = _FakeEvent("response.done", response=response)
    assert not handler._observe_response_done(done)
    assert handler._utterance_span_pcm == []
    assert handler._utterance_span_pcm_bytes == 0
    handler._finish_response_suppression(done)

    sender_task.cancel()
    await sender_task


@pytest.mark.asyncio
async def test_stopped_near_cap_span_survives_later_frames_until_transcript() -> None:
    """Stopped PCM is snapshotted without increasing the aggregate 15-second cap."""
    observed: list[conv_mod.CompletedUserUtterance] = []

    async def observer(utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        observed.append(utterance)
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(240_000, dtype=np.int32).astype(np.int16)

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler.receive((handler.SAMPLE_RATE, samples))
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=15_000)
    )
    assert handler._utterance_span_pcm_bytes == hf_mod._UTTERANCE_AUDIO_MAX_BYTES
    assert handler._audio_ring == bytearray()

    later_samples = np.arange(240_000, 240_160, dtype=np.int32).astype(np.int16)
    await handler.receive((handler.SAMPLE_RATE, later_samples))
    assert len(handler._audio_ring) + handler._utterance_span_pcm_bytes == (hf_mod._UTTERANCE_AUDIO_MAX_BYTES)

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    await _accept_response(handler)
    await completion_task
    sender_task.cancel()
    await sender_task

    assert len(observed) == 1
    assert observed[0].pcm16 == samples.tobytes()
    assert handler._utterance_span_pcm_bytes == 0


@pytest.mark.asyncio
async def test_audio_ring_cap_and_missing_boundary_fail_unavailable() -> None:
    """The 15-second cap never guesses at an evicted utterance boundary."""
    observer = AsyncMock(return_value={"status": "matched", "display_name": "Test Person"})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    samples = np.arange(250_000, dtype=np.int32).astype(np.int16)
    await handler.receive((handler.SAMPLE_RATE, samples))

    assert len(handler._audio_ring) == hf_mod._UTTERANCE_AUDIO_MAX_BYTES
    assert handler._audio_ring_start_sample == 10_000
    assert handler._audio_ring_end_sample == 250_000

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=15_625)
    )
    assert handler._utterance_observer_task is None
    observer.assert_not_awaited()

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    request = await _accept_response(handler)
    await completion_task
    sender_task.cancel()
    await sender_task

    output = request["response"]["input"][1]["output"]
    assert json.loads(output) == {"status": "unavailable"}


@pytest.mark.parametrize("failure_mode", ["timeout", "cancelled", "malformed"])
@pytest.mark.asyncio
async def test_observer_failure_uses_unavailable_context(failure_mode: str) -> None:
    """Observer failure cannot delay, mute, or claim identity for the response."""

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> Any:
        if failure_mode == "timeout":
            await asyncio.sleep(1)
            return {"status": "matched", "display_name": "Late"}
        if failure_mode == "cancelled":
            raise asyncio.CancelledError
        return {"status": []}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(
        observer,
        timeout_seconds=0.01 if failure_mode == "timeout" else 2.0,
    )
    handler.connection = AsyncMock()
    await handler.receive((handler.SAMPLE_RATE, np.ones(160, dtype=np.int16)))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )

    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    request = await _accept_response(handler)
    await completion_task
    sender_task.cancel()
    await sender_task

    assert json.loads(request["response"]["input"][1]["output"]) == {"status": "unavailable"}


def test_completed_utterance_observer_timeout_is_bounded() -> None:
    """Composition may extend provisioning work without allowing an unbounded callback."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    for timeout_seconds in (0.0, 120.1, float("inf")):
        with pytest.raises(ValueError, match="at most 120 seconds"):
            handler.set_completed_utterance_observer(
                AsyncMock(return_value={"status": "unavailable"}),
                timeout_seconds=timeout_seconds,
            )


@pytest.mark.asyncio
async def test_observer_timeout_discards_a_late_result_when_cancellation_is_suppressed() -> None:
    """A callback cannot extend latency or multiply while swallowing cancellation."""
    first_cancellation = asyncio.Event()
    second_cancellation = asyncio.Event()
    observer_calls = 0

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        nonlocal observer_calls
        observer_calls += 1
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if not first_cancellation.is_set():
                    first_cancellation.set()
                    continue
                second_cancellation.set()
                raise

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer, timeout_seconds=0.01)
    utterance = conv_mod.CompletedUserUtterance("item-1", handler.SAMPLE_RATE, b"\x00\x00")

    result = await handler._run_completed_utterance_observer(utterance)

    assert result == {"status": "unavailable"}
    await asyncio.wait_for(first_cancellation.wait(), timeout=0.5)
    assert len(handler._late_utterance_observer_tasks) == 1
    assert await handler._run_completed_utterance_observer(utterance) == {"status": "unavailable"}
    assert observer_calls == 1

    handler._reset_utterance_state()
    await asyncio.wait_for(second_cancellation.wait(), timeout=0.5)
    await asyncio.sleep(0)
    assert not handler._late_utterance_observer_tasks


@pytest.mark.asyncio
async def test_synchronous_observer_invocation_failure_uses_unavailable_context() -> None:
    """A callback that raises before returning an awaitable still fails soft."""

    def observer(_utterance: conv_mod.CompletedUserUtterance) -> Any:
        raise RuntimeError("observer construction failed")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    utterance = conv_mod.CompletedUserUtterance("item-1", handler.SAMPLE_RATE, b"\x00\x00")

    assert await handler._run_completed_utterance_observer(utterance) == {"status": "unavailable"}


@pytest.mark.parametrize("supersession", ["speech", "reconnect"])
@pytest.mark.asyncio
async def test_delayed_observer_is_cancelled_by_supersession(supersession: str) -> None:
    """Later speech or a reconnect cancels callback work and its pending response."""
    observer_started = asyncio.Event()

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        observer_started.set()
        await asyncio.Event().wait()
        return {"status": "matched", "display_name": "Late"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    await handler.receive((handler.SAMPLE_RATE, np.ones(320, dtype=np.int16)))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None
    await observer_started.wait()

    if supersession == "speech":
        await handler._observe_speech_started(
            _FakeEvent("input_audio_buffer.speech_started", item_id="item-2", audio_start_ms=20)
        )
        assert handler._audio_ring_start_sample == 160
    else:
        handler._reset_utterance_state()
        assert handler._audio_ring == bytearray()

    await _wait_until(completion_task.done)
    assert completion_task.cancelled()
    handler.connection.response.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_context_retries_once_without_identity(caplog: pytest.LogCaptureFixture) -> None:
    """A rejected in-band context falls back to a plain explicit response."""

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "matched", "display_name": "Test Person"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    await handler.receive((handler.SAMPLE_RATE, np.ones(160, dtype=np.int16)))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    completion_task = handler._utterance_completion_task
    assert completion_task is not None

    await _wait_until(lambda: handler.connection.response.create.await_count == 1)
    private_canary = "PRIVATE DISPLAY NAME MUST NOT ESCAPE"
    context_request = handler.connection.response.create.await_args_list[0].kwargs
    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(
                type="invalid_input_item",
                message=private_canary,
                event_id=context_request["event_id"],
            ),
        )
    )
    assert private_canary not in caplog.text
    assert handler.output_queue.empty()
    await _wait_until(lambda: handler.connection.response.create.await_count == 2)
    fallback = await _accept_response(handler, request_index=1)
    await completion_task
    await _wait_until(lambda: handler._active_utterance_token is None)

    caplog.clear()
    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(type="invalid_input_item", message=private_canary),
        )
    )
    generic_error = await handler.output_queue.get()
    assert private_canary not in caplog.text
    assert isinstance(generic_error, hf_mod.AdditionalOutputs)
    assert private_canary not in str(generic_error.args)
    assert generic_error.args == ({"role": "assistant", "content": "[error] Realtime request failed."},)

    sender_task.cancel()
    await sender_task

    assert "input" in handler.connection.response.create.await_args_list[0].kwargs["response"]
    assert "input" not in fallback["response"]


@pytest.mark.asyncio
async def test_supersession_between_queue_and_send_drops_request() -> None:
    """A new item invalidates an observer response waiting in the sender."""

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    await handler.receive((handler.SAMPLE_RATE, np.ones(320, dtype=np.int16)))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    handler._response_done_event.clear()
    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    await _wait_until(lambda: handler._active_utterance_token is not None)

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-2", audio_start_ms=20)
    )
    handler._response_done_event.set()
    await asyncio.sleep(0)
    sender_task.cancel()
    await sender_task

    handler.connection.response.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_supersession_after_send_cancels_late_acceptance() -> None:
    """A response accepted after its turn changed is cancelled and suppressed."""

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    await handler.receive((handler.SAMPLE_RATE, np.ones(320, dtype=np.int16)))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    await _wait_until(lambda: handler.connection.response.create.await_count == 1)
    request = handler.connection.response.create.await_args.kwargs

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-2", audio_start_ms=20)
    )
    handler.connection.response.cancel.assert_not_awaited()

    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker})
    handler._response_done_event.clear()
    assert handler._observe_response_created(_FakeEvent("response.created", response=response))
    await handler._cancel_stale_utterance_response()
    assert handler._suppress_active_response
    handler.connection.response.cancel.assert_awaited_once()
    assert handler._observe_response_done(_FakeEvent("response.done", response=response))

    await _wait_until(lambda: handler._active_utterance_token is None)
    sender_task.cancel()
    await sender_task


@pytest.mark.parametrize(
    ("supersede_after_timeout", "cancel_fails"),
    [(False, True), (True, False)],
)
@pytest.mark.asyncio
async def test_accepted_superseded_response_stays_suppressed_until_done(
    monkeypatch: Any,
    supersede_after_timeout: bool,
    cancel_fails: bool,
) -> None:
    """A stale response stays suppressed while a real next response recovers."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.connection = AsyncMock()
    if cancel_fails:
        handler.connection.response.cancel.side_effect = RuntimeError("cancel failed")
    await handler.receive((handler.SAMPLE_RATE, np.ones(320, dtype=np.int16)))
    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-1", audio_start_ms=0)
    )
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-1", audio_end_ms=10)
    )
    sender_task = asyncio.create_task(handler._response_sender_loop())
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-1"),
        "hello",
    )
    await _wait_until(lambda: handler.connection.response.create.await_count == 1)
    request = handler.connection.response.create.await_args.kwargs
    marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    response = SimpleNamespace(
        id="resp-stale",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
    )
    handler._response_done_event.clear()
    assert handler._observe_response_created(_FakeEvent("response.created", response=response))
    await _wait_until(lambda: handler._utterance_completion_task is None)
    assert not handler._response_event_is_suppressed(
        _FakeEvent("response.output_audio.delta", response_id="resp-stale")
    )
    if supersede_after_timeout:
        await _wait_until(lambda: handler._active_utterance_token is None)
    else:
        assert handler._active_utterance_token is not None

    await handler._observe_speech_started(
        _FakeEvent("input_audio_buffer.speech_started", item_id="item-2", audio_start_ms=20)
    )
    handler.connection.response.cancel.assert_awaited_once()
    await _wait_until(lambda: handler._active_utterance_token is None)

    encoded_audio = base64.b64encode(np.ones(16, dtype=np.int16).tobytes()).decode("ascii")
    late_audio = _FakeEvent("response.output_audio.delta", response_id="resp-stale", delta=encoded_audio)
    late_tool = _FakeEvent("response.function_call_arguments.done", response_id="resp-stale")
    assert not await handler._handle_response_audio_delta(late_audio)
    assert handler.output_queue.empty()
    assert handler._response_event_is_suppressed(late_tool)

    await handler.receive((handler.SAMPLE_RATE, np.ones(160, dtype=np.int16)))
    await handler._observe_speech_stopped(
        _FakeEvent("input_audio_buffer.speech_stopped", item_id="item-2", audio_end_ms=30)
    )
    handler._observe_completed_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-2"),
        "next turn",
    )
    next_completion_task = handler._utterance_completion_task
    assert next_completion_task is not None
    await _wait_until(lambda: handler.connection.response.create.await_count == 2)
    next_request = handler.connection.response.create.await_args_list[1].kwargs
    next_marker = next_request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    next_response = SimpleNamespace(
        id="resp-current",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: next_marker},
    )
    handler._response_done_event.clear()
    assert handler._observe_response_created(_FakeEvent("response.created", response=next_response))

    current_audio = _FakeEvent("response.output_audio.delta", response_id="resp-current", delta=encoded_audio)
    assert await handler._handle_response_audio_delta(current_audio)
    sample_rate, pcm = await handler.output_queue.get()
    assert sample_rate == handler.SAMPLE_RATE
    assert np.array_equal(pcm, np.ones((1, 16), dtype=np.int16))

    next_done = _FakeEvent("response.done", response=next_response)
    assert handler._observe_response_done(next_done)
    handler._finish_response_suppression(next_done)
    await next_completion_task
    assert handler._response_event_is_suppressed(late_audio)

    stale_done = _FakeEvent("response.done", response=SimpleNamespace(id="resp-stale"))
    handler._finish_response_suppression(stale_done)
    assert not handler._response_event_is_suppressed(late_audio)

    sender_task.cancel()
    await sender_task


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
async def test_tools_disabled_private_response_drops_injected_tool_call(monkeypatch: Any) -> None:
    """A private answer's injected tool event cannot reach the background manager."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="private-answer",
                call_id="injected-call",
                name="move_head",
                arguments='{"direction":"left"}',
            ),
        )
    )
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    async def classify_private_answer(_tool_specs: list[Any]) -> None:
        handler._response_purposes_by_id["private-answer"] = "search_answer"

    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", classify_private_answer)

    await handler._run_realtime_session()

    start_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_session_update_releases_observer_setup_lock(monkeypatch: Any) -> None:
    """Cancellation before connection ownership cannot strand observer setup."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    observer = AsyncMock(return_value={"status": "unknown"})
    observer.on_connection_reset = MagicMock()
    handler.set_completed_utterance_observer(observer)
    handler.client = _make_fake_realtime_client(session_update_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await handler._run_realtime_session()

    assert not handler._completed_utterance_observer_locked
    assert handler.connection is None
    observer.on_connection_reset.assert_not_called()
    handler.set_completed_utterance_observer(None)


def test_connection_reset_hook_lookup_is_fail_soft() -> None:
    """A malformed optional hook cannot replace session cleanup."""

    class Observer:
        async def __call__(self, _utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
            return {"status": "unavailable"}

        @property
        def on_connection_reset(self) -> Callable[[], None]:
            raise RuntimeError("broken descriptor")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(Observer())

    handler._notify_completed_utterance_observer_connection_reset()


@pytest.mark.asyncio
async def test_session_teardown_clears_tool_batch_state(monkeypatch: Any) -> None:
    """An interrupted tool call must not block responses after reconnect."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    observer = AsyncMock(return_value={"status": "unavailable"})
    observer.on_connection_reset = MagicMock()
    handler.set_completed_utterance_observer(observer)
    handler.client = _make_fake_realtime_client()
    handler._in_flight_tool_calls.add("call_interrupted")
    handler._internal_tool_calls.add("call_interrupted")
    handler._tool_batch_needs_response = True
    handler._startup_greeting_sent = True
    handler._startup_response_pending = True
    handler._pending_responses.put_nowait(hf_mod._QueuedResponse(kwargs={}))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    await handler._run_realtime_session()

    assert handler._in_flight_tool_calls == set()
    assert handler._internal_tool_calls == set()
    assert not handler._tool_batch_needs_response
    assert not handler._startup_greeting_sent
    assert not handler._startup_input_blocked
    assert not handler._startup_response_pending
    assert handler._pending_responses.empty()
    assert handler._response_done_event.is_set()
    assert handler._response_request_done_event.is_set()
    observer.on_connection_reset.assert_called_once_with()


@pytest.mark.asyncio
async def test_active_shutdown_notifies_observer_once(monkeypatch: Any) -> None:
    """Shutdown and the session's finally block share one reset notification."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    observer = AsyncMock(return_value={"status": "unavailable"})
    observer.on_connection_reset = MagicMock()
    handler.set_completed_utterance_observer(observer)
    handler.client = _make_fake_realtime_client(hold_open_until_close=True)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    session_task = asyncio.create_task(handler._run_realtime_session())
    await handler._connected_event.wait()
    await handler.shutdown()
    await session_task
    await handler.shutdown()

    observer.on_connection_reset.assert_called_once_with()


@pytest.mark.asyncio
async def test_observer_restart_waits_for_prior_session_reset(monkeypatch: Any) -> None:
    """A replacement observer session cannot start before predecessor teardown."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    order: list[str] = []
    connection_exit_started = asyncio.Event()
    allow_connection_exit = asyncio.Event()
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    observer = AsyncMock(return_value={"status": "unavailable"})
    observer.on_connection_reset = MagicMock(side_effect=lambda: order.append("reset"))
    handler.set_completed_utterance_observer(observer)
    handler.client = _make_fake_realtime_client(
        hold_open_until_close=True,
        connection_exit_callback=lambda: order.append("exit"),
        connection_exit_started=connection_exit_started,
        connection_exit_gate=allow_connection_exit,
    )
    replacement_client = _make_fake_realtime_client(session_update_callback=lambda: order.append("replacement"))
    monkeypatch.setattr(handler, "_build_realtime_client", AsyncMock(return_value=replacement_client))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    prior_session = asyncio.create_task(handler._run_realtime_session())
    await handler._connected_event.wait()
    restart = asyncio.create_task(handler._restart_session())
    await connection_exit_started.wait()
    assert order == ["reset"]
    allow_connection_exit.set()
    await restart
    await prior_session

    assert order[:3] == ["reset", "exit", "replacement"]


@pytest.mark.asyncio
async def test_observer_restart_aborts_when_predecessor_cannot_stop(monkeypatch: Any) -> None:
    """A stuck predecessor is bounded and never overlaps a replacement."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "_OBSERVER_SESSION_STOP_TIMEOUT", 0.01)

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    observer = AsyncMock(return_value={"status": "unavailable"})
    observer.on_connection_reset = MagicMock()
    handler.set_completed_utterance_observer(observer)
    handler.client = _make_fake_realtime_client(hold_open_until_close=True, close_unblocks=False)
    build_replacement = AsyncMock()
    monkeypatch.setattr(handler, "_build_realtime_client", build_replacement)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    prior_session = asyncio.create_task(handler._run_realtime_session())
    await handler._connected_event.wait()
    await handler._restart_session()

    build_replacement.assert_not_awaited()
    prior_session.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prior_session
    observer.on_connection_reset.assert_called_once_with()


@pytest.mark.asyncio
async def test_search_only_restart_waits_for_prior_session_teardown(monkeypatch: Any) -> None:
    """The optional search seam cannot let old teardown erase replacement state."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    order: list[str] = []
    connection_exit_started = asyncio.Event()
    allow_connection_exit = asyncio.Event()
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler.client = _make_fake_realtime_client(
        hold_open_until_close=True,
        connection_exit_callback=lambda: order.append("exit"),
        connection_exit_started=connection_exit_started,
        connection_exit_gate=allow_connection_exit,
    )
    replacement_client = _make_fake_realtime_client(session_update_callback=lambda: order.append("replacement"))
    monkeypatch.setattr(handler, "_build_realtime_client", AsyncMock(return_value=replacement_client))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    prior_session = asyncio.create_task(handler._run_realtime_session())
    await handler._connected_event.wait()
    restart = asyncio.create_task(handler._restart_session())
    await connection_exit_started.wait()
    assert order == []
    allow_connection_exit.set()
    await restart
    await prior_session

    assert order[:2] == ["exit", "replacement"]


@pytest.mark.parametrize("failure_site", ["generic_cleanup", "connection_exit"])
@pytest.mark.asyncio
async def test_teardown_failure_still_resets_observer_session(
    monkeypatch: Any,
    failure_site: str,
) -> None:
    """Fallible generic cleanup cannot strand observer reset ownership."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    observer = AsyncMock(return_value={"status": "unavailable"})
    observer.on_connection_reset = MagicMock()
    handler.set_completed_utterance_observer(observer)
    error = RuntimeError("cleanup failed")
    handler.client = _make_fake_realtime_client(
        connection_exit_error=error if failure_site == "connection_exit" else None,
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(
        type(handler.tool_manager),
        "shutdown",
        AsyncMock(side_effect=error if failure_site == "generic_cleanup" else None),
    )
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await handler._run_realtime_session()

    observer.on_connection_reset.assert_called_once_with()
    assert handler._observer_session_stopped.is_set()
    assert not handler._completed_utterance_observer_locked


@pytest.mark.asyncio
async def test_unrelated_response_created_does_not_reopen_microphone_input(monkeypatch: Any) -> None:
    """Only the sender correlated with startup may reopen microphone input."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    movement_manager = MagicMock()
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager))
    handler.client = _make_fake_realtime_client(events=(_FakeEvent("response.created"), _FakeEvent("response.done")))
    handler._startup_input_blocked = True
    input_blocked_when_speaking_started: list[bool] = []

    def observe_speaking(speaking: bool) -> None:
        if speaking:
            input_blocked_when_speaking_started.append(handler._startup_input_blocked)

    movement_manager.set_speaking.side_effect = observe_speaking
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    monkeypatch.setattr(handler, "_send_startup_greeting_prompt", AsyncMock())

    await handler._run_realtime_session()

    assert input_blocked_when_speaking_started == [True]


@pytest.mark.asyncio
async def test_startup_response_sender_reopens_microphone_after_created() -> None:
    """Only the metadata-tagged startup response may reopen microphone input."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._startup_input_blocked = True
    handler._startup_response_pending = True
    request_sent = asyncio.Event()
    request_kwargs: dict[str, Any] = {}

    async def create_response(**kwargs: Any) -> None:
        request_kwargs.update(kwargs)
        request_sent.set()

    handler.connection.response.create.side_effect = create_response
    sender_task = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create(_is_startup=True)
    await asyncio.wait_for(request_sent.wait(), timeout=1.0)
    handler._response_done_event.clear()

    metadata = request_kwargs["response"]["metadata"]
    marker = metadata[hf_mod._RESPONSE_REQUEST_METADATA_KEY]
    assert request_kwargs["event_id"].startswith("event_")

    unrelated = _FakeEvent(
        "response.created",
        response=SimpleNamespace(metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: "unrelated"}),
    )
    assert not handler._observe_response_created(unrelated)
    await asyncio.sleep(0)
    assert handler._startup_input_blocked

    matching = _FakeEvent(
        "response.created",
        response=SimpleNamespace(metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker}),
    )
    assert handler._observe_response_created(matching)

    async def wait_until_unblocked() -> None:
        while handler._startup_input_blocked:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_unblocked(), timeout=1.0)

    unrelated_done = _FakeEvent(
        "response.done",
        response=SimpleNamespace(metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: "unrelated"}),
    )
    assert not handler._observe_response_done(unrelated_done)
    await asyncio.sleep(0)
    assert handler._active_response_marker == marker

    matching_done = _FakeEvent(
        "response.done",
        response=SimpleNamespace(metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker}),
    )
    assert handler._observe_response_done(matching_done)

    async def wait_until_request_cleared() -> None:
        while handler._active_response_marker is not None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_request_cleared(), timeout=1.0)
    sender_task.cancel()
    await sender_task

    assert not handler._startup_input_blocked
    assert not handler._startup_response_pending


@pytest.mark.asyncio
async def test_policy_disabled_input_error_preserves_sender_recovery() -> None:
    """Without Stage 4 policy, eventless input errors retain the base sender behavior."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_event_id = "event_current"
    handler._response_started_or_rejected_event.clear()

    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(event_id=None, code="input_audio_buffer_failed", message="input failed"),
        )
    )

    assert handler._response_started_or_rejected_event.is_set()


@pytest.mark.asyncio
async def test_policy_disabled_eventless_response_error_wakes_sender() -> None:
    """Search hardening preserves immediate recovery for ordinary backend errors."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_event_id = "event-ordinary"
    handler._response_started_or_rejected_event.clear()

    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(event_id=None, code="server_error", message="ordinary failure"),
        )
    )

    assert handler._response_started_or_rejected_event.is_set()


@pytest.mark.asyncio
async def test_failed_startup_response_sender_reopens_microphone() -> None:
    """A startup send failure must fail open instead of muting indefinitely."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.connection.response.create.side_effect = RuntimeError("send failed")
    handler._startup_input_blocked = True
    handler._startup_response_pending = True
    sender_task = asyncio.create_task(handler._response_sender_loop())
    await handler._safe_response_create(_is_startup=True)

    async def wait_until_unblocked() -> None:
        while handler._startup_input_blocked:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_unblocked(), timeout=1.0)
    sender_task.cancel()
    await sender_task

    assert not handler._startup_input_blocked
    assert handler._startup_response_pending


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
async def test_emit_skips_idle_signal_while_startup_greeting_pending(monkeypatch: Any) -> None:
    """Idle behavior must not overtake configured startup recognition."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager))
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler._startup_input_blocked = True

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


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
async def test_startup_greeting_runs_configured_tool_before_model_response(monkeypatch: Any) -> None:
    """A configured greeting tool should use the normal result lifecycle before speech."""
    tool = MagicMock(needs_response=True)
    spec: hf_mod.ToolSpec = {
        "type": "function",
        "name": "recognize_person",
        "description": "Recognize the person in view.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "Open after recognition.")
    monkeypatch.setattr(hf_mod, "get_session_greeting_tool_name", lambda: "recognize_person")
    monkeypatch.setattr(hf_mod.core_tools, "ALL_TOOLS", {"recognize_person": tool})

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    create_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create_response)

    await handler._send_startup_greeting_prompt([spec])

    assert handler.connection.conversation.item.create.await_count == 2
    prompt_item = handler.connection.conversation.item.create.await_args_list[0].kwargs["item"]
    function_item = handler.connection.conversation.item.create.await_args_list[1].kwargs["item"]
    assert prompt_item["content"][0]["text"] == "Open after recognition."
    assert function_item["type"] == "function_call"
    assert "id" not in function_item
    assert function_item["name"] == "recognize_person"
    assert function_item["arguments"] == "{}"
    assert function_item["call_id"] in handler._in_flight_tool_calls
    assert handler._startup_input_blocked
    assert handler._startup_response_pending
    routine = start_tool.await_args.kwargs["tool_call_routine"]
    assert routine.tool_name == "recognize_person"
    assert routine.args_json_str == "{}"
    create_response.assert_not_awaited()

    await handler._handle_tool_result(
        ToolNotification(
            id=function_item["call_id"],
            tool_name="recognize_person",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"status": "matched", "display_name": "Test Person"},
        )
    )

    result_item = handler.connection.conversation.item.create.await_args_list[2].kwargs["item"]
    assert result_item["type"] == "function_call_output"
    assert result_item["call_id"] == function_item["call_id"]
    assert json.loads(result_item["output"]) == {
        "status": "matched",
        "display_name": "Test Person",
    }
    assert function_item["call_id"] not in handler._in_flight_tool_calls
    assert function_item["call_id"] not in handler._internal_tool_calls
    assert handler.output_queue.empty()
    assert handler._startup_input_blocked
    create_response.assert_awaited_once_with(_is_startup=True)

    created_items = handler.connection.conversation.item.create.await_count
    with pytest.raises(RuntimeError, match="startup greeting pending"):
        await handler.say("Do not overtake startup.")
    assert handler.connection.conversation.item.create.await_count == created_items

    await handler.receive((handler.SAMPLE_RATE, np.ones(160, dtype=np.int16)))
    handler.connection.input_audio_buffer.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_tool_result_timeout_reopens_microphone(monkeypatch: Any) -> None:
    """Failure to submit startup identity must not leave input muted forever."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._startup_input_blocked = True
    handler._internal_tool_calls.add("call_startup")
    handler._in_flight_tool_calls.add("call_startup")
    monkeypatch.setattr(
        handler,
        "_wait_for_response_done_before_tool_result",
        AsyncMock(return_value=False),
    )

    await handler._handle_tool_result(
        ToolNotification(
            id="call_startup",
            tool_name="recognize_person",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"status": "unknown"},
        )
    )

    assert not handler._startup_input_blocked
    handler.connection.conversation.item.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_greeting_without_configured_tool_uses_model_response(monkeypatch: Any) -> None:
    """Profiles without a greeting tool retain Pollen's ordinary response path."""
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "Open normally.")
    monkeypatch.setattr(hf_mod, "get_session_greeting_tool_name", lambda: None)

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    create_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create_response)

    await handler._send_startup_greeting_prompt([])

    handler.connection.conversation.item.create.assert_awaited_once()
    start_tool.assert_not_awaited()
    create_response.assert_awaited_once_with()
    assert not handler._startup_input_blocked


@pytest.mark.asyncio
async def test_invalid_configured_greeting_tool_fails_closed(monkeypatch: Any) -> None:
    """A missing configured greeting tool must not fall back to model-first speech."""
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "Do not speak.")
    monkeypatch.setattr(hf_mod, "get_session_greeting_tool_name", lambda: "missing_tool")
    monkeypatch.setattr(hf_mod.core_tools, "ALL_TOOLS", {})

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    create_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create_response)

    await handler._send_startup_greeting_prompt([])

    handler.connection.conversation.item.create.assert_not_awaited()
    start_tool.assert_not_awaited()
    create_response.assert_not_awaited()
    assert not handler._startup_input_blocked


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("needs_response", "parameters"),
    [
        (False, {"type": "object", "properties": {}, "additionalProperties": False}),
        (
            True,
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        ),
    ],
)
async def test_incompatible_configured_greeting_tool_fails_closed(
    monkeypatch: Any,
    needs_response: bool,
    parameters: dict[str, Any],
) -> None:
    """Startup tools must need a spoken response and accept no arguments."""
    tool = MagicMock(needs_response=needs_response)
    spec: hf_mod.ToolSpec = {
        "type": "function",
        "name": "configured_tool",
        "description": "Configured startup tool.",
        "parameters": parameters,
    }
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "Do not speak.")
    monkeypatch.setattr(hf_mod, "get_session_greeting_tool_name", lambda: "configured_tool")
    monkeypatch.setattr(hf_mod.core_tools, "ALL_TOOLS", {"configured_tool": tool})

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    create_response = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create_response)

    await handler._send_startup_greeting_prompt([spec])

    handler.connection.conversation.item.create.assert_not_awaited()
    start_tool.assert_not_awaited()
    create_response.assert_not_awaited()


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


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [
        ('{"query":" current score "}', ("current score", 3)),
        ('{"query":"current score","max_results":1}', ("current score", 1)),
        ('{"query":"current score","max_results":3}', ("current score", 3)),
        ("", None),
        ("[]", None),
        ('{"query":"a","query":"b"}', None),
        ('{"query":""}', None),
        ('{"query":"   "}', None),
        ('{"query":1}', None),
        ('{"query":"a","extra":true}', None),
        ('{"query":"a","max_results":true}', None),
        ('{"query":"a","max_results":1.0}', None),
        ('{"query":"a","max_results":0}', None),
        ('{"query":"a","max_results":4}', None),
        ('{"query":"a","max_results":5}', None),
        ('{"query":"a","max_results":10}', None),
        (json.dumps({"query": "a" * 257}), None),
        (json.dumps({"query": "😀" * 256}), ("😀" * 256, 3)),
        (json.dumps({"query": "\ud800"}), None),
    ],
)
def test_search_argument_parser_is_strict_and_bounded(
    args_json: str,
    expected: tuple[str, int] | None,
) -> None:
    """Only canonical query/count values can cross the official search seam."""
    assert HuggingFaceRealtimeHandler._parse_search_arguments(args_json) == expected


def test_search_policy_narrows_only_the_advertised_official_search_schema() -> None:
    """The model sees the same 1–3/default-3 band enforced by the policy parser."""
    search_spec = {
        "type": "function",
        "name": hf_mod._OFFICIAL_SEARCH_TOOL_NAME,
        "description": "Search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    }
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    ordinary_parameters = handler._get_session_config([search_spec])["tools"][0]["parameters"]
    handler.set_search_policy(AsyncMock())
    policy_parameters = handler._get_session_config([search_spec])["tools"][0]["parameters"]

    assert ordinary_parameters["properties"]["max_results"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 5,
    }
    assert policy_parameters["properties"]["max_results"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
        "default": 3,
    }
    assert search_spec["parameters"] is ordinary_parameters


def test_blocked_search_attempts_extend_the_rolling_rate_window(monkeypatch: Any) -> None:
    """Repeated blocked calls count, so a request storm cannot age itself out."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    attempts = iter((0.0, 1.0, 2.0, 58.0, 59.0, 60.0, 61.0))
    monkeypatch.setattr(hf_mod.time, "monotonic", lambda: next(attempts))

    assert [handler._consume_search_attempt() for _ in range(7)] == [True, True, True, False, False, False, False]


@pytest.mark.asyncio
async def test_late_tagged_response_cannot_rebind_to_newer_search_turn() -> None:
    """An explicit response retains its enqueue-time turn instead of latest-at-arrival."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-old"), "old search turn")
    request = await handler._enqueue_response_request()
    assert request.search_turn is not None
    _, marker, _ = handler._tag_response_request(request.kwargs)
    handler._search_turns_by_response_marker[marker] = request.search_turn

    handler._record_search_transcript(_FakeEvent("completed", item_id="item-new"), "new ordinary turn")
    untagged_late_response = SimpleNamespace(id="response-late-untagged", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=untagged_late_response))
    late_response = SimpleNamespace(
        id="response-late-old",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
    )
    handler._observe_response_created(_FakeEvent("response.created", response=late_response))

    assert "response-late-untagged" not in handler._search_turns_by_response_id
    assert "response-late-old" not in handler._search_turns_by_response_id
    assert handler._latest_search_turn is None
    assert not handler._unbound_search_turn_keys
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-next"), "next search turn")
    next_response = SimpleNamespace(id="response-next", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=next_response))
    assert handler._search_turns_by_response_id[next_response.id] == handler._latest_search_turn
    handler._discard_pending_responses()
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_reopened_transcript_keeps_search_on_the_same_audio_item() -> None:
    """A later transcript revision can own the response already bound to its audio item."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-reopened"),
        "search for the latest NASA art",
    )
    request = await handler._enqueue_response_request()
    assert request.search_turn is not None
    _, marker, _ = handler._tag_response_request(request.kwargs)
    handler._search_turns_by_response_marker[marker] = request.search_turn
    original_token = request.search_turn

    handler._invalidate_search_turn()
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-reopened"),
        "search for the latest NASA Artemis II",
    )
    handler._invalidate_search_turn()
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-reopened"),
        "search for the latest NASA Artemis II mission update",
    )
    response = SimpleNamespace(
        id="response-reopened",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
    )
    handler._observe_response_created(_FakeEvent("response.created", response=response))
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=response.id,
            call_id="call-reopened",
            arguments='{"query":"latest NASA Artemis II mission update"}',
        )
    )

    assert handler._active_search is not None
    assert handler._active_search.token.generation > original_token.generation
    assert handler._active_search.token.transcript.endswith("Artemis II mission update")
    assert not handler._unbound_search_turn_keys
    handler._discard_pending_responses()
    await handler._end_search_session()


@pytest.mark.parametrize(
    "revised_transcript",
    ("search current weather in Paris", "search local news"),
)
@pytest.mark.asyncio
async def test_observer_revision_keeps_its_own_response_request(revised_transcript: str) -> None:
    """An old response cannot seize a revision that queues an observer response."""

    async def observe(_utterance: conv_mod.CompletedUserUtterance) -> conv_mod.CompletedUtteranceResult:
        return {"status": "unavailable"}

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observe)
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-revised"), "search current weather")
    old_request = await handler._enqueue_response_request()
    assert old_request.search_turn is not None
    _, old_marker, _ = handler._tag_response_request(old_request.kwargs)
    handler._search_turns_by_response_marker[old_marker] = old_request.search_turn
    handler._invalidate_search_turn()
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-revised"),
        revised_transcript,
    )
    revised_request = await handler._enqueue_response_request()
    assert revised_request.search_turn is not None
    _, revised_marker, _ = handler._tag_response_request(revised_request.kwargs)
    handler._search_turns_by_response_marker[revised_marker] = revised_request.search_turn

    old_response = SimpleNamespace(
        id="response-old-revision",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: old_marker},
    )
    handler._observe_response_created(_FakeEvent("response.created", response=old_response))
    refusal = MagicMock()
    handler._schedule_unstarted_search = refusal
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=old_response.id,
            call_id="call-old-revision",
            arguments='{"query":"current weather"}',
        )
    )

    assert refusal.call_args.kwargs["outcome"] == "stale"
    assert handler._latest_search_turn == revised_request.search_turn
    revised_response = SimpleNamespace(
        id="response-revised",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: revised_marker},
    )
    handler._observe_response_created(_FakeEvent("response.created", response=revised_response))
    assert handler._search_turns_by_response_id[revised_response.id] == revised_request.search_turn
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_rewritten_same_item_fails_closed_without_desynchronizing_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A divergent revision is refused and cannot strand the next turn behind it."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-rewritten"), "search home address")
    response = SimpleNamespace(id="response-rewritten", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=response))
    handler._invalidate_search_turn()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-rewritten"), "search home town")
    refusal = MagicMock()
    monkeypatch.setattr(handler, "_schedule_unstarted_search", refusal)

    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=response.id,
            call_id="call-rewritten",
            arguments='{"query":"home address"}',
        )
    )

    assert handler._active_search is None
    assert refusal.call_args.kwargs["outcome"] == "stale"
    assert handler._latest_search_turn is None
    assert not handler._unbound_search_turn_keys
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-next"), "search current weather")
    next_response = SimpleNamespace(id="response-next", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=next_response))
    assert handler._search_turns_by_response_id[next_response.id] == handler._latest_search_turn
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_metadata_free_reopened_item_fails_closed_without_desynchronizing_next_turn() -> None:
    """An untagged response cannot leave a revised audio item ahead of the FIFO."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler._begin_search_session()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-reopened"), "search home address")
    handler._invalidate_search_turn()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-reopened"), "search home town")
    handler._invalidate_search_turn()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-reopened"), "search home town")

    reopened_response = SimpleNamespace(id="response-reopened", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=reopened_response))

    assert reopened_response.id not in handler._search_turns_by_response_id
    assert handler._latest_search_turn is None
    assert not handler._unbound_search_turn_keys
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-next"), "search current weather")
    next_response = SimpleNamespace(id="response-next", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=next_response))
    assert handler._search_turns_by_response_id[next_response.id] == handler._latest_search_turn
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_consumed_response_cannot_promote_onto_revised_search_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused response cannot spend another attempt from a later revision."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler._begin_search_session()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-revised"), "search current weather")
    response = SimpleNamespace(id="response-consumed", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=response))
    refusal = MagicMock()
    monkeypatch.setattr(handler, "_schedule_unstarted_search", refusal)
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=response.id,
            call_id="call-invalid",
            arguments="{}",
        )
    )
    assert refusal.call_args.kwargs["outcome"] == "invalid_arguments"
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-revised"),
        "search current weather in Paris",
    )

    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=response.id,
            call_id="call-replayed",
            arguments='{"query":"current weather in Paris"}',
        )
    )

    assert handler._active_search is None
    assert refusal.call_args.kwargs["outcome"] == "stale"
    assert handler._latest_search_turn is not None
    assert handler._latest_search_turn.transcript.endswith("in Paris")
    await handler._end_search_session()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_before_marker_response", (True, False))
async def test_marker_claimed_revision_survives_stale_fifo_response(
    monkeypatch: pytest.MonkeyPatch,
    stale_before_marker_response: bool,
) -> None:
    """A stale pending key cannot revoke a revision owned by an exact marker."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler._begin_search_session()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-revised"), "search current weather")
    old_response = SimpleNamespace(id="response-old", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=old_response))
    handler._invalidate_search_turn()
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-revised"),
        "search current weather in France",
    )
    handler._invalidate_search_turn()
    handler._record_search_transcript(
        _FakeEvent("completed", item_id="item-revised"),
        "search current weather in Paris, France",
    )
    latest_request = await handler._enqueue_response_request()
    assert latest_request.search_turn is not None
    _, latest_marker, _ = handler._tag_response_request(latest_request.kwargs)
    handler._search_turns_by_response_marker[latest_marker] = latest_request.search_turn
    latest_response = SimpleNamespace(
        id="response-latest",
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: latest_marker},
    )
    stale_response = SimpleNamespace(id="response-stale", metadata={})
    if stale_before_marker_response:
        handler._observe_response_created(_FakeEvent("response.created", response=stale_response))
    handler._observe_response_created(_FakeEvent("response.created", response=latest_response))
    refusal = MagicMock()
    monkeypatch.setattr(handler, "_schedule_unstarted_search", refusal)
    active_search = hf_mod._SearchCallState(
        call_id="call-latest",
        response_id=latest_response.id,
        response_done=hf_mod._SearchResponseDone(),
        token=latest_request.search_turn,
        query="current weather in Paris, France",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = active_search
    if not stale_before_marker_response:
        handler._observe_response_created(_FakeEvent("response.created", response=stale_response))

    assert handler._latest_search_turn == latest_request.search_turn
    assert handler._active_search is active_search
    assert not active_search.superseded.is_set()
    assert active_search.query == "current weather in Paris, France"
    assert not handler._unbound_search_turn_keys
    handler._revoke_search_transport(active_search)
    handler._active_search = None
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=old_response.id,
            call_id="call-old",
            arguments='{"query":"current weather"}',
        )
    )
    assert refusal.call_args.kwargs["outcome"] == "stale"
    assert handler._latest_search_turn == latest_request.search_turn
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_different_audio_item_cannot_rebind_search_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response from a prior audio item remains stale when a new turn completes."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-old"), "search old weather")
    response = SimpleNamespace(id="response-old", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=response))
    handler._invalidate_search_turn()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-new"), "search new weather")
    refusal = MagicMock()
    monkeypatch.setattr(handler, "_schedule_unstarted_search", refusal)

    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=response.id,
            call_id="call-old",
            arguments='{"query":"old weather"}',
        )
    )

    assert handler._active_search is None
    assert refusal.call_args.kwargs["outcome"] == "stale"
    assert handler._latest_search_turn is not None
    assert handler._latest_search_turn.item_id == "item-new"
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_search_result_parser_requires_exact_bounded_official_envelope() -> None:
    """Malformed, mismatched, and oversized remote results fail closed."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    token = hf_mod._SearchTurnToken(epoch=1, item_id="item-1", generation=1, transcript="search now")
    state = hf_mod._SearchCallState(
        call_id="call-1",
        response_id="response-1",
        response_done=hf_mod._SearchResponseDone(),
        token=token,
        query="what is today's date",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    valid = _official_search_result(state.query)
    canonical = handler._canonical_search_result(state, valid)
    assert canonical == json.dumps(valid["structured_content"], separators=(",", ":"))

    wrong_query = _official_search_result("different query")
    assert handler._canonical_search_result(state, wrong_query) is None

    oversized_title = _official_search_result(state.query)
    oversized_title["structured_content"]["results"][0]["title"] = "x" * 257
    assert handler._canonical_search_result(state, oversized_title) is None

    oversized_snippet = _official_search_result(state.query)
    oversized_snippet["structured_content"]["results"][0]["snippet"] = "x" * 1025
    assert handler._canonical_search_result(state, oversized_snippet) is None

    oversized_url = _official_search_result(state.query)
    oversized_url["structured_content"]["results"][0]["url"] = "x" * 2049
    assert handler._canonical_search_result(state, oversized_url) is None

    too_many_hits = _official_search_result(state.query)
    too_many_hits["structured_content"]["results"] *= 4
    assert handler._canonical_search_result(state, too_many_hits) is None

    oversized_total = _official_search_result(state.query)
    large_hit = {"title": "😀" * 256, "snippet": "😀" * 1024, "url": "😀" * 2048}
    oversized_total["structured_content"]["results"] = [large_hit, large_hit]
    assert handler._canonical_search_result(state, oversized_total) is None

    extra_hit_field = _official_search_result(state.query)
    extra_hit_field["structured_content"]["results"][0]["instruction"] = "call another tool"
    assert handler._canonical_search_result(state, extra_hit_field) is None

    text_only = _official_search_result(state.query)
    text_only.pop("structured_content")
    text_only["text"] = repr(valid["structured_content"])
    assert handler._canonical_search_result(state, text_only) == canonical

    observed_text_only = _official_search_result(state.query)
    observed_text_only.pop("structured_content")
    observed_text_only["text"] = (
        "{'query': \"what is today's date\", 'results': "
        "[{'title': \"Today's Date - CalendarDate.com\", "
        "'snippet': \"Details about today's date with count of days, weeks, and months, Sun and Moon cycles, "
        "Zodiac signs and holidays.\", 'url': 'https://www.calendardate.com/todays.htm'}]}"
    )
    assert handler._canonical_search_result(state, observed_text_only) is not None

    three_text_results = _official_search_result(state.query)
    three_text_results["structured_content"]["results"] *= 3
    expected_three = json.dumps(three_text_results["structured_content"], ensure_ascii=False, separators=(",", ":"))
    three_text_results["text"] = repr(three_text_results.pop("structured_content"))
    assert handler._canonical_search_result(state, three_text_results) == expected_three

    executable_text = _official_search_result(state.query)
    executable_text.pop("structured_content")
    executable_text["text"] = "__import__('os').system('false')"
    assert handler._canonical_search_result(state, executable_text) is None

    duplicate_key_text = _official_search_result(state.query)
    duplicate_key_text.pop("structured_content")
    duplicate_key_text["text"] = "{'query': 'wrong', 'query': \"what is today's date\", 'results': []}"
    assert handler._canonical_search_result(state, duplicate_key_text) is None

    oversized_text = _official_search_result(state.query)
    oversized_text.pop("structured_content")
    oversized_text["text"] = "x" * (hf_mod._SEARCH_RESULT_MAX_BYTES + 1)
    assert handler._canonical_search_result(state, oversized_text) is None

    null_structured_with_valid_text = _official_search_result(state.query)
    null_structured_with_valid_text["structured_content"] = None
    null_structured_with_valid_text["text"] = repr(valid["structured_content"])
    assert handler._canonical_search_result(state, null_structured_with_valid_text) == canonical

    malformed_structured_with_valid_text = _official_search_result(state.query)
    malformed_structured_with_valid_text["structured_content"] = {"unexpected": "payload"}
    malformed_structured_with_valid_text["text"] = repr(valid["structured_content"])
    assert handler._canonical_search_result(state, malformed_structured_with_valid_text) is None


@pytest.mark.asyncio
async def test_same_response_refused_search_siblings_queue_one_failure_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extra refused calls in one response resolve individually but speak only once."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    queue_failure = AsyncMock()
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    handler._search_response_done_events["response-search"] = response_done
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-search",
        generation=handler._search_turn_generation,
        transcript="search now",
    )
    handler._latest_search_turn = token
    handler._search_turns_by_response_id["response-search"] = token

    for index in range(5):
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="response-search",
                call_id=f"call-sibling-{index}",
                arguments="not-json",
            )
        )
    await _wait_until(lambda: handler.connection.conversation.item.create.await_count == 5)
    await _wait_until(lambda: not handler._search_tasks)

    assert [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list] == [
        {
            "type": "function_call_output",
            "call_id": f"call-sibling-{index}",
            "output": hf_mod._SEARCH_FAILURE_MARKER,
        }
        for index in range(5)
    ]
    queue_failure.assert_awaited_once()


@pytest.mark.asyncio
async def test_uncorrelated_search_refusals_log_marker_unavailable_and_speak_once(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed response correlation fails closed with bounded speech and logging."""
    caplog.set_level("INFO")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    queue_failure = AsyncMock()
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)

    for index in range(3):
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="x" * (hf_mod._SEARCH_ID_MAX_CHARS + 1),
                call_id=f"call-uncorrelated-{index}",
                arguments='{"query":"current score"}',
            )
        )
    await _wait_until(lambda: not handler._search_tasks)

    handler.connection.conversation.item.create.assert_not_awaited()
    queue_failure.assert_awaited_once()
    assert caplog.text.count("search_call outcome=marker_unavailable") == 3
    assert "call-uncorrelated" not in caplog.text


@pytest.mark.asyncio
async def test_duplicate_search_result_is_scrubbed_without_replacing_winner() -> None:
    """A result that loses the completion race is cleared without being parsed or copied."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    token = hf_mod._SearchTurnToken(epoch=1, item_id="item-search", generation=1, transcript="search now")
    result_future: asyncio.Future[hf_mod._SearchToolResult] = asyncio.get_running_loop().create_future()
    winner = hf_mod._SearchToolResult(canonical="already handled")
    result_future.set_result(winner)
    handler._active_search = hf_mod._SearchCallState(
        call_id="call-search",
        response_id="response-search",
        response_done=hf_mod._SearchResponseDone(),
        token=token,
        query="current score",
        max_results=3,
        result=result_future,
        superseded=asyncio.Event(),
    )
    handler.tool_manager = MagicMock()
    incoming = MagicMock()
    incoming.id = "call-search"
    incoming.tool_name = hf_mod._OFFICIAL_SEARCH_TOOL_NAME
    raw_result = {"private-result-canary": ["nested-private-canary"]}
    incoming.result = raw_result
    incoming.error = "private-error-canary"

    await handler._handle_search_tool_result(incoming)

    assert incoming.result is None
    assert incoming.error is None
    assert raw_result == {}
    assert result_future.result() is winner
    assert winner.canonical == "already handled"


@pytest.mark.asyncio
async def test_search_result_is_bounded_before_copy_and_raw_source_is_scrubbed() -> None:
    """Only canonical bounded fields survive handoff; unused raw structures are never copied."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    token = hf_mod._SearchTurnToken(epoch=1, item_id="item-search", generation=1, transcript="search now")
    result_future: asyncio.Future[hf_mod._SearchToolResult] = asyncio.get_running_loop().create_future()
    handler._active_search = hf_mod._SearchCallState(
        call_id="call-search",
        response_id="response-search",
        response_done=hf_mod._SearchResponseDone(),
        token=token,
        query="current score",
        max_results=3,
        result=result_future,
        superseded=asyncio.Event(),
    )
    handler.tool_manager = MagicMock()

    class CopyCanary:
        def __deepcopy__(self, _memo: dict[int, Any]) -> None:
            raise AssertionError("untrusted result was deep-copied")

    raw_result = _official_search_result("current score")
    raw_result["content_blocks"] = [CopyCanary()]
    incoming = MagicMock()
    incoming.id = "call-search"
    incoming.tool_name = hf_mod._OFFICIAL_SEARCH_TOOL_NAME
    incoming.result = raw_result
    incoming.error = None

    await handler._handle_search_tool_result(incoming)

    assert incoming.result is None
    assert incoming.error is None
    bounded = result_future.result()
    assert bounded.canonical == json.dumps(
        {
            "query": "current score",
            "results": [
                {
                    "title": "Current result",
                    "snippet": "A bounded result.",
                    "url": "https://example.com/current",
                }
            ],
        },
        separators=(",", ":"),
    )
    assert raw_result == {}


@pytest.mark.asyncio
async def test_search_policy_timeout_is_bounded_when_cancellation_is_suppressed() -> None:
    """A misbehaving local policy cannot hold the search coordinator past its deadline."""
    release_policy = asyncio.Event()
    policy_started = asyncio.Event()
    captured_requests: list[conv_mod.SearchPolicyRequest] = []

    async def cancellation_resistant_policy(
        request: conv_mod.SearchPolicyRequest,
    ) -> conv_mod.SearchPolicyDecision:
        captured_requests.append(request)
        policy_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            await release_policy.wait()
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(cancellation_resistant_policy, timeout_seconds=0.01)
    handler.set_search_space_gate(_allow_search_space_gate)
    token = hf_mod._SearchTurnToken(epoch=1, item_id="item-policy", generation=1, transcript="search now")
    state = hf_mod._SearchCallState(
        call_id="call-policy",
        response_id="response-policy",
        response_done=hf_mod._SearchResponseDone(),
        token=token,
        query="bounded query",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )

    decision = await asyncio.wait_for(handler._run_search_policy(state), timeout=0.1)

    assert decision is None
    assert policy_started.is_set()
    assert len(handler._late_search_policy_tasks) == 1
    assert captured_requests == [conv_mod.SearchPolicyRequest(item_id="", transcript="", query="", max_results=0)]
    assert state.policy_request is None
    release_policy.set()
    await _wait_until(lambda: not handler._late_search_policy_tasks)


@pytest.mark.asyncio
async def test_new_transcript_synchronously_revokes_active_search_transport() -> None:
    """A superseding turn clears every shared query/result holder before yielding control."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(AsyncMock())
    private_canary = "superseded-transport-query-canary"
    owned_arguments: dict[str, Any] = {"query": private_canary, "max_results": 3}
    private_arguments = hf_mod.RevocableMcpToolArguments(owned_arguments)
    private_result = hf_mod.RevocableMcpToolResult()
    raw_result = {"query": private_canary, "results": [{"snippet": "private-result-canary"}]}
    captured_result = private_result.capture(raw_result)
    result_future: asyncio.Future[hf_mod._SearchToolResult] = asyncio.get_running_loop().create_future()
    bounded_result = hf_mod._SearchToolResult(canonical=f'{{"query":"{private_canary}"}}')
    result_future.set_result(bounded_result)
    state = hf_mod._SearchCallState(
        call_id="call-active",
        response_id="response-active",
        response_done=hf_mod._SearchResponseDone(),
        token=hf_mod._SearchTurnToken(epoch=0, item_id="old-item", generation=0, transcript="old transcript"),
        query=private_canary,
        max_results=3,
        result=result_future,
        superseded=asyncio.Event(),
        private_arguments=private_arguments,
        private_result=private_result,
    )
    handler._active_search = state
    queued_notification = ToolNotification(
        id=state.call_id,
        tool_name=hf_mod._OFFICIAL_SEARCH_TOOL_NAME,
        is_idle_tool_call=False,
        status=ToolState.COMPLETED,
    )
    queued_notification.result = captured_result
    handler.tool_manager._notification_queue.put_nowait(queued_notification)

    handler._record_search_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="new-item"),
        "new unrelated turn",
    )

    assert state.superseded.is_set()
    assert private_arguments.revoked
    assert private_result.revoked
    assert owned_arguments == {}
    assert raw_result == {}
    assert queued_notification.result == {}
    assert bounded_result.canonical is None
    assert state.private_arguments is None
    assert state.private_result is None
    assert state.query == ""
    assert state.token.transcript == ""


@pytest.mark.asyncio
async def test_shutdown_revokes_search_transport_before_waiting_for_connection_close() -> None:
    """Public shutdown clears query ownership before its first external await."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    private_canary = "shutdown-transport-query-canary"
    owned_arguments: dict[str, Any] = {"query": private_canary}
    private_arguments = hf_mod.RevocableMcpToolArguments(owned_arguments)
    state = hf_mod._SearchCallState(
        call_id="call-active",
        response_id="response-active",
        response_done=hf_mod._SearchResponseDone(),
        token=hf_mod._SearchTurnToken(epoch=0, item_id="item", generation=0, transcript="search"),
        query=private_canary,
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
        private_arguments=private_arguments,
    )
    handler._active_search = state
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class Connection:
        async def close(self) -> None:
            close_started.set()
            await release_close.wait()

    handler.connection = Connection()  # type: ignore[assignment]
    handler.tool_manager = MagicMock()
    handler.tool_manager.shutdown = AsyncMock()
    shutdown_task = asyncio.create_task(handler.shutdown())
    await close_started.wait()

    assert private_arguments.revoked
    assert owned_arguments == {}
    assert state.query == ""

    release_close.set()
    await shutdown_task


def test_late_private_response_remains_classified_and_fully_suppressed() -> None:
    """A response created after local failure cannot leak text, tools, or stale audio."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    marker = "late-private-marker"
    handler._response_purposes_by_marker[marker] = "search_answer"
    handler._abandoned_private_response_markers.add(marker)
    response = SimpleNamespace(id="response-late", metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker})

    assert not handler._observe_response_created(_FakeEvent("response.created", response=response))
    late_event = _FakeEvent("response.output_audio.delta", response_id="response-late")
    assert handler._response_event_has_private_text(late_event)
    assert handler._response_event_has_tools_disabled(late_event)
    assert handler._response_event_is_suppressed(late_event)


@pytest.mark.asyncio
async def test_completed_private_response_keeps_late_event_tombstones(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Events arriving after response.done remain private until connection teardown."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    marker = "completed-private-marker"
    event_id = "event-completed-private"
    response_id = "response-completed-private"
    handler._response_purposes_by_marker[marker] = "search_answer"
    handler._response_event_ids_by_marker[marker] = event_id
    handler._response_purposes_by_event_id[event_id] = "search_answer"
    response = SimpleNamespace(
        id=response_id,
        metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        status="completed",
    )
    handler._observe_response_created(_FakeEvent("response.created", response=response))
    done = _FakeEvent("response.done", response=response)
    handler._observe_response_done(done)
    handler._finish_response_suppression(done)

    late_event = _FakeEvent("response.output_audio.delta", response_id=response_id)
    assert handler._response_event_purpose(late_event) == "search_answer"
    assert handler._response_event_is_suppressed(late_event)
    error_canary = "post-done-private-error-canary"
    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(event_id=event_id, message=error_canary, code="private_failure"),
        )
    )
    assert error_canary not in caplog.text
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_late_private_error_is_redacted_after_sender_state_resets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Late private backend failures cannot regain raw logs or UI errors."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_purpose = "ordinary"
    handler._response_purposes_by_event_id["event-private"] = "search_answer"
    error_canary = "private-error-canary"

    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(
                event_id="event-private",
                message=error_canary,
                code="private_failure",
            ),
        )
    )

    assert error_canary not in caplog.text
    assert handler.output_queue.empty()


@pytest.mark.asyncio
async def test_unknown_error_id_is_redacted_during_private_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An uncorrelated backend error cannot expose details during a private window."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_purpose = "search_answer"
    handler._active_response_event_id = "event-current-private"
    handler._response_purposes_by_event_id["event-current-private"] = "search_answer"
    handler._response_request_done_event.clear()
    error_canary = "unknown-id-private-error-canary"

    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(
                event_id="event-unknown",
                message=error_canary,
                code="response_failed",
            ),
        )
    )

    assert not handler._last_response_failed
    assert not handler._response_started_or_rejected_event.is_set()
    assert error_canary not in caplog.text
    generic_error = await handler.output_queue.get()
    assert generic_error.args == ({"role": "assistant", "content": "[error] Realtime request failed."},)


@pytest.mark.asyncio
async def test_input_buffer_error_does_not_fail_active_private_response() -> None:
    """A microphone error remains operationally ordinary during private generation."""
    movement_manager = MagicMock()
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager))
    handler._active_response_purpose = "search_answer"
    handler._active_response_event_id = "event-current-private"
    handler._response_purposes_by_event_id["event-current-private"] = "search_answer"
    handler._response_request_done_event.clear()

    await handler._handle_realtime_error(
        _FakeEvent(
            "error",
            error=SimpleNamespace(message="empty microphone buffer", code="input_audio_buffer_commit_empty"),
        )
    )

    assert not handler._last_response_failed
    assert not handler._response_started_or_rejected_event.is_set()
    assert not handler._response_request_done_event.is_set()
    movement_manager.set_listening.assert_called_once_with(False)
    if handler._utterance_completion_task is not None:
        await handler._utterance_completion_task


@pytest.mark.asyncio
async def test_eventless_abandoned_private_error_does_not_wake_current_private_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ambiguous old backend error cannot fail or complete a newer indicator."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_purpose = "search_indicator"
    handler._active_response_event_id = "event-current-private"
    handler._abandoned_private_response_markers.add("marker-old-private")
    handler._response_started_or_rejected_event.clear()
    handler._response_request_done_event.clear()
    error_canary = "eventless-old-private-error-canary"

    await handler._handle_realtime_error(
        _FakeEvent("error", error=SimpleNamespace(message=error_canary, code="private_failure"))
    )

    assert not handler._response_started_or_rejected_event.is_set()
    assert not handler._response_request_done_event.is_set()
    assert not handler._last_response_failed
    assert error_canary not in caplog.text
    assert handler.output_queue.empty()
    assert not handler._abandoned_private_response_markers


@pytest.mark.asyncio
async def test_private_response_without_metadata_fails_closed() -> None:
    """Missing echoed metadata suppresses all output instead of downgrading privacy."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_purpose = "search_answer"
    handler._active_response_marker = "expected-private-marker"
    handler._active_response_event_id = "event-private"
    response = SimpleNamespace(id="response-without-metadata", metadata={})

    assert not handler._observe_response_created(_FakeEvent("response.created", response=response))
    private_event = _FakeEvent(
        "response.output_audio.delta",
        response_id=response.id,
        delta=base64.b64encode(np.ones(16, dtype=np.int16).tobytes()).decode("ascii"),
    )
    assert handler._response_event_has_private_text(private_event)
    assert handler._response_event_has_tools_disabled(private_event)
    assert handler._response_event_is_suppressed(private_event)
    assert not await handler._handle_response_audio_delta(private_event)
    assert handler.output_queue.empty()


def test_abandoned_private_response_without_metadata_claims_one_tombstone() -> None:
    """A metadata-free late answer stays private without poisoning later responses."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    marker = "abandoned-private-marker"
    handler._response_purposes_by_marker[marker] = "search_answer"
    handler._abandoned_private_response_markers.add(marker)
    handler._active_response_purpose = "ordinary"

    late_response = SimpleNamespace(id="response-late-without-metadata", metadata={})
    assert not handler._observe_response_created(_FakeEvent("response.created", response=late_response))
    late_event = _FakeEvent("response.output_audio_transcript.done", response_id=late_response.id)
    assert handler._response_event_purpose(late_event) == "search_answer"
    assert handler._response_event_has_tools_disabled(late_event)
    assert handler._response_event_is_suppressed(late_event)
    assert handler._response_markers_by_id[late_response.id] == marker

    ordinary_response = SimpleNamespace(id="response-ordinary-without-metadata", metadata={})
    assert not handler._observe_response_created(_FakeEvent("response.created", response=ordinary_response))
    ordinary_event = _FakeEvent("response.output_audio_transcript.done", response_id=ordinary_response.id)
    assert handler._response_event_purpose(ordinary_event) == "ordinary"
    assert not handler._response_event_is_suppressed(ordinary_event)

    late_response.status = "completed"
    late_done = _FakeEvent("response.done", response=late_response)
    handler._observe_response_done(late_done)
    handler._finish_response_suppression(late_done)
    assert marker not in handler._abandoned_private_response_markers
    assert late_response.id in handler._private_response_tombstones


def test_active_private_response_without_metadata_binds_expected_marker() -> None:
    """An active metadata-free response cannot leave its marker to poison a later turn."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    marker = "active-private-marker"
    handler._active_response_purpose = "search_answer"
    handler._active_response_marker = marker
    handler._response_purposes_by_marker[marker] = "search_answer"
    response = SimpleNamespace(id="response-active-without-metadata", metadata={})

    assert not handler._observe_response_created(_FakeEvent("response.created", response=response))
    assert handler._response_markers_by_id[response.id] == marker
    handler._abandoned_private_response_markers.add(marker)
    handler._active_response_purpose = "ordinary"
    handler._active_response_marker = None

    next_response = SimpleNamespace(id="response-next-ordinary", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=next_response))
    next_event = _FakeEvent("response.output_audio_transcript.done", response_id=next_response.id)
    assert handler._response_event_purpose(next_event) == "ordinary"
    assert not handler._response_event_is_suppressed(next_event)


@pytest.mark.asyncio
async def test_new_speech_immediately_scrubs_and_suppresses_active_private_response() -> None:
    """Supersession closes the response.created-to-sender-abandonment audio race."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    marker = "active-private-superseded-marker"
    response_id = "response-active-private-superseded"
    private_canary = "active-private-superseded-canary"
    payload = {"response": {"input": [{"content": [{"text": private_canary}]}]}}
    handler._active_response_purpose = "search_answer"
    handler._active_response_marker = marker
    handler._active_response_id = response_id
    handler._active_private_response_payload = payload
    handler._response_purposes_by_marker[marker] = "search_answer"
    handler._response_purposes_by_id[response_id] = "search_answer"

    handler._record_search_transcript(_FakeEvent("completed", item_id="item-new"), "new turn")

    assert response_id in handler._private_response_tombstones
    assert marker in handler._abandoned_private_response_markers
    assert payload == {}
    audio_event = _FakeEvent(
        "response.output_audio.delta",
        response_id=response_id,
        delta=base64.b64encode(np.ones(16, dtype=np.int16).tobytes()).decode("ascii"),
    )
    assert not await handler._handle_response_audio_delta(audio_event)
    assert handler.output_queue.empty()


def test_begin_search_session_clears_stale_response_suppression() -> None:
    """A cancelled prior sender cannot mute a later search-only session."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._suppress_active_response = True
    handler._suppressed_response_ids.add("response-prior-session")

    handler._begin_search_session()

    assert not handler._suppress_active_response
    assert not handler._suppressed_response_ids


@pytest.mark.asyncio
async def test_cancelled_private_response_is_not_reported_completed() -> None:
    """An explicit cancelled terminal status fails the private response gate."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    sender = asyncio.create_task(handler._response_sender_loop())
    response_task = asyncio.create_task(
        handler._queue_private_search_statement(
            purpose="search_indicator",
            statement=hf_mod._SEARCH_INDICATOR_TEXT,
        )
    )
    try:
        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        request = handler.connection.response.create.await_args.kwargs
        marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        response = SimpleNamespace(
            id="response-cancelled-private",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
            status="cancelled",
        )
        handler._response_done_event.clear()
        assert handler._observe_response_created(_FakeEvent("response.created", response=response))
        done = _FakeEvent("response.done", response=response)
        assert handler._observe_response_done(done)
        handler._finish_response_suppression(done)

        assert await response_task == "failed"
    finally:
        sender.cancel()
        await sender


@pytest.mark.asyncio
async def test_private_response_timeout_scrubs_and_abandons_queued_request(monkeypatch: Any) -> None:
    """A private payload that times out in the queue can never be sent later."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_ACCEPTANCE_TIMEOUT", 0.01)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    private_canary = "queued-private-payload-canary"

    outcome = await handler._queue_search_response(
        purpose="search_answer",
        response={
            "conversation": "none",
            "input": handler._search_response_input(private_canary),
            "tool_choice": "none",
        },
    )

    assert outcome == "failed"
    request = handler._pending_responses.get_nowait()
    assert request.abandoned.is_set()
    assert request.kwargs == {}
    handler._pending_responses.put_nowait(request)
    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        await asyncio.sleep(0)
        handler.connection.response.create.assert_not_awaited()
    finally:
        sender.cancel()
        await sender


@pytest.mark.asyncio
async def test_blocked_response_create_retains_no_private_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancellation-resistant sender call keeps only scrubbed containers after its bound."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_CREATE_TIMEOUT", 0.01)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    started = asyncio.Event()
    release = asyncio.Event()
    captured: dict[str, Any] = {}
    private_canary = "blocked-response-create-private-canary"

    async def blocked_create(**kwargs: Any) -> None:
        captured.update(kwargs)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    handler.connection.response.create.side_effect = blocked_create
    sender = asyncio.create_task(handler._response_sender_loop())
    response_task = asyncio.create_task(
        handler._queue_search_response(
            purpose="search_answer",
            response={
                "conversation": "none",
                "input": handler._search_response_input(private_canary),
                "tool_choice": "none",
            },
        )
    )
    try:
        await started.wait()
        assert await asyncio.wait_for(response_task, timeout=0.2) == "failed"
        assert private_canary not in json.dumps(captured)
        assert handler._active_private_response_payload is None
        assert len(handler._late_response_create_tasks) == 1
        release.set()
        await _wait_until(lambda: not handler._late_response_create_tasks)
    finally:
        release.set()
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)


@pytest.mark.asyncio
async def test_private_response_timeout_cancels_and_suppresses_active_request(monkeypatch: Any) -> None:
    """A timed-out accepted answer is cancelled and cannot speak stale audio."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_ACCEPTANCE_TIMEOUT", 0.01)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    sender = asyncio.create_task(handler._response_sender_loop())
    response_task = asyncio.create_task(
        handler._queue_search_response(
            purpose="search_answer",
            response={
                "conversation": "none",
                "input": handler._search_response_input("active-private-payload-canary"),
                "tool_choice": "none",
            },
        )
    )
    try:
        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        request = handler.connection.response.create.await_args.kwargs
        marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        response = SimpleNamespace(
            id="response-timed-out-private",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        )
        assert handler._observe_response_created(_FakeEvent("response.created", response=response))

        assert await response_task == "failed"
        await _wait_until(lambda: handler.connection.response.cancel.await_count == 1)
        stale_audio = _FakeEvent(
            "response.output_audio.delta",
            response_id="response-timed-out-private",
            delta=base64.b64encode(np.ones(16, dtype=np.int16).tobytes()).decode("ascii"),
        )
        assert not await handler._handle_response_audio_delta(stale_audio)
        assert handler.output_queue.empty()
        await _wait_until(
            lambda: (
                sender.get_coro().cr_frame is not None and sender.get_coro().cr_frame.f_locals.get("send_kwargs") == {}
            )
        )
        sender_frame = sender.get_coro().cr_frame
        assert sender_frame is not None
        sender_request = sender_frame.f_locals["request"]
        assert sender_request.kwargs == {}
        assert sender_frame.f_locals["send_kwargs"] == {}
    finally:
        response_task.cancel()
        sender.cancel()
        await asyncio.gather(response_task, sender, return_exceptions=True)


@pytest.mark.asyncio
async def test_private_response_done_timeout_tombstones_and_flushes_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    """An accepted response that never finishes cannot speak queued or late private audio."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_DONE_TIMEOUT", 0.01)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._clear_queue = MagicMock()
    sender = asyncio.create_task(handler._response_sender_loop())
    response_task = asyncio.create_task(
        handler._queue_search_response(
            purpose="search_answer",
            response={
                "conversation": "none",
                "input": handler._search_response_input("response-done-timeout-canary"),
                "tool_choice": "none",
            },
        )
    )
    try:
        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        request = handler.connection.response.create.await_args.kwargs
        marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        response_id = "response-never-done-private"
        response = SimpleNamespace(
            id=response_id,
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        )
        assert handler._observe_response_created(_FakeEvent("response.created", response=response))
        handler.output_queue.put_nowait((handler.SAMPLE_RATE, np.ones((1, 16), dtype=np.int16)))

        assert await response_task == "failed"
        await _wait_until(lambda: handler.connection.response.cancel.await_count == 1)

        assert response_id in handler._private_response_tombstones
        assert handler.output_queue.empty()
        handler._clear_queue.assert_called_once_with()
        late_audio = _FakeEvent(
            "response.output_audio.delta",
            response_id=response_id,
            delta=base64.b64encode(np.ones(16, dtype=np.int16).tobytes()).decode("ascii"),
        )
        assert not await handler._handle_response_audio_delta(late_audio)
    finally:
        response_task.cancel()
        sender.cancel()
        await asyncio.gather(response_task, sender, return_exceptions=True)


@pytest.mark.asyncio
async def test_search_session_teardown_flushes_private_player_and_output_queue() -> None:
    """Reconnect teardown drops result-derived PCM and pending UI items before state reset."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._clear_queue = MagicMock()
    handler._response_purposes_by_id["response-private"] = "search_answer"
    private_payload = {"response": {"input": "private-result-canary"}}
    handler._active_response_purpose = "search_answer"
    handler._active_private_response_payload = private_payload
    handler.output_queue.put_nowait(AdditionalOutputs({"role": "user", "content": "private query"}))
    handler.output_queue.put_nowait((handler.SAMPLE_RATE, np.ones((1, 16), dtype=np.int16)))

    await handler._end_search_session()

    handler._clear_queue.assert_called_once_with()
    assert handler.output_queue.empty()
    assert private_payload == {}
    assert handler._response_purposes_by_id == {}


def test_unrelated_ordinary_completion_clears_pending_search_confirmation() -> None:
    """A delivered confirmation cannot survive a completed unrelated turn."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    cleanup = MagicMock()
    handler._pending_search_confirmation_cleanup = cleanup
    response = SimpleNamespace(id="response-unrelated", metadata={}, status="completed")

    assert not handler._observe_response_done(_FakeEvent("response.done", response=response))

    cleanup.assert_called_once_with()
    assert handler._pending_search_confirmation_cleanup is None


@pytest.mark.asyncio
async def test_scheduled_confirmation_reply_survives_selecting_response_done() -> None:
    """A claimed reply keeps consent pending until its scheduled policy task evaluates it."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    cleanup = MagicMock()
    handler._pending_search_confirmation_cleanup = cleanup
    handler._active_search = hf_mod._SearchCallState(
        call_id="call-confirmation-reply",
        response_id="response-confirmation-reply",
        response_done=hf_mod._SearchResponseDone(),
        token=hf_mod._SearchTurnToken(epoch=0, item_id="item", generation=0, transcript="yes"),
        query="pending query",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    response = SimpleNamespace(id="response-confirmation-reply", metadata={}, status="completed")

    assert not handler._observe_response_done(_FakeEvent("response.done", response=response))

    cleanup.assert_not_called()
    assert handler._pending_search_confirmation_cleanup is cleanup


@pytest.mark.asyncio
async def test_private_response_supersession_cancels_and_suppresses_active_request() -> None:
    """A new turn abandons an accepted private response without waiting for timeout."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    superseded = asyncio.Event()
    sender = asyncio.create_task(handler._response_sender_loop())
    response_task = asyncio.create_task(
        handler._queue_search_response(
            purpose="search_answer",
            response={
                "conversation": "none",
                "input": handler._search_response_input("superseded-private-payload-canary"),
                "tool_choice": "none",
            },
            abandon_on=superseded,
        )
    )
    try:
        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        request = handler.connection.response.create.await_args.kwargs
        marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        response = SimpleNamespace(
            id="response-superseded-private",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        )
        assert handler._observe_response_created(_FakeEvent("response.created", response=response))

        superseded.set()

        assert await response_task == "stale"
        await _wait_until(lambda: handler.connection.response.cancel.await_count == 1)
        stale_audio = _FakeEvent(
            "response.output_audio.delta",
            response_id="response-superseded-private",
            delta=base64.b64encode(np.ones(16, dtype=np.int16).tobytes()).decode("ascii"),
        )
        assert not await handler._handle_response_audio_delta(stale_audio)
        assert handler.output_queue.empty()
    finally:
        response_task.cancel()
        sender.cancel()
        await asyncio.gather(response_task, sender, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_preserves_private_classification_for_session_owner() -> None:
    """A live answer remains private until the event-owning loop drains."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._response_purposes_by_id["response-private"] = "search_answer"
    operations: list[str] = []

    class Connection:
        async def close(self) -> None:
            assert handler._response_purposes_by_id["response-private"] == "search_answer"
            operations.append("close")

    tool_manager = MagicMock()
    tool_manager.shutdown = AsyncMock(side_effect=lambda: operations.append("manager_shutdown"))
    handler.tool_manager = tool_manager
    handler.connection = Connection()

    await handler.shutdown()

    assert operations == ["close", "manager_shutdown"]
    assert handler._response_purposes_by_id == {"response-private": "search_answer"}


@pytest.mark.asyncio
async def test_shutdown_preserves_private_classification_when_connection_close_fails() -> None:
    """A failed close cannot downgrade a concurrently streaming private response."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._response_purposes_by_id["response-private"] = "search_answer"

    class Connection:
        async def close(self) -> None:
            raise RuntimeError("close failed")

    tool_manager = MagicMock()
    tool_manager.shutdown = AsyncMock()
    handler.tool_manager = tool_manager
    handler.connection = Connection()

    await handler.shutdown()

    assert handler._response_purposes_by_id["response-private"] == "search_answer"


@pytest.mark.parametrize("gate_mode", ["missing", "refused", "raised"])
@pytest.mark.asyncio
async def test_revision_gate_fails_closed_before_indicator_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    gate_mode: str,
) -> None:
    """An absent, refused, or failed metadata preflight cannot transmit a query."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    async def refuse_gate() -> bool:
        return False

    async def raise_gate() -> bool:
        raise RuntimeError("metadata failure")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    if gate_mode == "refused":
        handler.set_search_space_gate(refuse_gate)
    elif gate_mode == "raised":
        handler.set_search_space_gate(raise_gate)
    handler._begin_search_session()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-revision",
        generation=handler._search_turn_generation,
        transcript="search now",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-revision",
        response_id="response-revision",
        response_done=response_done,
        token=token,
        query="private-query-canary",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    send_marker = AsyncMock(return_value=True)
    queue_failure = AsyncMock()
    monkeypatch.setattr(handler, "_send_search_marker", send_marker)
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)

    await handler._coordinate_search(state)

    handler.tool_manager.start_tool.assert_not_awaited()
    send_marker.assert_awaited_once_with("call-revision", response_done, hf_mod._SEARCH_FAILURE_MARKER)
    queue_failure.assert_awaited_once_with(abandon_on=state.superseded)
    assert state.query == ""
    assert state.token.transcript == ""


@pytest.mark.asyncio
async def test_unbound_search_manifest_source_refuses_before_metadata_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checked Space cannot authorize a different manifest MCP destination."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    revision_gate = AsyncMock(return_value=True)
    handler.set_search_space_gate(revision_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-source-mismatch",
        generation=handler._search_turn_generation,
        transcript="search now",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-source-mismatch",
        response_id="response-source-mismatch",
        response_done=response_done,
        token=token,
        query="private-query-canary",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    handler._in_flight_tool_calls.add(state.call_id)
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    monkeypatch.setattr(hf_mod.core_tools, "resolve_expected_remote_mcp_tool", MagicMock(return_value=None))
    monkeypatch.setattr(handler, "_send_search_marker", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_queue_search_failure", AsyncMock())

    await handler._coordinate_search(state)

    revision_gate.assert_not_awaited()
    handler.tool_manager.start_tool.assert_not_awaited()
    assert state.query == ""
    assert not handler._in_flight_tool_calls


@pytest.mark.parametrize("response_status", ["cancelled", "failed", "incomplete", None])
@pytest.mark.asyncio
async def test_noncompleted_search_selecting_response_never_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    response_status: str | None,
) -> None:
    """Only an explicitly completed selecting response can release a query."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    queue_failure = AsyncMock()
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-terminal-status"), "search now")
    response = SimpleNamespace(id="response-terminal-status", metadata={}, status=response_status)
    handler._observe_response_created(_FakeEvent("response.created", response=response))
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=response.id,
            call_id="call-terminal-status",
            arguments='{"query":"current score"}',
        )
    )
    state = handler._active_search
    assert state is not None

    handler._observe_response_done(_FakeEvent("response.done", response=response))
    await _wait_until(lambda: handler._active_search is None)

    handler.tool_manager.start_tool.assert_not_awaited()
    handler.connection.conversation.item.create.assert_not_awaited()
    queue_failure.assert_awaited_once_with(abandon_on=state.superseded)


def test_search_space_gate_cannot_change_during_session() -> None:
    """Revision ownership is locked with the policy for one connection."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = MagicMock()

    with pytest.raises(RuntimeError, match="Space gate cannot change"):
        handler.set_search_space_gate(_allow_search_space_gate)


@pytest.mark.asyncio
async def test_search_supersession_during_revision_gate_never_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metadata await cannot revive a turn replaced by new speech."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    gate_started = asyncio.Event()
    release_gate = asyncio.Event()

    async def delayed_gate() -> bool:
        gate_started.set()
        await release_gate.wait()
        return True

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(delayed_gate)
    handler._begin_search_session()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-revision",
        generation=handler._search_turn_generation,
        transcript="search now",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-revision",
        response_id="response-revision",
        response_done=response_done,
        token=token,
        query="private-query-canary",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    send_marker = AsyncMock(return_value=True)
    queue_failure = AsyncMock()
    monkeypatch.setattr(handler, "_send_search_marker", send_marker)
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)

    coordinator = asyncio.create_task(handler._coordinate_search(state))
    await gate_started.wait()
    handler._invalidate_search_turn()
    release_gate.set()
    await coordinator

    handler.tool_manager.start_tool.assert_not_awaited()
    send_marker.assert_awaited_once_with("call-revision", response_done, hf_mod._SEARCH_FAILURE_MARKER)
    queue_failure.assert_not_awaited()
    assert state.query == ""
    assert state.token.transcript == ""


@pytest.mark.asyncio
async def test_started_search_failure_is_abandoned_by_new_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    """A generic search failure cannot finish speaking into a superseding turn."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    failure_started = asyncio.Event()

    async def wait_for_supersession(*, abandon_on: asyncio.Event | None = None) -> None:
        assert abandon_on is not None
        failure_started.set()
        await abandon_on.wait()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(AsyncMock(return_value=False))
    handler._begin_search_session()
    handler.connection = AsyncMock()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-failing-search",
        generation=handler._search_turn_generation,
        transcript="search now",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-failing-search",
        response_id="response-failing-search",
        response_done=response_done,
        token=token,
        query="current score",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    monkeypatch.setattr(handler, "_send_search_marker", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_queue_search_failure", wait_for_supersession)

    coordinator = asyncio.create_task(handler._coordinate_search(state))
    await failure_started.wait()
    assert handler._active_search is state
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-new-turn"), "new turn")
    await coordinator

    assert state.superseded.is_set()
    assert handler._active_search is None


@pytest.mark.asyncio
async def test_search_admission_blocks_early_ordinary_tool_followup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling result waits at admission and cannot add a second search-flow response."""
    policy_started = asyncio.Event()
    release_policy = asyncio.Event()

    async def delayed_refusal(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        policy_started.set()
        await release_policy.wait()
        return conv_mod.SearchPolicyDecision(outcome="refused")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(delayed_refusal)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler.tool_manager = MagicMock()
    safe_response_create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", safe_response_create)
    monkeypatch.setattr(handler, "_send_search_marker", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_queue_search_failure", AsyncMock())
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-early-sibling"), "search now")
    selecting_response = SimpleNamespace(id="response-early-sibling", metadata={}, status="completed")
    handler._observe_response_created(_FakeEvent("response.created", response=selecting_response))
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id=selecting_response.id,
            call_id="call-search-admitted",
            arguments='{"query":"current score"}',
        )
    )
    await policy_started.wait()
    assert "call-search-admitted" in handler._in_flight_tool_calls

    handler._in_flight_tool_calls.add("call-ordinary-sibling")
    await handler._handle_tool_result(
        ToolNotification(
            id="call-ordinary-sibling",
            tool_name="ordinary_sibling",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"status": "ok"},
        )
    )

    safe_response_create.assert_not_awaited()
    assert handler._tool_batch_needs_response
    assert handler._in_flight_tool_calls == {"call-search-admitted"}

    release_policy.set()
    await _wait_until(lambda: handler._active_search is None)
    safe_response_create.assert_not_awaited()
    assert not handler._tool_batch_needs_response


@pytest.mark.asyncio
async def test_search_completion_suppresses_waiting_ordinary_tool_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search flow owns the sole follow-up for its co-occurring tool batch."""

    async def refuse(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="refused")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(refuse)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-tool-batch",
        generation=handler._search_turn_generation,
        transcript="search and use another tool",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-search-batch",
        response_id="response-search-batch",
        response_done=response_done,
        token=token,
        query="current score",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    handler._in_flight_tool_calls.add(state.call_id)
    handler._tool_batch_needs_response = True
    safe_response_create = AsyncMock()
    monkeypatch.setattr(handler, "_send_search_marker", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_queue_search_failure", AsyncMock())
    monkeypatch.setattr(handler, "_safe_response_create", safe_response_create)

    await handler._coordinate_search(state)

    assert not handler._in_flight_tool_calls
    assert not handler._tool_batch_needs_response
    safe_response_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_search_sibling_cannot_follow_or_invalidate_delivered_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search response ownership persists when its ordinary sibling finishes last."""
    abandon_confirmation = MagicMock()

    async def require_confirmation(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(
            outcome="confirmation_required",
            confirmation_question="May I search for that personal detail?",
            on_confirmation_abandoned=abandon_confirmation,
        )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(require_confirmation)
    handler.connection = AsyncMock()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-slow-sibling",
        generation=handler._search_turn_generation,
        transcript="search my detail and use another tool",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-search-slow-batch",
        response_id="response-slow-batch",
        response_done=response_done,
        token=token,
        query="my personal detail",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    handler._in_flight_tool_calls.update((state.call_id, "call-slow-sibling"))
    handler._tool_call_response_ids["call-slow-sibling"] = state.response_id
    handler._response_done_event.clear()
    safe_response_create = AsyncMock()
    monkeypatch.setattr(handler, "_finish_search_confirmation", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_safe_response_create", safe_response_create)
    sibling_result = asyncio.create_task(
        handler._handle_tool_result(
            ToolNotification(
                id="call-slow-sibling",
                tool_name="ordinary_sibling",
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result={"status": "ok"},
            )
        )
    )
    await asyncio.sleep(0)
    handler._search_owned_response_ids.add(state.response_id)

    await handler._coordinate_search(state)

    assert handler._pending_search_confirmation_cleanup is abandon_confirmation
    assert handler._in_flight_tool_calls == {"call-slow-sibling"}
    handler._response_done_event.set()
    await sibling_result

    safe_response_create.assert_not_awaited()
    assert handler._pending_search_confirmation_cleanup is abandon_confirmation
    assert not handler._in_flight_tool_calls
    assert "call-slow-sibling" not in handler._tool_call_response_ids


@pytest.mark.asyncio
async def test_indicator_error_after_created_fails_search_without_dispatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A correlated private response failure cannot authorize remote dispatch."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-error"), "search now")
    original_response = SimpleNamespace(id="response-error", metadata={}, status="completed")
    handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="response-error",
                call_id="call-error",
                arguments='{"query":"current score"}',
            )
        )
        original_done = _FakeEvent("response.done", response=original_response)
        handler._observe_response_done(original_done)
        handler._finish_response_suppression(original_done)

        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        indicator_request = handler.connection.response.create.await_args_list[0].kwargs
        marker = indicator_request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        indicator_response = SimpleNamespace(
            id="response-error-indicator",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
        )
        assert handler._observe_response_created(_FakeEvent("response.created", response=indicator_response))
        error_canary = "private-indicator-error-canary"
        await handler._handle_realtime_error(
            _FakeEvent(
                "error",
                error=SimpleNamespace(
                    event_id=indicator_request["event_id"],
                    message=error_canary,
                    code="private_failure",
                ),
            )
        )

        await _accept_response(handler, 1, response_id="response-error-failure")
        await _wait_until(lambda: handler._active_search is None)
        handler.tool_manager.start_tool.assert_not_awaited()
        assert handler._latest_search_turn is None
        assert error_canary not in caplog.text
        marker_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
        assert marker_items == [
            {
                "type": "function_call_output",
                "call_id": "call-error",
                "output": hf_mod._SEARCH_FAILURE_MARKER,
            }
        ]
    finally:
        sender.cancel()
        await sender
        await handler._end_search_session()


@pytest.mark.asyncio
async def test_approved_search_orders_indicator_dispatch_marker_and_private_answer(
    caplog: pytest.LogCaptureFixture,
    _bind_reviewed_search_source: MagicMock,
) -> None:
    """One approved query uses the official manager and keeps result/answer text out of local sinks."""
    private_query = "privacy-canary-query"
    private_result = "privacy-canary-result"
    policy_requests: list[tuple[str, str, str, int]] = []

    async def approve(request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        policy_requests.append((request.item_id, request.transcript, request.query, request.max_results))
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    tool_manager = MagicMock()
    tool_manager.start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="search-tool-id"))
    tool_manager.cancel_tool = AsyncMock(return_value=True)
    tool_manager.discard_tool = MagicMock(return_value=True)
    handler.tool_manager = tool_manager
    handler._record_search_transcript(
        _FakeEvent("conversation.item.input_audio_transcription.completed", item_id="item-search"),
        f"please search for {private_query}",
    )
    original_response = SimpleNamespace(id="response-search", metadata={}, status="completed")
    assert not handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    handler._response_done_event.clear()

    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="response-search",
                call_id="call-search",
                name=hf_mod._OFFICIAL_SEARCH_TOOL_NAME,
                arguments=json.dumps({"query": private_query, "max_results": 2}),
            )
        )
        await asyncio.sleep(0)
        handler.connection.response.create.assert_not_awaited()
        handler.tool_manager.start_tool.assert_not_awaited()

        unrelated_response = SimpleNamespace(id="response-unrelated", metadata={})
        unrelated_done = _FakeEvent("response.done", response=unrelated_response)
        handler._observe_response_done(unrelated_done)
        handler._finish_response_suppression(unrelated_done)
        await asyncio.sleep(0)
        handler.connection.response.create.assert_not_awaited()
        handler.tool_manager.start_tool.assert_not_awaited()

        original_done = _FakeEvent("response.done", response=original_response)
        handler._observe_response_done(original_done)
        handler._finish_response_suppression(original_done)

        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        handler._response_purposes_by_event_id["event-old-private"] = "search_answer"
        handler._abandoned_private_response_markers.add("marker-old-private")
        await handler._handle_realtime_error(
            _FakeEvent(
                "error",
                error=SimpleNamespace(
                    event_id="event-old-private",
                    message="late-private-error-canary",
                    code="private_failure",
                ),
            )
        )
        await asyncio.sleep(0)
        handler.tool_manager.start_tool.assert_not_awaited()

        indicator_request = await _accept_response(handler, 0, response_id="response-indicator")
        indicator = indicator_request["response"]
        assert indicator["conversation"] == "none"
        assert indicator["tool_choice"] == "none"
        assert indicator["input"]
        assert private_query not in json.dumps(indicator)

        await _wait_until(lambda: handler.tool_manager.start_tool.await_count == 1)
        routine = handler.tool_manager.start_tool.await_args.kwargs["tool_call_routine"]
        assert routine.tool_name == hf_mod._OFFICIAL_SEARCH_TOOL_NAME
        assert routine.bound_remote_tool is _bind_reviewed_search_source
        assert routine.args_json_str == "{}"
        assert routine.private_arguments is not None
        assert routine.private_arguments.borrow() == {"query": private_query, "max_results": 2}
        assert handler.tool_manager.start_tool.await_args.kwargs["retain_result"] is False

        await handler._handle_tool_result(
            ToolNotification(
                id="call-search",
                tool_name=hf_mod._OFFICIAL_SEARCH_TOOL_NAME,
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result=_official_search_result(private_query, snippet=private_result),
            )
        )
        await _wait_until(lambda: handler.connection.response.create.await_count > 1)
        marker_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
        assert marker_items == [
            {
                "type": "function_call_output",
                "call_id": "call-search",
                "output": hf_mod._SEARCH_RESULT_MARKER,
            }
        ]

        answer_request = handler.connection.response.create.await_args_list[1].kwargs
        marker = answer_request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        answer_response = SimpleNamespace(
            id="response-answer",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
            status="completed",
        )
        assert handler._observe_response_created(_FakeEvent("response.created", response=answer_response))
        assert answer_request["response"]["conversation"] == "none"
        assert answer_request["response"]["tool_choice"] == "none"
        assert answer_request["response"]["input"]
        assert private_query in json.dumps(answer_request["response"])
        assert private_result in json.dumps(answer_request["response"])

        answer_text_event = _FakeEvent(
            "response.output_audio_transcript.done",
            response_id="response-answer",
            transcript="privacy-canary-answer",
        )
        assert handler._response_event_has_private_text(answer_text_event)
        assert handler._response_event_has_tools_disabled(
            _FakeEvent("response.function_call_arguments.done", response_id="response-answer")
        )
        audio = np.array([1, -1], dtype=np.int16).tobytes()
        assert await handler._handle_response_audio_delta(
            _FakeEvent(
                "response.output_audio.delta",
                response_id="response-answer",
                delta=base64.b64encode(audio),
            )
        )

        answer_done = _FakeEvent("response.done", response=answer_response)
        handler._observe_response_done(answer_done)
        handler._finish_response_suppression(answer_done)
        await _wait_until(lambda: handler._active_search is None)

        assert policy_requests == [("item-search", f"please search for {private_query}", private_query, 2)]
        sender_frame = sender.get_coro().cr_frame
        assert sender_frame is not None
        sender_request = sender_frame.f_locals["request"]
        assert sender_request.kwargs == {}
        assert sender_request.search_turn is None
        assert sender_frame.f_locals["send_kwargs"] == {}
        assert handler.tool_manager.discard_tool.call_count >= 1
        assert handler._latest_search_turn is None
        assert private_query not in caplog.text
        assert private_result not in caplog.text
        assert "late-private-error-canary" not in caplog.text
        queued_outputs = []
        while not handler.output_queue.empty():
            queued_outputs.append(handler.output_queue.get_nowait())
        assert len(queued_outputs) == 1
        assert isinstance(queued_outputs[0], tuple)
    finally:
        sender.cancel()
        await sender
        await handler._end_search_session()


@pytest.mark.asyncio
async def test_dispatched_search_supersession_cancels_without_stale_failure() -> None:
    """New speech promptly discards transport work without speaking into the new turn."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    tool_manager = MagicMock()
    tool_manager.start_tool = AsyncMock(return_value=SimpleNamespace(tool_id="search-tool-superseded"))
    tool_manager.cancel_tool = AsyncMock(return_value=True)
    tool_manager.discard_tool = MagicMock(return_value=True)
    tool_manager.discard_tool_call = MagicMock()
    handler.tool_manager = tool_manager
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-old"), "search old turn")
    original_response = SimpleNamespace(id="response-old", metadata={}, status="completed")
    handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="response-old",
                call_id="call-old",
                arguments='{"query":"old query"}',
            )
        )
        original_done = _FakeEvent("response.done", response=original_response)
        handler._observe_response_done(original_done)
        handler._finish_response_suppression(original_done)
        await _accept_response(handler, 0, response_id="response-old-indicator")
        await _wait_until(lambda: handler.tool_manager.start_tool.await_count == 1)
        state = handler._active_search
        assert state is not None

        handler._invalidate_search_turn()

        await _wait_until(lambda: handler._active_search is None)
        handler.tool_manager.cancel_tool.assert_awaited_once_with("search-tool-superseded", log=False)
        handler.tool_manager.discard_tool.assert_called_with("search-tool-superseded")
        assert handler.connection.response.create.await_count == 1
        assert handler._latest_search_turn is None
        assert state.query == ""
        assert state.token.transcript == ""
        marker_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
        assert marker_items == [
            {
                "type": "function_call_output",
                "call_id": "call-old",
                "output": hf_mod._SEARCH_FAILURE_MARKER,
            }
        ]

        late_result = ToolNotification(
            id="call-old",
            tool_name=hf_mod._OFFICIAL_SEARCH_TOOL_NAME,
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result=_official_search_result("old query"),
        )
        await handler._handle_tool_result(late_result)
        assert late_result.result is None
        assert handler.connection.response.create.await_count == 1
    finally:
        sender.cancel()
        await sender
        await handler._end_search_session()


@pytest.mark.asyncio
async def test_personal_search_with_sibling_queues_only_one_tools_disabled_confirmation() -> None:
    """A sibling tool cannot add a follow-up or invalidate the sole consent question."""
    question = "That would send a personal detail to Pollen's search service. Is that okay?"
    abandon_confirmation = MagicMock()

    async def require_confirmation(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(
            outcome="confirmation_required",
            confirmation_question=question,
            on_confirmation_abandoned=abandon_confirmation,
        )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(require_confirmation)
    revision_gate = AsyncMock(return_value=True)
    handler.set_search_space_gate(revision_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-personal"), "search my detail")
    original_response = SimpleNamespace(id="response-personal", metadata={}, status="completed")
    handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    handler._response_done_event.clear()
    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="response-personal",
                call_id="call-personal",
                arguments='{"query":"personal query"}',
            )
        )
        handler._tool_batch_needs_response = True
        original_done = _FakeEvent("response.done", response=original_response)
        handler._observe_response_done(original_done)
        handler._finish_response_suppression(original_done)

        confirmation_request = await _accept_response(handler, 0, response_id="response-confirmation")
        assert confirmation_request["response"]["tool_choice"] == "none"
        assert confirmation_request["response"]["instructions"].endswith(json.dumps(question))
        assert "conversation" not in confirmation_request["response"]
        handler.tool_manager.start_tool.assert_not_awaited()
        revision_gate.assert_not_awaited()
        marker_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
        assert marker_items == [
            {
                "type": "function_call_output",
                "call_id": "call-personal",
                "output": hf_mod._SEARCH_CONFIRMATION_MARKER,
            }
        ]
        await _wait_until(lambda: handler._active_search is None)
        abandon_confirmation.assert_not_called()
        assert handler._pending_search_confirmation_cleanup is abandon_confirmation
        assert handler.connection.response.create.await_count == 1
        assert not handler._tool_batch_needs_response
        assert handler._latest_search_turn is None
    finally:
        sender.cancel()
        await sender
        await handler._end_search_session()


@pytest.mark.asyncio
async def test_failed_confirmation_response_clears_policy_pending_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confirmation that was not delivered invokes its policy cleanup exactly once."""
    lifecycle: list[str] = []

    def abandon_confirmation() -> None:
        lifecycle.append("abandoned")

    async def require_confirmation(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(
            outcome="confirmation_required",
            confirmation_question="May I send that personal detail?",
            on_confirmation_abandoned=abandon_confirmation,
        )

    async def queue_failure(*, abandon_on: asyncio.Event | None = None) -> None:
        assert abandon_on is not None
        lifecycle.append("failure")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(require_confirmation)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-confirmation-failed"), "search my detail")
    original_response = SimpleNamespace(
        id="response-confirmation-selector",
        metadata={},
        status="completed",
    )
    handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id=original_response.id,
                call_id="call-confirmation-failed",
                arguments='{"query":"personal query"}',
            )
        )
        original_done = _FakeEvent("response.done", response=original_response)
        handler._observe_response_done(original_done)
        handler._finish_response_suppression(original_done)

        await _wait_until(lambda: handler.connection.response.create.await_count == 1)
        request = handler.connection.response.create.await_args.kwargs
        marker = request["response"]["metadata"][hf_mod._RESPONSE_REQUEST_METADATA_KEY]
        confirmation_response = SimpleNamespace(
            id="response-confirmation-cancelled",
            metadata={hf_mod._RESPONSE_REQUEST_METADATA_KEY: marker},
            status="cancelled",
        )
        assert handler._observe_response_created(_FakeEvent("response.created", response=confirmation_response))
        confirmation_done = _FakeEvent("response.done", response=confirmation_response)
        handler._observe_response_done(confirmation_done)
        handler._finish_response_suppression(confirmation_done)
        await _wait_until(lambda: handler._active_search is None)

        assert lifecycle == ["abandoned", "failure"]
        handler.tool_manager.start_tool.assert_not_awaited()
        assert not handler._search_confirmation_cleanup_failed
    finally:
        sender.cancel()
        await sender
        await handler._end_search_session()


@pytest.mark.asyncio
async def test_delivered_confirmation_cleanup_survives_missing_policy_reset_hook() -> None:
    """The decision-owned callback clears pending consent at reconnect by itself."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    cleanup = MagicMock()
    handler._pending_search_confirmation_cleanup = cleanup

    await handler._end_search_session()

    cleanup.assert_called_once_with()
    assert handler._pending_search_confirmation_cleanup is None
    assert not handler._search_confirmation_cleanup_failed


@pytest.mark.asyncio
async def test_throwing_delivered_confirmation_cleanup_latches_search_closed() -> None:
    """A reconnect cannot admit more searches after pending consent cleanup fails."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    def failing_cleanup() -> None:
        raise RuntimeError("private cleanup detail")

    handler._pending_search_confirmation_cleanup = failing_cleanup

    await handler._end_search_session()

    assert handler._pending_search_confirmation_cleanup is None
    assert handler._search_confirmation_cleanup_failed


@pytest.mark.asyncio
async def test_confirmation_without_cleanup_hook_latches_search_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A policy cannot leave inaccessible pending consent and keep searching."""

    async def unsafe_confirmation(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        return conv_mod.SearchPolicyDecision(
            outcome="confirmation_required",
            confirmation_question="May I send that personal detail?",
        )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(unsafe_confirmation)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    token = hf_mod._SearchTurnToken(
        epoch=handler._search_connection_epoch,
        item_id="item-unsafe-confirmation",
        generation=handler._search_turn_generation,
        transcript="search my detail",
    )
    handler._latest_search_turn = token
    response_done = hf_mod._SearchResponseDone(completed=True)
    response_done.event.set()
    state = hf_mod._SearchCallState(
        call_id="call-unsafe-confirmation",
        response_id="response-unsafe-confirmation",
        response_done=response_done,
        token=token,
        query="personal query",
        max_results=3,
        result=asyncio.get_running_loop().create_future(),
        superseded=asyncio.Event(),
    )
    handler._active_search = state
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    finish_confirmation = AsyncMock()
    queue_failure = AsyncMock()
    monkeypatch.setattr(handler, "_finish_search_confirmation", finish_confirmation)
    monkeypatch.setattr(handler, "_send_search_marker", AsyncMock(return_value=True))
    monkeypatch.setattr(handler, "_queue_search_failure", queue_failure)

    await handler._coordinate_search(state)

    finish_confirmation.assert_not_awaited()
    handler.tool_manager.start_tool.assert_not_awaited()
    queue_failure.assert_awaited_once_with(abandon_on=state.superseded)
    assert handler._search_confirmation_cleanup_failed


@pytest.mark.asyncio
async def test_invalid_search_resolves_with_marker_and_one_private_generic_failure() -> None:
    """A rejected exact tool call receives no dispatch and one content-free audible failure."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        raise AssertionError("Malformed arguments must not reach policy")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler.tool_manager = MagicMock()
    handler.tool_manager.start_tool = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-invalid"), "search now")
    original_response = SimpleNamespace(id="response-invalid", metadata={}, status="completed")
    handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    handler._response_done_event.clear()
    sender = asyncio.create_task(handler._response_sender_loop())
    try:
        handler._schedule_search_tool_call(
            _FakeEvent(
                "response.function_call_arguments.done",
                response_id="response-invalid",
                call_id="call-invalid",
                arguments='{"query":"private-canary","extra":true}',
            )
        )
        original_done = _FakeEvent("response.done", response=original_response)
        handler._observe_response_done(original_done)
        handler._finish_response_suppression(original_done)

        failure_request = await _accept_response(handler, 0, response_id="response-failure")
        response = failure_request["response"]
        assert response["conversation"] == "none"
        assert response["tool_choice"] == "none"
        assert "private-canary" not in json.dumps(response)
        handler.tool_manager.start_tool.assert_not_awaited()
        marker_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
        assert marker_items == [
            {
                "type": "function_call_output",
                "call_id": "call-invalid",
                "output": hf_mod._SEARCH_FAILURE_MARKER,
            }
        ]
        await _wait_until(lambda: not handler._search_tasks)
        assert handler.connection.response.create.await_count == 1
        assert handler._latest_search_turn is None
        assert len(handler._search_consumed_turns) == 1
    finally:
        sender.cancel()
        await sender
        await handler._end_search_session()


@pytest.mark.asyncio
async def test_unstarted_search_refusal_does_not_speak_after_new_transcript() -> None:
    """A malformed old call resolves silently once a newer user turn exists."""

    async def approve(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        raise AssertionError("Malformed arguments must not reach policy")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(approve)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-old"), "old search turn")
    original_response = SimpleNamespace(id="response-old-invalid", metadata={}, status="completed")
    handler._observe_response_created(_FakeEvent("response.created", response=original_response))
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id="response-old-invalid",
            call_id="call-old-invalid",
            arguments='{"query":"private-canary","extra":true}',
        )
    )

    handler._record_search_transcript(_FakeEvent("completed", item_id="item-new"), "new ordinary turn")
    original_done = _FakeEvent("response.done", response=original_response)
    handler._observe_response_done(original_done)
    handler._finish_response_suppression(original_done)
    await _wait_until(lambda: not handler._search_tasks)

    handler.connection.response.create.assert_not_awaited()
    marker_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
    assert marker_items == [
        {
            "type": "function_call_output",
            "call_id": "call-old-invalid",
            "output": hf_mod._SEARCH_FAILURE_MARKER,
        }
    ]
    assert handler._latest_search_turn is not None
    assert handler._latest_search_turn.item_id == "item-new"
    await handler._end_search_session()


@pytest.mark.asyncio
async def test_search_one_flight_and_rate_limit_survive_new_transcript() -> None:
    """A later transcript cannot release a running search or bypass the rolling attempt limit."""
    policy_gate = asyncio.Event()

    async def delayed_policy(_request: conv_mod.SearchPolicyRequest) -> conv_mod.SearchPolicyDecision:
        await policy_gate.wait()
        return conv_mod.SearchPolicyDecision(outcome="approved")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_search_policy(delayed_policy)
    handler.set_search_space_gate(_allow_search_space_gate)
    handler._begin_search_session()
    handler.connection = AsyncMock()
    handler._record_search_transcript(_FakeEvent("completed", item_id="item-a"), "search for a")
    response_a = SimpleNamespace(id="response-a", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=response_a))
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id="response-a",
            call_id="call-a",
            arguments='{"query":"a"}',
        )
    )
    await _wait_until(lambda: handler._active_search is not None)

    handler._record_search_transcript(_FakeEvent("completed", item_id="item-b"), "search for b")
    response_b = SimpleNamespace(id="response-b", metadata={})
    handler._observe_response_created(_FakeEvent("response.created", response=response_b))
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id="response-b",
            call_id="call-b",
            arguments='{"query":"b"}',
        )
    )
    handler._schedule_search_tool_call(
        _FakeEvent(
            "response.function_call_arguments.done",
            response_id="response-b",
            call_id="call-c",
            arguments='{"query":"c"}',
        )
    )
    assert handler._active_search is not None
    assert handler._active_search.call_id == "call-a"
    assert len(handler._search_attempt_times) == 3
    assert handler._consume_search_attempt() is False

    policy_gate.set()
    await handler._end_search_session()
