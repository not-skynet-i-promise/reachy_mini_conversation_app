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
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
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
    session_update_error: BaseException | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if session_update_error is not None:
                raise session_update_error
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


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    """Wait for an asynchronous test condition."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("Timed out waiting for condition")


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
    )
    handler._response_done_event.clear()
    assert handler._observe_response_created(_FakeEvent("response.created", response=response))
    done = _FakeEvent("response.done", response=response)
    assert handler._observe_response_done(done)
    handler._finish_response_suppression(done)
    return request


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
async def test_cancelled_session_update_releases_observer_setup_lock(monkeypatch: Any) -> None:
    """Cancellation before connection ownership cannot strand observer setup."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    async def observer(_utterance: conv_mod.CompletedUserUtterance) -> dict[str, str]:
        return {"status": "unknown"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.set_completed_utterance_observer(observer)
    handler.client = _make_fake_realtime_client(session_update_error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await handler._run_realtime_session()

    assert not handler._completed_utterance_observer_locked
    assert handler.connection is None
    handler.set_completed_utterance_observer(None)


@pytest.mark.asyncio
async def test_session_teardown_clears_tool_batch_state(monkeypatch: Any) -> None:
    """An interrupted tool call must not block responses after reconnect."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
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


def test_response_error_correlation_uses_client_event_id() -> None:
    """Errors for another explicit request must not wake the active sender."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_event_id = "event_current"

    assert not handler._error_matches_active_request(SimpleNamespace(event_id="event_other"))
    assert handler._error_matches_active_request(SimpleNamespace(event_id="event_current"))
    # Pinned speech-to-speech 0.2.11 omits the causing ID; serialization is the
    # compatibility correlation guarantee in that backend.
    assert handler._error_matches_active_request(SimpleNamespace(event_id=None))


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
