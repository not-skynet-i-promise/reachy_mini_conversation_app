import json
import time
import uuid
import base64
import random
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Final, Tuple, Optional, TypeAlias
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import Mapping, AsyncIterator

import httpx
import numpy as np
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from numpy.typing import NDArray
from huggingface_hub import get_token
from typing_extensions import Literal, TypedDict
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeToolsConfigParam,
    RealtimeFunctionToolParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from websockets.exceptions import ConnectionClosedError
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.config import (
    HF_LOCAL_CONNECTION_MODE,
    config,
    get_default_voice,
    get_available_voices,
    get_hf_direct_ws_url,
    parse_hf_realtime_url,
    get_hf_connection_selection,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_instructions,
    get_session_greeting_prompt,
    get_session_greeting_tool_name,
)
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.tools.core_tools import (
    ToolSpec,
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.conversation_handler import (
    DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS,
    ConversationHandler,
    CompletedUserUtterance,
    CompletedUtteranceResult,
    CompletedUtteranceObserver,
)
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


if TYPE_CHECKING:
    from openai.resources.realtime.realtime import AsyncRealtimeConnection


logger = logging.getLogger(__name__)

_RESPONSE_DONE_TIMEOUT: Final[float] = 30.0
_RESPONSE_REJECTION_RETRY_DELAY: Final[float] = 0.5
_RESPONSE_REQUEST_METADATA_KEY: Final[str] = "reachy_response_request"
_RESPONSE_ACCEPTANCE_TIMEOUT: Final[float] = 65.0
_OBSERVER_SESSION_STOP_TIMEOUT: Final[float] = 5.0
_UTTERANCE_AUDIO_MAX_BYTES: Final[int] = 480_000
_DISPLAY_NAME_MAX_CHARS: Final[int] = 100
_UTTERANCE_CONTEXT_FUNCTION_NAME: Final[str] = "voice_assessment"
_UNAVAILABLE_UTTERANCE_RESULT: Final[dict[str, str]] = {"status": "unavailable"}

_ResponseOutcome: TypeAlias = Literal["created", "failed", "stale"]


@dataclass(frozen=True)
class _UtteranceToken:
    epoch: int
    item_id: str
    generation: int
    discard_through_sample: int


@dataclass
class _QueuedResponse:
    kwargs: dict[str, Any]
    is_startup: bool = False
    utterance_token: _UtteranceToken | None = None
    outcome: asyncio.Future[_ResponseOutcome] | None = None


class InputTranscriptChunksByItem(BaseModel):
    """Current item_id and its accumulated deltas. Only one item at a time."""

    item_id: str | None = None
    deltas: list[str] = Field(default_factory=list)


def to_realtime_tools_config(tool_specs: list[ToolSpec]) -> RealtimeToolsConfigParam:
    """Convert app tool specs to the OpenAI-compatible realtime session shape."""
    realtime_tools: RealtimeToolsConfigParam = []
    for spec in tool_specs:
        realtime_tools.append(
            RealtimeFunctionToolParam(
                type="function",
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
            )
        )
    return realtime_tools


class HFNativeRateAudioPCM(TypedDict):
    """Hugging Face extension for native-rate PCM audio."""

    type: Literal["audio/pcm"]
    rate: None


def _native_rate_audio_pcm() -> HFNativeRateAudioPCM:
    """Return the Hugging Face native-rate PCM config."""
    return {"type": "audio/pcm", "rate": None}


def _build_openai_compatible_client_from_realtime_url(
    realtime_url: str,
    bearer_token: str | None,
) -> tuple[AsyncOpenAI, dict[str, str]]:
    """Build an OpenAI-compatible realtime client from a direct websocket/base URL."""
    parsed = parse_hf_realtime_url(realtime_url)
    client = AsyncOpenAI(
        api_key=bearer_token or "DUMMY",
        base_url=parsed.base_url,
        websocket_base_url=parsed.websocket_base_url,
    )
    return client, parsed.connect_query


class HuggingFaceRealtimeHandler(ConversationHandler):
    """Realtime stream handler for the Hugging Face OpenAI-compatible endpoint."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__()

        self.deps = deps

        self.client: AsyncOpenAI
        self.connection: "AsyncRealtimeConnection | None" = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.instance_path = instance_path
        self._voice_override: str | None = self._normalize_startup_voice(startup_voice)
        self._realtime_connect_query: dict[str, str] = {}

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_debounce_delay = 0.5  # seconds
        self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()
        self._observer_session_stopped: asyncio.Event = asyncio.Event()
        self._observer_session_stopped.set()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[_QueuedResponse] = asyncio.Queue()
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._response_request_done_event: asyncio.Event = asyncio.Event()
        self._response_request_done_event.set()
        self._response_started_or_rejected_event: asyncio.Event = asyncio.Event()
        self._last_response_rejected: bool = False
        self._last_response_created: bool = False
        self._active_response_event_id: str | None = None
        self._active_response_marker: str | None = None
        self._active_response_id: str | None = None
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None
        self._startup_greeting_sent = False
        self._startup_input_blocked = False
        self._startup_response_pending = False
        self._in_flight_tool_calls: set[str] = set()
        self._internal_tool_calls: set[str] = set()
        self._tool_batch_needs_response = False

        self._connection_epoch = 0
        self._utterance_generation = 0
        self._audio_ring = bytearray()
        self._audio_ring_start_sample = 0
        self._audio_ring_end_sample = 0
        self._audio_capture_compatible = True
        self._utterance_item_id: str | None = None
        self._utterance_spans: list[tuple[int, int]] = []
        self._utterance_span_pcm: list[bytes] = []
        self._utterance_span_pcm_bytes = 0
        self._utterance_spans_valid = True
        self._utterance_discard_through_sample: int | None = None
        self._utterance_segment_start_sample: int | None = None
        self._utterance_segment_valid = False
        self._utterance_observer_task: asyncio.Task[CompletedUtteranceResult] | None = None
        self._late_utterance_observer_tasks: set[asyncio.Future[CompletedUtteranceResult]] = set()
        self._utterance_observer_token: _UtteranceToken | None = None
        self._utterance_completion_task: asyncio.Task[None] | None = None
        self._active_utterance_token: _UtteranceToken | None = None
        self._suppress_active_response = False
        self._stale_response_cancel_sent = False
        self._suppressed_response_ids: set[str] = set()
        self._response_tokens_by_id: dict[str, _UtteranceToken] = {}
        self._completed_utterance_observer_locked = False

    def set_completed_utterance_observer(
        self,
        observer: CompletedUtteranceObserver | None,
        *,
        timeout_seconds: float = DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS,
    ) -> None:
        """Attach the observer before a realtime session is configured."""
        if self.connection is not None or self._completed_utterance_observer_locked:
            raise RuntimeError("The completed-utterance observer cannot change during a realtime session")
        super().set_completed_utterance_observer(observer, timeout_seconds=timeout_seconds)

    @staticmethod
    def _sanitize_tool_result_for_model(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Remove bulky transport-only fields before echoing tool output back to the model."""
        if tool_name == "camera" and "b64_im" in tool_result:
            sanitized = dict(tool_result)
            sanitized.pop("b64_im", None)
            sanitized["image_attached"] = True
            return sanitized
        return tool_result

    def _normalize_startup_voice(self, voice: str | None) -> str | None:
        """Return a valid persisted startup voice, or None."""
        return self._resolve_backend_voice(voice, source="persisted startup voice")

    @asynccontextmanager
    async def _realtime_connection(
        self,
        connect_kwargs: dict[str, Any],
    ) -> AsyncIterator["AsyncRealtimeConnection"]:
        """Signal observer teardown only after the connection context exits."""
        try:
            async with self.client.realtime.connect(**connect_kwargs) as connection:
                yield connection
        finally:
            self._observer_session_stopped.set()

    async def _wait_for_response_done_before_tool_result(self) -> bool:
        """Return whether the function-call response finished before sending tool output."""
        if self._response_done_event.is_set():
            return True

        try:
            await asyncio.wait_for(
                self._response_done_event.wait(),
                timeout=_RESPONSE_DONE_TIMEOUT,
            )
            return True
        except asyncio.TimeoutError:
            return False

    def _resolve_backend_voice(
        self,
        voice: str | None,
        *,
        source: str,
        fallback: str | None = None,
    ) -> str | None:
        """Return a backend-supported voice, optionally falling back when unsupported."""
        available_voices = get_available_voices()
        voice_value = (voice or "").strip()
        if not voice_value:
            return fallback

        voice_by_lowercase = {candidate.lower(): candidate for candidate in available_voices}
        normalized_voice = voice_by_lowercase.get(voice_value.lower())
        if normalized_voice is not None:
            return normalized_voice

        if voice:
            logger.warning(
                "Ignoring unsupported %s %r; expected one of %s",
                source,
                voice,
                available_voices,
            )
        return fallback

    def _get_session_config(self, tool_specs: list[ToolSpec]) -> RealtimeSessionCreateRequestParam:
        """Return the Hugging Face OpenAI-compatible session config."""
        turn_detection = ServerVad(type="server_vad", interrupt_response=True)
        if self._completed_utterance_observer is not None:
            turn_detection = ServerVad(
                type="server_vad",
                create_response=False,
                interrupt_response=True,
            )
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=get_session_instructions(self.instance_path),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    # The OpenAI SDK type only includes 24 kHz PCM, but the HF
                    # compatible server uses rate=None for native 16 kHz mode.
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language=config.REALTIME_TRANSCRIPTION_LANGUAGE,
                    ),
                    turn_detection=turn_detection,
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    def _is_connected(self) -> bool:
        """Return whether the realtime connection is open."""
        return self.connection is not None

    def _idle_behavior_ready(self) -> bool:
        """Hold idle behavior while a model response is still active."""
        return not self._startup_input_blocked and self._response_done_event.is_set()

    async def _cancel_partial_transcript_task(self) -> None:
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

    async def change_voice(self, voice: str) -> str:
        """Change only the voice, updating the active session when possible."""
        default_voice = get_default_voice()
        resolved_voice = (
            self._resolve_backend_voice(voice, source="requested voice", fallback=default_voice) or default_voice
        )
        self._voice_override = resolved_voice
        if self.connection is not None:
            try:
                await self.connection.session.update(
                    session=RealtimeSessionCreateRequestParam(
                        type="realtime",
                        audio=RealtimeAudioConfigParam(
                            output=RealtimeAudioConfigOutputParam(
                                voice=resolved_voice,
                            ),
                        ),
                    ),
                )
                return f"Voice changed to {resolved_voice}."
            except Exception as e:
                logger.warning("Failed to update live session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        default_voice = get_default_voice()
        voice = self._voice_override or get_session_voice(default=default_voice)
        return self._resolve_backend_voice(voice, source="session voice", fallback=default_voice) or default_voice

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions(self.instance_path)
                voice = self.get_current_voice()
            except Exception as e:
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Rebuild the tool registry
            core_tools.initialize_tools(force=True)

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session=RealtimeSessionCreateRequestParam(
                            type="realtime",
                            instructions=instructions,
                            audio=RealtimeAudioConfigParam(
                                output=RealtimeAudioConfigOutputParam(
                                    voice=voice,
                                ),
                            ),
                        ),
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, item_id: str, sequence_counter: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)

            input_transcript = self.input_transcript_chunks_by_item
            if input_transcript.item_id == item_id and len(input_transcript.deltas) - 1 == sequence_counter:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a Hugging Face partial transcript snapshot."""
        input_transcript.item_id = item_id
        input_transcript.deltas = [delta]

    def _cancel_utterance_tasks(self) -> None:
        """Cancel observer work owned by the superseded utterance generation."""
        tasks = (
            self._utterance_observer_task,
            self._utterance_completion_task,
            *tuple(self._late_utterance_observer_tasks),
        )
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
        self._utterance_observer_task = None
        self._utterance_observer_token = None
        self._utterance_completion_task = None

    def _resolve_response_outcome(self, request: _QueuedResponse, outcome: _ResponseOutcome) -> None:
        """Resolve an observer-owned response request once."""
        if request.outcome is not None and not request.outcome.done():
            request.outcome.set_result(outcome)

    def _is_current_utterance(self, token: _UtteranceToken) -> bool:
        """Return whether a token still owns the current observed turn."""
        return (
            self._completed_utterance_observer is not None
            and token.epoch == self._connection_epoch
            and token.item_id == self._utterance_item_id
            and token.generation == self._utterance_generation
        )

    def _purge_stale_utterance_responses(self) -> None:
        """Remove superseded observer requests without disturbing ordinary responses."""
        retained: list[_QueuedResponse] = []
        while True:
            try:
                request = self._pending_responses.get_nowait()
            except asyncio.QueueEmpty:
                break
            if request.utterance_token is not None and not self._is_current_utterance(request.utterance_token):
                self._resolve_response_outcome(request, "stale")
            else:
                retained.append(request)
        for request in retained:
            self._pending_responses.put_nowait(request)

    def _discard_audio_through(self, sample_position: int) -> None:
        """Discard retained PCM through one absolute sample position."""
        bounded_position = min(max(sample_position, self._audio_ring_start_sample), self._audio_ring_end_sample)
        dropped_samples = bounded_position - self._audio_ring_start_sample
        if dropped_samples > 0:
            del self._audio_ring[: dropped_samples * np.dtype(np.int16).itemsize]
            self._audio_ring_start_sample = bounded_position

    def _trim_audio_ring_to_capacity(self) -> None:
        """Keep stopped-segment snapshots plus the live ring within one PCM cap."""
        ring_capacity = max(0, _UTTERANCE_AUDIO_MAX_BYTES - self._utterance_span_pcm_bytes)
        excess_bytes = len(self._audio_ring) - ring_capacity
        if excess_bytes <= 0:
            return
        excess_bytes -= excess_bytes % np.dtype(np.int16).itemsize
        del self._audio_ring[:excess_bytes]
        self._audio_ring_start_sample += excess_bytes // np.dtype(np.int16).itemsize

    def _retain_sent_audio(self, sample_rate: int, audio_bytes: bytes, sample_count: int) -> None:
        """Retain one successfully sent normalized frame when observation is enabled."""
        frame_start = self._audio_ring_end_sample
        self._audio_ring_end_sample += sample_count
        if sample_rate != self.SAMPLE_RATE:
            self._audio_capture_compatible = False
            self._audio_ring.clear()
            self._audio_ring_start_sample = self._audio_ring_end_sample
            return
        if not self._audio_capture_compatible:
            self._audio_ring_start_sample = self._audio_ring_end_sample
            return
        if self._audio_ring_start_sample > frame_start:
            self._audio_ring_start_sample = frame_start
        self._audio_ring.extend(audio_bytes)
        self._trim_audio_ring_to_capacity()

    @classmethod
    def _sample_position(cls, timestamp_ms: object) -> int | None:
        """Convert an integer backend millisecond boundary to a sample position."""
        if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int) or timestamp_ms < 0:
            return None
        return timestamp_ms * cls.SAMPLE_RATE // 1000

    def _audio_span_pcm(self, start_sample: int, end_sample: int) -> bytes | None:
        """Copy one exact absolute sample span from the live audio ring."""
        if (
            start_sample < self._audio_ring_start_sample
            or end_sample > self._audio_ring_end_sample
            or end_sample <= start_sample
        ):
            return None
        sample_size = np.dtype(np.int16).itemsize
        start_byte = (start_sample - self._audio_ring_start_sample) * sample_size
        end_byte = (end_sample - self._audio_ring_start_sample) * sample_size
        return bytes(self._audio_ring[start_byte:end_byte])

    def _combined_utterance_pcm(self) -> bytes | None:
        """Return exact stopped-segment bytes retained under the aggregate cap."""
        if (
            not self._audio_capture_compatible
            or not self._utterance_spans_valid
            or not self._utterance_span_pcm
            or len(self._utterance_span_pcm) != len(self._utterance_spans)
            or self._utterance_span_pcm_bytes > _UTTERANCE_AUDIO_MAX_BYTES
        ):
            return None
        return b"".join(self._utterance_span_pcm)

    @staticmethod
    def _normalize_utterance_result(result: object) -> dict[str, str]:
        """Reduce observer output to the bounded model-visible identity contract."""
        if not isinstance(result, Mapping):
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        status = result.get("status")
        if not isinstance(status, str) or status not in {"matched", "unknown", "uncertain", "unavailable"}:
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        if status != "matched":
            return {"status": status}
        display_name = result.get("display_name")
        if not isinstance(display_name, str):
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        display_name = " ".join(display_name.split())
        if (
            not display_name
            or len(display_name) > _DISPLAY_NAME_MAX_CHARS
            or any(not character.isprintable() for character in display_name)
        ):
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        return {"status": "matched", "display_name": display_name}

    async def _run_completed_utterance_observer(
        self,
        utterance: CompletedUserUtterance,
    ) -> CompletedUtteranceResult:
        """Run the observer under the short fail-soft timeout."""
        observer = self._completed_utterance_observer
        if observer is None:
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        if self._late_utterance_observer_tasks:
            logger.warning("Prior completed-utterance observer is still stopping")
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        try:
            observer_task: asyncio.Future[CompletedUtteranceResult] = asyncio.ensure_future(observer(utterance))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Completed-utterance observer failed")
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        try:
            done, _ = await asyncio.wait((observer_task,), timeout=self._completed_utterance_timeout_seconds)
            if not done:
                observer_task.cancel()
                self._retain_late_utterance_observer_task(observer_task)
                logger.warning("Completed-utterance observer timed out")
                return dict(_UNAVAILABLE_UTTERANCE_RESULT)
            result = observer_task.result()
        except asyncio.CancelledError:
            observer_task.cancel()
            self._retain_late_utterance_observer_task(observer_task)
            raise
        except Exception:
            logger.warning("Completed-utterance observer failed")
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)
        try:
            return self._normalize_utterance_result(result)
        except Exception:
            logger.warning("Completed-utterance observer returned a malformed result")
            return dict(_UNAVAILABLE_UTTERANCE_RESULT)

    def _retain_late_utterance_observer_task(
        self,
        observer_task: asyncio.Future[CompletedUtteranceResult],
    ) -> None:
        """Keep cancellation-suppressing observer work owned until it finishes."""
        if observer_task not in self._late_utterance_observer_tasks:
            self._late_utterance_observer_tasks.add(observer_task)
            observer_task.add_done_callback(self._discard_late_utterance_observer_result)

    def _discard_late_utterance_observer_result(
        self,
        observer_task: asyncio.Future[CompletedUtteranceResult],
    ) -> None:
        """Consume a result that lost timeout or supersession ownership."""
        self._late_utterance_observer_tasks.discard(observer_task)
        try:
            observer_task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Late completed-utterance observer failed", exc_info=True)

    def _invalidate_utterance(self, *, preserve_spans: bool) -> None:
        """Advance the generation and cancel work from the prior segment state."""
        self._utterance_generation += 1
        self._cancel_utterance_tasks()
        self._utterance_segment_start_sample = None
        self._utterance_segment_valid = False
        if not preserve_spans:
            if self._utterance_discard_through_sample is not None:
                self._discard_audio_through(self._utterance_discard_through_sample)
            self._utterance_item_id = None
            self._utterance_spans = []
            self._utterance_span_pcm = []
            self._utterance_span_pcm_bytes = 0
            self._utterance_spans_valid = True
            self._utterance_discard_through_sample = None
        self._purge_stale_utterance_responses()

    def _reset_utterance_state(self) -> None:
        """Invalidate all per-connection observer, span, request, and PCM state."""
        self._connection_epoch += 1
        self._utterance_generation += 1
        self._cancel_utterance_tasks()
        self._audio_ring.clear()
        self._audio_ring_start_sample = 0
        self._audio_ring_end_sample = 0
        self._audio_capture_compatible = True
        self._utterance_item_id = None
        self._utterance_spans = []
        self._utterance_span_pcm = []
        self._utterance_span_pcm_bytes = 0
        self._utterance_spans_valid = True
        self._utterance_discard_through_sample = None
        self._utterance_segment_start_sample = None
        self._utterance_segment_valid = False
        self._active_utterance_token = None
        self._suppress_active_response = False
        self._stale_response_cancel_sent = False
        self._active_response_id = None
        self._suppressed_response_ids.clear()
        self._response_tokens_by_id.clear()
        self._purge_stale_utterance_responses()

    async def _cancel_stale_utterance_response(self) -> None:
        """Cancel an accepted observer response after its turn is superseded."""
        token = self._active_utterance_token
        stale_response_ids = {
            response_id
            for response_id, response_token in self._response_tokens_by_id.items()
            if not self._is_current_utterance(response_token)
        }
        self._suppressed_response_ids.update(stale_response_ids)
        token_is_stale = token is not None and not self._is_current_utterance(token)
        active_response_is_stale = self._active_response_id in stale_response_ids
        if not token_is_stale and not active_response_is_stale:
            return
        self._suppress_active_response = True
        if not self._last_response_created or self._stale_response_cancel_sent or self.connection is None:
            return
        self._stale_response_cancel_sent = True
        try:
            await self.connection.response.cancel()
        except Exception:
            logger.debug("Failed to cancel superseded observer response")

    async def _observe_speech_started(self, event: Any) -> None:
        """Start or reopen one backend-identified utterance segment."""
        item_id_value = getattr(event, "item_id", None)
        item_id = item_id_value if isinstance(item_id_value, str) and item_id_value else None
        preserve_spans = item_id is not None and item_id == self._utterance_item_id
        self._invalidate_utterance(preserve_spans=preserve_spans)
        if not preserve_spans:
            self._utterance_item_id = item_id

        start_sample = self._sample_position(getattr(event, "audio_start_ms", None))
        prior_end = self._utterance_spans[-1][1] if self._utterance_spans else self._audio_ring_start_sample
        self._utterance_segment_start_sample = start_sample
        self._utterance_segment_valid = (
            item_id is not None
            and self._audio_capture_compatible
            and start_sample is not None
            and self._audio_ring_start_sample <= start_sample <= self._audio_ring_end_sample
            and start_sample >= prior_end
        )
        await self._cancel_stale_utterance_response()

    async def _observe_speech_stopped(self, event: Any) -> None:
        """Close one segment and retain its combined PCM until the transcript is final."""
        item_id_value = getattr(event, "item_id", None)
        item_id = item_id_value if isinstance(item_id_value, str) and item_id_value else None
        end_sample = self._sample_position(getattr(event, "audio_end_ms", None))
        start_sample = self._utterance_segment_start_sample
        segment_valid = (
            self._utterance_segment_valid
            and item_id is not None
            and item_id == self._utterance_item_id
            and start_sample is not None
            and end_sample is not None
            and end_sample > start_sample
            and end_sample <= self._audio_ring_end_sample
        )
        segment_pcm = (
            self._audio_span_pcm(start_sample, end_sample)
            if segment_valid and start_sample is not None and end_sample is not None
            else None
        )

        self._invalidate_utterance(preserve_spans=True)
        if (
            segment_pcm is not None
            and start_sample is not None
            and end_sample is not None
            and self._utterance_span_pcm_bytes + len(segment_pcm) <= _UTTERANCE_AUDIO_MAX_BYTES
        ):
            self._utterance_spans.append((start_sample, end_sample))
            self._utterance_span_pcm.append(segment_pcm)
            self._utterance_span_pcm_bytes += len(segment_pcm)
            self._utterance_discard_through_sample = end_sample
            self._discard_audio_through(end_sample)
            self._trim_audio_ring_to_capacity()
        else:
            self._utterance_spans_valid = False
            self._utterance_discard_through_sample = self._audio_ring_end_sample

        discard_through = self._utterance_discard_through_sample
        if discard_through is None:
            discard_through = self._audio_ring_end_sample
        token = _UtteranceToken(
            epoch=self._connection_epoch,
            item_id=self._utterance_item_id or "",
            generation=self._utterance_generation,
            discard_through_sample=discard_through,
        )
        self._utterance_observer_token = token
        pcm16 = self._combined_utterance_pcm()
        observer = self._completed_utterance_observer
        if pcm16 is not None and observer is not None and token.item_id:
            utterance = CompletedUserUtterance(
                item_id=token.item_id,
                sample_rate=self.SAMPLE_RATE,
                pcm16=pcm16,
            )
            self._utterance_observer_task = asyncio.create_task(
                self._run_completed_utterance_observer(utterance),
                name="completed-utterance-observer",
            )

    def _utterance_response_kwargs(self, result: Mapping[str, str]) -> dict[str, Any]:
        """Build one in-band function-call/output pair for the current response."""
        call_id = f"call_{uuid.uuid4().hex}"
        return {
            "response": {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": _UTTERANCE_CONTEXT_FUNCTION_NAME,
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(dict(result)),
                    },
                ]
            }
        }

    async def _wait_for_response_outcome(self, future: asyncio.Future[_ResponseOutcome]) -> _ResponseOutcome:
        """Wait for correlated response acceptance without muting indefinitely."""
        try:
            return await asyncio.wait_for(future, timeout=_RESPONSE_ACCEPTANCE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for observer response acceptance")
            return "failed"

    async def _complete_observed_utterance(
        self,
        token: _UtteranceToken,
        observer_task: asyncio.Task[CompletedUtteranceResult] | None,
    ) -> None:
        """Attach one bounded result and queue the matching explicit response."""
        try:
            result = dict(_UNAVAILABLE_UTTERANCE_RESULT)
            if observer_task is not None:
                try:
                    result = dict(await observer_task)
                except asyncio.CancelledError:
                    if not self._is_current_utterance(token):
                        raise
                    logger.warning("Completed-utterance observer was cancelled")
            if not self._is_current_utterance(token):
                return

            outcome_future = await self._safe_response_create(
                _utterance_token=token,
                **self._utterance_response_kwargs(result),
            )
            if outcome_future is None:
                return
            outcome = await self._wait_for_response_outcome(outcome_future)
            if outcome == "failed" and self._is_current_utterance(token):
                logger.warning("Observer context was rejected; continuing without identity context")
                fallback_future = await self._safe_response_create(_utterance_token=token)
                if fallback_future is not None:
                    await self._wait_for_response_outcome(fallback_future)
        except asyncio.CancelledError:
            raise
        finally:
            if self._is_current_utterance(token):
                self._utterance_observer_task = None
                self._utterance_observer_token = None
                if self._utterance_completion_task is asyncio.current_task():
                    self._utterance_completion_task = None

    def _release_completed_utterance_audio(self, token: _UtteranceToken) -> None:
        """Release PCM only after the matching response commits the current turn."""
        if not self._is_current_utterance(token):
            return
        self._discard_audio_through(token.discard_through_sample)
        self._utterance_spans = []
        self._utterance_span_pcm = []
        self._utterance_span_pcm_bytes = 0
        self._utterance_spans_valid = True
        self._utterance_discard_through_sample = None

    def _observe_completed_transcript(self, event: Any, transcript: str) -> None:
        """Run the observer for one backend-completed transcript revision."""
        item_id_value = getattr(event, "item_id", None)
        item_id = item_id_value if isinstance(item_id_value, str) and item_id_value else None
        if not transcript:
            if item_id is None or item_id == self._utterance_item_id:
                self._invalidate_utterance(preserve_spans=False)
            return
        if self._utterance_item_id is not None and item_id != self._utterance_item_id:
            logger.debug("Ignoring completed transcript for superseded item %r", item_id)
            return
        if self._utterance_item_id is None:
            self._invalidate_utterance(preserve_spans=False)
            self._utterance_item_id = item_id or "missing-item"
            self._utterance_spans_valid = False

        token = self._utterance_observer_token
        observer_task = None
        if token is None or not self._is_current_utterance(token):
            discard_through = self._utterance_spans[-1][1] if self._utterance_spans else self._audio_ring_end_sample
            token = _UtteranceToken(
                epoch=self._connection_epoch,
                item_id=self._utterance_item_id,
                generation=self._utterance_generation,
                discard_through_sample=discard_through,
            )
        else:
            observer_task = self._utterance_observer_task

        if self._utterance_completion_task is not None and not self._utterance_completion_task.done():
            self._utterance_completion_task.cancel()
        self._utterance_completion_task = asyncio.create_task(
            self._complete_observed_utterance(token, observer_task),
            name="completed-utterance-response",
        )

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        self.client = await self._build_realtime_client()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    self.client = await self._build_realtime_client()
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            observer_session_was_live = self.connection is not None and self._completed_utterance_observer is not None
            if self.connection is not None:
                if observer_session_was_live:
                    try:
                        await self.connection.close()
                    except Exception as error:
                        logger.warning("Observer session close failed; restart aborted: %s", error)
                        return
                    self.connection = None
                else:
                    try:
                        await self.connection.close()
                    except Exception:
                        pass
                    finally:
                        self.connection = None
            if observer_session_was_live:
                try:
                    await asyncio.wait_for(
                        self._observer_session_stopped.wait(),
                        timeout=_OBSERVER_SESSION_STOP_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Observer session teardown timed out; restart aborted")
                    return

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: realtime client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            self.client = await self._build_realtime_client()
            asyncio.create_task(self._run_realtime_session(), name="realtime-session-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    def _discard_pending_responses(self) -> None:
        """Discard response requests left behind by a closed realtime session."""
        while True:
            try:
                request = self._pending_responses.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._resolve_response_outcome(request, "stale")

    def _tag_response_request(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        """Return one response.create request with private correlation identifiers."""
        marker = uuid.uuid4().hex
        event_id = f"event_{uuid.uuid4().hex}"
        tagged_kwargs = dict(kwargs)
        response_value = tagged_kwargs.get("response")
        response = dict(response_value) if isinstance(response_value, Mapping) else {}
        metadata_value = response.get("metadata")
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        metadata[_RESPONSE_REQUEST_METADATA_KEY] = marker
        response["metadata"] = metadata
        tagged_kwargs["response"] = response
        tagged_kwargs["event_id"] = event_id
        return tagged_kwargs, marker, event_id

    def _response_event_matches_active_request(self, event: Any) -> bool:
        """Return whether a server response carries the active request marker."""
        if self._active_response_marker is None:
            return False
        response = getattr(event, "response", None)
        metadata = getattr(response, "metadata", None)
        return (
            isinstance(metadata, Mapping)
            and metadata.get(_RESPONSE_REQUEST_METADATA_KEY) == self._active_response_marker
        )

    def _observe_response_created(self, event: Any) -> bool:
        """Wake the sender only for response.created from its tagged request."""
        if not self._response_event_matches_active_request(event):
            return False
        response = getattr(event, "response", None)
        response_id = getattr(response, "id", None)
        self._active_response_id = response_id if isinstance(response_id, str) and response_id else None
        if self._active_response_id is not None and self._active_utterance_token is not None:
            self._response_tokens_by_id[self._active_response_id] = self._active_utterance_token
        self._last_response_created = True
        self._response_started_or_rejected_event.set()
        return True

    def _observe_response_done(self, event: Any) -> bool:
        """Complete sender bookkeeping only for its tagged response."""
        self._response_done_event.set()
        response = getattr(event, "response", None)
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str):
            token = self._response_tokens_by_id.get(response_id)
            if token is not None:
                self._release_completed_utterance_audio(token)
        if not self._response_event_matches_active_request(event):
            return False
        self._response_request_done_event.set()
        self._response_started_or_rejected_event.set()
        return True

    def _response_event_is_suppressed(self, event: Any) -> bool:
        """Return whether output belongs to a superseded observer response."""
        if self._suppress_active_response:
            return True
        response_id = getattr(event, "response_id", None)
        return isinstance(response_id, str) and response_id in self._suppressed_response_ids

    def _finish_response_suppression(self, event: Any) -> None:
        """Release per-response suppression only after its actual done event."""
        response = getattr(event, "response", None)
        response_id = getattr(response, "id", None)
        if not isinstance(response_id, str):
            return
        self._suppressed_response_ids.discard(response_id)
        self._response_tokens_by_id.pop(response_id, None)
        if response_id == self._active_response_id:
            self._active_response_id = None

    async def _handle_response_audio_delta(self, event: Any) -> bool:
        """Queue current response audio and reject superseded response audio."""
        if self._response_event_is_suppressed(event):
            logger.debug("Dropping audio from superseded observer response")
            return False
        decoded_pcm_bytes = base64.b64decode(event.delta)
        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
        self._mark_activity("assistant_audio_delta")
        if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
            self._turn_first_audio_at = time.perf_counter()
            delta_ms = (self._turn_first_audio_at - self._turn_user_done_at) * 1000
            logger.info("Turn latency: first audio delta %.0f ms after user transcript", delta_ms)
        await self.output_queue.put((self.SAMPLE_RATE, decoded_pcm))
        return True

    def _error_matches_active_request(self, error: Any) -> bool:
        """Correlate an error when the backend includes the causing client event ID."""
        if self._active_response_event_id is None:
            return False
        error_event_id = getattr(error, "event_id", None)
        # speech-to-speech 0.2.11 does not copy the client event ID into errors.
        # The serial sender and startup input gates leave only the active explicit
        # response.create as a possible source in that compatibility case.
        return error_event_id is None or error_event_id == self._active_response_event_id

    async def _handle_realtime_error(self, event: Any) -> None:
        """Handle one backend error without exposing observer-owned context."""
        err = getattr(event, "error", None)
        msg = getattr(err, "message", str(err) if err else "unknown error")
        code = getattr(err, "code", "") or getattr(err, "type", "")
        observer_enabled = self._completed_utterance_observer is not None
        active_response_rejection = code == "conversation_already_has_active_response"
        observer_request_rejection = (
            not active_response_rejection
            and code != "input_audio_buffer_commit_empty"
            and self._active_utterance_token is not None
            and not self._last_response_created
            and self._error_matches_active_request(err)
        )

        if active_response_rejection and self._error_matches_active_request(err):
            # The sender retries ordinary requests; observer context requests
            # fail over to their plain response after their single attempt.
            self._last_response_rejected = True
            self._response_started_or_rejected_event.set()
            logger.debug("response.create rejected; worker will retry or fall back")
        elif active_response_rejection:
            logger.debug("Ignoring response.create rejection for a different request")
        elif observer_request_rejection:
            self._response_started_or_rejected_event.set()
            logger.warning("Observer response request was rejected")
        elif observer_enabled:
            if self._error_matches_active_request(err):
                self._response_started_or_rejected_event.set()
            logger.error("Realtime request failed while completed-utterance observation was active")
        else:
            if self._error_matches_active_request(err):
                self._response_started_or_rejected_event.set()
            logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

        if code == "input_audio_buffer_commit_empty":
            self.deps.movement_manager.set_listening(False)

        if (
            observer_enabled
            and not observer_request_rejection
            and code
            not in (
                "input_audio_buffer_commit_empty",
                "conversation_already_has_active_response",
            )
        ):
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": "[error] Realtime request failed."})
            )
        elif not observer_enabled and code not in (
            "input_audio_buffer_commit_empty",
            "conversation_already_has_active_response",
        ):
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))

    async def _safe_response_create(
        self,
        *,
        _is_startup: bool = False,
        _utterance_token: _UtteranceToken | None = None,
        **kwargs: Any,
    ) -> asyncio.Future[_ResponseOutcome] | None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        outcome = asyncio.get_running_loop().create_future() if _utterance_token is not None else None
        await self._pending_responses.put(
            _QueuedResponse(
                kwargs=kwargs,
                is_startup=_is_startup,
                utterance_token=_utterance_token,
                outcome=outcome,
            )
        )
        return outcome

    async def say(self, text: str) -> None:
        """Inject ``text`` as a turn and have the model voice it now.

        Mirrors the startup-greeting path: create a user message item, then
        queue a ``response.create`` through the serial sender. Not verbatim TTS
        (speech-to-speech may rephrase). Raises if the session is closed.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self.connection:
            raise RuntimeError("say: no active session")
        if self._startup_input_blocked:
            raise RuntimeError("say: startup greeting pending")
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )
        self._mark_activity("say")
        await self._safe_response_create()

    async def _send_startup_greeting_prompt(self, tool_specs: list[ToolSpec]) -> None:
        """Prompt the model to open the conversation once the session is ready."""
        if self._startup_greeting_sent or not self.connection:
            return

        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            self._startup_greeting_sent = True
            return

        try:
            greeting_tool_name = get_session_greeting_tool_name()
        except RuntimeError as exc:
            self._startup_greeting_sent = True
            logger.error("Startup greeting disabled: %s", exc)
            return

        greeting_tool_spec: ToolSpec | None = None
        if greeting_tool_name is not None:
            greeting_tool_spec = next(
                (spec for spec in tool_specs if spec["name"] == greeting_tool_name),
                None,
            )
            greeting_tool = core_tools.ALL_TOOLS.get(greeting_tool_name)
            greeting_tool_parameters = greeting_tool_spec["parameters"] if greeting_tool_spec is not None else {}
            if (
                greeting_tool_spec is None
                or greeting_tool is None
                or not greeting_tool.needs_response
                or greeting_tool_parameters.get("type") != "object"
                or bool(greeting_tool_parameters.get("properties"))
                or bool(greeting_tool_parameters.get("required"))
            ):
                self._startup_greeting_sent = True
                logger.error(
                    "Startup greeting disabled: configured tool %r must be enabled, "
                    "require no arguments, and produce a response",
                    greeting_tool_name,
                )
                return

            self._startup_input_blocked = True
            self._startup_response_pending = True

        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": greeting_prompt,
                        },
                    ],
                },
            )
            self._startup_greeting_sent = True
            self._mark_activity("startup_greeting_prompt")
            if greeting_tool_name is None:
                await self._safe_response_create()
                logger.info("Queued startup greeting prompt")
                return

            call_id = f"call_{uuid.uuid4().hex}"
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call",
                    "call_id": call_id,
                    "name": greeting_tool_name,
                    "arguments": "{}",
                    "status": "in_progress",
                },
            )
            self._in_flight_tool_calls.add(call_id)
            self._internal_tool_calls.add(call_id)
            try:
                await self.tool_manager.start_tool(
                    call_id=call_id,
                    tool_call_routine=ToolCallRoutine(
                        tool_name=greeting_tool_name,
                        args_json_str="{}",
                        deps=self.deps,
                    ),
                    is_idle_tool_call=False,
                )
            except Exception:
                self._in_flight_tool_calls.discard(call_id)
                self._internal_tool_calls.discard(call_id)
                raise
            logger.info("Started configured startup greeting tool: %s", greeting_tool_name)
        except Exception as e:
            self._startup_input_blocked = False
            logger.warning("Failed to queue startup greeting prompt: %s", e)

    async def _response_sender_loop(self) -> None:
        """Dedicated worker that sends ``response.create()`` calls serially.

        This logic was designed to comply with the response.create() docstring specification for event ordering:
        https://github.com/openai/openai-python/blob/3e0c05b84a2056870abf3bd6a5e7849020209cc3/src/openai/resources/realtime/realtime.py#L649C1-L651C30

        For each queued request the worker:
        1. Waits until no response is active (_response_done_event).
        2. Sends response.create().
        3. Waits until the receiver observes response.created or a rejection.
        4. Waits for the response cycle to complete (response.done).
        5. If the server rejected with active_response, retries from step 1.
        """
        deferred_request: _QueuedResponse | None = None
        while self.connection:
            try:
                request = deferred_request or await self._pending_responses.get()
                deferred_request = None
            except asyncio.CancelledError:
                return

            # Parallel tool calls enqueue duplicate empty requests; coalesce to one.
            while request.utterance_token is None and not request.kwargs and not self._pending_responses.empty():
                try:
                    candidate = self._pending_responses.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if candidate.utterance_token is not None or candidate.kwargs:
                    deferred_request = candidate
                    break
                request.is_startup = request.is_startup or candidate.is_startup

            token = request.utterance_token
            if token is not None and not self._is_current_utterance(token):
                self._resolve_response_outcome(request, "stale")
                continue

            sent = False
            startup_response_created = False
            # A rejected observer context must fall back to one plain response,
            # not repeat the same rejected input. Ordinary requests and that
            # plain fallback retain the established active-response retries.
            max_retries = 1 if token is not None and request.kwargs else 5
            attempts = 0
            self._active_utterance_token = token
            self._suppress_active_response = False
            self._stale_response_cancel_sent = False
            while not sent and self.connection and attempts < max_retries:
                if token is not None and not self._is_current_utterance(token):
                    self._resolve_response_outcome(request, "stale")
                    break
                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for previous response to finish; forcing ahead")
                    self._response_done_event.set()

                if not self.connection:
                    break
                if token is not None and not self._is_current_utterance(token):
                    self._resolve_response_outcome(request, "stale")
                    break

                self._last_response_rejected = False
                self._last_response_created = False
                self._response_request_done_event.clear()
                self._response_started_or_rejected_event.clear()
                self._active_response_id = None
                send_kwargs, response_marker, response_event_id = self._tag_response_request(request.kwargs)
                self._active_response_marker = response_marker
                self._active_response_event_id = response_event_id
                try:
                    await self.connection.response.create(**send_kwargs)
                except Exception as e:
                    if token is None:
                        logger.debug("_response_sender_loop: send failed: %s", e)
                    else:
                        logger.debug("_response_sender_loop: observer response send failed")
                    self._response_done_event.set()
                    break

                try:
                    await asyncio.wait_for(
                        self._response_started_or_rejected_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.created or response rejection")

                if token is not None and not self._is_current_utterance(token):
                    self._resolve_response_outcome(request, "stale")
                    await self._cancel_stale_utterance_response()
                    break

                # Check if the receiver loop observed an asynchronous rejection.
                if self._last_response_rejected:
                    attempts += 1
                    if attempts >= max_retries:
                        logger.debug("response.create rejected %d times; giving up", attempts)
                        break
                    logger.debug("response.create was rejected; retrying (%d/%d)", attempts, max_retries)
                    await asyncio.sleep(_RESPONSE_REJECTION_RETRY_DELAY)
                    continue

                if not self._last_response_created:
                    logger.debug("response.create ended without response.created; giving up")
                    break

                self._resolve_response_outcome(request, "created")

                if request.is_startup:
                    startup_response_created = True
                    self._startup_input_blocked = False
                    self._startup_response_pending = False

                try:
                    await asyncio.wait_for(
                        self._response_request_done_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.done; assuming response completed")
                    self._response_request_done_event.set()
                    self._response_done_event.set()
                    break

                sent = True

            if request.is_startup and not startup_response_created:
                self._startup_input_blocked = False
            if request.outcome is not None and not request.outcome.done():
                outcome: _ResponseOutcome = (
                    "stale" if token is not None and not self._is_current_utterance(token) else "failed"
                )
                self._resolve_response_outcome(request, outcome)
            self._active_utterance_token = None
            self._suppress_active_response = False
            self._stale_response_cancel_sent = False
            self._active_response_marker = None
            self._active_response_event_id = None

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        is_internal_tool_call = completed_tool.id in self._internal_tool_calls
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed with error: %s",
                completed_tool.tool_name,
                completed_tool.id,
                completed_tool.error,
            )
            tool_result = {"error": completed_tool.error}
            tool_result_for_model = tool_result
        elif completed_tool.result is not None:
            tool_result = completed_tool.result
            tool_result_for_model = (
                self._sanitize_tool_result_for_model(completed_tool.tool_name, tool_result)
                if isinstance(tool_result, dict)
                else tool_result
            )
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                completed_tool.tool_name,
                completed_tool.id,
            )
            if not is_internal_tool_call:
                logger.debug("Tool '%s' model-visible result: %s", completed_tool.tool_name, tool_result_for_model)
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        # Connection may have closed while tool was running
        if not self.connection:
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                completed_tool.tool_name,
                completed_tool.id,
            )
            if is_internal_tool_call:
                self._startup_input_blocked = False
            return

        try:
            send_result_to_model = not completed_tool.is_idle_tool_call
            if send_result_to_model:
                self._mark_activity("tool_result_ready")
            model_result_submitted = False
            if send_result_to_model and isinstance(completed_tool.id, str):
                if not await self._wait_for_response_done_before_tool_result():
                    send_result_to_model = False
                if not send_result_to_model:
                    logger.warning(
                        "Dropping realtime model result for tool '%s' (id=%s) because response.done was not observed",
                        completed_tool.tool_name,
                        completed_tool.id,
                    )
                elif not self.connection:
                    logger.warning(
                        "Connection closed before sending tool '%s' (id=%s) result back",
                        completed_tool.tool_name,
                        completed_tool.id,
                    )
                    if is_internal_tool_call:
                        self._startup_input_blocked = False
                    return
                else:
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": completed_tool.id,
                            "output": json.dumps(tool_result_for_model),
                        },
                    )
                    model_result_submitted = True

            if is_internal_tool_call and not model_result_submitted:
                self._startup_input_blocked = False

            if not is_internal_tool_call:
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(tool_result_for_model),
                        },
                    ),
                )

            if model_result_submitted and completed_tool.tool_name == "camera" and "b64_im" in tool_result:
                # use raw base64, don't json.dumps (which adds quotes)
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                image_width = tool_result.get("image_width")
                image_height = tool_result.get("image_height")
                jpeg_bytes_value = tool_result.get("jpeg_bytes")
                jpeg_bytes = jpeg_bytes_value if isinstance(jpeg_bytes_value, int) else (len(b64_im) * 3) // 4
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_im}",
                            },
                        ],
                    },
                )
                if isinstance(image_width, int) and isinstance(image_height, int):
                    logger.info(
                        "Added camera image to conversation frame=%sx%s jpeg_bytes=%s",
                        image_width,
                        image_height,
                        jpeg_bytes,
                    )
                else:
                    logger.info(
                        "Added camera image to conversation jpeg_bytes=%s",
                        jpeg_bytes,
                    )

            if isinstance(completed_tool.id, str):
                self._in_flight_tool_calls.discard(completed_tool.id)
                self._internal_tool_calls.discard(completed_tool.id)

            tool = core_tools.ALL_TOOLS.get(completed_tool.tool_name)
            # Always surface errors, skip the spoken follow-up for tools that opt out.
            if model_result_submitted and (completed_tool.error is not None or tool is None or tool.needs_response):
                self._tool_batch_needs_response = True

            # Parallel tool calls in one turn: respond once every result is in, not per tool.
            if self._tool_batch_needs_response and not self._in_flight_tool_calls:
                self._tool_batch_needs_response = False
                await self._safe_response_create(_is_startup=is_internal_tool_call)

        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._response_done_event.set()
            self._response_request_done_event.set()
            if is_internal_tool_call:
                self._startup_input_blocked = False
        except Exception:
            if is_internal_tool_call:
                self._startup_input_blocked = False
            raise

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        observer_session_established = False
        tool_specs = get_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
        connect_kwargs: dict[str, Any] = {}
        if self._realtime_connect_query:
            connect_kwargs["extra_query"] = self._realtime_connect_query
        async with self._realtime_connection(connect_kwargs) as conn:
            self._completed_utterance_observer_locked = True
            try:
                session_config = self._get_session_config(tool_specs)
                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
            except asyncio.CancelledError:
                self._completed_utterance_observer_locked = False
                raise
            except Exception:
                self._completed_utterance_observer_locked = False
                logger.exception("Realtime session.update failed; aborting startup")
                raise

            logger.info("Realtime session updated successfully")
            observer_session_established = self._completed_utterance_observer is not None

            self._discard_pending_responses()
            self._response_done_event.set()
            self._response_request_done_event.set()
            self._response_started_or_rejected_event.clear()
            self._last_response_rejected = False
            self._last_response_created = False
            self._active_response_event_id = None
            self._active_response_marker = None
            self._active_response_id = None
            if self._completed_utterance_observer is not None:
                self._reset_utterance_state()

            # Reset the partial-transcript accumulator for each new session
            self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

            # Manage events received from the realtime server.
            self.connection = conn
            if self._completed_utterance_observer is not None:
                self._observer_session_stopped.clear()
            try:
                self._connected_event.set()
            except Exception:
                pass

            response_sender_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start the response sender worker
                response_sender_task = asyncio.create_task(self._response_sender_loop(), name="response-sender")
                await self._send_startup_greeting_prompt(tool_specs)

                async for event in self.connection:
                    logger.debug("Realtime event: %s", event.type)
                    if event.type == "input_audio_buffer.speech_started":
                        self._mark_activity("user_speech_started")
                        self._turn_user_done_at = None
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        if self._clear_queue:
                            self._clear_queue()
                        if self._completed_utterance_observer is not None:
                            await self._observe_speech_started(event)
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self._mark_activity("user_speech_stopped")
                        if self._completed_utterance_observer is not None:
                            await self._observe_speech_stopped(event)
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type == "response.output_audio.done":
                        if self._response_event_is_suppressed(event):
                            logger.debug("Dropping audio completion from superseded observer response")
                            continue
                        self.deps.movement_manager.set_speaking(False)
                        logger.debug("response completed")

                    if event.type == "response.output_text.delta":
                        if self._response_event_is_suppressed(event):
                            continue
                        logger.debug("response text delta")

                    if event.type == "response.output_text.done":
                        if self._response_event_is_suppressed(event):
                            logger.debug("Dropping text from superseded observer response")
                            continue
                        logger.debug("response text done: %s", event.text)

                    if event.type == "response.created":
                        matched_request = self._observe_response_created(event)
                        if matched_request:
                            await self._cancel_stale_utterance_response()
                        if not self._suppress_active_response:
                            self._mark_activity("response_created")
                            self.deps.movement_manager.set_speaking(True)
                            self._response_done_event.clear()
                            if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                                self._turn_response_created_at = time.perf_counter()
                                delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                                logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)
                            logger.debug("Response created (active)")
                        else:
                            self._response_done_event.clear()
                            logger.debug("Suppressing superseded observer response")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        # Resume tracking for responses that emit no audio (text-only / tool-only).
                        self.deps.movement_manager.set_speaking(False)
                        self._observe_response_done(event)
                        self._finish_response_suppression(event)
                        logger.debug("Response done")

                    if event.type == "conversation.item.input_audio_transcription.delta":
                        self._mark_activity("user_transcription_delta")
                        logger.debug(f"User partial transcript: {event.delta}")

                        item_id = event.item_id
                        delta = event.delta or ""

                        input_transcript = self.input_transcript_chunks_by_item
                        self._record_partial_transcript_delta(input_transcript, item_id, delta)

                        current_partial = "".join(input_transcript.deltas)
                        sequence_counter = len(input_transcript.deltas) - 1

                        await self._cancel_partial_transcript_task()

                        # Start new debounce timer with the last delta
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(current_partial, item_id, sequence_counter)
                        )

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        self._mark_activity("user_transcription_completed")
                        raw_transcript = event.transcript or ""
                        transcript = raw_transcript.strip()
                        logger.debug("User transcript: %s", raw_transcript)
                        self.deps.movement_manager.set_listening(False)

                        await self._cancel_partial_transcript_task()

                        if not transcript:
                            logger.debug("Ignoring empty user transcript")
                            if self._completed_utterance_observer is not None:
                                self._observe_completed_transcript(event, transcript)
                            continue

                        self._turn_user_done_at = time.perf_counter()
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        self._in_flight_tool_calls.clear()
                        self._tool_batch_needs_response = False

                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                        self._emit_transcript("user", transcript, True)
                        if self._completed_utterance_observer is not None:
                            self._observe_completed_transcript(event, transcript)

                    # Handle assistant transcription
                    if event.type == "response.output_audio_transcript.done":
                        if self._response_event_is_suppressed(event):
                            logger.debug("Dropping transcript from superseded observer response")
                            continue
                        self._mark_activity("assistant_transcript_done")
                        logger.debug(f"Assistant transcript: {event.transcript}")
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event.transcript})
                        )
                        self._emit_transcript("assistant", event.transcript or "", True)

                    # Handle audio delta
                    if event.type == "response.output_audio.delta":
                        if not await self._handle_response_audio_delta(event):
                            continue
                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        if self._response_event_is_suppressed(event):
                            logger.debug("Dropping tool call from superseded observer response")
                            continue
                        self._mark_activity("tool_call_received")
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, args=%s",
                            tool_name,
                            call_id,
                            args_json_str,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name,
                                type(tool_name).__name__,
                                args_json_str,
                                type(args_json_str).__name__,
                                call_id,
                            )
                            continue

                        self._in_flight_tool_calls.add(call_id)
                        background_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=args_json_str,
                                deps=self.deps,
                            ),
                            is_idle_tool_call=False,
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. The tool is now running. Tool ID: {background_tool.tool_id}",
                                },
                            ),
                        )
                        logger.info(
                            "Started background tool: %s (id=%s, call_id=%s)",
                            tool_name,
                            background_tool.tool_id,
                            call_id,
                        )

                    # server error
                    if event.type == "error":
                        await self._handle_realtime_error(event)
            finally:
                try:
                    # Stop the response sender worker.
                    if response_sender_task is not None:
                        response_sender_task.cancel()
                        try:
                            await response_sender_task
                        except asyncio.CancelledError:
                            pass

                    # Stop background tool manager tasks (listener + cleanup) in all paths.
                    await self.tool_manager.shutdown()
                    self._in_flight_tool_calls.clear()
                    self._internal_tool_calls.clear()
                    self._tool_batch_needs_response = False
                    self._startup_input_blocked = False
                    if self._startup_response_pending:
                        self._startup_greeting_sent = False
                        self._startup_response_pending = False
                    self._discard_pending_responses()
                    self._response_done_event.set()
                    self._response_request_done_event.set()
                    self._active_response_event_id = None
                    self._active_response_marker = None
                    self._active_response_id = None
                finally:
                    try:
                        if observer_session_established:
                            self._notify_completed_utterance_observer_connection_reset()
                        if self._completed_utterance_observer is not None:
                            self._reset_utterance_state()
                    finally:
                        self._completed_utterance_observer_locked = False

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the realtime server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for the realtime API.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection or self._startup_input_blocked:
            return

        sample_rate, audio_frame = frame
        if audio_frame.size == 0:
            return

        # Reshape if needed
        if audio_frame.ndim == 2:
            # channels-last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to the realtime input buffer (guard against races during reconnect).
        try:
            audio_bytes = audio_frame.tobytes()
            audio_message = base64.b64encode(audio_bytes).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return
        if self._completed_utterance_observer is not None:
            self._retain_sent_audio(sample_rate, audio_bytes, audio_frame.size)

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._startup_input_blocked = False
        # Unblock the response sender worker so it can exit
        self._response_done_event.set()
        self._response_request_done_event.set()

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()

        await self._cancel_partial_transcript_task()

        if self._completed_utterance_observer is not None:
            self._reset_utterance_state()

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_available_voices(self) -> list[str]:
        """Return the available Hugging Face voices."""
        return get_available_voices()

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the Hugging Face OpenAI-compatible realtime client."""
        configured_bearer_token = (config.HF_TOKEN or "").strip()
        connection_selection = get_hf_connection_selection()
        direct_realtime_url = get_hf_direct_ws_url()
        if connection_selection.mode == HF_LOCAL_CONNECTION_MODE:
            if not direct_realtime_url:
                raise RuntimeError("HF_REALTIME_WS_URL must be set when HF_REALTIME_CONNECTION_MODE=local")
            client, connect_query = _build_openai_compatible_client_from_realtime_url(
                direct_realtime_url,
                configured_bearer_token,
            )
            self._realtime_connect_query = connect_query
            logger.info("Using direct Hugging Face realtime endpoint %s", direct_realtime_url)
            return client

        session_url = connection_selection.session_url
        if not session_url:
            raise RuntimeError("Built-in Hugging Face session proxy URL is unavailable")
        if direct_realtime_url:
            logger.info("HF_REALTIME_CONNECTION_MODE=deployed; ignoring HF_REALTIME_WS_URL.")

        bearer_token = configured_bearer_token or (get_token() or "").strip()
        allocator_headers = {"User-Agent": "reachy-mini-conversation-app"}
        if bearer_token:
            allocator_headers["X-Reachy-Mini-Authorization"] = f"Bearer {bearer_token}"
        allocator_payload: dict[str, str] = {}
        try:
            hardware_id = self.deps.reachy_mini.client.get_status(wait=False).hardware_id
        except (AssertionError, ConnectionError, TimeoutError) as e:
            logger.warning("Daemon status unavailable for realtime session allocation: %s", e)
        else:
            if hardware_id:
                allocator_payload["hardware_id"] = hardware_id

        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(session_url, headers=allocator_headers, json=allocator_payload)
            response.raise_for_status()
            payload = response.json()

        connect_url = payload.get("connect_url")
        if not isinstance(connect_url, str) or not connect_url:
            raise RuntimeError(f"Session allocator response did not contain a valid connect_url: {payload!r}")

        parsed_connect_url = parse_hf_realtime_url(connect_url)
        if not parsed_connect_url.has_realtime_path:
            raise ValueError(f"Expected realtime connect URL ending with /realtime, got: {connect_url}")

        logger.info("Allocated realtime session %s", payload.get("session_id") or "<unknown>")
        client, connect_query = _build_openai_compatible_client_from_realtime_url(
            connect_url,
            bearer_token,
        )
        self._realtime_connect_query = connect_query
        return client
