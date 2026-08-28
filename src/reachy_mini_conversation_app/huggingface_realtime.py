import json
import time
import uuid
import base64
import random
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Final, Tuple, Optional
from dataclasses import dataclass

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
    set_custom_profile,
    get_available_voices,
    get_hf_direct_ws_url,
    parse_hf_realtime_url,
    get_hf_connection_selection,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_instructions,
    get_session_greeting_prompt,
)
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.tools.core_tools import (
    ToolSpec,
    ToolDependencies,
    RealtimeToolResult,
    get_tool_specs,
)
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
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
_RESPONSE_CREATE_ID_METADATA_KEY: Final[str] = "reachy_mini_response_create_id"
_PRIVATE_TOOL_DELETE_TIMEOUT: Final[float] = 5.0


@dataclass(frozen=True)
class _PendingPrivateToolCall:
    """A private tool call held until history deletion is acknowledged."""

    event_id: str
    item_id: str
    tool_name: str
    arguments: str
    call_id: str


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

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._response_sender_task: asyncio.Task[None] | None = None
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._response_started_or_rejected_event: asyncio.Event = asyncio.Event()
        self._last_response_rejected: bool = False
        self._pending_response_create_id: str | None = None
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None
        self._startup_greeting_sent = False
        self._in_flight_tool_calls: set[str] = set()
        self._tool_batch_needs_response = False
        self._isolated_response_ids: dict[str, None] = {}
        self._redacted_tool_calls: set[str] = set()
        self._pending_private_tool_calls: dict[str, _PendingPrivateToolCall] = {}
        self._private_tool_delete_tasks: dict[str, asyncio.Task[None]] = {}
        self._private_tool_delete_terminal = False
        # A reconnect may be requested before the detached receive iterator has
        # observed transport closure. Keep teardown and replacement activation
        # strictly ordered because the tool manager, response queue, and privacy
        # deletion fence are handler-scoped resources.
        self._realtime_session_lock = asyncio.Lock()

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
                    turn_detection=ServerVad(type="server_vad", interrupt_response=True),
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
        return self._response_done_event.is_set()

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
        """Apply a personality to the active or next realtime connection."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            instructions = get_session_instructions(self.instance_path)
            voice = self.get_current_voice()
            core_tools.initialize_tools(force=True)
        except Exception as exc:
            set_custom_profile(previous_profile)
            logger.error("Failed to resolve personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

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
                logger.info("Applied personality via live update: %s", profile or "default")
            except Exception as exc:
                logger.warning("Live update failed; will restart session: %s", exc)

            try:
                await self._restart_session()
                return "Applied personality and restarted realtime session."
            except Exception as exc:
                logger.warning("Failed to restart session after apply: %s", exc)
                return "Applied personality. Will take effect on next connection."

        logger.info(
            "Applied personality recorded: %s (no live connection; will apply on next session)",
            profile or "default",
        )
        return "Applied personality. Will take effect on next connection."

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
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

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

    async def _safe_response_create(self, **kwargs: Any) -> None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        await self._pending_responses.put(kwargs)

    def _drain_pending_responses(self) -> None:
        """Release every queued response payload, including isolated raw input."""
        while not self._pending_responses.empty():
            try:
                request = self._pending_responses.get_nowait()
            except asyncio.QueueEmpty:
                break
            request.clear()

    async def _stop_response_sender(self, task: asyncio.Task[None] | None = None) -> None:
        """Stop a response sender and release its current request."""
        sender = task or self._response_sender_task
        if sender is None:
            return
        if self._response_sender_task is sender:
            self._response_sender_task = None
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)

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
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )
        self._mark_activity("say")
        await self._safe_response_create()

    async def _send_startup_greeting_prompt(self) -> None:
        """Prompt the model to open the conversation once the session is ready."""
        if self._startup_greeting_sent or not self.connection:
            return

        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            self._startup_greeting_sent = True
            return

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
            await self._safe_response_create()
            logger.info("Queued startup greeting prompt")
        except Exception as e:
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
        while self.connection:
            try:
                kwargs = await self._pending_responses.get()
            except asyncio.CancelledError:
                return
            response_payload: dict[str, Any] | None = None
            try:
                sent = False
                max_retries = 5
                attempts = 0
                while not sent and self.connection and attempts < max_retries:
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

                    create_id = f"response_create_{uuid.uuid4().hex}"
                    response = kwargs.setdefault("response", {})
                    if not isinstance(response, dict):
                        logger.error("response.create payload must be a dictionary")
                        break
                    response_payload = response
                    metadata = response.setdefault("metadata", {})
                    if not isinstance(metadata, dict):
                        logger.error("response.create metadata must be a dictionary")
                        break
                    metadata[_RESPONSE_CREATE_ID_METADATA_KEY] = create_id
                    kwargs["event_id"] = create_id
                    self._last_response_rejected = False
                    self._pending_response_create_id = create_id
                    self._response_started_or_rejected_event.clear()
                    try:
                        await self.connection.response.create(**kwargs)
                    except Exception as e:
                        # An isolated response payload can be reflected in a
                        # provider exception. Keep diagnostics content-free.
                        logger.debug(
                            "_response_sender_loop: response.create send failed (%s)",
                            type(e).__name__,
                        )
                        self._response_done_event.set()
                        if self._pending_response_create_id == create_id:
                            self._pending_response_create_id = None
                        break

                    try:
                        await asyncio.wait_for(
                            self._response_started_or_rejected_event.wait(),
                            timeout=_RESPONSE_DONE_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.error("Timed out waiting for the correlated response.create outcome; closing session")
                        if self._pending_response_create_id == create_id:
                            self._pending_response_create_id = None
                        connection = self.connection
                        self.connection = None
                        if connection is not None:
                            try:
                                await connection.close()
                            except Exception as e:
                                logger.debug("Failed to close uncorrelated realtime session: %s", e)
                        break

                    if self._pending_response_create_id == create_id:
                        self._pending_response_create_id = None

                    if self._last_response_rejected:
                        attempts += 1
                        if attempts >= max_retries:
                            logger.debug("response.create rejected %d times; giving up", attempts)
                            break
                        logger.debug("response.create was rejected; retrying (%d/%d)", attempts, max_retries)
                        await asyncio.sleep(_RESPONSE_REJECTION_RETRY_DELAY)
                        continue

                    # The server accepted all request fields; release raw input
                    # before waiting for its potentially long response cycle.
                    response_payload.clear()
                    kwargs.clear()
                    try:
                        await asyncio.wait_for(
                            self._response_done_event.wait(),
                            timeout=_RESPONSE_DONE_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        logger.debug("Timed out waiting for response.done; assuming response completed")
                        self._response_done_event.set()
                        break

                    sent = True
            finally:
                self._pending_response_create_id = None
                if response_payload is not None:
                    response_payload.clear()
                kwargs.clear()

    def _accept_correlated_response_create(self, event: object) -> bool:
        """Wake the sender only for the response.create request it issued."""
        response = getattr(event, "response", None)
        metadata = getattr(response, "metadata", None)
        create_id = metadata.get(_RESPONSE_CREATE_ID_METADATA_KEY) if isinstance(metadata, dict) else None
        if create_id != self._pending_response_create_id:
            return False
        self._last_response_rejected = False
        self._response_started_or_rejected_event.set()
        return True

    def _reject_correlated_response_create(self, event_id: object, *, active_response: bool) -> bool:
        """Wake the sender for its rejection and preserve any active response fence."""
        if event_id != self._pending_response_create_id:
            return False
        if active_response:
            self._response_done_event.clear()
        self._last_response_rejected = True
        self._response_started_or_rejected_event.set()
        return True

    async def _start_realtime_tool_call(
        self,
        *,
        tool_name: str,
        arguments: str,
        call_id: str,
        exposed_arguments: object,
    ) -> None:
        """Start a validated call after any required privacy fence."""
        self._in_flight_tool_calls.add(call_id)
        background_tool = await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(
                tool_name=tool_name,
                args_json_str=arguments,
                deps=self.deps,
            ),
            is_idle_tool_call=False,
        )
        await self.output_queue.put(
            AdditionalOutputs(
                {
                    "role": "assistant",
                    "content": (
                        f"🛠️ Used tool {tool_name} with args {exposed_arguments}. "
                        f"The tool is now running. Tool ID: {background_tool.tool_id}"
                    ),
                },
            ),
        )
        logger.info(
            "Started background tool: %s (id=%s, call_id=%s)",
            tool_name,
            background_tool.tool_id,
            call_id,
        )

    async def _acknowledge_private_tool_delete(self, item_id: str) -> None:
        """Start a private tool only while its deletion fence is usable."""
        pending = self._pending_private_tool_calls.get(item_id)
        if pending is None:
            return
        if self._private_tool_delete_terminal or self.connection is None:
            logger.warning("Ignoring late private tool deletion acknowledgement for item=%s", item_id)
            return
        self._pending_private_tool_calls.pop(item_id, None)
        self._cancel_private_tool_delete_timeout(pending.item_id)
        self._redacted_tool_calls.add(pending.call_id)
        await self._start_realtime_tool_call(
            tool_name=pending.tool_name,
            arguments=pending.arguments,
            call_id=pending.call_id,
            exposed_arguments="[redacted]",
        )

    async def _expire_private_tool_delete(self, item_id: str, event_id: str) -> None:
        """Close a session that never proves deletion of private arguments."""
        try:
            await asyncio.sleep(_PRIVATE_TOOL_DELETE_TIMEOUT)
            pending = self._pending_private_tool_calls.get(item_id)
            if pending is None or pending.event_id != event_id:
                return
            logger.error("Private realtime tool history deletion was not acknowledged; closing session")
            # Mark the connection unusable before a best-effort transport close.
            # Keep the pending call as a privacy guard until session cleanup, so
            # a close failure can never admit another speech turn or tool call.
            self._private_tool_delete_terminal = True
            connection = self.connection
            self.connection = None
            if connection is not None:
                try:
                    await connection.close()
                except Exception as e:
                    logger.debug("Failed to close realtime session after private deletion timeout: %s", e)
        finally:
            self._private_tool_delete_tasks.pop(item_id, None)

    def _cancel_private_tool_delete_timeout(self, item_id: str) -> None:
        task = self._private_tool_delete_tasks.pop(item_id, None)
        if task is not None:
            task.cancel()

    async def _clear_private_tool_deletes(self, *, reset_terminal: bool = False) -> None:
        """Cancel deletion timers and release raw arguments without reopening a live loop."""
        tasks = list(self._private_tool_delete_tasks.values())
        self._private_tool_delete_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_private_tool_calls.clear()
        if reset_terminal:
            self._private_tool_delete_terminal = False

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        isolated_result: RealtimeToolResult | None = None
        arguments_redacted = completed_tool.id in self._redacted_tool_calls
        tool = core_tools.get_tools().get(completed_tool.tool_name)
        if completed_tool.error is not None:
            error = "Tool failed" if arguments_redacted else completed_tool.error
            logger.error(
                "Tool '%s' (id=%s) failed with error: %s",
                completed_tool.tool_name,
                completed_tool.id,
                error,
            )
            tool_result = {"error": error}
            tool_result_for_model = tool_result
        elif isinstance(completed_tool.result, RealtimeToolResult):
            isolated_result = completed_tool.result
            tool_result = {"status": isolated_result.model_status}
            tool_result_for_model = tool_result
            logger.info(
                "Tool '%s' (id=%s) produced an isolated result.",
                completed_tool.tool_name,
                completed_tool.id,
            )
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
            logger.debug("Tool '%s' model-visible result: %s", completed_tool.tool_name, tool_result_for_model)
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        # Connection may have closed while tool was running
        if not self.connection:
            if isinstance(completed_tool.id, str):
                self._in_flight_tool_calls.discard(completed_tool.id)
            self._redacted_tool_calls.discard(completed_tool.id)
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                completed_tool.tool_name,
                completed_tool.id,
            )
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
                    return
                else:
                    if arguments_redacted:
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call",
                                "call_id": completed_tool.id,
                                "name": completed_tool.tool_name,
                                "arguments": "{}",
                            },
                        )
                    await self.connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": completed_tool.id,
                            "output": json.dumps(tool_result_for_model),
                        },
                    )
                    model_result_submitted = True

            if isolated_result is None:
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

            if model_result_submitted and isolated_result is not None:
                response: dict[str, Any] = {
                    "conversation": "none",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": isolated_result.isolated_input,
                                }
                            ],
                        }
                    ],
                    "instructions": isolated_result.isolated_instructions,
                    "tool_choice": "none",
                }
                await self._safe_response_create(response=response)
            # Always surface errors, skip the spoken follow-up for tools that opt out.
            elif model_result_submitted and (completed_tool.error is not None or tool is None or tool.needs_response):
                self._tool_batch_needs_response = True

            # Parallel tool calls in one turn: respond once every result is in, not per tool.
            if self._tool_batch_needs_response and not self._in_flight_tool_calls:
                self._tool_batch_needs_response = False
                await self._safe_response_create()

        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self._tool_batch_needs_response = False
            self.connection = None
            self._response_done_event.set()
        except Exception:
            # The model and client histories no longer agree if a tool result
            # fails mid-batch. End this session instead of prompting from a
            # partially delivered batch.
            self._tool_batch_needs_response = False
            self._private_tool_delete_terminal = True
            connection = self.connection
            self.connection = None
            self._response_done_event.set()
            await self._stop_response_sender()
            if connection is not None:
                try:
                    await connection.close()
                except Exception as error:
                    logger.debug("Failed to close realtime session after tool result failure: %s", error)
            raise
        finally:
            if isinstance(completed_tool.id, str):
                # Callback failures are isolated by BackgroundToolManager; do
                # not let one failed delivery strand the rest of its batch.
                self._in_flight_tool_calls.discard(completed_tool.id)
            self._redacted_tool_calls.discard(completed_tool.id)

    async def _run_realtime_session(self) -> None:
        """Run one session after every older session has fully cleaned up."""
        async with self._realtime_session_lock:
            await self._run_realtime_session_serialized()

    async def _run_realtime_session_serialized(self) -> None:
        """Establish and manage a single realtime session."""
        tool_specs = get_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
        connect_kwargs: dict[str, Any] = {}
        if self._realtime_connect_query:
            connect_kwargs["extra_query"] = self._realtime_connect_query
        async with self.client.realtime.connect(**connect_kwargs) as conn:
            try:
                session_config = self._get_session_config(tool_specs)
                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                raise

            logger.info("Realtime session updated successfully")

            # Reset the partial-transcript accumulator for each new session
            self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

            # Manage events received from the realtime server.
            self.connection = conn
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
                self._response_sender_task = response_sender_task
                await self._send_startup_greeting_prompt()

                async for event in conn:
                    if self.connection is not conn:
                        logger.debug("Dropping event from detached realtime session")
                        break
                    logger.debug("Realtime event: %s", event.type)
                    if event.type == "conversation.item.deleted":
                        item_id = getattr(event, "item_id", None)
                        if isinstance(item_id, str):
                            await self._acknowledge_private_tool_delete(item_id)
                        continue

                    if event.type == "input_audio_buffer.speech_started":
                        if self._private_tool_delete_terminal or self._pending_private_tool_calls:
                            raise RuntimeError(
                                "private realtime tool history deletion was not acknowledged before the next turn"
                            )
                        self._mark_activity("user_speech_started")
                        self._turn_user_done_at = None
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        if self._clear_queue:
                            self._clear_queue()
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self._mark_activity("user_speech_stopped")
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type == "response.output_audio.done":
                        self.deps.movement_manager.set_speaking(False)
                        logger.debug("response completed")

                    if event.type == "response.output_text.delta":
                        logger.debug("response text delta")

                    if event.type == "response.output_text.done":
                        response_id = getattr(event, "response_id", None)
                        if response_id not in self._isolated_response_ids:
                            logger.debug("response text done: %s", event.text)

                    if event.type == "response.created":
                        self._mark_activity("response_created")
                        self.deps.movement_manager.set_speaking(True)
                        self._response_done_event.clear()
                        response = getattr(event, "response", None)
                        response_id = getattr(response, "id", None)
                        self._accept_correlated_response_create(event)
                        if isinstance(response_id, str) and getattr(response, "conversation_id", None) is None:
                            self._isolated_response_ids[response_id] = None
                            while len(self._isolated_response_ids) > 128:
                                self._isolated_response_ids.pop(next(iter(self._isolated_response_ids)))
                        if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                            self._turn_response_created_at = time.perf_counter()
                            delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)
                        logger.debug("Response created (active)")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        # Resume tracking for responses that emit no audio (text-only / tool-only).
                        self.deps.movement_manager.set_speaking(False)
                        self._response_done_event.set()
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
                            continue

                        self._turn_user_done_at = time.perf_counter()
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        self._in_flight_tool_calls.clear()
                        self._tool_batch_needs_response = False

                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                        self._emit_transcript("user", transcript, True)

                    # Handle assistant transcription
                    if event.type == "response.output_audio_transcript.done":
                        response_id = getattr(event, "response_id", None)
                        transcript = event.transcript or ""
                        if response_id in self._isolated_response_ids:
                            logger.debug("Suppressed isolated assistant transcript")
                            continue
                        self._mark_activity("assistant_transcript_done")
                        logger.debug(f"Assistant transcript: {event.transcript}")
                        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": transcript}))
                        self._emit_transcript("assistant", transcript, True)

                    # Handle audio delta
                    if event.type == "response.output_audio.delta":
                        decoded_pcm_bytes = base64.b64decode(event.delta)
                        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
                        self._mark_activity("assistant_audio_delta")
                        if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
                            self._turn_first_audio_at = time.perf_counter()
                            delta_ms = (self._turn_first_audio_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: first audio delta %.0f ms after user transcript", delta_ms)
                        await self.output_queue.put(
                            (
                                self.SAMPLE_RATE,
                                decoded_pcm,
                            ),
                        )
                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        if self._private_tool_delete_terminal or self._pending_private_tool_calls:
                            raise RuntimeError(
                                "private realtime tool history deletion was not acknowledged before another tool call"
                            )
                        self._mark_activity("tool_call_received")
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        tool = core_tools.get_tools().get(tool_name) if isinstance(tool_name, str) else None
                        redact_arguments = tool is not None and not tool.expose_arguments
                        exposed_arguments = "[redacted]" if redact_arguments else args_json_str
                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, args=%s",
                            tool_name,
                            call_id,
                            exposed_arguments,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name,
                                type(tool_name).__name__,
                                exposed_arguments,
                                type(args_json_str).__name__,
                                call_id,
                            )
                            continue

                        if redact_arguments:
                            tool_item_id = getattr(event, "item_id", None)
                            if not isinstance(tool_item_id, str) or not tool_item_id:
                                raise RuntimeError("private realtime tool call is missing its history item id")
                            delete_event_id = f"event_{uuid.uuid4().hex}"
                            pending = _PendingPrivateToolCall(
                                event_id=delete_event_id,
                                item_id=tool_item_id,
                                tool_name=tool_name,
                                arguments=args_json_str,
                                call_id=call_id,
                            )
                            if tool_item_id in self._pending_private_tool_calls:
                                raise RuntimeError("duplicate private realtime tool history item id")
                            self._pending_private_tool_calls[tool_item_id] = pending
                            try:
                                await self.connection.conversation.item.delete(
                                    item_id=tool_item_id,
                                    event_id=delete_event_id,
                                )
                            except Exception:
                                self._pending_private_tool_calls.pop(tool_item_id, None)
                                raise RuntimeError("private realtime tool history deletion send failed") from None
                            self._private_tool_delete_tasks[tool_item_id] = asyncio.create_task(
                                self._expire_private_tool_delete(tool_item_id, delete_event_id),
                                name="private-tool-history-delete-timeout",
                            )
                            continue

                        await self._start_realtime_tool_call(
                            tool_name=tool_name,
                            arguments=args_json_str,
                            call_id=call_id,
                            exposed_arguments=exposed_arguments,
                        )

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        error_event_id = getattr(err, "event_id", None)
                        failed_private_item = next(
                            (
                                item_id
                                for item_id, pending in self._pending_private_tool_calls.items()
                                if pending.event_id == error_event_id
                            ),
                            None,
                        )
                        if failed_private_item is not None:
                            self._pending_private_tool_calls.pop(failed_private_item, None)
                            self._cancel_private_tool_delete_timeout(failed_private_item)
                            raise RuntimeError("private realtime tool history deletion was rejected")
                        code = getattr(err, "code", "") or getattr(err, "type", "")
                        correlated_rejection = self._reject_correlated_response_create(
                            error_event_id,
                            active_response=code == "conversation_already_has_active_response",
                        )

                        if code == "conversation_already_has_active_response":
                            # response.create was rejected.  The sender worker
                            # is waiting on _response_done_event; when the active
                            # response finishes it will wake up and see this flag.
                            if correlated_rejection:
                                logger.debug(
                                    "Correlated response.create rejected; worker will retry after active response finishes"
                                )
                        else:
                            # Provider error text can reflect the isolated input
                            # that produced it. Never copy it into logs or UI.
                            logger.error("Realtime request failed")

                        if code == "input_audio_buffer_commit_empty":
                            self.deps.movement_manager.set_listening(False)

                        # Only show user-facing errors, not internal state errors.
                        if code not in (
                            "input_audio_buffer_commit_empty",
                            "conversation_already_has_active_response",
                        ):
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {
                                        "role": "assistant",
                                        "content": "[error] Realtime request failed",
                                    }
                                )
                            )
            finally:
                if self.connection is conn:
                    self.connection = None
                    self._connected_event.clear()
                self._isolated_response_ids.clear()
                # Stop the response sender worker.
                if response_sender_task is not None:
                    await self._stop_response_sender(response_sender_task)

                # Stop background tool manager tasks (listener + cleanup) in all paths.
                await self.tool_manager.shutdown()
                self._drain_pending_responses()
                # The receive iterator has now stopped, so no event from this
                # session can cross a previously tripped deletion fence.
                await self._clear_private_tool_deletes(reset_terminal=True)
                self._redacted_tool_calls.clear()

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the realtime server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for the realtime API.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        _, audio_frame = frame
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
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Fail closed before detaching the transport. connection.close() is
        # best-effort and an old receive iterator may still deliver events;
        # only that iterator's finally block may reopen the fence.
        self._private_tool_delete_terminal = True
        connection = self.connection
        self.connection = None
        # Unblock the response sender worker after closing admission to sends.
        self._response_done_event.set()
        await self._stop_response_sender()

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()
        self._drain_pending_responses()
        await self._clear_private_tool_deletes()

        await self._cancel_partial_transcript_task()

        if connection:
            try:
                await connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
        self._drain_pending_responses()

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
