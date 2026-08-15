from __future__ import annotations
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from math import isfinite
from typing import Literal, ClassVar, Protocol, TypeAlias, runtime_checkable
from dataclasses import dataclass
from collections.abc import Mapping, Callable, Awaitable

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from reachy_mini_conversation_app.idle_policy import start_idle_tool_call
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)

DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS = 2.0
MAX_COMPLETED_UTTERANCE_TIMEOUT_SECONDS = 120.0
DEFAULT_SEARCH_POLICY_TIMEOUT_SECONDS = 10.0
MAX_SEARCH_POLICY_TIMEOUT_SECONDS = 120.0
MAX_SEARCH_PROVIDER_INDICATOR_CHARS = 512


AudioFrame: TypeAlias = tuple[int, NDArray[np.int16]]
HandlerOutput: TypeAlias = AudioFrame | AdditionalOutputs | None
QueueItem: TypeAlias = AudioFrame | AdditionalOutputs


@dataclass(frozen=True)
class CompletedUserUtterance:
    """One completed mono PCM16 utterance from the active realtime input."""

    item_id: str
    sample_rate: int
    pcm16: bytes


# None preserves accepted-turn lifecycle hooks without adding model-visible context.
CompletedUtteranceResult: TypeAlias = Mapping[str, str] | None
CompletedUtteranceObserver: TypeAlias = Callable[[CompletedUserUtterance], Awaitable[CompletedUtteranceResult]]


@runtime_checkable
class AcceptedTranscriptObserver(Protocol):
    """Optional lifecycle hook for a current nonempty completed transcript."""

    def on_transcript_accepted(self, item_id: str) -> None:
        """Observe the accepted backend item without retaining transcript text."""
        ...


@runtime_checkable
class AcceptedTranscriptTextObserver(Protocol):
    """Optional transient-text hook for a current accepted transcript."""

    def on_transcript_observed(self, item_id: str, transcript: str) -> None:
        """Inspect the accepted transcript without retaining its text."""
        ...


@dataclass
class SearchPolicyRequest:
    """One locally correlated and bounded request for the official search tool."""

    item_id: str
    transcript: str
    query: str
    max_results: int
    requested_provider: str | None = None


@dataclass(frozen=True)
class SearchProviderSelection:
    """One approved request-local provider; None explicitly selects Pollen's official search."""

    provider: SearchProvider | None


@dataclass(frozen=True)
class SearchPolicyDecision:
    """A local decision made before any official search transport is started."""

    outcome: Literal["approved", "confirmation_required", "refused"]
    confirmation_question: str | None = None
    on_confirmation_abandoned: Callable[[], None] | None = None
    provider_selection: SearchProviderSelection | None = None


SearchPolicy: TypeAlias = Callable[[SearchPolicyRequest], Awaitable[SearchPolicyDecision]]
SearchSpaceGate: TypeAlias = Callable[[], Awaitable[bool]]


@dataclass(frozen=True)
class SearchSource:
    """One cited source returned by an injected search provider."""

    title: str
    url: str


@dataclass(frozen=True)
class SearchProviderResult:
    """One bounded answer and its cited sources."""

    answer: str
    sources: tuple[SearchSource, ...]


SearchProviderCall: TypeAlias = Callable[[str, int], Awaitable[SearchProviderResult]]


@dataclass(frozen=True)
class SearchProvider:
    """An explicitly configured replacement for the bundled search transport."""

    indicator_text: str
    search: SearchProviderCall


SearchAttemptStage: TypeAlias = Literal[
    "attempt",
    "policy",
    "confirmation",
    "provider",
    "progress",
    "answer",
    "failure",
    "terminal",
]
SearchAttemptOutcome: TypeAlias = Literal[
    "requested",
    "approved",
    "confirmation_required",
    "refused",
    "dispatched",
    "first_pcm",
    "response_done",
    "playback_drained",
    "abandoned",
    "completed",
    "failed",
    "invalid",
    "rate_limited",
    "stale",
    "replayed",
    "superseded",
    "cancelled",
    "timeout",
    "unavailable",
]
SearchElapsedBucket: TypeAlias = Literal["under_1s", "1_to_3s", "3_to_10s", "10_to_30s", "over_30s"]


@dataclass(frozen=True)
class SearchAttemptEvent:
    """One bounded content-free search lifecycle observation."""

    supervisor_generation: int
    child_generation: int
    attempt_seq: int
    event_seq: int
    stage: SearchAttemptStage
    outcome: SearchAttemptOutcome
    elapsed_bucket: SearchElapsedBucket


SearchAttemptObserver: TypeAlias = Callable[[SearchAttemptEvent], None]


def validate_search_attempt_observer(
    observer: SearchAttemptObserver | None,
    *,
    supervisor_generation: int,
    child_generation: int,
) -> None:
    """Reject malformed search-observer composition before startup side effects."""
    if not all(
        type(generation) is int and 0 <= generation <= 2**63 - 1
        for generation in (supervisor_generation, child_generation)
    ):
        raise ValueError("Search observer generations must be non-negative 63-bit integers")
    if observer is not None and not callable(observer):
        raise ValueError("The search attempt observer must be callable")


def validate_completed_utterance_timeout_seconds(timeout_seconds: float) -> None:
    """Reject observer timeouts outside the bounded composition contract."""
    if not isfinite(timeout_seconds) or not (0.0 < timeout_seconds <= MAX_COMPLETED_UTTERANCE_TIMEOUT_SECONDS):
        raise ValueError("Completed-utterance observer timeout must be greater than zero and at most 120 seconds")


def validate_search_policy_timeout_seconds(timeout_seconds: float) -> None:
    """Reject search-policy timeouts outside the bounded composition contract."""
    if not isfinite(timeout_seconds) or not (0.0 < timeout_seconds <= MAX_SEARCH_POLICY_TIMEOUT_SECONDS):
        raise ValueError("Search-policy timeout must be greater than zero and at most 120 seconds")


def validate_search_provider(provider: SearchProvider | None) -> None:
    """Reject malformed provider composition before it can cause side effects."""
    if provider is None:
        return
    indicator = getattr(provider, "indicator_text", None)
    search = getattr(provider, "search", None)
    if (
        not isinstance(indicator, str)
        or not indicator
        or indicator != indicator.strip()
        or len(indicator) > MAX_SEARCH_PROVIDER_INDICATOR_CHARS
        or not callable(search)
    ):
        raise ValueError("The search provider is invalid")
    try:
        indicator.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("The search provider is invalid") from exc


def validate_search_provider_selection(selection: SearchProviderSelection | None) -> None:
    """Reject malformed request-local provider selections before dispatch."""
    if selection is None:
        return
    if type(selection) is not SearchProviderSelection:
        raise ValueError("The search provider selection is invalid")
    validate_search_provider(selection.provider)


class ConversationHandler(AsyncStreamHandler, ABC):
    """Shared app handler contract and idle behavior for realtime conversation backends."""

    IDLE_BEHAVIOR_THRESHOLD_S: ClassVar[float] = 180.0

    deps: ToolDependencies
    tool_manager: BackgroundToolManager
    output_queue: asyncio.Queue[QueueItem]
    last_activity_time: float
    last_idle_behavior_time: float
    _activity_observer: Callable[[str], None] | None = None
    _transcript_observer: Callable[[str, str, bool], None] | None = None
    _completed_utterance_observer: CompletedUtteranceObserver | None = None
    _completed_utterance_timeout_seconds = DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS
    _search_policy: SearchPolicy | None = None
    _search_policy_timeout_seconds = DEFAULT_SEARCH_POLICY_TIMEOUT_SECONDS
    _search_space_gate: SearchSpaceGate | None = None
    _search_provider: SearchProvider | None = None
    _search_attempt_observer: SearchAttemptObserver | None = None
    _search_attempt_supervisor_generation = 0
    _search_attempt_child_generation = 0

    def __init__(self) -> None:
        """Initialize the stream handler and shared idle/activity tracking."""
        super().__init__()
        self.last_activity_time = time.monotonic()
        self.last_idle_behavior_time = self.last_activity_time

    def set_activity_observer(self, observer: Callable[[str], None] | None) -> None:
        """Attach or detach an activity observer. Pass None to clear."""
        self._activity_observer = observer

    def set_transcript_observer(self, observer: Callable[[str, str, bool], None] | None) -> None:
        """Attach/detach a transcript observer, called (role, text, final)."""
        self._transcript_observer = observer

    def set_completed_utterance_observer(
        self,
        observer: CompletedUtteranceObserver | None,
        *,
        timeout_seconds: float = DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS,
    ) -> None:
        """Attach or detach the completed-user-utterance observer."""
        validate_completed_utterance_timeout_seconds(timeout_seconds)
        self._completed_utterance_observer = observer
        self._completed_utterance_timeout_seconds = timeout_seconds

    def set_search_policy(
        self,
        policy: SearchPolicy | None,
        *,
        timeout_seconds: float = DEFAULT_SEARCH_POLICY_TIMEOUT_SECONDS,
    ) -> None:
        """Attach or detach the local policy for the official search tool."""
        validate_search_policy_timeout_seconds(timeout_seconds)
        if policy is None and self._search_provider is not None:
            raise ValueError("A search provider requires a search policy")
        self._search_policy = policy
        self._search_policy_timeout_seconds = timeout_seconds

    def set_search_space_gate(self, gate: SearchSpaceGate | None) -> None:
        """Attach the anonymous official-Space revision gate."""
        self._search_space_gate = gate

    def set_search_provider(self, provider: SearchProvider | None) -> None:
        """Attach an explicitly configured search provider."""
        validate_search_provider(provider)
        if provider is not None and self._search_policy is None:
            raise ValueError("A search provider requires a search policy")
        self._search_provider = provider

    def set_search_attempt_observer(
        self,
        observer: SearchAttemptObserver | None,
        *,
        supervisor_generation: int,
        child_generation: int,
    ) -> None:
        """Attach a content-free search lifecycle observer."""
        validate_search_attempt_observer(
            observer,
            supervisor_generation=supervisor_generation,
            child_generation=child_generation,
        )
        self._search_attempt_observer = observer
        self._search_attempt_supervisor_generation = supervisor_generation
        self._search_attempt_child_generation = child_generation

    def _notify_completed_utterance_observer_connection_reset(self) -> None:
        """Let an observer discard provisional state when a live session ends."""
        try:
            observer = self._completed_utterance_observer
            callback = getattr(observer, "on_connection_reset", None)
            if not callable(callback):
                return
            callback()
        except Exception:
            logger.warning("Completed-utterance observer connection-reset hook failed", exc_info=True)

    def _notify_completed_utterance_observer_transcript_accepted(self, item_id: str, transcript: str) -> None:
        """Notify an observer only after a current nonempty transcript is accepted."""
        observer = self._completed_utterance_observer
        if isinstance(observer, AcceptedTranscriptObserver):
            try:
                observer.on_transcript_accepted(item_id)
            except Exception:
                logger.warning("Completed-utterance observer transcript hook failed", exc_info=True)
        if isinstance(observer, AcceptedTranscriptTextObserver):
            try:
                observer.on_transcript_observed(item_id, transcript)
            except Exception:
                logger.warning("Completed-utterance observer transcript-text hook failed", exc_info=True)

    def _notify_search_policy_connection_reset(self) -> None:
        """Let a search policy discard pending consent when a live session ends."""
        try:
            policy = self._search_policy
            callback = getattr(policy, "on_connection_reset", None)
            if not callable(callback):
                return
            callback()
        except Exception:
            logger.warning("Search policy connection-reset hook failed")

    def _emit_transcript(self, role: str, text: str, final: bool = True) -> None:
        """Forward one transcript chunk to the observer, if attached."""
        observer = self._transcript_observer
        if observer is not None and text:
            try:
                observer(role, text, final)
            except Exception:
                logger.debug("transcript observer raised (ignored)", exc_info=True)

    def _mark_activity(self, reason: str) -> None:
        """Record non-idle conversation activity for the idle timer."""
        self.last_activity_time = time.monotonic()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)
        if self._activity_observer is not None:
            try:
                self._activity_observer(reason)
            except Exception:
                logger.debug("activity observer raised (ignored)", exc_info=True)

    def _idle_behavior_ready(self) -> bool:
        """Return whether idle behavior may run now. Backends can add guards."""
        return True

    async def emit(self) -> HandlerOutput:
        """Emit the next queued output, triggering local idle behavior when due."""
        now = time.monotonic()
        idle_duration = now - self.last_activity_time
        idle_behavior_duration = now - self.last_idle_behavior_time
        if (
            idle_duration > self.IDLE_BEHAVIOR_THRESHOLD_S
            and idle_behavior_duration > self.IDLE_BEHAVIOR_THRESHOLD_S
            and self._idle_behavior_ready()
            and self.deps.movement_manager.is_idle()
        ):
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle tool skipped (connection closed?): %s", e)
                return None
            self.last_idle_behavior_time = now
        handler_output = await wait_for_item(self.output_queue)
        return handler_output

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Run a locally selected idle tool without sending an idle turn to the model."""
        if not self._is_connected():
            logger.debug("No active session; cannot run idle tool")
            return

        available_tool_names = {spec["name"] for spec in get_tool_specs()}
        await start_idle_tool_call(
            deps=self.deps,
            tool_manager=self.tool_manager,
            output_queue=self.output_queue,
            available_tool_names=available_tool_names,
            idle_duration=idle_duration,
        )

    @abstractmethod
    def _is_connected(self) -> bool:
        """Return whether the backend session/connection is currently open."""
        ...

    @abstractmethod
    async def start_up(self) -> None:
        """Start the realtime handler."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shut down the realtime handler."""
        ...

    def shutdown_complete(self) -> bool:
        """Return whether the handler owns no remaining conversation work."""
        return self.tool_manager.shutdown_complete()

    @abstractmethod
    async def receive(self, frame: AudioFrame) -> None:
        """Receive an input audio frame."""
        ...

    @abstractmethod
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile."""
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[str]:
        """Return voices available for the active backend."""
        ...

    @abstractmethod
    def get_current_voice(self) -> str:
        """Return the current voice."""
        ...

    @abstractmethod
    async def change_voice(self, voice: str) -> str:
        """Change the current voice."""
        ...

    @abstractmethod
    async def say(self, text: str) -> None:
        """Make the robot speak ``text`` now (injected turn; not verbatim TTS).

        The backend is speech-to-speech, so ``text`` is an instruction the
        model voices, not a guaranteed-literal string. Raises if no session is
        open.
        """
        ...
