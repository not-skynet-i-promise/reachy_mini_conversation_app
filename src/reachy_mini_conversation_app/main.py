"""Entrypoint for the Reachy Mini conversation app."""

from __future__ import annotations
import os
import sys
import stat
import time
import asyncio
import logging
import argparse
import threading
from typing import TYPE_CHECKING, Any, NoReturn, Optional
from pathlib import Path
from collections.abc import Callable, Awaitable, MutableMapping

from fastapi import FastAPI, Request, Response

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    log_connection_troubleshooting,
)
from reachy_mini_conversation_app.config import (
    RECOVERY_CONNECTION_ACK_FD_ENV,
    RECOVERY_CONNECTION_ACK_NONCE_ENV,
    load_dotenv_without_recovery_connection_authority,
)
from reachy_mini_conversation_app.conversation_handler import (
    DEFAULT_SEARCH_POLICY_TIMEOUT_SECONDS,
    DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS,
    DEFAULT_PRIVATE_TRANSCRIPT_ROUTER_TIMEOUT_SECONDS,
    SearchPolicy,
    SearchProvider,
    SearchAttemptObserver,
    SearchAttemptSequence,
    PrivateTranscriptRouter,
    CompletedUtteranceObserver,
    validate_search_provider,
    validate_search_attempt_observer,
    validate_private_transcript_router,
    validate_search_policy_timeout_seconds,
    validate_completed_utterance_timeout_seconds,
)


if TYPE_CHECKING:
    from reachy_mini_conversation_app.console import LocalStream


_RECOVERY_CONNECTION_ACK_FD_ENV = RECOVERY_CONNECTION_ACK_FD_ENV
_RECOVERY_CONNECTION_ACK_NONCE_ENV = RECOVERY_CONNECTION_ACK_NONCE_ENV
_RECOVERY_CONNECTION_ACK_MAGIC = b"RCA1"
STALE_CONNECTION_EXIT_STATUS = 76
_STALE_CONNECTION_TERMINATOR: Callable[[int], NoReturn] = os._exit


class _StaleConnectionExit:
    """Give one live app instance a local-only terminal path."""

    def __init__(self, request_event: threading.Event) -> None:
        self._request_event = request_event
        self._lock = threading.Lock()
        self._accepting = False
        self._accepted = False
        self._thread: threading.Thread | None = None

    def arm(self) -> None:
        try:
            thread = threading.Thread(
                target=self._await_request,
                daemon=True,
                name="stale-connection-exit",
            )
            with self._lock:
                if self._request_event.is_set():
                    raise RuntimeError("Stale-connection exit was requested before the conversation loop")
                self._accepting = True
                self._thread = thread
            thread.start()
        except BaseException:
            with self._lock:
                self._accepting = False
                self._thread = None
            raise

    def _await_request(self) -> None:
        while not self._request_event.wait(0.05):
            with self._lock:
                if not self._accepting:
                    return
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            self._accepted = True
            _STALE_CONNECTION_TERMINATOR(STALE_CONNECTION_EXIT_STATUS)

    def close_for_ordinary_cleanup(self) -> bool:
        with self._lock:
            self._accepting = False
            accepted = self._accepted
        if not accepted and self._thread is not None:
            self._thread.join(timeout=0.1)
        return not accepted


def _consume_recovery_connection_acknowledgment_environment(
    environment: MutableMapping[str, str] | None = None,
) -> tuple[int, bytes] | None:
    selected_environment = os.environ if environment is None else environment
    encoded_fd = selected_environment.pop(_RECOVERY_CONNECTION_ACK_FD_ENV, None)
    encoded_nonce = selected_environment.pop(_RECOVERY_CONNECTION_ACK_NONCE_ENV, None)
    if encoded_fd is None and encoded_nonce is None:
        return None

    owned_fd: int | None = None
    try:
        if encoded_fd is None:
            raise ValueError
        if (
            len(encoded_fd) > 10
            or not encoded_fd.isascii()
            or not encoded_fd.isdecimal()
            or encoded_fd != str(int(encoded_fd))
        ):
            raise ValueError
        descriptor = int(encoded_fd)
        if descriptor <= 2:
            raise ValueError
        owned_fd = descriptor
        os.set_inheritable(descriptor, False)
        descriptor_stat = os.fstat(descriptor)
        if os.name != "posix":
            raise ValueError
        if not stat.S_ISFIFO(descriptor_stat.st_mode):
            raise ValueError
        if (
            encoded_nonce is None
            or len(encoded_nonce) != 64
            or any(character not in "0123456789abcdef" for character in encoded_nonce)
        ):
            raise ValueError
        nonce = bytes.fromhex(encoded_nonce)
        os.set_blocking(descriptor, False)
        return descriptor, _RECOVERY_CONNECTION_ACK_MAGIC + nonce
    except (OSError, ValueError):
        if owned_fd is not None:
            try:
                os.close(owned_fd)
            except OSError:
                raise RuntimeError("Invalid recovery connection acknowledgment configuration") from None
        raise RuntimeError("Invalid recovery connection acknowledgment configuration") from None


def _close_recovery_connection_acknowledgment(
    recovery_connection_acknowledgment: tuple[int, bytes] | None,
) -> None:
    if recovery_connection_acknowledgment is None:
        return
    try:
        os.close(recovery_connection_acknowledgment[0])
    except OSError:
        pass


def _initialize_robot_and_acknowledge_connection(
    args: argparse.Namespace,
    robot: ReachyMini | None,
    logger: logging.Logger,
    recovery_connection_acknowledgment: tuple[int, bytes] | None,
) -> ReachyMini:
    descriptor = recovery_connection_acknowledgment[0] if recovery_connection_acknowledgment is not None else None
    try:
        if recovery_connection_acknowledgment is not None and robot is not None:
            raise RuntimeError("Recovery connection acknowledgment requires app-owned robot initialization")

        if robot is None:
            try:
                robot_kwargs: dict[str, object] = {}
                if args.robot_name is not None:
                    robot_kwargs["robot_name"] = args.robot_name
                if args.robot_host is not None:
                    robot_kwargs["host"] = args.robot_host
                    robot_kwargs["connection_mode"] = "network"
                if args.robot_host is None:
                    logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
                else:
                    logger.info("Initializing ReachyMini with an explicit network daemon host")
                robot = ReachyMini(**robot_kwargs)
            except TimeoutError as error:
                logger.error("Connection timeout: Failed to connect to Reachy Mini daemon. Details: %s", error)
                log_connection_troubleshooting(logger, args.robot_name)
                raise SystemExit(1) from error
            except ConnectionError as error:
                logger.error("Connection failed: Unable to establish connection to Reachy Mini. Details: %s", error)
                log_connection_troubleshooting(logger, args.robot_name)
                raise SystemExit(1) from error
            except Exception as error:
                logger.error(
                    "Unexpected error during robot initialization: %s: %s",
                    type(error).__name__,
                    error,
                )
                logger.error("Please check your configuration and try again.")
                raise SystemExit(1) from error

        if recovery_connection_acknowledgment is not None:
            assert descriptor is not None
            record = recovery_connection_acknowledgment[1]
            try:
                written = os.write(descriptor, record)
            except OSError:
                logger.error("Recovery connection acknowledgment failed")
                raise RuntimeError("Recovery connection acknowledgment failed") from None
            if written != len(record):
                logger.error("Recovery connection acknowledgment failed")
                raise RuntimeError("Recovery connection acknowledgment failed")
        return robot
    finally:
        pending_error = sys.exc_info()[0] is not None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                logger.error("Failed to close the recovery connection acknowledgment descriptor")
                if not pending_error:
                    raise RuntimeError("Recovery connection acknowledgment failed") from None


def _start_inactivity_timeout_thread(
    timeout_minutes: float,
    stream_manager: LocalStream,
    logger: logging.Logger,
    app_stop_event: threading.Event | None,
    go_to_sleep: Callable[[], dict[str, Any]] | None = None,
) -> threading.Thread:
    """Start a daemon that puts the app to sleep after inactivity."""
    timeout_seconds = timeout_minutes * 60.0

    def poll_inactivity_timeout() -> None:
        logger.info("App inactivity timeout enabled: %.1f minutes.", timeout_minutes)
        while app_stop_event is None or not app_stop_event.is_set():
            elapsed = stream_manager.seconds_since_activity()
            if elapsed >= timeout_seconds:
                logger.info("No activity for %.1f minutes; going to sleep.", elapsed / 60.0)
                try:
                    if go_to_sleep is not None:
                        go_to_sleep()
                    else:
                        stream_manager.close()
                except Exception as e:
                    logger.error("Error while going to sleep after inactivity timeout: %s", e)
                    try:
                        stream_manager.close()
                    except Exception as close_error:
                        logger.error("Error while closing stream manager after inactivity timeout: %s", close_error)
                return
            time.sleep(1.0)

    thread = threading.Thread(target=poll_inactivity_timeout, daemon=True)
    thread.start()
    return thread


def main(
    completed_utterance_observer: CompletedUtteranceObserver | None = None,
    completed_utterance_timeout_seconds: float = DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS,
    search_policy: SearchPolicy | None = None,
    search_policy_timeout_seconds: float = DEFAULT_SEARCH_POLICY_TIMEOUT_SECONDS,
    search_provider: SearchProvider | None = None,
    search_attempt_observer: SearchAttemptObserver | None = None,
    search_attempt_supervisor_generation: int = 0,
    search_attempt_child_generation: int = 0,
    private_transcript_router: PrivateTranscriptRouter | None = None,
    private_transcript_router_timeout_seconds: float = DEFAULT_PRIVATE_TRANSCRIPT_ROUTER_TIMEOUT_SECONDS,
    graceful_shutdown_event: threading.Event | None = None,
    graceful_shutdown_complete_event: threading.Event | None = None,
    stale_connection_exit_event: threading.Event | None = None,
) -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    if args.command == "tool-spaces":
        from reachy_mini_conversation_app.tool_spaces import handle_tool_spaces_command

        logger = setup_logger(args.debug)
        try:
            raise SystemExit(handle_tool_spaces_command(args))
        except Exception as exc:
            logger.error("tool-spaces command failed: %s", exc)
            raise SystemExit(1) from exc
    run(
        args,
        completed_utterance_observer=completed_utterance_observer,
        completed_utterance_timeout_seconds=completed_utterance_timeout_seconds,
        search_policy=search_policy,
        search_policy_timeout_seconds=search_policy_timeout_seconds,
        search_provider=search_provider,
        search_attempt_observer=search_attempt_observer,
        search_attempt_supervisor_generation=search_attempt_supervisor_generation,
        search_attempt_child_generation=search_attempt_child_generation,
        private_transcript_router=private_transcript_router,
        private_transcript_router_timeout_seconds=private_transcript_router_timeout_seconds,
        graceful_shutdown_event=graceful_shutdown_event,
        graceful_shutdown_complete_event=graceful_shutdown_complete_event,
        stale_connection_exit_event=stale_connection_exit_event,
    )


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
    load_instance_runtime_settings: bool = True,
    completed_utterance_observer: CompletedUtteranceObserver | None = None,
    completed_utterance_timeout_seconds: float = DEFAULT_COMPLETED_UTTERANCE_TIMEOUT_SECONDS,
    search_policy: SearchPolicy | None = None,
    search_policy_timeout_seconds: float = DEFAULT_SEARCH_POLICY_TIMEOUT_SECONDS,
    search_provider: SearchProvider | None = None,
    search_attempt_observer: SearchAttemptObserver | None = None,
    search_attempt_supervisor_generation: int = 0,
    search_attempt_child_generation: int = 0,
    private_transcript_router: PrivateTranscriptRouter | None = None,
    private_transcript_router_timeout_seconds: float = DEFAULT_PRIVATE_TRANSCRIPT_ROUTER_TIMEOUT_SECONDS,
    graceful_shutdown_event: threading.Event | None = None,
    graceful_shutdown_complete_event: threading.Event | None = None,
    stale_connection_exit_event: threading.Event | None = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    recovery_acknowledgment_environment = {}
    for environment_name in (
        _RECOVERY_CONNECTION_ACK_FD_ENV,
        _RECOVERY_CONNECTION_ACK_NONCE_ENV,
    ):
        environment_value = os.environ.pop(environment_name, None)
        if environment_value is not None:
            recovery_acknowledgment_environment[environment_name] = environment_value

    try:
        recovery_connection_acknowledgment = _consume_recovery_connection_acknowledgment_environment(
            recovery_acknowledgment_environment
        )
    except RuntimeError:
        logging.getLogger(__name__).error("Invalid recovery connection acknowledgment configuration")
        raise

    try:
        if not isinstance(load_instance_runtime_settings, bool):
            raise ValueError("load_instance_runtime_settings must be a boolean")
        validate_completed_utterance_timeout_seconds(completed_utterance_timeout_seconds)
        validate_search_policy_timeout_seconds(search_policy_timeout_seconds)
        validate_search_provider(search_provider)
        if search_provider is not None and search_policy is None:
            raise ValueError("A search provider requires a search policy")
        if search_attempt_observer is not None and search_policy is None:
            raise ValueError("A search attempt observer requires a search policy")
        validate_search_attempt_observer(
            search_attempt_observer,
            supervisor_generation=search_attempt_supervisor_generation,
            child_generation=search_attempt_child_generation,
        )
        validate_private_transcript_router(
            private_transcript_router,
            timeout_seconds=private_transcript_router_timeout_seconds,
        )
        if private_transcript_router is not None and completed_utterance_observer is not None:
            raise ValueError("Private transcript routing cannot be combined with the completed-utterance observer")
        if (graceful_shutdown_event is None) != (graceful_shutdown_complete_event is None):
            raise ValueError("Graceful shutdown requires distinct request and completion events")
        if graceful_shutdown_event is not None:
            if (
                not isinstance(graceful_shutdown_event, threading.Event)
                or not isinstance(graceful_shutdown_complete_event, threading.Event)
                or graceful_shutdown_event is graceful_shutdown_complete_event
                or graceful_shutdown_event is app_stop_event
                or graceful_shutdown_complete_event is app_stop_event
                or graceful_shutdown_complete_event.is_set()
            ):
                raise ValueError("Graceful shutdown requires distinct request and completion events")
        if stale_connection_exit_event is not None and (
            not isinstance(stale_connection_exit_event, threading.Event)
            or stale_connection_exit_event.is_set()
            or stale_connection_exit_event is app_stop_event
            or stale_connection_exit_event is graceful_shutdown_event
            or stale_connection_exit_event is graceful_shutdown_complete_event
        ):
            raise ValueError("Stale-connection exit requires a fresh distinct request event")
        if stale_connection_exit_event is not None and recovery_connection_acknowledgment is None:
            raise ValueError("Stale-connection exit requires recovery connection acknowledgment")
        diagnostic_antenna_expression = getattr(args, "diagnostic_antenna_expression", False)
        if not isinstance(diagnostic_antenna_expression, bool):
            raise ValueError("diagnostic_antenna_expression must be a boolean")

        # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
        from reachy_mini_conversation_app.moves import MovementManager
        from reachy_mini_conversation_app.config import (
            HF_LOCAL_CONNECTION_MODE,
            set_instance_path,
            get_hf_connection_selection,
            resolve_app_timeout_minutes,
            refresh_runtime_config_from_env,
            has_private_mcp_local_realtime_boundary,
        )
        from reachy_mini_conversation_app.startup_settings import (
            StartupSettings,
            load_startup_settings_into_runtime,
        )

        logger = setup_logger(args.debug)
        logger.info("Starting Reachy Mini Conversation App")
        if diagnostic_antenna_expression:
            logger.warning("Diagnostic antenna expression enabled; use only for supervised diagnosis")
        set_instance_path(instance_path)
        startup_settings = StartupSettings()

        if instance_path is not None and load_instance_runtime_settings:
            try:
                env_path = Path(instance_path) / ".env"
                if env_path.exists():
                    load_dotenv_without_recovery_connection_authority(str(env_path), override=True)
                    refresh_runtime_config_from_env()
                    logger.info("Loaded instance configuration from %s", env_path)
            except Exception as e:
                logger.warning("Failed to load instance configuration: %s", e)

            try:
                startup_settings = load_startup_settings_into_runtime(instance_path)
            except Exception as e:
                logger.warning("Failed to load startup settings: %s", e)

        # Instance settings never gain recovery-handshake authority.
        os.environ.pop(_RECOVERY_CONNECTION_ACK_FD_ENV, None)
        os.environ.pop(_RECOVERY_CONNECTION_ACK_NONCE_ENV, None)

        logger.info(
            "Configured Hugging Face realtime backend, connection mode: %s",
            get_hf_connection_selection().mode,
        )
        if private_transcript_router is not None and not has_private_mcp_local_realtime_boundary():
            raise ValueError("Private transcript routing requires an explicit loopback realtime backend")

        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, initialize_tools
        from reachy_mini_conversation_app.conversation_handler import ConversationHandler
    except BaseException:
        _close_recovery_connection_acknowledgment(recovery_connection_acknowledgment)
        raise

    initialization_acknowledgment = recovery_connection_acknowledgment
    recovery_connection_acknowledgment = None
    robot = _initialize_robot_and_acknowledge_connection(
        args,
        robot,
        logger,
        initialization_acknowledgment,
    )

    if args.no_wobble:
        try:
            robot.disable_wobbling()
        except Exception as e:
            logger.error("Failed to disable head wobbling before startup wake-up: %s", e)
            raise RuntimeError("Failed to disable head wobbling before startup wake-up") from e

    app_lifecycle.wake_up_if_sleeping(robot, logger)

    movement_manager = MovementManager(
        current_robot=robot,
        diagnostic_antenna_expression=diagnostic_antenna_expression,
    )

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        instance_path=instance_path,
        camera_enabled=not args.no_camera,
    )
    search_attempt_sequence = SearchAttemptSequence() if search_attempt_observer is not None else None

    def build_handler(startup_voice: Optional[str] = None) -> ConversationHandler:
        """Build a Hugging Face realtime handler for the current runtime config."""
        from reachy_mini_conversation_app.search_space_gate import build_official_search_space_gate
        from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler

        hf_connection_selection = get_hf_connection_selection()
        transport_label = (
            "Hugging Face direct websocket"
            if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
            else "Hugging Face session proxy"
        )
        logger.info("Using Hugging Face realtime handler (%s)", transport_label)
        handler = HuggingFaceRealtimeHandler(
            deps,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )
        if completed_utterance_observer is not None:
            handler.set_completed_utterance_observer(
                completed_utterance_observer,
                timeout_seconds=completed_utterance_timeout_seconds,
            )
        if search_policy is not None:
            handler.set_search_policy(search_policy, timeout_seconds=search_policy_timeout_seconds)
            handler.set_search_space_gate(build_official_search_space_gate())
            if search_provider is not None:
                handler.set_search_provider(search_provider)
            if search_attempt_observer is not None:
                handler.set_search_attempt_observer(
                    search_attempt_observer,
                    supervisor_generation=search_attempt_supervisor_generation,
                    child_generation=search_attempt_child_generation,
                    sequence=search_attempt_sequence,
                )
        if private_transcript_router is not None:
            handler.set_private_transcript_router(
                private_transcript_router,
                timeout_seconds=private_transcript_router_timeout_seconds,
            )
        return handler

    handler = build_handler(startup_settings.voice)

    stream_manager: LocalStream | None = None
    own_ui_server = None

    effective_settings_app = settings_app
    if args.ui and settings_app is None:
        effective_settings_app = FastAPI()

        @effective_settings_app.middleware("http")
        async def _no_cache(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            """Serve everything no-store so browsers don't keep stale UI modules."""
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response

    stream_manager = LocalStream(
        handler,
        robot,
        settings_app=effective_settings_app,
        instance_path=instance_path,
        load_instance_runtime_settings=load_instance_runtime_settings,
        handler_factory=build_handler,
        startup_voice=startup_settings.voice,
    )

    # The page is served immediately, so the API must be live before the slow startup work below.
    if effective_settings_app is not None:
        stream_manager._init_settings_ui_if_needed()

    go_to_sleep_lock = threading.Lock()
    go_to_sleep_requested = threading.Event()
    sleep_failure_result: dict[str, Any] | None = None

    def go_to_sleep_and_stop_app() -> dict[str, Any]:
        """Put Reachy to sleep, then stop the current app."""
        nonlocal sleep_failure_result

        if not go_to_sleep_lock.acquire(blocking=False):
            return {"status": "already_requested"}

        try:
            if go_to_sleep_requested.is_set():
                if sleep_failure_result is not None:
                    return sleep_failure_result.copy()
                return {"status": "already_requested"}
            go_to_sleep_requested.set()

            logger.info("Going to sleep before stopping conversation app.")
            sleep_error: str | None = None
            sleep_failure_action = "movement"

            try:
                robot.disable_wobbling()
            except Exception as e:
                logger.debug("Error disabling wobbling before sleep: %s", e)

            movement_manager.stop(reset_to_neutral=False)

            try:
                robot.goto_sleep()
            except Exception as e:
                sleep_error = f"{type(e).__name__}: {e}"
                logger.error("Failed to move Reachy Mini to sleep pose: %s", e)

            if sleep_error is None:
                try:
                    robot.disable_motors()
                except Exception as e:
                    sleep_failure_action = "motor disable"
                    sleep_error = f"{type(e).__name__}: {e}"
                    logger.error("Failed to disable Reachy Mini motors after sleep: %s", e)

            if sleep_error is not None:
                # A failed sleep transition may leave an unknown pose or torque state.
                sleep_failure_result = {
                    "status": "sleep_failed",
                    "stop_current_app_requested": False,
                    "local_stop_requested": False,
                    "error": f"go_to_sleep {sleep_failure_action} failed: {sleep_error}",
                }
                return sleep_failure_result.copy()

            stop_current_app_requested = False
            if app_stop_event is None or not app_stop_event.is_set():
                stop_current_app_requested = app_lifecycle.request_stop_current_app(robot, logger)
            local_stop_requested = True
            if app_stop_event is not None:
                app_stop_event.set()
            else:
                try:
                    stream_manager.close()
                except Exception as e:
                    local_stop_requested = False
                    logger.error("Error while closing stream manager after go_to_sleep: %s", e)

            return {
                "status": "sleeping",
                "stop_current_app_requested": stop_current_app_requested,
                "local_stop_requested": local_stop_requested,
            }
        finally:
            go_to_sleep_lock.release()

    deps.go_to_sleep = go_to_sleep_and_stop_app

    def run_go_to_sleep_tool() -> dict[str, Any]:
        return app_lifecycle.run_go_to_sleep_tool(deps, logger)

    if args.ui and settings_app is None and effective_settings_app is not None:
        import uvicorn

        own_ui_server = uvicorn.Server(
            uvicorn.Config(effective_settings_app, host="0.0.0.0", port=7860, log_level="warning")
        )
        threading.Thread(target=own_ui_server.run, daemon=True, name="ui-server").start()
        logger.info("Web UI available at http://localhost:7860")

    try:
        initialize_tools(instance_path=instance_path)
    except Exception as e:
        logger.error("Failed to initialize tools: %s", e)
        sys.exit(1)

    graceful_shutdown_thread: threading.Thread | None = None
    stale_connection_exit: _StaleConnectionExit | None = None
    try:
        # Each async service → its own thread/loop
        movement_manager.start()
        if not args.no_wobble:
            # Audio-reactive head motion is driven by the daemon's wobbler, which
            # taps the media pipeline at push_audio_sample. The console stream pushes
            # assistant audio through that pipeline directly.
            robot.enable_wobbling()

        timeout_minutes = resolve_app_timeout_minutes()
        if timeout_minutes is not None:
            _start_inactivity_timeout_thread(
                timeout_minutes,
                stream_manager,
                logger,
                app_stop_event,
                run_go_to_sleep_tool,
            )

        if graceful_shutdown_event is not None and graceful_shutdown_complete_event is not None:

            def poll_graceful_shutdown_event() -> None:
                graceful_shutdown_event.wait()
                logger.info("Graceful shutdown requested; quiescing the conversation before sleep.")
                if not stream_manager.quiesce_for_shutdown():
                    movement_manager.stop(reset_to_neutral=False)
                    logger.error("Graceful shutdown stopped before the safe-rest transition")
                    return
                result = run_go_to_sleep_tool()
                if result.get("status") != "sleeping":
                    logger.error("Graceful shutdown stopped because the safe-rest transition failed")
                    return
                graceful_shutdown_complete_event.set()

            graceful_shutdown_thread = threading.Thread(
                target=poll_graceful_shutdown_event,
                daemon=True,
                name="graceful-shutdown",
            )
            graceful_shutdown_thread.start()

        def poll_stop_event() -> None:
            """Poll the stop event to allow graceful shutdown.

            Deliberately does NOT put the robot to sleep: an external stop
            (mobile app, dashboard, app switch) means "stop this app", not
            "power the robot down" — the daemon returns it to the neutral
            pose afterwards, awake and ready for the next app. Sleeping is
            reserved for the explicit paths (the voice go_to_sleep tool and
            the inactivity timeout).
            """
            if app_stop_event is not None:
                app_stop_event.wait()

            logger.info("App stop event detected, shutting down...")
            try:
                stream_manager.close()
            except Exception as e:
                logger.error(f"Error while closing stream manager: {e}")

        if app_stop_event:
            threading.Thread(target=poll_stop_event, daemon=True).start()

        stale_connection_exit = (
            _StaleConnectionExit(stale_connection_exit_event) if stale_connection_exit_event is not None else None
        )
        if stale_connection_exit is not None:
            stale_connection_exit.arm()
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        ordinary_cleanup = stale_connection_exit is None or stale_connection_exit.close_for_ordinary_cleanup()
        if ordinary_cleanup:
            # Stop target writes before any later cleanup that could fail.
            movement_manager.stop(reset_to_neutral=False)
            if (
                graceful_shutdown_thread is not None
                and graceful_shutdown_event is not None
                and graceful_shutdown_event.is_set()
            ):
                graceful_shutdown_thread.join()
            if own_ui_server is not None:
                own_ui_server.should_exit = True

            try:
                robot.disable_wobbling()
            except Exception as e:
                logger.debug(f"Error disabling wobbling during shutdown: {e}")

            # Ensure media is explicitly closed before disconnecting
            try:
                robot.media.close()
            except Exception as e:
                logger.debug(f"Error closing media during shutdown: {e}")

            # prevent connection to keep alive some threads
            robot.client.disconnect()
            time.sleep(1)
            logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
