"""Tests for app-level runtime behavior."""

import sys
import typing
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.main as main_mod
import reachy_mini_conversation_app.moves as moves_mod
import reachy_mini_conversation_app.utils as utils_mod
import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.console as console_mod
import reachy_mini_conversation_app.startup_settings as startup_settings_mod
import reachy_mini_conversation_app.tools.core_tools as core_tools_mod
import reachy_mini_conversation_app.huggingface_realtime as huggingface_realtime_mod


@pytest.mark.parametrize(
    ("robot_host", "expected_kwargs"),
    [
        (None, {"robot_name": "kitchen"}),
        (
            "reachy-mini.local",
            {
                "robot_name": "kitchen",
                "host": "reachy-mini.local",
                "connection_mode": "network",
            },
        ),
    ],
)
def test_standalone_robot_connection_uses_the_selected_sdk_mode(
    monkeypatch: pytest.MonkeyPatch,
    robot_host: str | None,
    expected_kwargs: dict[str, object],
) -> None:
    """The standalone app should preserve auto mode or bind an explicit network host."""

    class ConstructionObserved(BaseException):
        pass

    constructor = MagicMock(side_effect=ConstructionObserved)
    monkeypatch.setattr(main_mod, "ReachyMini", constructor)
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(config_mod, "set_instance_path", MagicMock())
    monkeypatch.setattr(
        config_mod,
        "get_hf_connection_selection",
        MagicMock(return_value=SimpleNamespace(mode="test", has_target=False)),
    )
    monkeypatch.setattr(
        startup_settings_mod,
        "StartupSettings",
        MagicMock(return_value=SimpleNamespace(voice=None)),
    )
    args = SimpleNamespace(
        debug=False,
        robot_name="kitchen",
        robot_host=robot_host,
        no_camera=True,
        no_wobble=False,
        ui=False,
    )

    with pytest.raises(ConstructionObserved):
        main_mod.run(args)

    constructor.assert_called_once_with(**expected_kwargs)


def test_robot_host_cli_option_selects_the_explicit_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone parser should expose the same host passed to the SDK."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["reachy-mini-conversation-app", "--robot-host", "reachy-mini.local"],
    )

    args, unknown = utils_mod.parse_args()

    assert args.robot_host == "reachy-mini.local"
    assert unknown == []


def test_main_forwards_completed_utterance_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Programmatic callers should be able to compose an utterance observer."""
    args = SimpleNamespace(command=None)
    observer = MagicMock()
    run = MagicMock()
    monkeypatch.setattr(main_mod, "parse_args", MagicMock(return_value=(args, [])))
    monkeypatch.setattr(main_mod, "run", run)

    main_mod.main(observer)

    run.assert_called_once_with(
        args,
        completed_utterance_observer=observer,
        completed_utterance_timeout_seconds=2.0,
        search_policy=None,
        search_policy_timeout_seconds=10.0,
        search_provider=None,
        graceful_shutdown_event=None,
        graceful_shutdown_complete_event=None,
    )


def test_run_rejects_invalid_observer_timeout_before_robot_startup() -> None:
    """Public composition errors cannot occur after robot side effects."""
    robot = MagicMock()

    with pytest.raises(ValueError, match="at most 120 seconds"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            completed_utterance_timeout_seconds=121.0,
        )

    assert robot.mock_calls == []


def test_public_observer_annotations_are_runtime_resolvable() -> None:
    """Composition tooling should be able to inspect the public callback type."""
    assert "completed_utterance_observer" in typing.get_type_hints(main_mod.main)
    assert "completed_utterance_observer" in typing.get_type_hints(main_mod.run)
    assert "search_policy" in typing.get_type_hints(main_mod.main)
    assert "search_policy" in typing.get_type_hints(main_mod.run)
    assert "search_provider" in typing.get_type_hints(main_mod.main)
    assert "search_provider" in typing.get_type_hints(main_mod.run)
    assert "graceful_shutdown_event" in typing.get_type_hints(main_mod.main)
    assert "graceful_shutdown_event" in typing.get_type_hints(main_mod.run)


def test_graceful_shutdown_requires_paired_distinct_events_before_robot_startup() -> None:
    """A malformed shutdown capability cannot reach robot initialization."""
    robot = MagicMock()
    request = threading.Event()

    with pytest.raises(ValueError, match="distinct request and completion events"):
        main_mod.run(MagicMock(), robot=robot, graceful_shutdown_event=request)
    with pytest.raises(ValueError, match="distinct request and completion events"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            graceful_shutdown_event=request,
            graceful_shutdown_complete_event=request,
        )
    complete = threading.Event()
    complete.set()
    with pytest.raises(ValueError, match="distinct request and completion events"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            graceful_shutdown_event=request,
            graceful_shutdown_complete_event=complete,
        )

    assert robot.mock_calls == []


def test_inactivity_timeout_thread_goes_to_sleep() -> None:
    """The watchdog should use the shared sleep shutdown path once activity is too old."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_called_once_with()
    stream_manager.close.assert_not_called()


def test_inactivity_timeout_thread_closes_stream_manager_without_sleep_callback() -> None:
    """The watchdog should still close the stream when no sleep callback is available."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()


def _run_sleep_scenario(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sleep_fails: bool,
    disable_fails: bool = False,
    use_stop_event: bool,
    no_wobble: bool = False,
    completed_utterance_observer: object | None = None,
    completed_utterance_timeout_seconds: float = 2.0,
    search_policy: object | None = None,
    search_policy_timeout_seconds: float = 10.0,
    search_provider: object | None = None,
    rebuild_handler: bool = False,
    graceful_shutdown: bool = False,
    quiesce_succeeds: bool = True,
) -> dict[str, object]:
    """Run the app through one go_to_sleep tool call with hardware-free doubles."""
    operations: list[str] = []
    startup_operations: list[str] = []
    robot = MagicMock()
    robot.disable_wobbling.side_effect = lambda: startup_operations.append("disable_wobbling")
    robot.enable_wobbling.side_effect = lambda: startup_operations.append("enable_wobbling")

    def _goto_sleep() -> None:
        operations.append("sleep")
        if sleep_fails:
            raise RuntimeError("motor fault")

    robot.goto_sleep.side_effect = _goto_sleep

    def _disable_motors() -> None:
        operations.append("disable_motors")
        if disable_fails:
            raise RuntimeError("torque fault")

    robot.disable_motors.side_effect = _disable_motors
    movement_manager = MagicMock()
    movement_manager.start.side_effect = lambda: startup_operations.append("movement_manager_start")
    stream_manager = MagicMock()
    stream_manager.close.side_effect = lambda: operations.append("local_stream_close")
    stream_manager.quiesce_for_shutdown.side_effect = lambda: operations.append("quiesce") or quiesce_succeeds

    class _RecordingStopEvent(threading.Event):
        def set(self) -> None:
            operations.append("local_stop_event")
            super().set()

    stop_event = _RecordingStopEvent() if use_stop_event else None
    graceful_shutdown_event = threading.Event() if graceful_shutdown else None
    graceful_shutdown_complete_event = threading.Event() if graceful_shutdown else None
    if graceful_shutdown_event is not None:
        graceful_shutdown_event.set()
    request_stop_current_app = MagicMock(side_effect=lambda _robot, _logger: operations.append("stop") or True)
    monkeypatch.setattr(main_mod.app_lifecycle, "request_stop_current_app", request_stop_current_app)
    monkeypatch.setattr(
        main_mod.app_lifecycle,
        "wake_up_if_sleeping",
        MagicMock(side_effect=lambda *_args: startup_operations.append("wake_check")),
    )
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod.time, "sleep", MagicMock())
    thread_targets: list[object] = []

    class _DeferredThread:
        def __init__(self, *, target: object, **_kwargs: object) -> None:
            self.target = target

        def start(self) -> None:
            thread_targets.append(self.target)

    monkeypatch.setattr(
        main_mod,
        "threading",
        SimpleNamespace(Event=threading.Event, Lock=threading.Lock, Thread=_DeferredThread),
    )
    movement_manager_factory = MagicMock(
        side_effect=lambda **_kwargs: startup_operations.append("movement_manager_construct") or movement_manager
    )
    monkeypatch.setattr(moves_mod, "MovementManager", movement_manager_factory)
    monkeypatch.setattr(config_mod, "set_instance_path", MagicMock())
    monkeypatch.setattr(
        config_mod,
        "get_hf_connection_selection",
        MagicMock(return_value=SimpleNamespace(mode="test", has_target=False)),
    )
    monkeypatch.setattr(config_mod, "resolve_app_timeout_minutes", MagicMock(return_value=None))
    monkeypatch.setattr(
        startup_settings_mod,
        "StartupSettings",
        MagicMock(return_value=SimpleNamespace(voice=None)),
    )
    handlers = [MagicMock(), MagicMock()]
    handler_factory = MagicMock(side_effect=handlers)
    monkeypatch.setattr(huggingface_realtime_mod, "HuggingFaceRealtimeHandler", handler_factory)
    monkeypatch.setattr(console_mod, "LocalStream", MagicMock(return_value=stream_manager))
    monkeypatch.setattr(core_tools_mod, "initialize_tools", MagicMock())

    observed: dict[str, object] = {}

    def _launch() -> None:
        observed["startup_operations"] = startup_operations.copy()
        deps = handler_factory.call_args.args[0]
        if rebuild_handler:
            console_mod.LocalStream.call_args.kwargs["handler_factory"]()
        if graceful_shutdown:
            graceful_target = next(
                target
                for target in thread_targets
                if getattr(target, "__name__", "") == "poll_graceful_shutdown_event"
            )
            graceful_target()
        else:
            observed["result"] = deps.go_to_sleep()
            if sleep_fails or disable_fails:
                observed["retry_result"] = deps.go_to_sleep()
        observed["stop_event_set"] = stop_event.is_set() if stop_event is not None else False
        observed["graceful_shutdown_complete"] = (
            graceful_shutdown_complete_event.is_set() if graceful_shutdown_complete_event is not None else False
        )
        observed["quiesce_calls"] = stream_manager.quiesce_for_shutdown.call_count
        observed["stream_close_calls"] = stream_manager.close.call_count
        observed["daemon_stop_calls"] = request_stop_current_app.call_count
        observed["goto_sleep_calls"] = robot.goto_sleep.call_count
        observed["disable_motors_calls"] = robot.disable_motors.call_count
        observed["movement_stop_calls"] = movement_manager.stop.call_count

    stream_manager.launch.side_effect = _launch
    args = SimpleNamespace(debug=False, robot_name=None, no_camera=True, no_wobble=no_wobble, ui=False)
    main_mod.run(
        args,
        robot=robot,
        app_stop_event=stop_event,
        completed_utterance_observer=completed_utterance_observer,
        completed_utterance_timeout_seconds=completed_utterance_timeout_seconds,
        search_policy=search_policy,
        search_policy_timeout_seconds=search_policy_timeout_seconds,
        search_provider=search_provider,
        graceful_shutdown_event=graceful_shutdown_event,
        graceful_shutdown_complete_event=graceful_shutdown_complete_event,
    )
    observed["operations"] = operations
    observed["enable_wobbling_calls"] = robot.enable_wobbling.call_count
    observed["disable_motors_calls_after_shutdown"] = robot.disable_motors.call_count
    observed["handlers"] = handlers
    return observed


def test_completed_utterance_observer_attaches_to_initial_and_rebuilt_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional observer should survive runtime handler reconstruction."""
    observer = MagicMock()

    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        completed_utterance_observer=observer,
        completed_utterance_timeout_seconds=120.0,
        rebuild_handler=True,
    )

    for handler in observed["handlers"]:
        handler.set_completed_utterance_observer.assert_called_once_with(
            observer,
            timeout_seconds=120.0,
        )


def test_completed_utterance_observer_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standard app path should preserve the existing automatic response mode."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
    )

    observed["handlers"][0].set_completed_utterance_observer.assert_not_called()


def test_search_policy_attaches_to_initial_and_rebuilt_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional search policy should survive runtime handler reconstruction."""
    policy = MagicMock()

    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        search_policy=policy,
        search_policy_timeout_seconds=120.0,
        rebuild_handler=True,
    )

    for handler in observed["handlers"]:
        handler.set_search_policy.assert_called_once_with(policy, timeout_seconds=120.0)
        handler.set_search_space_gate.assert_called_once()
        assert callable(handler.set_search_space_gate.call_args.args[0])


def test_search_policy_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standard app path should not install a search boundary."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
    )

    observed["handlers"][0].set_search_policy.assert_not_called()
    observed["handlers"][0].set_search_space_gate.assert_not_called()
    observed["handlers"][0].set_search_provider.assert_not_called()


def test_search_provider_preserves_official_gate_for_request_local_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured provider should retain the official request-local option."""
    policy = MagicMock()
    provider = main_mod.SearchProvider(indicator_text="I'll check the configured search.", search=MagicMock())

    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        search_policy=policy,
        search_provider=provider,
        rebuild_handler=True,
    )

    for handler in observed["handlers"]:
        handler.set_search_policy.assert_called_once_with(policy, timeout_seconds=10.0)
        handler.set_search_provider.assert_called_once_with(provider)
        handler.set_search_space_gate.assert_called_once()
        assert callable(handler.set_search_space_gate.call_args.args[0])


def test_search_provider_requires_policy_before_robot_startup() -> None:
    """A transport cannot bypass the policy by being composed alone."""
    robot = MagicMock()
    provider = main_mod.SearchProvider(indicator_text="I'll check the configured search.", search=MagicMock())

    with pytest.raises(ValueError, match="requires a search policy"):
        main_mod.run(MagicMock(), robot=robot, search_provider=provider)

    assert robot.mock_calls == []


def test_invalid_search_provider_is_rejected_before_robot_startup() -> None:
    """Malformed provider fields cannot wake or initialize the robot."""
    robot = MagicMock()
    provider = main_mod.SearchProvider(indicator_text=" ", search=MagicMock())

    with pytest.raises(ValueError, match="search provider is invalid"):
        main_mod.run(MagicMock(), robot=robot, search_policy=MagicMock(), search_provider=provider)

    assert robot.mock_calls == []


@pytest.mark.parametrize(
    ("no_wobble", "expected_operations"),
    [
        (
            False,
            ["wake_check", "movement_manager_construct", "movement_manager_start", "enable_wobbling"],
        ),
        (
            True,
            ["disable_wobbling", "wake_check", "movement_manager_construct", "movement_manager_start"],
        ),
    ],
)
def test_wobble_mode_is_established_before_conversation_launch(
    monkeypatch: pytest.MonkeyPatch,
    no_wobble: bool,
    expected_operations: list[str],
) -> None:
    """Diagnostic mode disables wobble before wake and never enables it before launch."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        no_wobble=no_wobble,
    )

    assert observed["startup_operations"] == expected_operations
    assert observed["enable_wobbling_calls"] == (0 if no_wobble else 1)
    assert observed["disable_motors_calls_after_shutdown"] == 1


@pytest.mark.parametrize("failure", ["disable_wobbling", "wake_check"])
def test_startup_failure_aborts_before_movement_manager(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Wobble setup and wake-check failures must prevent later app-owned motion."""
    robot = MagicMock()
    wake_up_if_sleeping = MagicMock()
    movement_manager_factory = MagicMock()

    if failure == "disable_wobbling":
        robot.disable_wobbling.side_effect = RuntimeError("daemon unavailable")
        expected_error = "Failed to disable head wobbling"
    else:
        wake_up_if_sleeping.side_effect = RuntimeError("wake failed")
        expected_error = "wake failed"

    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", wake_up_if_sleeping)
    monkeypatch.setattr(moves_mod, "MovementManager", movement_manager_factory)
    monkeypatch.setattr(config_mod, "set_instance_path", MagicMock())
    monkeypatch.setattr(
        config_mod,
        "get_hf_connection_selection",
        MagicMock(return_value=SimpleNamespace(mode="test", has_target=False)),
    )
    monkeypatch.setattr(
        startup_settings_mod,
        "StartupSettings",
        MagicMock(return_value=SimpleNamespace(voice=None)),
    )

    args = SimpleNamespace(debug=False, robot_name=None, no_camera=True, no_wobble=True, ui=False)
    with pytest.raises(RuntimeError, match=expected_error):
        main_mod.run(args, robot=robot)

    if failure == "disable_wobbling":
        wake_up_if_sleeping.assert_not_called()
    movement_manager_factory.assert_not_called()
    robot.disable_motors.assert_not_called()


@pytest.mark.parametrize("use_stop_event", [True, False])
def test_sleep_success_requests_app_stop_after_motor_disable(
    monkeypatch: pytest.MonkeyPatch,
    use_stop_event: bool,
) -> None:
    """The app should stop locally and remotely only after sleep is limp."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=use_stop_event,
    )

    assert observed["result"] == {
        "status": "sleeping",
        "stop_current_app_requested": True,
        "local_stop_requested": True,
    }
    local_stop = "local_stop_event" if use_stop_event else "local_stream_close"
    assert observed["operations"] == ["sleep", "disable_motors", "stop", local_stop]
    assert observed["stop_event_set"] is use_stop_event
    assert observed["stream_close_calls"] == (0 if use_stop_event else 1)
    assert observed["daemon_stop_calls"] == 1
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == 1
    assert observed["disable_motors_calls_after_shutdown"] == 1
    assert observed["movement_stop_calls"] == 1


def test_graceful_shutdown_quiesces_before_sleep_and_signals_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A service stop is acknowledged only after quiesce, sleep, and motor disable."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        graceful_shutdown=True,
    )

    assert observed["operations"] == [
        "quiesce",
        "sleep",
        "disable_motors",
        "stop",
        "local_stream_close",
    ]
    assert observed["graceful_shutdown_complete"] is True
    assert observed["quiesce_calls"] == 1
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == 1


def test_graceful_shutdown_quiesce_failure_never_starts_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure to stop conversation activity must leave torque unchanged."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        graceful_shutdown=True,
        quiesce_succeeds=False,
    )

    assert observed["operations"] == ["quiesce"]
    assert observed["graceful_shutdown_complete"] is False
    assert observed["goto_sleep_calls"] == 0
    assert observed["disable_motors_calls"] == 0
    assert observed["movement_stop_calls"] == 1


@pytest.mark.parametrize(
    ("sleep_fails", "disable_fails", "expected_operations"),
    [
        (True, False, ["quiesce", "sleep"]),
        (False, True, ["quiesce", "sleep", "disable_motors"]),
    ],
)
def test_graceful_shutdown_sleep_failure_is_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
    sleep_fails: bool,
    disable_fails: bool,
    expected_operations: list[str],
) -> None:
    """A failed rest or motor transition must keep the shutdown gate closed."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=sleep_fails,
        disable_fails=disable_fails,
        use_stop_event=False,
        graceful_shutdown=True,
    )

    assert observed["operations"] == expected_operations
    assert observed["graceful_shutdown_complete"] is False
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == int(disable_fails)


@pytest.mark.parametrize("use_stop_event", [True, False])
def test_sleep_failure_keeps_app_running_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    use_stop_event: bool,
) -> None:
    """A failed sleep move should preserve support and leave the app alive."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=True,
        use_stop_event=use_stop_event,
    )

    assert observed["result"] == {
        "status": "sleep_failed",
        "stop_current_app_requested": False,
        "local_stop_requested": False,
        "error": "go_to_sleep movement failed: RuntimeError: motor fault",
    }
    assert observed["retry_result"] == observed["result"]
    assert observed["operations"] == ["sleep"]
    assert observed["stop_event_set"] is False
    assert observed["stream_close_calls"] == 0
    assert observed["daemon_stop_calls"] == 0
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == 0
    assert observed["movement_stop_calls"] == 1


@pytest.mark.parametrize("use_stop_event", [True, False])
def test_motor_disable_failure_keeps_app_running_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    use_stop_event: bool,
) -> None:
    """A failed motor disable should leave the app alive and the request latched."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        disable_fails=True,
        use_stop_event=use_stop_event,
    )

    assert observed["result"] == {
        "status": "sleep_failed",
        "stop_current_app_requested": False,
        "local_stop_requested": False,
        "error": "go_to_sleep motor disable failed: RuntimeError: torque fault",
    }
    assert observed["retry_result"] == observed["result"]
    assert observed["operations"] == ["sleep", "disable_motors"]
    assert observed["stop_event_set"] is False
    assert observed["stream_close_calls"] == 0
    assert observed["daemon_stop_calls"] == 0
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == 1
    assert observed["movement_stop_calls"] == 1
