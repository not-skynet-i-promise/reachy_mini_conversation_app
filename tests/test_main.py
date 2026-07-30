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

    run.assert_called_once_with(args, completed_utterance_observer=observer)


def test_public_observer_annotations_are_runtime_resolvable() -> None:
    """Composition tooling should be able to inspect the public callback type."""
    assert "completed_utterance_observer" in typing.get_type_hints(main_mod.main)
    assert "completed_utterance_observer" in typing.get_type_hints(main_mod.run)


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
    use_stop_event: bool,
    no_wobble: bool = False,
    completed_utterance_observer: object | None = None,
    rebuild_handler: bool = False,
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
    movement_manager = MagicMock()
    movement_manager.start.side_effect = lambda: startup_operations.append("movement_manager_start")
    stream_manager = MagicMock()
    stream_manager.close.side_effect = lambda: operations.append("local_stream_close")

    class _RecordingStopEvent(threading.Event):
        def set(self) -> None:
            operations.append("local_stop_event")
            super().set()

    stop_event = _RecordingStopEvent() if use_stop_event else None
    request_stop_current_app = MagicMock(side_effect=lambda _robot, _logger: operations.append("stop") or True)
    monkeypatch.setattr(main_mod.app_lifecycle, "request_stop_current_app", request_stop_current_app)
    monkeypatch.setattr(
        main_mod.app_lifecycle,
        "wake_up_if_sleeping",
        MagicMock(side_effect=lambda *_args: startup_operations.append("wake_check")),
    )
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod.time, "sleep", MagicMock())
    monkeypatch.setattr(main_mod.threading, "Thread", MagicMock())
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
        observed["result"] = deps.go_to_sleep()
        if sleep_fails:
            observed["retry_result"] = deps.go_to_sleep()
        observed["stop_event_set"] = stop_event.is_set() if stop_event is not None else False
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
        rebuild_handler=True,
    )

    for handler in observed["handlers"]:
        handler.set_completed_utterance_observer.assert_called_once_with(observer)


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
    assert observed["disable_motors_calls_after_shutdown"] == 0


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
def test_sleep_success_requests_app_stop_after_movement(
    monkeypatch: pytest.MonkeyPatch,
    use_stop_event: bool,
) -> None:
    """The app should stop locally and remotely only after a successful sleep move."""
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
    assert observed["operations"] == ["sleep", "stop", local_stop]
    assert observed["stop_event_set"] is use_stop_event
    assert observed["stream_close_calls"] == (0 if use_stop_event else 1)
    assert observed["daemon_stop_calls"] == 1
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == 0
    assert observed["movement_stop_calls"] == 1


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
