"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.main as main_mod
import reachy_mini_conversation_app.moves as moves_mod
import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.console as console_mod
import reachy_mini_conversation_app.startup_settings as startup_settings_mod
import reachy_mini_conversation_app.tools.core_tools as core_tools_mod
import reachy_mini_conversation_app.huggingface_realtime as huggingface_realtime_mod


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
) -> dict[str, object]:
    """Run the app through one go_to_sleep tool call with hardware-free doubles."""
    operations: list[str] = []
    robot = MagicMock()

    def _goto_sleep() -> None:
        operations.append("sleep")
        if sleep_fails:
            raise RuntimeError("motor fault")

    robot.goto_sleep.side_effect = _goto_sleep
    movement_manager = MagicMock()
    stream_manager = MagicMock()
    stream_manager.close.side_effect = lambda: operations.append("local_stream_close")

    class _RecordingStopEvent(threading.Event):
        def set(self) -> None:
            operations.append("local_stop_event")
            super().set()

    stop_event = _RecordingStopEvent() if use_stop_event else None
    request_stop_current_app = MagicMock(side_effect=lambda _robot, _logger: operations.append("stop") or True)
    monkeypatch.setattr(main_mod.app_lifecycle, "request_stop_current_app", request_stop_current_app)
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod.time, "sleep", MagicMock())
    monkeypatch.setattr(main_mod.threading, "Thread", MagicMock())
    monkeypatch.setattr(moves_mod, "MovementManager", MagicMock(return_value=movement_manager))
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
    handler_factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(huggingface_realtime_mod, "HuggingFaceRealtimeHandler", handler_factory)
    monkeypatch.setattr(console_mod, "LocalStream", MagicMock(return_value=stream_manager))
    monkeypatch.setattr(core_tools_mod, "initialize_tools", MagicMock())

    observed: dict[str, object] = {}

    def _launch() -> None:
        deps = handler_factory.call_args.args[0]
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
    args = SimpleNamespace(debug=False, robot_name=None, no_camera=True, ui=False)
    main_mod.run(args, robot=robot, app_stop_event=stop_event)
    observed["operations"] = operations
    return observed


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
    assert observed["retry_result"] == {"status": "already_requested"}
    assert observed["operations"] == ["sleep"]
    assert observed["stop_event_set"] is False
    assert observed["stream_close_calls"] == 0
    assert observed["daemon_stop_calls"] == 0
    assert observed["goto_sleep_calls"] == 1
    assert observed["disable_motors_calls"] == 0
    assert observed["movement_stop_calls"] == 1
