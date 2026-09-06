"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from pathlib import Path
from argparse import Namespace
from unittest.mock import MagicMock, call, patch

import pytest

import reachy_mini_conversation_app.main as main_mod
from reachy_mini_conversation_app.config import config


@pytest.mark.parametrize(
    "flag, no_camera, enabled",
    [
        (None, False, False),
        ("false", False, False),
        ("true", False, True),
        ("invalid", False, False),
        ("true", True, False),
    ],
)
def test_startup_head_tracking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str | None, no_camera: bool, enabled: bool
) -> None:
    """Startup honors late instance configuration and camera availability before streaming."""
    monkeypatch.delenv("REACHY_MINI_HEAD_TRACKING", raising=False)
    monkeypatch.setenv("REACHY_MINI_APP_TIMEOUT_MINUTES", "0")
    (tmp_path / ".env").write_text("" if flag is None else f"REACHY_MINI_HEAD_TRACKING={flag}\n", encoding="utf-8")
    args = Namespace(debug=False, robot_name=None, no_camera=no_camera, ui=False)
    with (
        patch.dict(config.__dict__),
        patch("reachy_mini_conversation_app.moves.MovementManager") as movement_class,
        patch("reachy_mini_conversation_app.console.LocalStream") as stream_class,
        patch("reachy_mini_conversation_app.huggingface_realtime.HuggingFaceRealtimeHandler"),
        patch("reachy_mini_conversation_app.startup_settings.load_startup_settings_into_runtime") as settings,
        patch.object(main_mod.app_lifecycle, "wake_up_if_sleeping"),
        patch.object(main_mod.app_lifecycle, "initialize_tools_with_default_fallback"),
        patch.object(main_mod.time, "sleep"),
    ):
        config.REACHY_MINI_HEAD_TRACKING = False
        settings.return_value = SimpleNamespace(voice=None)
        startup = MagicMock()
        startup.attach_mock(movement_class.return_value, "movement")
        startup.attach_mock(stream_class.return_value, "stream")

        main_mod.run(args, robot=MagicMock(), instance_path=str(tmp_path))

        expected = [call.movement.start()]
        if enabled:
            expected.append(call.movement.set_head_tracking(True))
        else:
            movement_class.return_value.set_head_tracking.assert_not_called()
        expected.extend([call.stream.launch(), call.movement.stop(reset_to_neutral=False)])
        assert startup.mock_calls == expected


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
