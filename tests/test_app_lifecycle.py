from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call

import numpy as np
import pytest

from reachy_mini.io.protocol import MotorControlMode
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def test_request_stop_current_app_posts_to_daemon(monkeypatch) -> None:
    """The app stop request should call the connected Reachy daemon endpoint."""

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://192.168.1.42:8000/api/apps/stop-current-app"
        assert request.get_method() == "POST"
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert app_lifecycle.request_stop_current_app(robot, MagicMock())


def test_wake_up_if_sleeping_enables_motors_before_wake_up() -> None:
    """Startup should enable sleeping motors before playing the wake-up movement."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    robot.client.get_status.assert_not_called()
    assert robot.method_calls == [
        call.get_current_head_pose(),
        call.enable_motors(),
        call.wake_up(),
    ]


@pytest.mark.parametrize(
    "motor_mode",
    [MotorControlMode.Enabled, MotorControlMode.GravityCompensation],
)
def test_wake_up_if_sleeping_skips_active_motors_outside_sleep_pose(
    motor_mode: MotorControlMode,
) -> None:
    """Startup should leave active motors outside the sleep pose alone."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.client.get_status.return_value = SimpleNamespace(
        backend_status=SimpleNamespace(motor_control_mode=motor_mode)
    )

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_enables_disabled_motors_in_place() -> None:
    """A displaced limp robot should regain torque before app-owned motion starts."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.client.get_status.side_effect = [
        SimpleNamespace(backend_status=SimpleNamespace(motor_control_mode=MotorControlMode.Disabled)),
        SimpleNamespace(backend_status=SimpleNamespace(motor_control_mode=MotorControlMode.Disabled)),
        SimpleNamespace(backend_status=SimpleNamespace(motor_control_mode=MotorControlMode.Enabled)),
    ]

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_called_once_with()
    robot.wake_up.assert_not_called()
    assert robot.method_calls == [
        call.get_current_head_pose(),
        call.client.get_status(),
        call.enable_motors(),
        call.client.get_status(timeout=ANY),
        call.client.get_status(timeout=ANY),
    ]


@pytest.mark.parametrize(
    "status_outcome",
    [
        TimeoutError("unavailable"),
        SimpleNamespace(backend_status=None),
    ],
)
def test_wake_up_if_sleeping_raises_when_motor_status_is_unavailable(status_outcome: object) -> None:
    """Unknown torque state must abort before app-owned motion starts."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    if isinstance(status_outcome, Exception):
        robot.client.get_status.side_effect = status_outcome
    else:
        robot.client.get_status.return_value = status_outcome

    with pytest.raises(RuntimeError, match="Could not read robot motor mode"):
        app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_raises_when_in_place_enable_fails() -> None:
    """A failed in-place enable must abort before later app-owned motion."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.client.get_status.return_value = SimpleNamespace(
        backend_status=SimpleNamespace(motor_control_mode=MotorControlMode.Disabled)
    )
    robot.enable_motors.side_effect = RuntimeError("motor fault")

    with pytest.raises(RuntimeError, match="Failed to confirm enabled robot motors"):
        app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_called_once_with()
    robot.wake_up.assert_not_called()


@pytest.mark.parametrize(
    "confirmation_outcome",
    [
        TimeoutError("unavailable"),
        SimpleNamespace(backend_status=None),
        SimpleNamespace(backend_status=SimpleNamespace(motor_control_mode=MotorControlMode.GravityCompensation)),
    ],
)
def test_wake_up_if_sleeping_requires_enabled_status_after_send(
    confirmation_outcome: object,
) -> None:
    """A sent torque command is not success until a fresh status confirms it."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    disabled = SimpleNamespace(backend_status=SimpleNamespace(motor_control_mode=MotorControlMode.Disabled))
    robot.client.get_status.side_effect = [disabled, confirmation_outcome]

    with pytest.raises(RuntimeError, match="Failed to confirm enabled robot motors"):
        app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_called_once_with()
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_raises_when_pose_read_fails() -> None:
    """Startup must abort instead of handing an unknown pose to the movement manager."""
    robot = MagicMock()
    robot.get_current_head_pose.side_effect = ConnectionError("unavailable")

    with pytest.raises(RuntimeError, match="Could not read a valid robot pose"):
        app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


@pytest.mark.parametrize(
    "head_pose",
    [
        np.zeros((3, 3)),
        np.full((4, 4), np.nan),
    ],
)
def test_wake_up_if_sleeping_raises_for_invalid_pose(head_pose: np.ndarray) -> None:
    """Malformed pose data must not be treated as a valid awake pose."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = head_pose

    with pytest.raises(RuntimeError, match="Could not read a valid robot pose"):
        app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_raises_when_wake_fails() -> None:
    """A failed wake must abort before later app-owned motion can start."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    robot.wake_up.side_effect = RuntimeError("motor fault")

    with pytest.raises(RuntimeError, match="Failed to run wake-up movement"):
        app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_called_once_with()
    robot.wake_up.assert_called_once_with()


def test_run_go_to_sleep_tool_uses_runtime_callback() -> None:
    """Synchronous lifecycle paths should enter through the go_to_sleep tool."""
    expected = {"status": "sleeping"}
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert result == expected
    go_to_sleep.assert_called_once_with()
