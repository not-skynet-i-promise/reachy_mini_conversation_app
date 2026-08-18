import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app.moves import BreathingMove, MovementManager
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _neutral_robot() -> MagicMock:
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    return robot


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = _neutral_robot()
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_manager_starts_from_robot_readback() -> None:
    """The first control target should preserve the pose left by wake-up."""
    robot = MagicMock()
    current_head_pose = create_head_pose(0, 0, 5, 0, 0, 0, degrees=True, mm=True)
    robot.get_current_head_pose.return_value = current_head_pose
    robot.get_current_joint_positions.return_value = ([0.1] + [0.0] * 6, [-0.2, 0.2])

    manager = MovementManager(robot)

    head, antennas, body_yaw = manager._get_primary_pose(manager._now())
    assert np.allclose(head, current_head_pose)
    assert antennas == (-0.2, 0.2)
    assert body_yaw == 0.1


def test_manager_fails_closed_without_initial_readback() -> None:
    """A missing initial pose should prevent the control loop from starting."""
    robot = _neutral_robot()
    robot.get_current_head_pose.side_effect = RuntimeError("readback unavailable")

    with pytest.raises(RuntimeError, match="readback unavailable"):
        MovementManager(robot)

    robot.set_target.assert_not_called()


def test_idle_pose_holds_stable_neutral_after_interpolation() -> None:
    """Idle should settle at the neutral head and antenna pose without periodic motion."""
    neutral_antennas = (-0.1745, 0.1745)
    move = BreathingMove(np.eye(4), (-0.3, 0.3))

    tick = 1.0 / 60.0
    _, before_handoff, _ = move.evaluate(move.interpolation_duration - tick)
    head_at_handoff, at_handoff, body_at_handoff = move.evaluate(move.interpolation_duration)
    head_later, antennas_later, body_later = move.evaluate(move.interpolation_duration + 10.0)

    assert before_handoff is not None
    assert head_at_handoff is not None
    assert at_handoff is not None
    assert head_later is not None
    assert antennas_later is not None
    assert np.max(np.abs(at_handoff - before_handoff)) < np.deg2rad(0.1)
    assert np.allclose(head_at_handoff, np.eye(4))
    assert np.allclose(at_handoff, neutral_antennas)
    assert body_at_handoff == 0.0
    assert np.allclose(head_later, head_at_handoff)
    assert np.allclose(antennas_later, at_handoff)
    assert body_later == body_at_handoff


def test_diagnostic_antenna_expression_is_mirrored_and_continuous() -> None:
    """The opt-in diagnostic should ramp a mirrored sway around official neutral."""
    neutral_right, neutral_left = (-0.1745, 0.1745)
    move = BreathingMove(
        np.eye(4),
        (-0.3, 0.3),
        diagnostic_antenna_expression=True,
    )

    tick = 1.0 / 60.0
    _, before_handoff, _ = move.evaluate(move.interpolation_duration - tick)
    head_at_handoff, at_handoff, body_at_handoff = move.evaluate(move.interpolation_duration)
    head_after_tick, after_tick, body_after_tick = move.evaluate(move.interpolation_duration + tick)
    head_at_peak, at_peak, body_at_peak = move.evaluate(move.interpolation_duration + 2.5)

    assert before_handoff is not None
    assert head_at_handoff is not None
    assert at_handoff is not None
    assert head_after_tick is not None
    assert after_tick is not None
    assert head_at_peak is not None
    assert at_peak is not None
    assert np.allclose(at_handoff, (neutral_right, neutral_left))
    assert np.max(np.abs(after_tick - at_handoff)) < np.deg2rad(0.1)
    velocity_before = (at_handoff - before_handoff) / tick
    velocity_after = (after_tick - at_handoff) / tick
    assert np.max(np.abs(velocity_after - velocity_before)) < np.deg2rad(1.0)
    assert at_peak[0] - neutral_right == pytest.approx(-(at_peak[1] - neutral_left))
    assert at_peak[0] - neutral_right == pytest.approx(np.deg2rad(5))
    assert at_peak[0] < 0 < at_peak[1]
    assert np.allclose(head_at_handoff, np.eye(4))
    assert np.allclose(head_after_tick, head_at_handoff)
    assert np.allclose(head_at_peak, head_at_handoff)
    assert body_at_handoff == body_after_tick == body_at_peak == 0.0


def test_listening_freezes_and_continuously_releases_diagnostic_antennas() -> None:
    """Listening should hold both channels and blend from that exact hold on release."""
    manager = MovementManager(_neutral_robot(), diagnostic_antenna_expression=True)
    now = [10.0]
    manager._now = lambda: now[0]
    manager._listening_debounce_s = 0.0
    manager._last_listening_toggle_time = 0.0
    frozen = (-0.1, 0.1)
    next_target = (-0.3, 0.3)
    manager._last_commanded_pose = (np.eye(4), frozen, 0.0)

    manager.set_listening(True)
    manager._poll_signals(now[0])
    manager._publish_shared_state()
    assert manager._calculate_blended_antennas(next_target) == frozen

    now[0] += 0.2
    assert manager._calculate_blended_antennas((0.05, -0.05)) == frozen
    manager.set_listening(False)
    manager._poll_signals(now[0])
    manager._publish_shared_state()
    assert manager._calculate_blended_antennas(next_target) == frozen

    now[0] += manager._antenna_blend_duration / 2.0
    halfway = manager._calculate_blended_antennas(next_target)
    assert halfway == pytest.approx((-0.2, 0.2))


def test_stop_ends_diagnostic_target_writes() -> None:
    """Stopping the movement owner should prevent every later target write."""
    robot = _neutral_robot()
    manager = MovementManager(robot, diagnostic_antenna_expression=True)
    manager.idle_inactivity_delay = 0.0
    manager._manage_breathing(manager._now())
    assert isinstance(manager.move_queue[0], BreathingMove)
    assert manager.move_queue[0].diagnostic_antenna_expression is True

    manager.start()
    assert _wait_for(lambda: robot.set_target.call_count > 0)
    manager.stop(reset_to_neutral=False)
    calls_at_stop = robot.set_target.call_count
    time.sleep(0.05)

    assert manager._thread is None
    assert robot.set_target.call_count == calls_at_stop


def test_breathing_interpolates_body_yaw_to_neutral() -> None:
    """Breathing should not reset a nonzero body yaw in its first tick."""
    move = BreathingMove(np.eye(4), (-0.1745, 0.1745), interpolation_start_body_yaw=0.2)

    _, _, at_start = move.evaluate(0.0)
    _, _, at_quarter = move.evaluate(move.interpolation_duration / 4.0)
    _, _, at_midpoint = move.evaluate(move.interpolation_duration / 2.0)
    _, _, at_handoff = move.evaluate(move.interpolation_duration)

    assert at_start == 0.2
    assert at_quarter == pytest.approx(0.179296875)
    assert at_midpoint == 0.1
    assert at_handoff == 0.0


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    manager = MovementManager(robot)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = _neutral_robot()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_speaking_anchor_holds_during_idle_move() -> None:
    """Speaking should not drop a tracked head pose when the idle move is active."""
    robot = _neutral_robot()
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor
    manager.state.current_move = BreathingMove(np.eye(4), (-0.1745, 0.1745))
    manager.state.move_start_time = manager._now() - 2.0

    head, antennas, body_yaw = manager._get_primary_pose(manager._now())

    assert np.allclose(head, anchor)
    assert antennas == (-0.1745, 0.1745)
    assert body_yaw == 0.0
