"""Tests for app-level runtime behavior."""

import os
import sys
import typing
import threading
import subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

import reachy_mini_conversation_app.main as main_mod
import reachy_mini_conversation_app.moves as moves_mod
import reachy_mini_conversation_app.utils as utils_mod
import reachy_mini_conversation_app.config as config_mod
import reachy_mini_conversation_app.console as console_mod
import reachy_mini_conversation_app.startup_settings as startup_settings_mod
import reachy_mini_conversation_app.huggingface_realtime as huggingface_realtime_mod


_POSIX_RECOVERY_ACK = pytest.mark.skipif(
    os.name != "posix",
    reason="the inherited recovery acknowledgment pipe is a POSIX maintenance capability",
)


def _recovery_ack_args() -> SimpleNamespace:
    return SimpleNamespace(
        debug=False,
        robot_name=None,
        robot_host=None,
        no_camera=True,
        no_wobble=False,
        diagnostic_antenna_expression=False,
        ui=False,
    )


def _patch_recovery_ack_preconstructor(
    monkeypatch: pytest.MonkeyPatch,
    constructor: MagicMock,
) -> MagicMock:
    logger = MagicMock()
    monkeypatch.setattr(main_mod, "ReachyMini", constructor)
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
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
    return logger


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
        diagnostic_antenna_expression=False,
        ui=False,
    )

    with pytest.raises(ConstructionObserved):
        main_mod.run(args)

    constructor.assert_called_once_with(**expected_kwargs)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_acknowledges_after_sdk_connection_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The maintenance proof must sit exactly between SDK connection and setup."""

    class SetupObserved(BaseException):
        pass

    read_descriptor, write_descriptor = os.pipe()
    nonce = bytes(range(32))
    expected_record = b"RCA1" + nonce
    robot = MagicMock()
    order: list[str] = []

    def construct_robot(**_kwargs: object) -> MagicMock:
        order.append("connected")
        return robot

    original_write = os.write

    def write_record(descriptor: int, record: bytes) -> int:
        order.append("acknowledged")
        return original_write(descriptor, record)

    def observe_first_setup(_robot: MagicMock, _logger: MagicMock) -> None:
        order.append("setup")
        assert os.read(read_descriptor, len(expected_record)) == expected_record
        assert os.read(read_descriptor, 1) == b""
        raise SetupObserved

    constructor = MagicMock(side_effect=construct_robot)
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setattr(main_mod.os, "write", MagicMock(side_effect=write_record))
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", observe_first_setup)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, nonce.hex())

    try:
        with pytest.raises(SetupObserved):
            main_mod.run(_recovery_ack_args())
    finally:
        os.close(read_descriptor)

    assert order == ["connected", "acknowledged", "setup"]
    main_mod.os.write.assert_called_once_with(write_descriptor, expected_record)
    with pytest.raises(OSError):
        os.fstat(write_descriptor)
    assert main_mod._RECOVERY_CONNECTION_ACK_FD_ENV not in os.environ
    assert main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV not in os.environ


@pytest.mark.parametrize(
    ("encoded_descriptor", "encoded_nonce"),
    [
        (None, "00" * 32),
        ("", "00" * 32),
        ("+4", "00" * 32),
        ("04", "00" * 32),
        ("-1", "00" * 32),
        ("2", "00" * 32),
        ("٤", "00" * 32),
        ("99999999999", "00" * 32),
    ],
)
def test_recovery_connection_ack_rejects_invalid_descriptor_before_sdk_connection(
    monkeypatch: pytest.MonkeyPatch,
    encoded_descriptor: str | None,
    encoded_nonce: str,
) -> None:
    """Malformed descriptor authority cannot reach robot construction."""
    constructor = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    if encoded_descriptor is not None:
        monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, encoded_descriptor)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, encoded_nonce)

    with pytest.raises(RuntimeError, match="Invalid recovery connection acknowledgment configuration"):
        main_mod.run(_recovery_ack_args())

    constructor.assert_not_called()
    assert main_mod._RECOVERY_CONNECTION_ACK_FD_ENV not in os.environ
    assert main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV not in os.environ


@pytest.mark.parametrize(
    "encoded_nonce",
    [
        None,
        "",
        "00" * 31,
        "00" * 32 + "0",
        "AA" * 32,
        "gg" * 32,
        " " + "00" * 32,
    ],
)
def test_recovery_connection_ack_rejects_invalid_nonce_and_closes_pipe(
    monkeypatch: pytest.MonkeyPatch,
    encoded_nonce: str | None,
) -> None:
    """Invalid nonce authority must close the inherited pipe before startup."""
    read_descriptor, write_descriptor = os.pipe()
    constructor = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    if encoded_nonce is not None:
        monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, encoded_nonce)

    try:
        with pytest.raises(RuntimeError, match="Invalid recovery connection acknowledgment configuration"):
            main_mod.run(_recovery_ack_args())
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)

    constructor.assert_not_called()
    with pytest.raises(OSError):
        os.fstat(write_descriptor)


def test_recovery_connection_ack_rejects_non_pipe_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A regular file cannot receive the recovery connection proof."""
    descriptor = os.open(tmp_path / "not-a-pipe", os.O_CREAT | os.O_WRONLY, 0o600)
    constructor = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    with pytest.raises(RuntimeError, match="Invalid recovery connection acknowledgment configuration"):
        main_mod.run(_recovery_ack_args())

    constructor.assert_not_called()
    with pytest.raises(OSError):
        os.fstat(descriptor)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_owns_and_cloexecs_before_fstat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a failing descriptor inspection occurs after ownership and CLOEXEC."""
    read_descriptor, write_descriptor = os.pipe()
    os.set_inheritable(write_descriptor, True)
    original_fstat = os.fstat

    def fail_fstat(descriptor: int) -> os.stat_result:
        assert descriptor == write_descriptor
        assert not os.get_inheritable(descriptor)
        raise OSError("inspection failed")

    monkeypatch.setattr(main_mod.os, "fstat", fail_fstat)
    environment = {
        main_mod._RECOVERY_CONNECTION_ACK_FD_ENV: str(write_descriptor),
        main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV: "00" * 32,
    }

    try:
        with pytest.raises(RuntimeError, match="Invalid recovery connection acknowledgment configuration"):
            main_mod._consume_recovery_connection_acknowledgment_environment(environment)
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)

    with pytest.raises(OSError):
        original_fstat(write_descriptor)


def test_recovery_connection_ack_fails_closed_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported pipe semantics cannot leave the inherited descriptor open."""
    read_descriptor, write_descriptor = os.pipe()
    environment = {
        main_mod._RECOVERY_CONNECTION_ACK_FD_ENV: str(write_descriptor),
        main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV: "00" * 32,
    }
    monkeypatch.setattr(main_mod.os, "name", "unsupported")

    try:
        with pytest.raises(RuntimeError, match="Invalid recovery connection acknowledgment configuration"):
            main_mod._consume_recovery_connection_acknowledgment_environment(environment)
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)

    with pytest.raises(OSError):
        os.fstat(write_descriptor)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_rejects_pipe_read_end_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipe read end cannot impersonate the supervisor's writer."""
    read_descriptor, write_descriptor = os.pipe()
    constructor = MagicMock(return_value=MagicMock())
    setup = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", setup)
    stale_exit_terminator = MagicMock()
    monkeypatch.setattr(main_mod, "_STALE_CONNECTION_TERMINATOR", stale_exit_terminator)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(read_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    try:
        with pytest.raises(RuntimeError, match="Recovery connection acknowledgment failed"):
            main_mod.run(
                _recovery_ack_args(),
                stale_connection_exit_event=threading.Event(),
            )
    finally:
        os.close(write_descriptor)

    setup.assert_not_called()
    stale_exit_terminator.assert_not_called()
    with pytest.raises(OSError):
        os.fstat(read_descriptor)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_full_pipe_fails_without_blocking_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backpressure fails closed instead of stalling connected startup."""
    read_descriptor, write_descriptor = os.pipe()
    os.set_blocking(write_descriptor, False)
    while True:
        try:
            os.write(write_descriptor, b"x" * 4096)
        except BlockingIOError:
            break
    constructor = MagicMock(return_value=MagicMock())
    setup = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", setup)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    try:
        with pytest.raises(RuntimeError, match="Recovery connection acknowledgment failed"):
            main_mod.run(_recovery_ack_args())
    finally:
        os.close(read_descriptor)

    setup.assert_not_called()
    with pytest.raises(OSError):
        os.fstat(write_descriptor)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_rejects_injected_robot_and_closes_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied robot cannot claim a constructor connection proof."""
    read_descriptor, write_descriptor = os.pipe()
    constructor = MagicMock()
    robot = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    try:
        with pytest.raises(RuntimeError, match="requires app-owned robot initialization"):
            main_mod.run(_recovery_ack_args(), robot=robot)
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)

    constructor.assert_not_called()
    assert robot.mock_calls == []
    with pytest.raises(OSError):
        os.fstat(write_descriptor)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_closes_pipe_when_sdk_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed SDK connection cannot leave a reusable acknowledgment pipe."""

    class ConstructionFailed(BaseException):
        pass

    read_descriptor, write_descriptor = os.pipe()
    constructor = MagicMock(side_effect=ConstructionFailed)
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    try:
        with pytest.raises(ConstructionFailed):
            main_mod.run(_recovery_ack_args())
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)

    with pytest.raises(OSError):
        os.fstat(write_descriptor)


@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_is_owned_before_preconstructor_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setup failure cannot retain or inherit the already captured proof pipe."""

    class SetupFailed(BaseException):
        pass

    read_descriptor, write_descriptor = os.pipe()
    os.set_inheritable(write_descriptor, True)
    constructor = MagicMock()
    monkeypatch.setattr(main_mod, "ReachyMini", constructor)

    def fail_setup_logger(_debug: bool) -> MagicMock:
        assert not os.get_inheritable(write_descriptor)
        raise SetupFailed

    monkeypatch.setattr(main_mod, "setup_logger", fail_setup_logger)
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    try:
        with pytest.raises(SetupFailed):
            main_mod.run(_recovery_ack_args())
        assert os.read(read_descriptor, 1) == b""
    finally:
        os.close(read_descriptor)

    constructor.assert_not_called()
    with pytest.raises(OSError):
        os.fstat(write_descriptor)


@pytest.mark.parametrize("write_outcome", ["error", "partial"])
@_POSIX_RECOVERY_ACK
def test_recovery_connection_ack_write_failure_stops_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    write_outcome: str,
) -> None:
    """A missing atomic record cannot continue into robot setup."""
    read_descriptor, write_descriptor = os.pipe()
    constructor = MagicMock(return_value=MagicMock())
    setup = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", setup)
    if write_outcome == "error":
        monkeypatch.setattr(main_mod.os, "write", MagicMock(side_effect=OSError))
    else:
        monkeypatch.setattr(main_mod.os, "write", MagicMock(return_value=35))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(write_descriptor))
    monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    try:
        with pytest.raises(RuntimeError, match="Recovery connection acknowledgment failed"):
            main_mod.run(_recovery_ack_args())
    finally:
        os.close(read_descriptor)

    setup.assert_not_called()
    with pytest.raises(OSError):
        os.fstat(write_descriptor)


@pytest.mark.parametrize("load_instance_runtime_settings", [True, False])
def test_run_can_keep_instance_storage_without_loading_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    load_instance_runtime_settings: bool,
) -> None:
    """Programmatic callers can isolate instance storage from persisted settings."""

    class StreamObserved(BaseException):
        pass

    instance_path = str(tmp_path)
    (tmp_path / ".env").write_text("HF_REALTIME_CONNECTION_MODE=deployed\n", encoding="utf-8")
    (tmp_path / "startup_settings.json").write_text(
        '{"profile":"user_personalities/unreviewed"}\n',
        encoding="utf-8",
    )
    load_dotenv = MagicMock()
    load_startup_settings = MagicMock(return_value=SimpleNamespace(voice=None))
    set_instance_path = MagicMock()
    stream_constructor = MagicMock(side_effect=StreamObserved)
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(moves_mod, "MovementManager", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(console_mod, "LocalStream", stream_constructor)
    monkeypatch.setattr(huggingface_realtime_mod, "HuggingFaceRealtimeHandler", MagicMock())
    monkeypatch.setattr(config_mod, "set_instance_path", set_instance_path)
    monkeypatch.setattr(config_mod, "refresh_runtime_config_from_env", MagicMock())
    monkeypatch.setattr(
        config_mod,
        "get_hf_connection_selection",
        MagicMock(return_value=SimpleNamespace(mode="test", has_target=False)),
    )
    monkeypatch.setattr(config_mod, "load_dotenv", load_dotenv)
    monkeypatch.setattr(startup_settings_mod, "load_startup_settings_into_runtime", load_startup_settings)

    args = SimpleNamespace(
        debug=False,
        robot_name=None,
        robot_host=None,
        no_camera=True,
        no_wobble=False,
        diagnostic_antenna_expression=False,
        ui=False,
    )
    with pytest.raises(StreamObserved):
        main_mod.run(
            args,
            robot=MagicMock(),
            instance_path=instance_path,
            load_instance_runtime_settings=load_instance_runtime_settings,
        )

    set_instance_path.assert_called_once_with(instance_path)
    if load_instance_runtime_settings:
        load_dotenv.assert_called_once()
        load_startup_settings.assert_called_once_with(instance_path)
    else:
        load_dotenv.assert_not_called()
        load_startup_settings.assert_not_called()
    assert stream_constructor.call_args.kwargs["load_instance_runtime_settings"] is load_instance_runtime_settings


def test_instance_dotenv_cannot_inject_recovery_connection_acknowledgment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persisted instance settings cannot acquire the supervisor proof pipe."""

    class StreamObserved(BaseException):
        pass

    read_descriptor, write_descriptor = os.pipe()
    nonce = "00" * 32
    (tmp_path / ".env").write_text(
        f"{main_mod._RECOVERY_CONNECTION_ACK_FD_ENV}={write_descriptor}\n"
        f"{main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV}={nonce}\n",
        encoding="utf-8",
    )
    robot = MagicMock()
    stream_constructor = MagicMock(side_effect=StreamObserved)
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(moves_mod, "MovementManager", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(console_mod, "LocalStream", stream_constructor)
    monkeypatch.setattr(huggingface_realtime_mod, "HuggingFaceRealtimeHandler", MagicMock())
    monkeypatch.setattr(config_mod, "set_instance_path", MagicMock())
    monkeypatch.setattr(config_mod, "refresh_runtime_config_from_env", MagicMock())
    monkeypatch.setattr(
        config_mod,
        "get_hf_connection_selection",
        MagicMock(return_value=SimpleNamespace(mode="test", has_target=False)),
    )
    monkeypatch.setattr(
        startup_settings_mod,
        "load_startup_settings_into_runtime",
        MagicMock(return_value=SimpleNamespace(voice=None)),
    )

    try:
        with pytest.raises(StreamObserved):
            main_mod.run(
                _recovery_ack_args(),
                robot=robot,
                instance_path=str(tmp_path),
            )
        os.fstat(write_descriptor)
        os.set_blocking(read_descriptor, False)
        with pytest.raises(BlockingIOError):
            os.read(read_descriptor, 1)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)

    assert main_mod._RECOVERY_CONNECTION_ACK_FD_ENV not in os.environ
    assert main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV not in os.environ


def test_late_instance_dotenv_reload_cannot_authorize_a_later_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stream reload cannot turn persisted values into authority on a later run."""

    class StreamObserved(BaseException):
        pass

    read_descriptor, write_descriptor = os.pipe()
    (tmp_path / ".env").write_text(
        f"{main_mod._RECOVERY_CONNECTION_ACK_FD_ENV}={write_descriptor}\n"
        f"{main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV}={'00' * 32}\n",
        encoding="utf-8",
    )
    stream = console_mod.LocalStream(
        MagicMock(),
        MagicMock(),
        instance_path=str(tmp_path),
    )
    monkeypatch.setattr(console_mod, "has_hf_realtime_target", lambda: False)

    stream.launch()

    assert main_mod._RECOVERY_CONNECTION_ACK_FD_ENV not in os.environ
    assert main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV not in os.environ

    constructor = MagicMock()
    _patch_recovery_ack_preconstructor(monkeypatch, constructor)
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", MagicMock())
    monkeypatch.setattr(moves_mod, "MovementManager", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(console_mod, "LocalStream", MagicMock(side_effect=StreamObserved))
    monkeypatch.setattr(huggingface_realtime_mod, "HuggingFaceRealtimeHandler", MagicMock())
    write_record = MagicMock()
    monkeypatch.setattr(main_mod.os, "write", write_record)

    try:
        with pytest.raises(StreamObserved):
            main_mod.run(_recovery_ack_args(), robot=MagicMock())
        os.fstat(write_descriptor)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)

    constructor.assert_not_called()
    write_record.assert_not_called()


@pytest.mark.parametrize("invalid_value", [None, 0, 1, "", "False"])
def test_run_rejects_non_boolean_instance_runtime_settings_before_robot_startup(
    invalid_value: object,
) -> None:
    """The instance-settings authority boundary requires an actual boolean."""
    robot = MagicMock()

    with pytest.raises(ValueError, match="must be a boolean"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            load_instance_runtime_settings=invalid_value,  # type: ignore[arg-type]
        )

    assert robot.mock_calls == []


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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["reachy-mini-conversation-app"], False),
        (["reachy-mini-conversation-app", "--diagnostic-antenna-expression"], True),
    ],
)
def test_diagnostic_antenna_expression_cli_is_default_off(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: bool,
) -> None:
    """Only the explicit diagnostic flag should enable idle antenna expression."""
    monkeypatch.setattr(sys, "argv", argv)

    args, unknown = utils_mod.parse_args()

    assert args.diagnostic_antenna_expression is expected
    assert unknown == []


def test_diagnostic_antenna_expression_rejects_non_boolean_before_robot_startup() -> None:
    """Programmatic callers cannot enable movement with a truthy non-boolean value."""
    robot = MagicMock()
    args = MagicMock(diagnostic_antenna_expression="false")

    with pytest.raises(ValueError, match="diagnostic_antenna_expression must be a boolean"):
        main_mod.run(args, robot=robot)

    assert robot.mock_calls == []


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
        search_attempt_observer=None,
        search_attempt_supervisor_generation=0,
        search_attempt_child_generation=0,
        private_transcript_router=None,
        private_transcript_router_timeout_seconds=2.0,
        graceful_shutdown_event=None,
        graceful_shutdown_complete_event=None,
        stale_connection_exit_event=None,
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
    assert "search_attempt_observer" in typing.get_type_hints(main_mod.main)
    assert "search_attempt_observer" in typing.get_type_hints(main_mod.run)
    assert "private_transcript_router" in typing.get_type_hints(main_mod.main)
    assert "private_transcript_router" in typing.get_type_hints(main_mod.run)
    assert "graceful_shutdown_event" in typing.get_type_hints(main_mod.main)
    assert "graceful_shutdown_event" in typing.get_type_hints(main_mod.run)
    assert "stale_connection_exit_event" in typing.get_type_hints(main_mod.main)
    assert "stale_connection_exit_event" in typing.get_type_hints(main_mod.run)


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


@pytest.mark.parametrize("invalid_request", [object(), "event", 1])
def test_stale_connection_exit_rejects_malformed_event_before_robot_startup(
    invalid_request: object,
) -> None:
    """Malformed composition cannot reach robot initialization."""
    robot = MagicMock()

    with pytest.raises(ValueError, match="fresh distinct request event"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            stale_connection_exit_event=invalid_request,  # type: ignore[arg-type]
        )

    assert robot.mock_calls == []


def test_stale_connection_exit_rejects_early_or_aliased_event_before_robot_startup() -> None:
    """The terminal request must begin as one unshared capability."""
    robot = MagicMock()
    pre_set = threading.Event()
    pre_set.set()
    app_stop = threading.Event()
    graceful_request = threading.Event()
    graceful_complete = threading.Event()

    with pytest.raises(ValueError, match="fresh distinct request event"):
        main_mod.run(MagicMock(), robot=robot, stale_connection_exit_event=pre_set)
    with pytest.raises(ValueError, match="fresh distinct request event"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            app_stop_event=app_stop,
            stale_connection_exit_event=app_stop,
        )
    for aliased_event in (graceful_request, graceful_complete):
        with pytest.raises(ValueError, match="fresh distinct request event"):
            main_mod.run(
                MagicMock(),
                robot=robot,
                graceful_shutdown_event=graceful_request,
                graceful_shutdown_complete_event=graceful_complete,
                stale_connection_exit_event=aliased_event,
            )

    assert robot.mock_calls == []


def test_stale_connection_exit_requires_same_run_recovery_acknowledgment() -> None:
    """The local exit seam cannot outlive its connection proof."""
    with pytest.raises(ValueError, match="requires recovery connection acknowledgment"):
        main_mod.run(
            MagicMock(),
            robot=MagicMock(),
            stale_connection_exit_event=threading.Event(),
        )


def test_stale_connection_exit_latches_once_with_dedicated_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate event sets select exactly one dedicated terminal result."""
    request = threading.Event()
    terminated = threading.Event()
    statuses: list[int] = []

    def terminate(status: int) -> None:
        statuses.append(status)
        terminated.set()

    monkeypatch.setattr(main_mod, "_STALE_CONNECTION_TERMINATOR", terminate)
    stale_exit = main_mod._StaleConnectionExit(request)
    stale_exit.arm()

    request.set()
    request.set()

    assert terminated.wait(1.0)
    assert not stale_exit.close_for_ordinary_cleanup()
    assert statuses == [76]


def test_stale_connection_exit_ignores_absent_and_late_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the live acceptance window makes later requests inert."""
    request = threading.Event()
    terminated = threading.Event()
    terminate = MagicMock(side_effect=lambda _status: terminated.set())
    monkeypatch.setattr(main_mod, "_STALE_CONNECTION_TERMINATOR", terminate)
    stale_exit = main_mod._StaleConnectionExit(request)
    stale_exit.arm()

    assert stale_exit.close_for_ordinary_cleanup()
    request.set()

    assert not terminated.wait(0.2)
    terminate.assert_not_called()


def test_stale_connection_exit_bypasses_python_cleanup(tmp_path: Path) -> None:
    """The real terminal path bypasses finally, atexit, and destructors."""
    marker = tmp_path / "cleanup-ran"
    repository_root = Path(__file__).resolve().parents[1]
    script = """
import atexit
import sys
import threading
import time
from pathlib import Path

from reachy_mini_conversation_app.main import _StaleConnectionExit

marker = Path(sys.argv[1])

class CleanupTrap:
    def __del__(self):
        marker.write_text("destructor", encoding="utf-8")

trap = CleanupTrap()
atexit.register(marker.write_text, "atexit", encoding="utf-8")
request = threading.Event()
stale_exit = _StaleConnectionExit(request)
try:
    stale_exit.arm()
    request.set()
    while True:
        time.sleep(0.1)
finally:
    marker.write_text("finally", encoding="utf-8")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")

    completed = subprocess.run(
        [sys.executable, "-c", script, str(marker)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == main_mod.STALE_CONNECTION_EXIT_STATUS
    assert not marker.exists()


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
    diagnostic_antenna_expression: bool = False,
    include_diagnostic_antenna_expression: bool = True,
    completed_utterance_observer: object | None = None,
    completed_utterance_timeout_seconds: float = 2.0,
    search_policy: object | None = None,
    search_policy_timeout_seconds: float = 10.0,
    search_provider: object | None = None,
    search_attempt_observer: object | None = None,
    search_attempt_supervisor_generation: int = 0,
    search_attempt_child_generation: int = 0,
    rebuild_handler: bool = False,
    graceful_shutdown: bool = False,
    quiesce_succeeds: bool = True,
    graceful_shutdown_on_join: bool = False,
    stale_connection_exit: bool = False,
    stale_connection_exit_before_arm: bool = False,
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
    stale_exit_enabled = stale_connection_exit or stale_connection_exit_before_arm
    stale_connection_exit_event = threading.Event() if stale_exit_enabled else None
    recovery_ack_read: int | None = None
    run_robot: MagicMock | None = robot
    if stale_exit_enabled:
        recovery_ack_read, recovery_ack_write = os.pipe()
        monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_FD_ENV, str(recovery_ack_write))
        monkeypatch.setenv(main_mod._RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)
        monkeypatch.setattr(main_mod, "ReachyMini", MagicMock(return_value=robot))
        run_robot = None
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
    graceful_join_calls = 0

    class _DeferredThread:
        def __init__(self, *, target: object, **_kwargs: object) -> None:
            self.target = target

        def start(self) -> None:
            thread_targets.append(self.target)

        def join(self, timeout: float | None = None) -> None:
            del timeout
            nonlocal graceful_join_calls
            if getattr(self.target, "__name__", "") != "poll_graceful_shutdown_event":
                return
            graceful_join_calls += 1
            if graceful_shutdown_on_join:
                self.target()

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

    def _initialize_tools(**_kwargs: object) -> None:
        if stale_connection_exit_before_arm:
            assert stale_connection_exit_event is not None
            stale_connection_exit_event.set()

    active_core_tools = sys.modules["reachy_mini_conversation_app.tools.core_tools"]
    monkeypatch.setattr(active_core_tools, "initialize_tools", MagicMock(side_effect=_initialize_tools))
    stale_exit_statuses: list[int] = []
    monkeypatch.setattr(
        main_mod,
        "_STALE_CONNECTION_TERMINATOR",
        lambda status: stale_exit_statuses.append(status),
    )

    observed: dict[str, object] = {}

    def _launch() -> None:
        observed["startup_operations"] = startup_operations.copy()
        deps = handler_factory.call_args.args[0]
        if rebuild_handler:
            console_mod.LocalStream.call_args.kwargs["handler_factory"]()
        if stale_connection_exit_event is not None:
            stale_connection_exit_event.set()
            stale_target = next(
                target for target in thread_targets if getattr(target, "__name__", "") == "_await_request"
            )
            stale_target()
        elif graceful_shutdown:
            if not graceful_shutdown_on_join:
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
    arg_values: dict[str, object] = {
        "debug": False,
        "robot_name": None,
        "robot_host": None,
        "no_camera": True,
        "no_wobble": no_wobble,
        "ui": False,
    }
    if include_diagnostic_antenna_expression:
        arg_values["diagnostic_antenna_expression"] = diagnostic_antenna_expression
    args = SimpleNamespace(**arg_values)
    try:
        try:
            main_mod.run(
                args,
                robot=run_robot,
                app_stop_event=stop_event,
                completed_utterance_observer=completed_utterance_observer,
                completed_utterance_timeout_seconds=completed_utterance_timeout_seconds,
                search_policy=search_policy,
                search_policy_timeout_seconds=search_policy_timeout_seconds,
                search_provider=search_provider,
                search_attempt_observer=search_attempt_observer,
                search_attempt_supervisor_generation=search_attempt_supervisor_generation,
                search_attempt_child_generation=search_attempt_child_generation,
                graceful_shutdown_event=graceful_shutdown_event,
                graceful_shutdown_complete_event=graceful_shutdown_complete_event,
                stale_connection_exit_event=stale_connection_exit_event,
            )
        except RuntimeError as error:
            if not stale_connection_exit_before_arm:
                raise
            observed["run_error"] = str(error)
        if recovery_ack_read is not None:
            observed["recovery_ack"] = os.read(recovery_ack_read, 36)
            assert os.read(recovery_ack_read, 1) == b""
    finally:
        if recovery_ack_read is not None:
            os.close(recovery_ack_read)
    observed["operations"] = operations
    observed["graceful_shutdown_complete"] = (
        graceful_shutdown_complete_event.is_set() if graceful_shutdown_complete_event is not None else False
    )
    observed["enable_wobbling_calls"] = robot.enable_wobbling.call_count
    observed["disable_motors_calls_after_shutdown"] = robot.disable_motors.call_count
    observed["disable_wobbling_calls_after_shutdown"] = robot.disable_wobbling.call_count
    observed["media_close_calls_after_shutdown"] = robot.media.close.call_count
    observed["client_disconnect_calls_after_shutdown"] = robot.client.disconnect.call_count
    observed["movement_stop_calls_after_shutdown"] = movement_manager.stop.call_count
    observed["movement_manager_kwargs"] = movement_manager_factory.call_args.kwargs
    observed["robot"] = robot
    observed["graceful_join_calls"] = graceful_join_calls
    observed["stale_exit_statuses"] = stale_exit_statuses
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
    observed["handlers"][0].set_search_attempt_observer.assert_not_called()


def test_search_attempt_observer_attaches_to_initial_and_rebuilt_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content-free search diagnostics retain supervisor generations across rebuilds."""
    policy = MagicMock()
    observer = MagicMock()

    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        search_policy=policy,
        search_attempt_observer=observer,
        search_attempt_supervisor_generation=17,
        search_attempt_child_generation=3,
        rebuild_handler=True,
    )

    sequences: list[object] = []
    for handler in observed["handlers"]:
        handler.set_search_attempt_observer.assert_called_once_with(
            observer,
            supervisor_generation=17,
            child_generation=3,
            sequence=ANY,
        )
        sequences.append(handler.set_search_attempt_observer.call_args.kwargs["sequence"])
    assert sequences[0] is sequences[1]


@pytest.mark.parametrize("invalid_generation", [True, -1, 2**63, 1.0, "1"])
def test_search_observer_rejects_invalid_generation_before_robot_startup(
    invalid_generation: object,
) -> None:
    """Malformed process correlation cannot reach robot initialization."""
    robot = MagicMock()

    with pytest.raises(ValueError, match="non-negative 63-bit integers"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            search_policy=MagicMock(),
            search_attempt_observer=MagicMock(),
            search_attempt_supervisor_generation=invalid_generation,  # type: ignore[arg-type]
        )

    assert robot.mock_calls == []


def test_search_observer_requires_policy_and_callable_before_robot_startup() -> None:
    """Diagnostics cannot create a search authority or defer composition errors."""
    robot = MagicMock()

    with pytest.raises(ValueError, match="requires a search policy"):
        main_mod.run(MagicMock(), robot=robot, search_attempt_observer=MagicMock())
    with pytest.raises(ValueError, match="must be callable"):
        main_mod.run(
            MagicMock(),
            robot=robot,
            search_policy=MagicMock(),
            search_attempt_observer=object(),  # type: ignore[arg-type]
        )

    assert robot.mock_calls == []


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


def test_diagnostic_antenna_expression_flag_reaches_movement_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diagnostic CLI value should reach only the app's movement owner."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        diagnostic_antenna_expression=True,
    )

    assert observed["movement_manager_kwargs"] == {
        "current_robot": observed["robot"],
        "diagnostic_antenna_expression": True,
    }


def test_legacy_args_keep_diagnostic_antenna_expression_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Namespaces created before the diagnostic flag should retain fixed-neutral idle."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        include_diagnostic_antenna_expression=False,
    )

    assert observed["movement_manager_kwargs"] == {
        "current_robot": observed["robot"],
        "diagnostic_antenna_expression": False,
    }


def test_wobbling_startup_failure_stops_actual_movement_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post-start failure must join the real movement loop before propagating."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = moves_mod.create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [-0.1745, 0.1745])
    first_target = threading.Event()
    robot.set_target.side_effect = lambda **_kwargs: first_target.set()

    def fail_wobbling_startup() -> None:
        assert first_target.wait(timeout=1.0)
        raise RuntimeError("wobbling startup failed")

    robot.enable_wobbling.side_effect = fail_wobbling_startup
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(main_mod, "time", SimpleNamespace(sleep=MagicMock()))
    monkeypatch.setattr(main_mod.app_lifecycle, "wake_up_if_sleeping", MagicMock())
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
    monkeypatch.setattr(huggingface_realtime_mod, "HuggingFaceRealtimeHandler", MagicMock())
    monkeypatch.setattr(console_mod, "LocalStream", MagicMock(return_value=MagicMock()))
    active_core_tools = sys.modules["reachy_mini_conversation_app.tools.core_tools"]
    monkeypatch.setattr(active_core_tools, "initialize_tools", MagicMock())
    args = SimpleNamespace(
        debug=False,
        robot_name=None,
        robot_host=None,
        no_camera=True,
        no_wobble=False,
        diagnostic_antenna_expression=True,
        ui=False,
    )

    with pytest.raises(RuntimeError, match="wobbling startup failed"):
        main_mod.run(args, robot=robot)

    calls_at_failure = robot.set_target.call_count
    assert calls_at_failure > 0
    threading.Event().wait(timeout=0.05)
    assert robot.set_target.call_count == calls_at_failure


def test_stale_connection_exit_skips_every_ordinary_cleanup_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A latched stale exit cannot enter ordinary external cleanup."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        stale_connection_exit=True,
    )

    assert observed["stale_exit_statuses"] == [main_mod.STALE_CONNECTION_EXIT_STATUS]
    assert observed["recovery_ack"] == b"RCA1" + (b"\x00" * 32)
    assert observed["operations"] == []
    assert observed["quiesce_calls"] == 0
    assert observed["stream_close_calls"] == 0
    assert observed["daemon_stop_calls"] == 0
    assert observed["goto_sleep_calls"] == 0
    assert observed["disable_motors_calls_after_shutdown"] == 0
    assert observed["disable_wobbling_calls_after_shutdown"] == 0
    assert observed["media_close_calls_after_shutdown"] == 0
    assert observed["client_disconnect_calls_after_shutdown"] == 0
    assert observed["movement_stop_calls_after_shutdown"] == 0


def test_stale_connection_request_during_startup_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh request set after validation but before arm cannot become terminal."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        stale_connection_exit_before_arm=True,
    )

    assert observed["run_error"] == "Stale-connection exit was requested before the conversation loop"
    assert observed["recovery_ack"] == b"RCA1" + (b"\x00" * 32)
    assert observed["stale_exit_statuses"] == []
    assert observed["movement_stop_calls_after_shutdown"] == 1
    assert observed["disable_wobbling_calls_after_shutdown"] == 1
    assert observed["media_close_calls_after_shutdown"] == 1
    assert observed["client_disconnect_calls_after_shutdown"] == 1


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

    args = SimpleNamespace(
        debug=False,
        robot_name=None,
        no_camera=True,
        no_wobble=True,
        diagnostic_antenna_expression=False,
        ui=False,
    )
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


def test_graceful_shutdown_waits_for_sleep_after_conversation_loop_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversation teardown cannot disconnect before the shutdown thread finishes."""
    observed = _run_sleep_scenario(
        monkeypatch,
        sleep_fails=False,
        use_stop_event=False,
        graceful_shutdown=True,
        graceful_shutdown_on_join=True,
    )

    assert observed["operations"] == [
        "quiesce",
        "sleep",
        "disable_motors",
        "stop",
        "local_stream_close",
    ]
    assert observed["graceful_shutdown_complete"] is True
    assert observed["graceful_join_calls"] == 1


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
