"""Tests for configuration helpers."""

import os
import sys
import subprocess
from pathlib import Path

import pytest

from reachy_mini_conversation_app import config


def test_fresh_config_import_rejects_cwd_dotenv_recovery_authority(tmp_path: Path) -> None:
    """The normal fresh-import path cannot acquire recovery authority from CWD."""
    (tmp_path / ".env").write_text(
        f"{config.RECOVERY_CONNECTION_ACK_FD_ENV}=99\n{config.RECOVERY_CONNECTION_ACK_NONCE_ENV}={'ff' * 32}\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("REACHY_MINI_SKIP_DOTENV", None)
    environment.pop(config.RECOVERY_CONNECTION_ACK_FD_ENV, None)
    environment.pop(config.RECOVERY_CONNECTION_ACK_NONCE_ENV, None)
    source_root = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = str(source_root)
    script = (
        "import os\n"
        "from reachy_mini_conversation_app import main\n"
        "assert main._RECOVERY_CONNECTION_ACK_FD_ENV not in os.environ\n"
        "assert main._RECOVERY_CONNECTION_ACK_NONCE_ENV not in os.environ\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("inherited", [False, True])
def test_dotenv_cannot_add_or_override_recovery_connection_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inherited: bool,
) -> None:
    """Only process inheritance can supply the recovery acknowledgment capability."""
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        f"{config.RECOVERY_CONNECTION_ACK_FD_ENV}=99\n"
        f"{config.RECOVERY_CONNECTION_ACK_NONCE_ENV}={'ff' * 32}\n"
        "RECOVERY_ACK_DOTENV_CONTROL=loaded\n",
        encoding="utf-8",
    )
    if inherited:
        monkeypatch.setenv(config.RECOVERY_CONNECTION_ACK_FD_ENV, "7")
        monkeypatch.setenv(config.RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)
    else:
        monkeypatch.delenv(config.RECOVERY_CONNECTION_ACK_FD_ENV, raising=False)
        monkeypatch.delenv(config.RECOVERY_CONNECTION_ACK_NONCE_ENV, raising=False)
    monkeypatch.delenv("RECOVERY_ACK_DOTENV_CONTROL", raising=False)

    config.load_dotenv_without_recovery_connection_authority(str(dotenv_path), override=True)

    if inherited:
        assert config.os.environ[config.RECOVERY_CONNECTION_ACK_FD_ENV] == "7"
        assert config.os.environ[config.RECOVERY_CONNECTION_ACK_NONCE_ENV] == "00" * 32
    else:
        assert config.RECOVERY_CONNECTION_ACK_FD_ENV not in config.os.environ
        assert config.RECOVERY_CONNECTION_ACK_NONCE_ENV not in config.os.environ
    assert config.os.environ["RECOVERY_ACK_DOTENV_CONTROL"] == "loaded"


def test_dotenv_failure_restores_recovery_connection_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed dotenv load cannot strand replaced supervisor capability values."""
    monkeypatch.setenv(config.RECOVERY_CONNECTION_ACK_FD_ENV, "7")
    monkeypatch.setenv(config.RECOVERY_CONNECTION_ACK_NONCE_ENV, "00" * 32)

    def fail_load(**_kwargs: object) -> None:
        config.os.environ[config.RECOVERY_CONNECTION_ACK_FD_ENV] = "99"
        config.os.environ[config.RECOVERY_CONNECTION_ACK_NONCE_ENV] = "ff" * 32
        raise RuntimeError("dotenv failed")

    monkeypatch.setattr(config, "load_dotenv", fail_load)

    with pytest.raises(RuntimeError, match="dotenv failed"):
        config.load_dotenv_without_recovery_connection_authority("unused", override=True)

    assert config.os.environ[config.RECOVERY_CONNECTION_ACK_FD_ENV] == "7"
    assert config.os.environ[config.RECOVERY_CONNECTION_ACK_NONCE_ENV] == "00" * 32


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected
