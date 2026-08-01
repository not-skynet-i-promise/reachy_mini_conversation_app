"""Offline tests for the anonymous official search Space revision gate."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import reachy_mini_conversation_app.search_space_gate as gate_mod


def _space_info(
    *,
    sha: str | None = gate_mod.EXPECTED_SEARCH_SPACE_REVISION,
    disabled: bool | None = False,
    private: bool | None = False,
    stage: str | None = "RUNNING",
) -> Any:
    runtime = None if stage is None else SimpleNamespace(stage=stage)
    return SimpleNamespace(sha=sha, disabled=disabled, private=private, runtime=runtime)


@pytest.mark.asyncio
async def test_official_search_space_gate_is_zero_input_anonymous_and_exact() -> None:
    """The only outbound call contains fixed metadata fields and no request content."""
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeClient:
        def space_info(self, repo_id: str, **kwargs: Any) -> Any:
            calls.append((repo_id, kwargs))
            return _space_info()

    gate = gate_mod.build_official_search_space_gate(client=FakeClient())

    assert await gate()
    assert calls == [
        (
            gate_mod.OFFICIAL_SEARCH_SPACE_SLUG,
            {
                "expand": ["disabled", "private", "runtime", "sha"],
                "timeout": 5.0,
                "token": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_default_search_space_client_disables_tokens_at_client_and_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither the client nor the metadata request can read a cached token."""
    constructed: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    class FakeHfApi:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

        def space_info(self, _repo_id: str, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return _space_info()

    monkeypatch.setattr(gate_mod, "HfApi", FakeHfApi)

    assert await gate_mod.build_official_search_space_gate()()
    assert constructed == [{"token": False}]
    assert calls[0]["token"] is False
    assert "headers" not in calls[0]
    assert "cookies" not in calls[0]


@pytest.mark.parametrize(
    "info",
    [
        _space_info(sha=None),
        _space_info(sha="different-revision"),
        _space_info(disabled=None),
        _space_info(disabled=True),
        _space_info(private=None),
        _space_info(private=True),
        _space_info(stage=None),
        _space_info(stage="BUILDING"),
        _space_info(stage="RUNNING_BUILDING"),
        _space_info(stage="RUNNING_APP_STARTING"),
        _space_info(stage="PAUSED"),
        _space_info(stage="STOPPED"),
    ],
)
def test_search_space_metadata_fails_closed(info: Any) -> None:
    """Missing, private, disabled, transitional, and mismatched metadata refuses."""
    assert not gate_mod._space_is_expected_revision(info, gate_mod.EXPECTED_SEARCH_SPACE_REVISION)


@pytest.mark.asyncio
async def test_search_space_gate_converts_metadata_errors_to_refusal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Private/missing/malformed HTTP outcomes remain bounded and content-free."""
    caplog.set_level(logging.INFO, logger=gate_mod.__name__)

    class FailingClient:
        def space_info(self, _repo_id: str, **_kwargs: Any) -> Any:
            raise KeyError("private-metadata-canary")

    gate = gate_mod.build_official_search_space_gate(client=FailingClient())

    assert not await gate()
    assert "private-metadata-canary" not in caplog.text
    assert "search_space_gate outcome=metadata_unavailable" in caplog.text


@pytest.mark.asyncio
async def test_search_space_gate_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck official client cannot hold the search coordinator indefinitely."""
    release = threading.Event()

    class BlockingClient:
        def space_info(self, _repo_id: str, **_kwargs: Any) -> Any:
            release.wait(timeout=1.0)
            return _space_info()

    monkeypatch.setattr(gate_mod, "_SEARCH_SPACE_GATE_TIMEOUT_SECONDS", 0.01)
    gate = gate_mod.build_official_search_space_gate(client=BlockingClient())
    try:
        assert not await asyncio.wait_for(gate(), timeout=0.1)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_search_space_gate_propagates_outer_cancellation() -> None:
    """Session teardown owns cancellation rather than turning it into a verdict."""
    release = threading.Event()
    started = threading.Event()

    class BlockingClient:
        def space_info(self, _repo_id: str, **_kwargs: Any) -> Any:
            started.set()
            release.wait(timeout=1.0)
            return _space_info()

    task = asyncio.create_task(gate_mod.build_official_search_space_gate(client=BlockingClient())())
    try:
        await asyncio.to_thread(started.wait, 0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
