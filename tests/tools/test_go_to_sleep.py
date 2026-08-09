import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep


def test_go_to_sleep_has_no_required_arguments() -> None:
    """The tool should be callable without a confirmation argument."""
    assert GoToSleep.parameters_schema == {
        "type": "object",
        "properties": {},
        "required": [],
    }


@pytest.mark.asyncio
async def test_go_to_sleep_returns_unavailable_without_runtime_callback() -> None:
    """The tool should fail gracefully if the runtime did not inject a sleep callback."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await GoToSleep()(deps)

    assert result == {"error": "go_to_sleep is unavailable in this runtime"}


@pytest.mark.asyncio
async def test_go_to_sleep_calls_runtime_callback() -> None:
    """The tool should delegate the actual movement and app stop to the host runtime."""
    expected = {
        "status": "sleeping",
        "stop_current_app_requested": True,
        "local_stop_requested": True,
    }
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = await GoToSleep()(deps)

    assert result == expected
    go_to_sleep.assert_called_once_with()


@pytest.mark.asyncio
async def test_go_to_sleep_cancellation_waits_for_callback_to_finish() -> None:
    """Cancellation cannot hide a still-running robot sleep callback."""
    callback_started = threading.Event()
    release_callback = threading.Event()

    def go_to_sleep() -> dict[str, str]:
        callback_started.set()
        release_callback.wait(timeout=2.0)
        return {"status": "sleeping"}

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )
    task = asyncio.create_task(GoToSleep()(deps))
    assert await asyncio.to_thread(callback_started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_callback.set()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_go_to_sleep_repeated_cancellation_waits_for_callback_to_finish() -> None:
    """Repeated manager cancellation cannot release a running executor callback."""
    callback_started = threading.Event()
    release_callback = threading.Event()

    def go_to_sleep() -> dict[str, str]:
        callback_started.set()
        release_callback.wait(timeout=2.0)
        return {"status": "sleeping"}

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )
    task = asyncio.create_task(GoToSleep()(deps))
    assert await asyncio.to_thread(callback_started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_callback.set()

    with pytest.raises(asyncio.CancelledError):
        await task
