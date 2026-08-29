from __future__ import annotations
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.idle_policy as idle_policy_mod
from reachy_mini_conversation_app.tools.move_head import MoveHead
from reachy_mini_conversation_app.tools.core_tools import Tool, AcceptedUserTurn, ToolDependencies
from reachy_mini_conversation_app.tools.idle_do_nothing import IdleDoNothing


class FakeIdleTool(Tool):
    """Tool with an idle tool name but not the idle tool class."""

    _auto_register = False
    name = "idle_do_nothing"
    description = "Fake idle tool with the same public name."
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return a fake result."""
        return {"status": "fake"}


def test_choose_idle_tool_call_uses_registered_name_for_matching_class() -> None:
    """A matching idle class should use the registered tool instance name."""
    tool = IdleDoNothing()
    tool.name = "profile_idle_do_nothing"

    selected = idle_policy_mod.choose_idle_tool_call(
        ["profile_idle_do_nothing"],
        tool_registry={"profile_idle_do_nothing": tool},
    )

    assert selected == ("profile_idle_do_nothing", {"reason": "random idle policy selected stillness"})


def test_choose_idle_tool_call_rejects_same_name_with_unmatched_class() -> None:
    """A matching public name alone should not make a tool eligible."""
    selected = idle_policy_mod.choose_idle_tool_call(
        ["idle_do_nothing"],
        tool_registry={"idle_do_nothing": FakeIdleTool()},
    )

    assert selected is None


def test_choose_idle_tool_call_keeps_args_with_weighted_candidate(monkeypatch) -> None:
    """Argument generation should stay attached to the weighted candidate."""
    monkeypatch.setattr(idle_policy_mod.random, "choice", lambda _choices: "right")

    selected = idle_policy_mod.choose_idle_tool_call(
        ["move_head"],
        tool_registry={"move_head": MoveHead()},
    )

    assert selected == ("move_head", {"direction": "right"})


@pytest.mark.asyncio
async def test_start_idle_tool_call_drops_accepted_user_turn(monkeypatch: Any) -> None:
    """Idle actions must never inherit authority from the last user turn."""
    accepted_turn = AcceptedUserTurn(item_id="item-1", transcript="Look to your left")
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        accepted_user_turn=accepted_turn,
    )
    tool_manager = MagicMock()
    tool_manager.start_tool = AsyncMock(return_value=MagicMock(tool_id="idle-tool-1"))
    monkeypatch.setattr(
        idle_policy_mod,
        "choose_idle_tool_call",
        lambda _available_tool_names: ("idle_do_nothing", {}),
    )

    await idle_policy_mod.start_idle_tool_call(
        deps=deps,
        tool_manager=tool_manager,
        output_queue=asyncio.Queue(),
        available_tool_names=["idle_do_nothing"],
        idle_duration=60.0,
    )

    routine = tool_manager.start_tool.await_args.kwargs["tool_call_routine"]
    assert routine.deps is not deps
    assert routine.deps.accepted_user_turn is None
    assert deps.accepted_user_turn is accepted_turn
    assert accepted_turn.is_current
