from __future__ import annotations
import logging
import argparse
import warnings
from typing import Optional


def parse_args() -> tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--no-wobble",
        default=False,
        action="store_true",
        help="Disable audio-reactive head wobbling before wake-up and for the session",
    )
    parser.add_argument(
        "--ui",
        default=False,
        action="store_true",
        help="Serve the web UI at http://127.0.0.1:7860/, in addition to console mode",
    )
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="[Optional] Robot name to target. Must match the daemon's --robot-name when connecting to a specific robot, mainly useful for development with multiple robots.",
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default=None,
        help="[Optional] Connect directly to this trusted network daemon host instead of auto-detecting localhost first.",
    )
    subparsers = parser.add_subparsers(dest="command")
    tool_spaces_parser = subparsers.add_parser("tool-spaces", help="Manage installed Hugging Face Space tool sources")
    tool_spaces_subparsers = tool_spaces_parser.add_subparsers(dest="tool_spaces_command", required=True)

    add_parser = tool_spaces_subparsers.add_parser("add", help="Install one Space tool source by slug")
    add_parser.add_argument("space_slug", help="Hugging Face Space slug in the form owner/space-name")
    add_parser.add_argument(
        "--install-only",
        action="store_true",
        default=False,
        help="Install the Space without enabling its tools in any profile.",
    )
    add_parser.add_argument(
        "--profile",
        dest="profile",
        default=None,
        metavar="PROFILE",
        help="Enable tools in this profile instead of the active profile.",
    )

    add_server_parser = tool_spaces_subparsers.add_parser(
        "add-server",
        help="Install one generic local MCP server with a cached prompt and isolated no-retry tools",
    )
    add_server_parser.add_argument("alias", help="Stable local server alias used to namespace its tools")
    add_server_parser.add_argument("mcp_url", help="Fixed loopback HTTP or remote HTTPS Streamable HTTP endpoint")
    add_server_parser.add_argument("--prompt", required=True, help="Exact no-argument MCP prompt to cache")
    add_server_parser.add_argument("--install-only", action="store_true", default=False)
    add_server_parser.add_argument("--profile", default=None, help="Profile in which to enable the discovered tools")

    remove_parser = tool_spaces_subparsers.add_parser("remove", help="Remove one installed Space tool source")
    remove_parser.add_argument("space_slug", help="Installed Hugging Face Space slug in the form owner/space-name")

    remove_server_parser = tool_spaces_subparsers.add_parser("remove-server", help="Remove one generic MCP server")
    remove_server_parser.add_argument("alias", help="Installed generic MCP server alias")

    tool_spaces_subparsers.add_parser("list", help="List installed Space tool sources")
    return parser.parse_known_args()


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger


def log_connection_troubleshooting(logger: logging.Logger, robot_name: Optional[str]) -> None:
    """Log troubleshooting steps for connection issues."""
    logger.error("Troubleshooting steps:")
    logger.error("  1. Verify reachy-mini-daemon is running")

    if robot_name is not None:
        logger.error(f"  2. Daemon must be started with: --robot-name '{robot_name}'")
    else:
        logger.error("  2. If daemon uses --robot-name, add the same flag here: --robot-name <name>")

    logger.error("  3. For wireless: check network connectivity")
    logger.error("  4. Review daemon logs")
    logger.error("  5. Restart the daemon")
