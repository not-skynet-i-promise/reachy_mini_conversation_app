"""Anonymous revision gate for Pollen's official hosted search Space."""

from __future__ import annotations
import asyncio
import logging
from typing import Final, Protocol, cast
from collections.abc import Callable, Awaitable

from huggingface_hub import HfApi, SpaceInfo


logger = logging.getLogger(__name__)

OFFICIAL_SEARCH_SPACE_SLUG: Final[str] = "pollen-robotics/reachy-mini-search-tool"
EXPECTED_SEARCH_SPACE_REVISION: Final[str] = "962e6d05e349a49187e78d8a86656da25ed857d3"
_SEARCH_SPACE_METADATA_FIELDS: Final[tuple[str, ...]] = ("disabled", "private", "runtime", "sha")
_SEARCH_SPACE_HTTP_TIMEOUT_SECONDS: Final[float] = 5.0
_SEARCH_SPACE_GATE_TIMEOUT_SECONDS: Final[float] = 8.0


class SpaceMetadataClient(Protocol):
    """The narrow official-client surface allowed to cross this gate."""

    def space_info(
        self,
        repo_id: str,
        *,
        expand: list[str],
        timeout: float,
        token: bool | str | None,
    ) -> SpaceInfo:
        """Return public Space metadata without accepting request content."""


def _space_is_expected_revision(info: SpaceInfo, expected_revision: str) -> bool:
    """Require one public, enabled, running Space at the advertised revision."""
    if info.disabled is not False or info.private is not False or info.sha != expected_revision:
        return False
    runtime = info.runtime
    if runtime is None:
        return False
    # The Hub client exposes the advertised SHA and runtime stage, but no
    # independently typed deployed SHA. Reject transitional RUNNING_* stages,
    # where an older container can still serve while a new revision builds.
    stage = runtime.stage
    return isinstance(stage, str) and stage == "RUNNING"


def _read_space_metadata(client: SpaceMetadataClient, slug: str) -> SpaceInfo:
    """Make the one anonymous official metadata call with an HTTP bound."""
    return client.space_info(
        slug,
        expand=list(_SEARCH_SPACE_METADATA_FIELDS),
        timeout=_SEARCH_SPACE_HTTP_TIMEOUT_SECONDS,
        token=False,
    )


def build_official_search_space_gate(
    *,
    client: SpaceMetadataClient | None = None,
    slug: str = OFFICIAL_SEARCH_SPACE_SLUG,
    expected_revision: str = EXPECTED_SEARCH_SPACE_REVISION,
) -> Callable[[], Awaitable[bool]]:
    """Build a bounded zero-input gate that cannot receive query content."""
    # token=False bypasses cached-token lookup. space_info is metadata-only and
    # does not use the Hub download cache; the launcher separately disables
    # implicit tokens before import for defense in depth.
    resolved_client = client if client is not None else cast(SpaceMetadataClient, HfApi(token=False))

    async def gate() -> bool:
        try:
            info = await asyncio.wait_for(
                asyncio.to_thread(_read_space_metadata, resolved_client, slug),
                timeout=_SEARCH_SPACE_GATE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info("search_space_gate outcome=metadata_unavailable")
            return False
        if not _space_is_expected_revision(info, expected_revision):
            logger.info("search_space_gate outcome=revision_refused")
            return False
        return True

    return gate
