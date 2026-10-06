from __future__ import annotations

import pytest
from sse_starlette.sse import AppStatus


@pytest.fixture(autouse=True)
def _fresh_sse_exit_event():
    # sse-starlette caches an exit event on the first event loop that streams; each TestClient
    # runs its own loop, so a second SSE test in the same process would hit "bound to a
    # different event loop". Production has one loop, so this only matters under pytest.
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None
