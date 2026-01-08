from __future__ import annotations

import asyncio

import pytest

from pacs_rag.mcp_client import (
    McpSession,
    McpToolCallError,
    McpRetryPolicy,
    build_stdio_server_params,
)


class FakeResult:
    def __init__(self, payload: object, is_error: bool = False) -> None:
        self.isError = is_error
        self.structuredContent = {"result": payload}
        self.content = []


class FakeSession:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TimeoutSession:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls += 1
        event = asyncio.Event()
        await event.wait()


def _run(coro):
    return asyncio.run(coro)


def test_call_tool_retries_on_os_error() -> None:
    policy = McpRetryPolicy(timeout_seconds=0.01, max_attempts=2, backoff_seconds=(0,))
    server_params = build_stdio_server_params("dicom-mcp")
    client = McpSession(server_params, retry_policy=policy)
    client._session = FakeSession([OSError("boom"), FakeResult({"ok": True})])

    result = _run(client.call_tool("query_studies", {}))

    assert result == {"ok": True}
    assert client._session.calls == 2


def test_call_tool_no_retry_for_non_idempotent() -> None:
    policy = McpRetryPolicy(
        timeout_seconds=0.01,
        max_attempts=3,
        backoff_seconds=(0,),
        non_idempotent_tools=frozenset({"move_study"}),
    )
    server_params = build_stdio_server_params("dicom-mcp")
    client = McpSession(server_params, retry_policy=policy)
    client._session = TimeoutSession()

    with pytest.raises(McpToolCallError):
        _run(client.call_tool("move_study", {"study_instance_uid": "1"}))

    assert client._session.calls == 1
