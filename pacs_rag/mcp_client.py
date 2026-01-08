from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpRetryPolicy:
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (0.5, 1.0, 2.0)
    non_idempotent_tools: frozenset[str] = frozenset({"move_study", "move_series"})


class McpToolExecutionError(RuntimeError):
    def __init__(self, tool: str, payload: Any) -> None:
        super().__init__(f"MCP tool '{tool}' returned error")
        self.tool = tool
        self.payload = payload


class McpToolCallError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def _backoff_for_attempt(attempt: int, policy: McpRetryPolicy) -> float:
    if not policy.backoff_seconds:
        return 0.0
    index = min(attempt - 1, len(policy.backoff_seconds) - 1)
    return policy.backoff_seconds[index]


def _summarize_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return {"type": "list", "count": len(payload)}
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload.keys())
        return {"type": "dict", "keys": keys[:20]}
    if isinstance(payload, str):
        return {"type": "str", "length": len(payload)}
    return {"type": type(payload).__name__}


def build_stdio_server_params(
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> StdioServerParameters:
    return StdioServerParameters(
        command=command,
        args=args or [],
        cwd=cwd,
        env=env,
    )


class McpSession:
    def __init__(
        self,
        server_params: StdioServerParameters,
        retry_policy: McpRetryPolicy | None = None,
    ) -> None:
        self._server_params = server_params
        self._retry_policy = retry_policy or McpRetryPolicy()
        self._stdio_cm = None
        self._session_cm: ClientSession | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "McpSession":
        self._stdio_cm = stdio_client(self._server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)
        self._stdio_cm = None
        self._session_cm = None
        self._session = None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session not initialized")
        arguments = arguments or {}
        policy = self._retry_policy
        max_attempts = policy.max_attempts
        if name in policy.non_idempotent_tools:
            max_attempts = 1

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._call_tool_once(name, arguments, policy.timeout_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                retryable = _is_retryable(name, exc, policy)
                details = _build_error_details(
                    name=name,
                    arguments=arguments,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    policy=policy,
                    retryable=retryable,
                    error=exc,
                )
                if not retryable or attempt >= max_attempts:
                    log.error("MCP tool call failed", extra={"extra_data": details})
                    raise McpToolCallError("MCP tool call failed", details) from exc
                backoff_seconds = _backoff_for_attempt(attempt, policy)
                log.warning(
                    "MCP tool call failed, retrying",
                    extra={"extra_data": {**details, "backoff_seconds": backoff_seconds}},
                )
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds)
        raise RuntimeError("MCP tool call retry loop exited unexpectedly")

    async def _call_tool_once(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout_seconds: float,
    ) -> Any:
        if self._session is None:
            raise RuntimeError("MCP session not initialized")
        call = self._session.call_tool(name=name, arguments=arguments)
        result = await asyncio.wait_for(call, timeout=timeout_seconds)
        payload = _extract_tool_payload(result)
        if result.isError:
            raise McpToolExecutionError(name, payload)
        return payload

    async def query_studies(self, **kwargs) -> Any:
        return await self.call_tool("query_studies", kwargs)

    async def query_series(self, **kwargs) -> Any:
        return await self.call_tool("query_series", kwargs)


def _is_retryable(tool: str, error: BaseException, policy: McpRetryPolicy) -> bool:
    if tool in policy.non_idempotent_tools:
        return False
    if isinstance(error, McpToolExecutionError):
        return False
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    return False


def _build_error_details(
    *,
    name: str,
    arguments: dict[str, Any],
    attempt: int,
    max_attempts: int,
    policy: McpRetryPolicy,
    retryable: bool,
    error: BaseException,
) -> dict[str, Any]:
    payload_summary = None
    if isinstance(error, McpToolExecutionError):
        payload_summary = _summarize_payload(error.payload)
    return {
        "tool": name,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "retryable": retryable,
        "timeout_seconds": policy.timeout_seconds,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "argument_keys": sorted(arguments.keys()),
        "payload_summary": payload_summary,
        "non_idempotent": name in policy.non_idempotent_tools,
    }


def _extract_tool_payload(result) -> Any:
    if result.structuredContent is not None:
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
            return payload["result"]
        return payload
    for block in result.content:
        if getattr(block, "type", None) == "text":
            text = block.text.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
                    return payload["result"]
                return payload
            except json.JSONDecodeError:
                return text
    return None
