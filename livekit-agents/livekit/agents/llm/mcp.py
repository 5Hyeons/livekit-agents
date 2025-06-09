import json
import logging
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

try:
    from mcp import ClientSession, stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters
    from mcp.types import JSONRPCMessage
except ImportError as e:
    raise ImportError(
        "The 'mcp' package is required to run the MCP server integration but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[mcp]'"
    ) from e

logger = logging.getLogger(__name__)

from .tool_context import RawFunctionTool, ToolError, function_tool

MCPTool = RawFunctionTool


class MCPServer(ABC):
    def __init__(self, *, client_session_timeout_seconds: float) -> None:
        self._client: ClientSession | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._read_timeout = client_session_timeout_seconds

        self._cache_dirty = True
        self._lk_tools: list[MCPTool] | None = None

    @property
    def initialized(self) -> bool:
        return self._client is not None

    def invalidate_cache(self) -> None:
        self._cache_dirty = True

    async def initialize(self) -> None:
        try:
            receive_stream, send_stream = await self._exit_stack.enter_async_context(
                self.client_streams()
            )
            self._client = await self._exit_stack.enter_async_context(
                ClientSession(
                    receive_stream,
                    send_stream,
                    read_timeout_seconds=timedelta(seconds=self._read_timeout)
                    if self._read_timeout
                    else None,
                )
            )
            await self._client.initialize()
            self._initialized = True
        except Exception:
            await self.aclose()
            raise

    async def list_tools(self) -> list[MCPTool]:
        if self._client is None:
            raise RuntimeError("MCPServer isn't initialized")

        if not self._cache_dirty and self._lk_tools is not None:
            return self._lk_tools

        tools = await self._client.list_tools()
        lk_tools = [
            self._make_function_tool(tool.name, tool.description, tool.inputSchema)
            for tool in tools.tools
        ]

        self._lk_tools = lk_tools
        self._cache_dirty = False
        return lk_tools

    def _make_function_tool(
        self, name: str, description: str | None, input_schema: dict
    ) -> MCPTool:
        async def _tool_called(raw_arguments: dict) -> Any:
            logger.debug(f"MCP tool '{name}' (server: {self}) called with args: {raw_arguments}")
            if not self.initialized or self._client is None:
                logger.error(f"MCP tool '{name}' invocation attempt while server not initialized or client is None. Server: {self}")
                raise ToolError(
                    f"MCP tool '{name}' invocation failed: MCP client is not available. "
                    "Please check that the MCPServer is properly initialized and running."
                )

            try:
                if self._client is None:
                    logger.error(f"Critical: MCP client became None just before call_tool for tool '{name}'. Server: {self}")
                    raise ToolError("MCP client unexpectedly became None before tool call.")
                
                tool_result = await self._client.call_tool(name, raw_arguments)
            except Exception as e:
                logger.error(f"Exception during MCPServer '{self}' call_tool for '{name}': {type(e).__name__} - {e}", exc_info=True)
                raise ToolError(
                    f"MCP tool '{name}' (server: {self}) invocation failed due to a network-related issue: {type(e).__name__} - {e}. "
                    "This might indicate the MCP server is unavailable or a network problem."
                ) from e

            if tool_result.isError:
                error_str = "\n".join(str(part) for part in tool_result.content)
                logger.warning(f"MCP tool '{name}' (server: {self}) returned error: {error_str}")
                raise ToolError(error_str)

            # TODO(theomonnom): handle images & binary messages
            if len(tool_result.content) == 1:
                result = tool_result.content[0].model_dump_json()
                logger.debug(f"MCP tool '{name}' (server: {self}) succeeded with result: {result}")
                return result
            elif len(tool_result.content) > 1:
                result_list = [item.model_dump() for item in tool_result.content]
                logger.debug(f"MCP tool '{name}' (server: {self}) succeeded with multiple results: {result_list}")
                return json.dumps(result_list)

            logger.warning(f"MCP tool '{name}' (server: {self}) completed without producing a valid result structure.")
            raise ToolError(
                f"Tool '{name}' completed without producing a result. "
                "This might indicate an issue with internal processing."
            )

        created_tool = function_tool(
            _tool_called,
            raw_schema={"name": name, "description": description, "parameters": input_schema},
        )
        created_tool._mcp_server_origin = self  # Add a reference to the MCPServer instance
        return created_tool

    async def aclose(self) -> None:
        try:
            await self._exit_stack.aclose()
        finally:
            self._client = None
            self._lk_tools = None

    @abstractmethod
    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]: ...


class MCPServerHTTP(MCPServer):
    # SSE is going to get replaced soon: https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206

    def __init__(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.url = url
        self.headers = headers
        self._timeout = timeout
        self._see_read_timeout = sse_read_timeout

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self._timeout,
            sse_read_timeout=self._see_read_timeout,
        )

    def __repr__(self) -> str:
        return f"MCPServerHTTP(url={self.url})"


class MCPServerStdio(MCPServer):
    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        client_session_timeout_seconds: float = 5,
    ) -> None:
        super().__init__(client_session_timeout_seconds=client_session_timeout_seconds)
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd

    def client_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        return stdio_client(
            StdioServerParameters(command=self.command, args=self.args, env=self.env, cwd=self.cwd)
        )

    def __repr__(self) -> str:
        return f"MCPServerStdio(command={self.command}, args={self.args}, cwd={self.cwd})"
