"""
MCP (Model Context Protocol) client for interacting with external tools and servers.
Refactored to use the official MCP SDK with SSE (Server-Sent Events) transport.
"""

import json
import asyncio
import logging
from typing import Any, Optional
from dataclasses import dataclass

from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel


@dataclass
class ToolCall:
    """Represents a tool call extracted from model output."""

    name: str
    parameters: dict[str, Any]
    call_id: Optional[str] = None


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    url: str
    api_key: Optional[str] = None
    timeout: int = 30
    headers: Optional[dict[str, str]] = None


class MCPClient:
    """Client for interacting with MCP servers via SSE transport and managing tool calls."""

    def __init__(self, server_configs: list[MCPServerConfig]):
        """Initialize MCP client with server configurations."""
        self.server_configs = {config.name: config for config in server_configs}
        self.logger = logging.getLogger(__name__)

        # Active sessions and their tools
        self.sessions: dict[str, ClientSession] = {}
        self.available_tools: dict[str, dict[str, Any]] = {}
        self.tool_to_server: dict[str, str] = {}

    async def connect_to_server(self, server_name: str) -> None:
        """Connect to an MCP server via SSE and initialize the session."""
        if server_name not in self.server_configs:
            raise ValueError(f"Server {server_name} not configured")

        if server_name in self.sessions:
            self.logger.info(f"Already connected to {server_name}")
            return

        config: MCPServerConfig = self.server_configs[server_name]

        try:
            # Prepare headers
            headers = config.headers.copy() if config.headers else {}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            # Connect to the server using SSE transport
            read_stream, write_stream = await sse_client(
                url=config.url, headers=headers, timeout=config.timeout
            )

            # Create and initialize session
            session = ClientSession(read_stream, write_stream)
            await session.initialize()

            self.sessions[server_name] = session

            # Discover tools from this server
            await self._discover_tools_from_server(server_name, session)

            self.logger.info(f"Connected to MCP server via SSE: {server_name}")

        except Exception as e:
            self.logger.error(f"Error connecting to {server_name}: {e}")
            raise

    async def _discover_tools_from_server(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Discover available tools from a connected MCP server."""
        try:
            # List available tools from the server
            tools_response = await session.list_tools()

            for tool in tools_response.tools:
                tool_info = {
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                }

                self.available_tools[tool.name] = tool_info
                self.tool_to_server[tool.name] = server_name

                self.logger.info(
                    f"Registered tool '{tool.name}' from server '{server_name}'"
                )

        except Exception as e:
            self.logger.error(f"Error discovering tools from {server_name}: {e}")
            raise

    async def connect_all_servers(self) -> None:
        """Connect to all configured MCP servers."""
        tasks = [
            self.connect_to_server(server_name)
            for server_name in self.server_configs.keys()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any connection failures
        for server_name, result in zip(self.server_configs.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to connect to {server_name}: {result}")

    def parse_tool_calls(self, model_output: str) -> list[ToolCall]:
        """Parse tool calls from model output."""
        tool_calls = []

        # Look for tool usage blocks in the output
        lines = model_output.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if line == "<tool_use>":
                # Found start of tool use block
                tool_call = self._parse_single_tool_call(lines, i)
                if tool_call:
                    tool_calls.append(tool_call)

                # Skip to end of this tool block
                while i < len(lines) and lines[i].strip() != "</tool_use>":
                    i += 1
            i += 1

        return tool_calls

    def _parse_single_tool_call(
        self, lines: list[str], start_idx: int
    ) -> Optional[ToolCall]:
        """Parse a single tool call block."""
        tool_name = None
        parameters = {}
        call_id = None

        i = start_idx + 1
        while i < len(lines):
            line = lines[i].strip()

            if line == "</tool_use>":
                break
            elif line == "<tool_name>":
                # Get tool name
                i += 1
                if i < len(lines):
                    tool_name = lines[i].strip()
            elif line == "<call_id>":
                # Get call ID if present
                i += 1
                if i < len(lines):
                    call_id = lines[i].strip()
            elif line == "<parameters>":
                # Parse parameters JSON
                i += 1
                param_lines = []
                while i < len(lines) and lines[i].strip() != "</parameters>":
                    param_lines.append(lines[i])
                    i += 1

                try:
                    param_text = "\n".join(param_lines)
                    parameters = json.loads(param_text)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing parameters: {e}")
                    return None

            i += 1

        if tool_name and tool_name in self.available_tools:
            return ToolCall(name=tool_name, parameters=parameters, call_id=call_id)

        return None

    async def execute_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute a tool call on the appropriate MCP server."""
        if tool_call.name not in self.available_tools:
            return {"error": f"Unknown tool: {tool_call.name}", "success": False}

        server_name = self.tool_to_server.get(tool_call.name)
        if not server_name or server_name not in self.sessions:
            return {
                "error": f"Server for tool {tool_call.name} not connected",
                "success": False,
            }

        session = self.sessions[server_name]

        try:
            # Call the tool
            result = await session.call_tool(
                tool_call.name, arguments=tool_call.parameters
            )

            # Extract content from the result
            content = self._extract_tool_result_content(result)

            return {
                "result": content,
                "success": not result.isError,
                "tool": tool_call.name,
                "call_id": tool_call.call_id,
            }

        except Exception as e:
            self.logger.error(f"Error executing {tool_call.name}: {e}")
            return {
                "error": str(e),
                "success": False,
                "tool": tool_call.name,
                "call_id": tool_call.call_id,
            }

    def _extract_tool_result_content(self, result) -> Any:
        """Extract content from CallToolResult."""
        if not result.content:
            return None

        # Handle multiple content items
        if len(result.content) == 1:
            content_item = result.content[0]
            if hasattr(content_item, "text"):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return content_item.text
            return content_item
        else:
            extracted = []
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        extracted.append(json.loads(item.text))
                    except json.JSONDecodeError:
                        extracted.append(item.text)
                else:
                    extracted.append(item)
            return extracted

    async def execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[dict[str, Any]]:
        """Execute multiple tool calls concurrently."""
        tasks = [self.execute_tool_call(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "error": str(result),
                        "success": False,
                        "tool": tool_calls[i].name,
                        "call_id": tool_calls[i].call_id,
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def reconnect_server(self, server_name: str) -> None:
        """Reconnect to a specific server if connection is lost."""
        if server_name in self.sessions:
            try:
                await self.sessions[server_name].close()
            except Exception as e:
                self.logger.warning(f"Error closing old session for {server_name}: {e}")

            del self.sessions[server_name]

        await self.connect_to_server(server_name)

    async def close(self):
        """Close all MCP server connections."""
        for server_name, session in self.sessions.items():
            try:
                await session.close()
                self.logger.info(f"Closed connection to {server_name}")
            except Exception as e:
                self.logger.error(f"Error closing {server_name}: {e}")

        self.sessions.clear()
        self.available_tools.clear()
        self.tool_to_server.clear()

    def get_available_tools_description(self) -> str:
        """Get a description of all available tools for the model."""
        if not self.available_tools:
            return "No tools available. Please connect to MCP servers first."

        descriptions = []
        descriptions.append("Available tools:")

        for tool_name, config in self.available_tools.items():
            descriptions.append(f"\n- {tool_name}: {config['description']}")

            # Show input schema if available
            schema = config.get("input_schema", {})
            if schema and "properties" in schema:
                params = []
                required = schema.get("required", [])
                for param_name, param_info in schema["properties"].items():
                    param_type = param_info.get("type", "any")
                    is_required = "*" if param_name in required else ""
                    params.append(f"{param_name}({param_type}{is_required})")

                if params:
                    descriptions.append(f"  Parameters: {', '.join(params)}")

        descriptions.append("\n\nTo use a tool, format your response like:")
        descriptions.append("<tool_use>")
        descriptions.append("<tool_name>tool_name</tool_name>")
        descriptions.append("<parameters>")
        descriptions.append('{"param1": "value1", "param2": "value2"}')
        descriptions.append("</parameters>")
        descriptions.append("</tool_use>")

        return "\n".join(descriptions)

    def get_tools_for_prompt(self) -> list[dict[str, Any]]:
        """Get tools in a format suitable for model prompts."""
        tools = []
        for tool_name, config in self.available_tools.items():
            tools.append(
                {
                    "name": tool_name,
                    "description": config["description"],
                    "input_schema": config["input_schema"],
                }
            )
        return tools

    async def list_resources(self, server_name: str) -> list[dict[str, Any]]:
        """List available resources from a specific server."""
        if server_name not in self.sessions:
            raise ValueError(f"Not connected to server: {server_name}")

        session = self.sessions[server_name]
        try:
            resources_response = await session.list_resources()
            return [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": resource.mimeType,
                }
                for resource in resources_response.resources
            ]
        except Exception as e:
            self.logger.error(f"Error listing resources from {server_name}: {e}")
            return []

    async def read_resource(self, server_name: str, uri: str) -> dict[str, Any]:
        """Read a resource from a specific server."""
        if server_name not in self.sessions:
            raise ValueError(f"Not connected to server: {server_name}")

        session = self.sessions[server_name]
        try:
            resource_response = await session.read_resource(uri)
            return {
                "uri": uri,
                "contents": resource_response.contents,
                "success": True,
            }
        except Exception as e:
            self.logger.error(f"Error reading resource {uri} from {server_name}: {e}")
            return {
                "uri": uri,
                "error": str(e),
                "success": False,
            }
