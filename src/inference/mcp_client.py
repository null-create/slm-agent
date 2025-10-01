"""
MCP (Model Context Protocol) client for interacting with external tools and servers.
"""

import os
import json
import asyncio
import httpx
import logging
from typing import Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

from pydantic import BaseModel, ValidationError


@dataclass
class ToolCall:
    """Represents a tool call extracted from model output."""

    name: str
    parameters: dict[str, Any]
    call_id: Optional[str] = None


class MCPServer(BaseModel):
    """Configuration for an MCP server."""

    name: str
    base_url: str
    auth_token: Optional[str] = None
    timeout: int = 30


class MCPClient:
    """Client for interacting with MCP servers and managing tool calls."""

    def __init__(self, servers: list[MCPServer]):
        """Initialize MCP client with server configurations."""
        self.servers = {server.name: server for server in servers}
        self.logger = logging.getLogger(__name__)
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Available tools registry
        self.available_tools = {}
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize available tools from all servers."""
        # TODO: this needs to be dynamically built based off what the server tells us
        self.available_tools = {
            "file_reader": {
                "server": "file_server",
                "endpoint": "/read",
                "description": "Read and analyze files",
                "parameters": {
                    "file_path": {"type": "string", "required": True},
                    "operation": {"type": "string", "default": "read"},
                },
            },
            "web_search": {
                "server": "web_search",
                "endpoint": "/search",
                "description": "Perform a web search and return results",
                "parameters": {"query": {"type": "string", "required": True}},
            },
        }

    async def discover_tools(self, server_name: str) -> dict[str, Any]:
        """Discover available tools from a specific MCP server."""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not configured")

        server = self.servers[server_name]

        try:
            response = await self.http_client.get(
                urljoin(server.base_url, "/tools"),
                headers=self._get_auth_headers(server),
            )
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            self.logger.error(f"Error discovering tools from {server_name}: {e}")
            return []

    def _get_auth_headers(self, server: MCPServer) -> dict[str, str]:
        """Get authentication headers for server requests."""
        headers = {"Content-Type": "application/json"}
        if server.auth_token:
            headers["Authorization"] = f"Bearer {server.auth_token}"
        return headers

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
            return ToolCall(name=tool_name, parameters=parameters)

        return None

    async def execute_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute a tool call on the appropriate MCP server."""
        if tool_call.name not in self.available_tools:
            return {"error": f"Unknown tool: {tool_call.name}", "success": False}

        tool_config = self.available_tools[tool_call.name]
        server_name = tool_config["server"]

        if server_name not in self.servers:
            return {"error": f"Server {server_name} not configured", "success": False}

        server = self.servers[server_name]
        endpoint = tool_config["endpoint"]

        try:
            # Validate parameters
            validated_params = self._validate_parameters(
                tool_call.parameters, tool_config["parameters"]
            )

            # Make request to MCP server
            response = await self.http_client.post(
                urljoin(server.base_url, endpoint),
                json=validated_params,
                headers=self._get_auth_headers(server),
            )

            response.raise_for_status()
            result = response.json()

            return {"result": result, "success": True, "tool": tool_call.name}

        except httpx.RequestError as e:
            self.logger.error(f"Error executing {tool_call.name}: {e}")
            return {"error": str(e), "success": False, "tool": tool_call.name}
        except ValidationError as e:
            return {
                "error": f"Parameter validation failed: {e}",
                "success": False,
                "tool": tool_call.name,
            }

    def _validate_parameters(
        self, params: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and fill default parameters."""
        validated = {}

        for param_name, param_config in schema.items():
            if param_name in params:
                validated[param_name] = params[param_name]
            elif param_config.get("required", False):
                raise ValidationError(f"Required parameter {param_name} missing")
            elif "default" in param_config:
                validated[param_name] = param_config["default"]

        # NOTE: this could actually be an issue. Double check this!
        # Add any extra parameters that aren't in schema
        for param_name, value in params.items():
            if param_name not in schema:
                validated[param_name] = value

        return validated

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
                    {"error": str(result), "success": False, "tool": tool_calls[i].name}
                )
            else:
                processed_results.append(result)

        return processed_results

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()

    def get_available_tools_description(self) -> str:
        """Get a description of all available tools for the model."""
        descriptions = []
        descriptions.append("Available tools:")

        for tool_name, config in self.available_tools.items():
            descriptions.append(f"- {tool_name}: {config['description']}")
            params = ", ".join(
                [
                    f"{name}({cfg.get('type', 'any')}{'*' if cfg.get('required') else ''})"
                    for name, cfg in config["parameters"].items()
                ]
            )
            descriptions.append(f"  Parameters: {params}")

        descriptions.append("\nTo use a tool, format your response like:")
        descriptions.append("<tool_use>")
        descriptions.append("<tool_name>tool_name</tool_name>")
        descriptions.append("<parameters>")
        descriptions.append('{"param1": "value1", "param2": "value2"}')
        descriptions.append("</parameters>")
        descriptions.append("</tool_use>")

        return "\n".join(descriptions)


class MockMCPClient(MCPClient):
    """Mock MCP client for testing without real servers."""

    def __init__(self):
        """Initialize mock client with fake servers."""
        mock_servers = [
            MCPServer(name="search_server", base_url="http://localhost:8001"),
            MCPServer(name="math_server", base_url="http://localhost:8002"),
            MCPServer(name="weather_server", base_url="http://localhost:8003"),
            MCPServer(name="file_server", base_url="http://localhost:8004"),
        ]
        super().__init__(mock_servers)

    async def execute_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute mock tool calls with simulated responses."""

        mock_responses = {
            "web_search": {
                "results": [
                    {
                        "title": "Sample Result 1",
                        "url": "https://example1.com",
                        "snippet": "Mock search result",
                    },
                    {
                        "title": "Sample Result 2",
                        "url": "https://example2.com",
                        "snippet": "Another mock result",
                    },
                ],
                "query": tool_call.parameters.get("query", ""),
            },
            "calculator": {
                "result": 42,
                "expression": tool_call.parameters.get("expression", ""),
                "explanation": "Mock calculation result",
            },
            "weather": {
                "location": tool_call.parameters.get("location", "Unknown"),
                "temperature": 22,
                "condition": "Partly Cloudy",
                "humidity": 65,
            },
            "file_reader": {
                "content": "Mock file content",
                "file_path": tool_call.parameters.get("file_path", ""),
                "size": 1024,
            },
        }

        if tool_call.name in mock_responses:
            return {
                "result": mock_responses[tool_call.name],
                "success": True,
                "tool": tool_call.name,
            }
        else:
            return {"error": f"Unknown tool: {tool_call.name}", "success": False}
