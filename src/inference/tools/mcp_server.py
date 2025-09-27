from pydantic import BaseModel, Field
from typing import Any, Callable, Collection, List, Literal, Optional

from mcp.types import Tool
from mcp.server.fastmcp import FastMCP

from file_reader.file_reader import make_file_read_tool
from web_search.web_search import make_web_search_tool


class MCPServerConfig(BaseModel):
    """
    Configuration for creating a FastMCP server.
    Mirrors the FastMCP init signature with validation & defaults.
    """

    # Core
    name: Optional[str] = Field(default=None, description="Server name")
    instructions: Optional[str] = Field(
        default=None, description="Human-readable instructions for the server"
    )

    # Auth / security
    auth_server_provider: Optional[Callable[...]] = None
    token_verifier: Optional[Any] = None
    auth: Optional[Any] = None
    transport_security: Optional[Any] = None

    # Tools / events
    tools: Optional[List[Tool]] = [make_file_read_tool(), make_web_search_tool()]
    event_store: Optional[Any] = None

    # Logging / debug
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Network / transport
    host: str = "127.0.0.1"
    port: int = 8000
    mount_path: str = "/"
    sse_path: str = "/sse"
    message_path: str = "/messages/"
    streamable_http_path: str = "/mcp"

    # Output / mode
    json_response: bool = False
    stateless_http: bool = False

    # Warnings
    warn_on_duplicate_resources: bool = True
    warn_on_duplicate_tools: bool = True
    warn_on_duplicate_prompts: bool = True

    # Misc
    dependencies: Collection[str] = ()
    lifespan: Optional[Any] = None  # AbstractAsyncContextManager

    def new_server(self) -> FastMCP:
        """
        Instantiate a FastMCP server using this config.
        """
        return FastMCP(
            name=self.name,
            instructions=self.instructions,
            auth_server_provider=self.auth_server_provider,
            token_verifier=self.token_verifier,
            event_store=self.event_store,
            tools=self.tools,
            debug=self.debug,
            log_level=self.log_level,
            host=self.host,
            port=self.port,
            mount_path=self.mount_path,
            sse_path=self.sse_path,
            message_path=self.message_path,
            streamable_http_path=self.streamable_http_path,
            json_response=self.json_response,
            stateless_http=self.stateless_http,
            warn_on_duplicate_resources=self.warn_on_duplicate_resources,
            warn_on_duplicate_tools=self.warn_on_duplicate_tools,
            warn_on_duplicate_prompts=self.warn_on_duplicate_prompts,
            dependencies=self.dependencies,
            lifespan=self.lifespan,
            auth=self.auth,
            transport_security=self.transport_security,
        )


if __name__ == "__main__":
    mcp_server: FastMCP = MCPServerConfig().new_server()
    mcp_server.run()
