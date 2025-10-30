import os
import json
import logging
from typing import Any

# MCP Resources
from mcp.types import (
    Tool,
    TextContent,
)
from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession

# Tool implementations
from file_reader.file_reader import (
    FileReaderError,
    FileReadInput,
    FileChunk,
    FileReadInput,
    FileReadOutput,
    read_file,
    make_file_read_tool,
    validate_file_access,
)
from web_search.web_search import (
    DDGSBackend,
    WebSearchError,
    WebSearchInput,
    WebSearchOutput,
    WebSearchResultItem,
    WebSearchInput,
    WebSearchOutput,
    WebSearchResultItem,
    make_web_search_tool,
    run_web_search,
)

# Server resources
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request

# Streaming Configuration
MAX_STREAMING_SIZE = 10 * 1024 * 1024  # 10MB - above this, we use true streaming
MAX_CONCAT_SIZE = 1 * 1024 * 1024  # 1MB - above this, warn about memory usage

# Load server configurations
config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "server.json")
if not os.path.exists(config_file):
    raise FileNotFoundError(f"{config_file} file not found")

with open(config_file, "r") as f:
    configs = json.load(f)

server_configs = configs["server"]

# Set up logging
logging.basicConfig(
    level=server_configs.get("log_level", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_server")


async def handle_websearch_tool_call(name: str, arguments: dict) -> dict[str, Any]:
    """
    Handle tool execution requests.
    """
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    # Extract arguments
    query = arguments.get("query")
    if not query:
        raise ValueError("Missing required argument: query")

    num_results = arguments.get("num_results", 10)

    # Validate arguments
    if not isinstance(query, str) or len(query.strip()) == 0:
        raise ValueError("Query must be a non-empty string")

    try:
        logger.info(f"Performing web search for query: {query}")

        # Execute search
        search_engine = DDGSBackend()
        payload = WebSearchInput(query=query, limit=num_results)

        results = await run_web_search(payload.model_dump(), search_engine)

        # Parse results
        return WebSearchOutput(
            query=payload.query.strip(),
            results=[WebSearchResultItem(**result) for result in results],
        ).model_dump()

    except WebSearchError as e:
        logger.error(f"Web search error: {str(e)}")
        return {"websearch-error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error during web search: {str(e)}")
        return {"error": str(e)}


async def handle_file_read_tool_call(
    name: str, arguments: dict[str, Any]
) -> list[TextContent]:
    """Handle tool execution requests."""

    if name == "read_file":
        return await handle_read_file(arguments)
    elif name == "read_file_info":
        return await handle_read_file_info(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_read_file_info(arguments: dict[str, Any]) -> list[TextContent]:
    """Get file information without reading contents."""

    try:
        from file_reader import validate_file_access
        from pathlib import Path

        path = arguments.get("path", "")
        chunk_size = arguments.get("chunk_size", 4096)

        if not path:
            raise ValueError("Path is required")

        # Validate and get file info
        file_path = validate_file_access(path)
        file_size = file_path.stat().st_size

        # Calculate chunks needed
        chunks_needed = (file_size + chunk_size - 1) // chunk_size  # Ceiling division

        # Determine recommended approach
        if file_size > MAX_STREAMING_SIZE:
            approach = "streaming (large file)"
        elif file_size > MAX_CONCAT_SIZE:
            approach = "concatenated with memory warning"
        else:
            approach = "concatenated (optimal)"

        info = f"""File Information:
Path: {file_path}
Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)
Chunks needed: {chunks_needed:,} (at {chunk_size:,} bytes per chunk)
Recommended approach: {approach}

Memory usage estimate:
- Streaming: ~{chunk_size:,} bytes per chunk
- Full concatenation: ~{file_size:,} bytes"""

        return [TextContent(type="text", text=info)]

    except Exception as e:
        error_msg = f"Error getting file info: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_read_file(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle file reading with intelligent streaming vs concatenation."""

    try:
        # Validate inputs
        file_input = FileReadInput.model_validate(arguments)
        force_streaming = arguments.get("force_streaming", False)

        file_path = validate_file_access(file_input.path, file_input.max_file_size)
        file_size = file_path.stat().st_size

        # Decide on approach based on file size
        if force_streaming or file_size > MAX_STREAMING_SIZE:
            return await handle_streaming_read(arguments, file_size)
        else:
            return await handle_concatenated_read(arguments, file_size)

    except FileReaderError as e:
        error_msg = str(e)
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_concatenated_read(
    arguments: dict[str, Any], file_size: int
) -> list[TextContent]:
    """Handle small to medium files by concatenating all chunks."""

    try:
        chunks = []
        chunk_count = 0
        has_error = False
        error_msg = ""

        # Memory warning for larger files
        if file_size > MAX_CONCAT_SIZE:
            logger.warning(f"Reading large file ({file_size:,} bytes) into memory")

        # Stream and collect chunks
        for chunk_data in read_file({"input": arguments}):
            chunk = FileChunk.model_validate(chunk_data)
            chunk_count += 1

            if chunk.error:
                has_error = True
                error_msg = chunk.error
                break

            if chunk.chunk:
                chunks.append(chunk.chunk)

            if chunk.eof:
                break

        if has_error:
            return [TextContent(type="text", text=f"Error: {error_msg}")]

        # Concatenate and return
        content = "".join(chunks)

        logger.info(
            f"Successfully read file: {arguments.get('path')} "
            f"({len(content):,} characters, {chunk_count} chunks, {file_size:,} bytes)"
        )

        return [TextContent(type="text", text=content)]

    except Exception as e:
        error_msg = f"Error in concatenated read: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]


async def handle_streaming_read(
    arguments: dict[str, Any], file_size: int
) -> list[TextContent]:
    """Handle large files with streaming approach."""

    try:
        chunk_size = arguments.get("chunk_size", 4096)
        estimated_chunks = (file_size + chunk_size - 1) // chunk_size

        # For very large files, we provide a streaming summary instead of full content
        logger.info(
            f"Streaming large file: {arguments.get('path')} "
            f"({file_size:,} bytes, ~{estimated_chunks:,} chunks)"
        )

        # Read first few chunks and last few chunks as a sample
        chunks = []
        chunks_read = 0
        first_chunks = []
        preview_len = 10  # limit preview length to 10 chunks
        content_preview = []
        has_error = False
        error_msg = ""

        # Read first 5 chunks for preview
        for chunk_data in read_file({"input": arguments}):
            chunk = FileChunk.model_validate(chunk_data)
            chunks_read += 1

            if chunk.error:
                has_error = True
                error_msg = chunk.error
                break

            if chunk.chunk and len(first_chunks) < 5:
                first_chunks.append(chunk.chunk)
                if chunks_read < preview_len:
                    content_preview.append(
                        f"Chunk {chunk.index}: {len(chunk.chunk)} chars"
                    )
                chunks.append(chunk.chunk)

            # NOTE: temp until we implement different approach to handle the streaming responses
            # if chunks_read >= 10 or chunk.eof:
            if chunk.eof:
                break

        if has_error:
            return [TextContent(type="text", text=f"Error: {error_msg}")]

        # Create streaming summary and include full file content in the message at the bottom for now.
        #
        # NOTE: will processing the file in chunks with the LLM be faster or slower? Inference calls would
        # probably be sequential, so it would probably bottleneck there if that's the case.
        preview_content = "".join(first_chunks)
        full_content = "".join(chunks)

        result = f"""Large File Streaming Summary:
File: {arguments.get('path')}
Size: {file_size:,} bytes
Estimated chunks: {estimated_chunks:,}
Chunks processed: {chunks_read}

Content Preview (first {len(first_chunks)} chunks):
{preview_content[:1000]}{"..." if len(preview_content) > 1000 else ""}

Chunk Details:
{chr(10).join(content_preview)}

Full Content:\n
{full_content}
"""

        return [TextContent(type="text", text=result)]

    except Exception as e:
        error_msg = f"Error in streaming read: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]


if __name__ == "__main__":
    server = Server("slm-mcp-server")

    # Register tools with the server
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [make_web_search_tool(), make_file_read_tool()]

    # Tool call handler
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        logger.info(f"Tool called: {name} with arguments: {arguments}")

        if name == "web_search":
            result = await handle_websearch_tool_call(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        elif name == "file_reader":
            return await handle_file_read_tool_call(name, arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def health_check(request: Request):
        """Health check endpoint"""
        return JSONResponse({"status": "healthy", "service": "mcp-server"})

    async def mcp_handler(request: Request):
        """Handle MCP requests over HTTP"""
        try:
            if request.method == "POST":
                # Get the JSON-RPC request
                body = await request.json()

                # Create a session for this request
                session = ServerSession(server, InitializationOptions())
                response = await session.handle_request(body)

                return JSONResponse(response)
            else:
                return JSONResponse(
                    {
                        "error": "Method not allowed",
                        "message": "Only POST requests are supported",
                    },
                    status_code=405,
                )
        except Exception as e:
            logger.error(f"Error handling MCP request: {str(e)}")
            return JSONResponse(
                {"error": "Internal server error", "message": str(e)}, status_code=500
            )

    # Create the Starlette application
    app = Starlette(
        routes=[
            Route("/health", health_check, methods=["GET"]),
            Route("/mcp", mcp_handler, methods=["POST", "GET"]),
            Route(
                "/mcp/", mcp_handler, methods=["POST", "GET"]
            ),  # Handle trailing slash
            Route("/mcp/search", mcp_handler, methods=["POST", "GET"]),
        ]
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Start the server
    uvicorn.run(
        app,
        host=server_configs.get("host", "localhost"),
        port=server_configs.get("port", 9000),
        log_level=server_configs.get("log_level", "info").lower(),
        access_log=True,
    )
