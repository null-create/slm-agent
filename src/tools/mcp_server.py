import os
import json
import logging
from typing import Any

from mcp.server import FastMCP

# Tool implementations
from file_reader.file_reader import (
    FileReaderError,
    FileReadInput,
    FileChunk,
    validate_file_access,
    read_file,
)
from web_search.web_search import (
    DDGSBackend,
    WebSearchError,
    WebSearchInput,
    WebSearchOutput,
    WebSearchResultItem,
    run_web_search,
)

# Streaming Configuration
MAX_STREAMING_SIZE = 10 * 1024 * 1024  # 10MB - above this, we use true streaming
MAX_CONCAT_SIZE = 1 * 1024 * 1024  # 1MB - above this, warn about memory usage

# Streamable HTTP options
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = os.getenv("MCP_PORT", "9000")

# Load server configurations
config_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "server.json")
if not os.path.exists(config_file):
    raise FileNotFoundError(f"{config_file} file not found")

with open(config_file, "r") as f:
    configs = json.load(f)

server_configs = configs["server"]

# Set up logging
logging.basicConfig(
    level=server_configs.get("log-level", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP server
mcp = FastMCP(name="slm-mcp-server", host=HOST, port=PORT, debug=True)


@mcp.tool(name="web_search")
async def web_search(query: str, num_results: int = 10) -> dict[str, Any]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (default: 10)

    Returns:
        Dictionary containing search results with query and result items
    """
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


@mcp.tool(name="read_file_info")
async def read_file_info(path: str, chunk_size: int = 4096) -> str:
    """
    Get file information without reading contents.

    Args:
        path: Path to the file
        chunk_size: Size of chunks for estimation (default: 4096)

    Returns:
        File information including size, estimated chunks, and recommended approach
    """
    try:
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

        return info

    except Exception as e:
        error_msg = f"Error getting file info: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


@mcp.tool(name="read_file")
async def read_file(
    path: str,
    chunk_size: int = 4096,
    max_file_size: int = 100 * 1024 * 1024,
    force_streaming: bool = False,
) -> str:
    """
    Read a file with intelligent streaming vs concatenation.

    Args:
        path: Path to the file to read
        chunk_size: Size of chunks to read (default: 4096)
        max_file_size: Maximum allowed file size in bytes (default: 100MB)
        force_streaming: Force streaming mode regardless of file size

    Returns:
        File contents or streaming summary for large files
    """
    try:
        # Validate inputs
        file_input = FileReadInput(
            path=path, chunk_size=chunk_size, max_file_size=max_file_size
        )

        file_path = validate_file_access(file_input.path, file_input.max_file_size)
        file_size = file_path.stat().st_size

        # Decide on approach based on file size
        if force_streaming or file_size > MAX_STREAMING_SIZE:
            return await _handle_streaming_read(file_input.model_dump(), file_size)
        else:
            return await _handle_concatenated_read(file_input.model_dump(), file_size)

    except FileReaderError as e:
        error_msg = str(e)
        logger.error(error_msg)
        return f"Error: {error_msg}"

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


async def _handle_concatenated_read(arguments: dict[str, Any], file_size: int) -> str:
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
            return f"Error: {error_msg}"

        # Concatenate and return
        content = "".join(chunks)

        logger.info(
            f"Successfully read file: {arguments.get('path')} "
            f"({len(content):,} characters, {chunk_count} chunks, {file_size:,} bytes)"
        )

        return content

    except Exception as e:
        error_msg = f"Error in concatenated read: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


async def _handle_streaming_read(arguments: dict[str, Any], file_size: int) -> str:
    """Handle large files with streaming approach."""
    try:
        chunk_size = arguments.get("chunk_size", 4096)
        estimated_chunks = (file_size + chunk_size - 1) // chunk_size

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

            if chunk.eof:
                break

        if has_error:
            return f"Error: {error_msg}"

        # Create streaming summary and include full file content
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

Full Content:

{full_content}
"""

        return result

    except Exception as e:
        error_msg = f"Error in streaming read: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


if __name__ == "__main__":
    try:
        logger.info(f"Starting server at: {HOST}:{PORT}")
        mcp.run(transport="streamable-http")
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
