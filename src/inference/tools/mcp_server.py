import base64
import mimetypes
from pathlib import Path
from typing import Optional, Generator

from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

from server_config import MCPServerConfig


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    base_url: str
    auth_token: Optional[str] = None
    timeout: int = 30


# Initialize a FastMCP server using config wrapper class
mcp_config = MCPServerConfig()
mcp: FastMCP = mcp_config.new_server()


@mcp.tool()
def file_reader_non_streaming(file_path: str, max_bytes: int = 5 * 1024 * 1024) -> str:
    """
    Read a file and return its contents.

    - For text files → return text as a string.
    - For binary files (png, jpg, so, etc.) → return base64 data URI.
    - Raises FileNotFoundError if the path doesn't exist.
    - Limits read to `max_bytes` bytes (default: 5 MB).
    """
    p = Path(file_path)

    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    # Guess MIME type from extension
    mime_type, _ = mimetypes.guess_type(p.name)
    mime_type = mime_type or "application/octet-stream"

    # Decide whether to treat as text or binary
    is_text = False
    if mime_type.startswith("text/"):
        is_text = True
    elif mime_type in {"application/json", "application/xml"}:
        is_text = True

    if is_text:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    else:
        with p.open("rb") as f:
            raw = f.read(max_bytes)
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{b64}"


@mcp.tool()
def file_reader(file_path: str, chunk_size: int = 65536) -> Generator[str, None, None]:
    """
    Stream a file's contents in chunks.

    - For text → yields UTF-8 text chunks.
    - For binary → yields base64 data URI *header* once, then base64 payload chunks.
    - `chunk_size` controls how many bytes per read (default: 64 KiB).
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    mime_type, _ = mimetypes.guess_type(p.name)
    mime_type = mime_type or "application/octet-stream"

    is_text = False
    if mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
    }:
        is_text = True

    if is_text:
        # Stream text in chunks
        with p.open("r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    else:
        # Stream binary: yield data URI header first, then base64 chunks
        yield f"data:{mime_type};base64,"
        with p.open("rb") as f:
            while True:
                raw: bytes = f.read(chunk_size)
                if not raw:
                    break
                yield base64.b64encode(raw).decode("ascii")


# Add a resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Return a greeting for a given name."""
    return f"Hello, {name}!"


# Add a prompt template
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """
    Generate a greeting prompt.
    style can be "friendly", "formal", etc.
    """
    if style == "formal":
        return f"Please compose a formal greeting to {name}."
    else:
        return f"Please compose a warm, friendly greeting to {name}."


if __name__ == "__main__":
    # Run the server. Default transport is stdio
    config = MCPServerConfig()
    mcp.run()
