"""
file_reader.py

Streaming file reader tool that can be served by an MCP server
Improved version with proper error handling and large file support
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Generator, Any, Optional
import logging

from mcp.types import Tool
from pydantic import BaseModel, Field, validator, field_validator

logger = logging.getLogger(__name__)


class FileReadInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file")
    encoding: str = Field("utf-8", description="Text encoding to use when reading")
    chunk_size: int = Field(
        4096, description="Size of each chunk in bytes", ge=512, le=1024 * 1024
    )  # 512B to 1MB
    max_file_size: Optional[int] = Field(
        100 * 1024 * 1024, description="Maximum file size in bytes (default: 100MB)"
    )

    @field_validator("path")
    def validate_path(cls, v):
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()


class FileReadOutput(BaseModel):
    path: str
    content: str
    total_chunks: int
    file_size: int


class FileChunk(BaseModel):
    """Represents a single streamed chunk of text."""

    chunk: str
    index: int
    eof: bool = False
    chunk_size: int = 0
    error: Optional[str] = None


class FileReaderError(Exception):
    """Custom exception for file reader errors."""

    pass


def validate_file_access(path: str, max_file_size: Optional[int] = None) -> Path:
    """Validate file exists, is readable, and within size limits."""
    expanded_path = Path(path).expanduser().resolve()

    # Security: Ensure we're not accessing restricted paths
    # You might want to add more sophisticated path validation here
    if not expanded_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not expanded_path.is_file():
        raise FileReaderError(f"Path is not a file: {path}")

    # Check file size
    file_size = expanded_path.stat().st_size
    if max_file_size and file_size > max_file_size:
        raise FileReaderError(
            f"File too large: {file_size} bytes (max: {max_file_size})"
        )

    return expanded_path


def stream_file_chunks(
    path: str,
    encoding: str = "utf-8",
    chunk_size: int = 4096,
    max_file_size: Optional[int] = None,
) -> Generator[FileChunk, None, None]:
    """Stream the file in chunks with proper error handling."""

    try:
        # Validate file first
        file_path = validate_file_access(path, max_file_size)

        logger.info(f"Starting to stream file: {file_path} (chunk_size: {chunk_size})")

        idx = 0

        # Use context manager for proper resource cleanup
        with open(file_path, "r", encoding=encoding, buffering=chunk_size) as f:
            while True:
                try:
                    data = f.read(chunk_size)

                    if not data:
                        # End of file reached
                        yield FileChunk(chunk="", index=idx, eof=True, chunk_size=0)
                        logger.info(
                            f"Finished streaming file: {file_path} ({idx} chunks)"
                        )
                        break

                    yield FileChunk(
                        chunk=data, index=idx, eof=False, chunk_size=len(data)
                    )
                    idx += 1

                except UnicodeDecodeError as e:
                    # Handle encoding errors gracefully
                    error_msg = f"Encoding error at chunk {idx}: {str(e)}"
                    logger.warning(error_msg)
                    yield FileChunk(
                        chunk="", index=idx, eof=True, chunk_size=0, error=error_msg
                    )
                    break

                except Exception as e:
                    # Handle other read errors
                    error_msg = f"Read error at chunk {idx}: {str(e)}"
                    logger.error(error_msg)
                    yield FileChunk(
                        chunk="", index=idx, eof=True, chunk_size=0, error=error_msg
                    )
                    break

    except (FileNotFoundError, PermissionError, FileReaderError) as e:
        # Yield error chunk for initial file access errors
        error_msg = str(e)
        logger.error(f"File access error: {error_msg}")
        yield FileChunk(chunk="", index=0, eof=True, chunk_size=0, error=error_msg)

    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        yield FileChunk(chunk="", index=0, eof=True, chunk_size=0, error=error_msg)


def read_file(invocation: dict) -> Generator[dict[str, Any], Any, None]:
    """Main file reader handler with improved error handling."""
    try:
        payload = FileReadInput.model_validate(invocation.get("input", invocation))

        for file_chunk in stream_file_chunks(
            payload.path, payload.encoding, payload.chunk_size, payload.max_file_size
        ):
            yield file_chunk.model_dump()

            # If there's an error, stop streaming
            if file_chunk.error:
                break

    except Exception as e:
        # Handle validation errors
        error_msg = f"Input validation error: {str(e)}"
        logger.error(error_msg)
        yield FileChunk(
            chunk="", index=0, eof=True, chunk_size=0, error=error_msg
        ).model_dump()


def read_file_complete(
    path: str,
    encoding: str = "utf-8",
    chunk_size: int = 4096,
    max_file_size: Optional[int] = None,
) -> FileReadOutput:
    """Read entire file and return as a single output (for smaller files)."""

    chunks = []
    total_chunks = 0
    error = None

    for chunk_data in stream_file_chunks(path, encoding, chunk_size, max_file_size):
        chunk = FileChunk.model_validate(chunk_data)

        if chunk.error:
            error = chunk.error
            break

        if chunk.chunk:
            chunks.append(chunk.chunk)

        total_chunks += 1

        if chunk.eof:
            break

    if error:
        raise FileReaderError(error)

    content = "".join(chunks)
    file_path = validate_file_access(path)
    file_size = file_path.stat().st_size

    return FileReadOutput(
        path=str(file_path),
        content=content,
        total_chunks=total_chunks,
        file_size=file_size,
    )


# Convenience function for testing
def read_file_simple(path: str, encoding: str = "utf-8") -> str:
    """Simple synchronous file reader for small files."""
    try:
        result = read_file_complete(path, encoding)
        return result.content
    except FileReaderError as e:
        raise e
    except Exception as e:
        raise FileReaderError(f"Failed to read file: {str(e)}")


def make_file_read_tool() -> Tool:
    """Return a Tool that streams a file using a generator."""
    return Tool(
        id="file_read_stream",
        name="File Read (streaming)",
        description="Reads a text file and streams chunks back.",
        input_schema=FileReadInput.model_dump_json(),
        output_schema=FileChunk.model_dump_json(),
        streaming=True,
    )
