"""
file_reader.py
A pair of MCP tools for reading files: one non-streaming, one streaming.

Requirements:
    pip install modelcontextprotocol fastapi uvicorn pydantic
"""

from __future__ import annotations
import os
from typing import Generator, Any

from pydantic import BaseModel, Field
from mcp import Tool, schema

# ---------- Schemas ----------


class FileReadInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file")
    encoding: str = Field("utf-8", description="Text encoding to use when reading")


class FileReadOutput(BaseModel):
    path: str
    content: str


class FileChunk(BaseModel):
    """Represents a single streamed chunk of text."""

    chunk: str
    index: int
    eof: bool = False


# ---------- Tool implementations ----------


def make_file_read_tool() -> Tool:
    """Read the whole file at once."""

    def handler(invocation: dict) -> dict:
        payload = FileReadInput.model_validate(invocation.get("input", invocation))
        p = os.path.expanduser(payload.path)
        with open(p, "r", encoding=payload.encoding) as f:
            data = f.read()
        return FileReadOutput(path=p, content=data).model_dump()

    return Tool(
        id="file_read",
        name="File Read (non-streaming)",
        description="Reads the entire contents of a text file at once.",
        input_schema=schema.from_pydantic(FileReadInput),
        output_schema=schema.from_pydantic(FileReadOutput),
        handler=handler,
    )


# --- Streaming version ---


def stream_file_chunks(
    path: str, encoding: str = "utf-8", chunk_size: int = 4096
) -> Generator[FileChunk, None, None]:
    """Stream the file in chunks."""
    p = os.path.expanduser(path)
    with open(p, "r", encoding=encoding) as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                yield FileChunk(chunk="", index=idx, eof=True)
                break
            yield FileChunk(chunk=data, index=idx, eof=False)
            idx += 1


def make_file_read_stream_tool() -> Tool:
    """Return a Tool that streams a file using a generator."""

    def handler(invocation: dict) -> Generator[dict[str, Any], Any, None]:
        payload = FileReadInput.model_validate(invocation.get("input", invocation))
        for part in stream_file_chunks(payload.path, payload.encoding):
            yield part.model_dump()

    return Tool(
        id="file_read_stream",
        name="File Read (streaming)",
        description="Reads a text file and streams chunks back.",
        input_schema=FileReadInput.model_dump_json(),
        output_schema=FileChunk.model_dump_json(),
        handler=handler,
        streaming=True,  # <-- important for MCP
    )


# # ---------- Entrypoint ----------


# def run_server(host: str = "0.0.0.0", port: int = 8001):
#     mcp = FastMCP(
#         name="File Reader MCP",
#         instructions="Provides tools for reading local files (whole or streaming).",
#         tools=[make_file_read_tool(), make_file_read_stream_tool()],
#     )
#     mcp.serve(host=host, port=port)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--host", default="0.0.0.0")
#     parser.add_argument("--port", type=int, default=8001)
#     args = parser.parse_args()
#     run_server(args.host, args.port)
