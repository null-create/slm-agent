"""
web_search.py
A minimal but production-minded MCP "web_search" tool using the MCP Python SDK.

See: https://pypi.org/project/ddgs/
"""

from __future__ import annotations

from typing import Any, Optional
import logging

from ddgs import DDGS, DDGSException
from pydantic import BaseModel, Field
from mcp import Tool

logger = logging.getLogger("mcp_web_search")
logging.basicConfig(level=logging.INFO)

# ---------- Pydantic schemas for the tool ----------


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: int = Field(
        5, ge=1, le=1000, description="Maximum number of results to return"
    )


class WebSearchResultItem(BaseModel):
    title: str
    snippet: str
    url: str
    source: Optional[str] = None
    rank: int


class WebSearchOutput(BaseModel):
    query: str
    results: list[WebSearchResultItem]


# ---------- Backend interface(s) ----------


class SearchBackend:
    """Abstract backend interface for web search. Implement `search` for a provider."""

    def search_text(self, query: str, limit: int = 5) -> list[WebSearchResultItem]:
        raise NotImplementedError


class DDGSBackend(SearchBackend):
    def __init__(self):
        self.search_engine = DDGS(verify=False)

    def search_text(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            return self.search_engine.text(query=query, max_results=limit)
        except DDGSException as e:
            return [{"message": "search engine failure: " + str(e)}]
        except Exception as e:
            return [{"message": "unexpected error occurred: " + str(e)}]


def make_web_search_tool(backend: SearchBackend) -> Tool:
    """
    Create an MCP Tool that can be registered with FastMCP.
    """

    # Tool handler function
    def _handler(invocation: dict) -> dict[str, Any]:
        try:
            payload = WebSearchInput.model_validate(invocation.get("input", invocation))
        except Exception as e:
            logger.exception("Invalid tool invocation")
            raise

        query = payload.query.strip()
        limit = payload.limit

        # Execute backend search
        try:
            items = backend.search_text(query=query, limit=limit)
        except Exception as e:
            logger.exception("Search backend error")
            # Convert to an MCP-appropriate error response if the SDK has helpers
            raise

        out = WebSearchOutput(query=query, results=items)
        # Return plain dict (MCP SDK will handle JSON serialization)
        return out.model_dump()

    # Construct and return the Tool object expected by FastMCP
    return Tool(
        id="web_search",
        name="Web Search",
        description="Perform a web search and return structured results (title, snippet, url).",
        input_schema=WebSearchInput.model_dump_json(),
        output_schema=WebSearchOutput.model_dump_json(),
        handler=_handler,
        # optional: capabilities, tags, icon, etc.
        keywords=["search", "web", "google", "bing", "query"],
    )
