"""alphaXiv MCP client, over streamable HTTP.

The only external dependency in riff, and retrieval only: pre-structured content
for arXiv links, full-text search for the scouts, repository files when grounding
a claim against real code. Every judgement stays on Gemini.

A session is initialised per call rather than kept open. It costs two extra round
trips, but ingest runs once per paper and the scouts batch, so the price is
irrelevant next to sharing mutable session state across FastAPI's worker threads.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from .config import settings

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "riff", "version": "0.2.0"}


class AlphaXivError(RuntimeError):
    """The MCP server refused a request, or answered something unusable."""


def _text_of(result: dict[str, Any]) -> str:
    """Flatten an MCP tool result into the text riff actually consumes."""
    if result.get("isError"):
        raise AlphaXivError(_join_text(result) or "tool reported an error")
    text = _join_text(result)
    if not text:
        raise AlphaXivError("tool returned no text content")
    return text


def _join_text(result: dict[str, Any]) -> str:
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def _parse_sse(body: str) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from an SSE stream.

    The server may answer the same request with either plain JSON or SSE, so
    both shapes are handled rather than assuming one.
    """
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


class _Session:
    def __init__(self) -> None:
        cfg = settings()
        if not cfg.alphaxiv_api_key:
            raise AlphaXivError("RIFF_ALPHAXIV_API_KEY is not set")
        self._url = cfg.alphaxiv_mcp_url
        self._client = httpx.Client(
            timeout=httpx.Timeout(cfg.alphaxiv_timeout_seconds),
            headers={
                "Authorization": f"Bearer {cfg.alphaxiv_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            follow_redirects=True,
        )
        self._session_id: str | None = None
        self._request_id = 0

    def __enter__(self) -> "_Session":
        self._initialise()
        return self

    def __exit__(self, *_: object) -> None:
        try:
            if self._session_id:
                self._client.delete(self._url, headers=self._session_header())
        except httpx.HTTPError:
            # The session expires on its own; failing to close it is not an error
            # worth surfacing over a result we already have.
            pass
        finally:
            self._client.close()

    def _session_header(self) -> dict[str, str]:
        return {"Mcp-Session-Id": self._session_id} if self._session_id else {}

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        response = self._client.post(
            self._url, json=payload, headers=self._session_header()
        )
        if response.status_code >= 400:
            raise AlphaXivError(
                f"{response.status_code} from alphaXiv MCP: {response.text[:400]}"
            )

        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        if "id" not in payload:  # a notification expects no reply
            return None

        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            for message in _parse_sse(response.text):
                if message.get("id") == payload["id"]:
                    return message
            raise AlphaXivError("no response for request in SSE stream")

        try:
            return response.json()
        except ValueError as exc:
            raise AlphaXivError(f"unparseable response: {response.text[:200]}") from exc

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        message = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }
        )
        if message is None:
            raise AlphaXivError(f"no response to {method}")
        if "error" in message:
            error = message["error"]
            raise AlphaXivError(
                f"{method} failed: {error.get('message')} ({error.get('code')})"
            )
        return message.get("result", {})

    def _initialise(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return _text_of(
            self._request("tools/call", {"name": name, "arguments": arguments})
        )

    def list_tools(self) -> list[str]:
        result = self._request("tools/list", {})
        return [tool.get("name", "") for tool in result.get("tools", [])]


def call(tool: str, arguments: dict[str, Any]) -> str:
    with _Session() as session:
        return session.call_tool(tool, arguments)


def available_tools() -> list[str]:
    """Used by the health check to prove the MCP dependency is reachable."""
    with _Session() as session:
        return session.list_tools()


def paper_content(url: str, full_text: bool = False) -> str:
    """Pre-structured paper content, so riff never parses a PDF itself.

    The default report is written for model consumption and is what sectioning
    wants. ``full_text`` is the fallback when a paper has no report yet.
    """
    return call("get_paper_content", {"url": url, "fullText": full_text})


def paper_queries(paper: str, queries: list[str]) -> str:
    """Page-level excerpts for specific questions, citation-ready.

    Multiple queries against one paper cost barely more than one, so callers
    should batch every question they have about a paper into a single call.
    """
    return call("answer_pdf_queries", {"paper": paper, "queries": queries})


def repository_files(github_url: str, path: str = "/") -> str:
    """Real code for grounding a claim. ``/`` returns the tree plus top-level files."""
    return call(
        "read_files_from_github_repository", {"githubUrl": github_url, "path": path}
    )


def discover_papers(
    keywords: list[str],
    question: str,
    difficulty: int = 5,
    published_after: str | None = None,
) -> str:
    """One ranked search. The scouts' only retrieval call per open hypothesis."""
    arguments: dict[str, Any] = {
        "keywords": keywords,
        "question": question,
        "difficulty": difficulty,
    }
    if published_after:
        arguments["published_after"] = published_after
    return call("discover_papers", arguments)
