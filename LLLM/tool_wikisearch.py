"""
Wikimedia wiki search and page retrieval tool for model tool-use loops.

The tool exposes one ``wikisearch`` function with two actions:
``search`` queries a Wikimedia wiki through the MediaWiki API, and ``open``
retrieves a plain-text extract for a wiki page URL or page title.
"""

from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass
from typing import cast
from urllib.parse import quote, unquote, urlparse

import requests

from .generator_with_tool import Tool

_DEFAULT_WIKI = "https://en.wikipedia.org"
_REQUEST_TIMEOUT_SECONDS = 10
_HEADERS = {
    "User-Agent": (
        "LLLM wikisearch tool/0.1 "
        "(https://github.com/; contact: llm-tool@example.invalid)"
    )
}
_SUPPORTED_HOST_SUFFIXES = (
    ".wikipedia.org",
    ".wiktionary.org",
    ".wikibooks.org",
    ".wikiquote.org",
    ".wikisource.org",
    ".wikiversity.org",
    ".wikivoyage.org",
    ".wikinews.org",
    ".wikimedia.org",
)
_SUPPORTED_EXACT_HOSTS = {"www.wikidata.org"}

WIKISEARCH_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "wikisearch",
        "description": (
            "Use action='search' to find relevant wiki pages. "
            "Use action='open' to open a wiki URL or title. "
            "Only supports Wikimedia-compatible wiki pages. "
            "When looking for data, do not stop at the action='search' "
            "step: action='search' is made to find pages. Open URL with action='open' "
            "to find the actual data in pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "open"],
                    "description": "Use 'search' for a wiki query or 'open' for a page.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query. Required when action is 'search'.",
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Wikimedia page URL to retrieve. Required for open unless "
                        "title is provided."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Wiki page title to retrieve. Required for open unless "
                        "url is provided."
                    ),
                },
                "wiki": {
                    "type": "string",
                    "description": (
                        "Optional Wikimedia wiki base URL, such as "
                        "https://fr.wikipedia.org. Defaults to English Wikipedia."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results to return, from 1 to 10.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return for opened pages.",
                },
            },
            "required": ["action"],
        },
    },
}


@dataclass(frozen=True)
class _WikiPage:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class _OpenTarget:
    base_url: str
    title: str


def wikisearch_tool() -> Tool:
    """Return the ready-to-register ``wikisearch`` tool."""
    return Tool(
        schema=copy.deepcopy(WIKISEARCH_TOOL_SCHEMA), execute=execute_wikisearch
    )


def execute_wikisearch(arguments: dict[str, object]) -> str:
    """Execute the wikisearch tool."""
    action = arguments.get("action")
    if not isinstance(action, str):
        raise ValueError("action must be a string")
    if action == "search":
        query = _require_non_empty_string(arguments, "query")
        wiki = _optional_wiki(arguments)
        max_results = _optional_int(arguments, "max_results", default=5)
        if max_results < 1 or max_results > 10:
            raise ValueError("max_results must be between 1 and 10")
        return _execute_search(query, wiki, max_results)
    if action == "open":
        target = _open_target(arguments)
        max_chars = _optional_int(arguments, "max_chars", default=6000)
        if max_chars < 500 or max_chars > 20000:
            raise ValueError("max_chars must be between 500 and 20000")
        return _execute_open(target, max_chars)
    raise ValueError("action must be 'search' or 'open'")


def _execute_search(
    query: str, wiki: str, max_results: int, include_snippet: bool = False
) -> str:
    payload = _api_get(
        wiki,
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max_results),
            "format": "json",
            "utf8": "1",
        },
    )
    raw_results = _query_dict(payload).get("search", [])
    if not isinstance(raw_results, list) or not raw_results:
        return f"No wiki results found for: {query}"

    results: list[_WikiPage] = []
    for raw_result in raw_results[:max_results]:
        if not isinstance(raw_result, dict):
            continue
        title = raw_result.get("title")
        if not isinstance(title, str) or not title:
            continue
        snippet = raw_result.get("snippet")
        results.append(
            _WikiPage(
                title=title,
                url=_page_url(wiki, title),
                snippet=_clean_snippet(snippet if isinstance(snippet, str) else ""),
            )
        )
    if not results:
        return f"No wiki results found for: {query}"

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}\nURL: {result.url}")
        if result.snippet and include_snippet is True:
            lines.append(f"Snippet: {result.snippet}")
    return "\n".join(lines)


def _execute_open(target: _OpenTarget, max_chars: int) -> str:
    payload = _api_get(
        target.base_url,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": target.title,
            "format": "json",
            "utf8": "1",
        },
    )
    page = _first_page(payload)
    title = page.get("title")
    if not isinstance(title, str) or not title:
        title = target.title
    if page.get("missing") is not None:
        raise ValueError(f"wiki page not found: {target.title}")
    extract = page.get("extract")
    if not isinstance(extract, str):
        raise ValueError(f"wiki page has no extract: {title}")

    output = f"URL: {_page_url(target.base_url, title)}\nTitle: {title}\n\n{extract}"
    return _truncate(output, max_chars)


def _api_get(base_url: str, params: dict[str, str]) -> dict[str, object]:
    url = f"{base_url}/w/api.php"
    try:
        response = requests.get(
            url,
            params=params,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise ValueError(f"wiki request failed: {error}") from error
    try:
        final_url = getattr(response, "url", url)
        if isinstance(final_url, str):
            _validate_wikimedia_base(_origin(final_url))
        if response.status_code < 200 or response.status_code >= 300:
            raise ValueError(f"wiki request failed: HTTP {response.status_code}")
        try:
            payload = cast(object, response.json())
        except ValueError as error:
            raise ValueError("wiki request failed: response was not JSON") from error
    finally:
        response.close()

    if not isinstance(payload, dict):
        raise ValueError("wiki request failed: response JSON was not an object")
    if "error" in payload:
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            info = error_payload.get("info")
            if isinstance(info, str) and info:
                raise ValueError(f"wiki API error: {info}")
        raise ValueError("wiki API error")
    return cast(dict[str, object], payload)


def _query_dict(payload: dict[str, object]) -> dict[str, object]:
    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError("wiki response missing query object")
    return cast(dict[str, object], query)


def _first_page(payload: dict[str, object]) -> dict[str, object]:
    pages = _query_dict(payload).get("pages")
    if not isinstance(pages, dict):
        raise ValueError("wiki response missing pages object")
    for page in pages.values():
        if isinstance(page, dict):
            return cast(dict[str, object], page)
    raise ValueError("wiki response contained no pages")


def _open_target(arguments: dict[str, object]) -> _OpenTarget:
    url = arguments.get("url")
    if isinstance(url, str) and url.strip():
        return _target_from_url(url.strip())

    title = _require_non_empty_string(arguments, "title")
    wiki = _optional_wiki(arguments)
    return _OpenTarget(base_url=wiki, title=title)


def _target_from_url(url: str) -> _OpenTarget:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an http or https URL")
    base_url = _origin(url)
    _validate_wikimedia_base(base_url)
    if parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
        if title:
            return _OpenTarget(base_url=base_url, title=title)
    raise ValueError("url must be a Wikimedia page URL under /wiki/")


def _optional_wiki(arguments: dict[str, object]) -> str:
    value = arguments.get("wiki", _DEFAULT_WIKI)
    if not isinstance(value, str):
        raise ValueError("wiki must be a string")
    base_url = _origin(value.strip())
    _validate_wikimedia_base(base_url)
    return base_url


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("wiki must be an http or https URL")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def _validate_wikimedia_base(base_url: str) -> None:
    parsed = urlparse(base_url)
    host = parsed.netloc.lower()
    if host in _SUPPORTED_EXACT_HOSTS:
        return
    if any(
        host.endswith(suffix) and host != suffix[1:]
        for suffix in _SUPPORTED_HOST_SUFFIXES
    ):
        return
    raise ValueError("wiki must be a supported Wikimedia wiki URL")


def _page_url(base_url: str, title: str) -> str:
    return f"{base_url}/wiki/{quote(title.replace(' ', '_'), safe=':_()')}"


def _clean_snippet(snippet: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", html.unescape(snippet))
    collapsed = re.sub(r"\s+", " ", without_tags).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", collapsed)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n[truncated]"
    return text[: max_chars - len(marker)].rstrip() + marker


def _require_non_empty_string(arguments: dict[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value.strip()


def _optional_int(arguments: dict[str, object], name: str, *, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value
