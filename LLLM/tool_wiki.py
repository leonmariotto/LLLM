"""
Wikimedia wiki search and page retrieval tool for model tool-use loops.

The tool exposes one ``wiki`` function with four actions:
``search`` queries a Wikimedia wiki through the MediaWiki API, ``open``
retrieves a plain-text extract for a wiki page URL or page title,
``search_in_page`` searches inside a known wiki page, and ``read_chunk``
continues a previous truncated response without another network request.
"""

from __future__ import annotations

from collections import deque
import copy
import html
import re
from dataclasses import dataclass
from typing import cast
from urllib.parse import quote, unquote, urlparse

import requests

from .tool_common import Tool

_DEFAULT_WIKI = "https://en.wikipedia.org"
_REQUEST_TIMEOUT_SECONDS = 10
_HEADERS = {
    "User-Agent": (
        "LLLM wiki tool/0.1 "
        "(https://github.com/leonmariotto/LLLM; contact: leon2mariotto@gmail.com)"
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
_SEARCH_IN_PAGE_CONTEXT_CHARS = 300
_TRUNCATED_MARKER = "\n[truncated]"

WIKI_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "Wikipedia/Wikimedia tool. Choose one action. "
            "1. action='search': find page titles for a topic. Use query as the "
            "topic, like 'Paris' or 'CAC 40'. "
            "2. action='open': read a known page. Use title or url. "
            "3. action='search_in_page': find an exact word or short phrase inside "
            "a known page. Use title or url for the page. Use query only for the "
            "short text to find, not the whole user question. Good queries: "
            "'market capitalization', 'population', 'Montmartre', 'CEO'. Bad "
            "queries: 'what is the market cap of CAC 40', 'tell me about Paris'. "
            "4. action='read_chunk': read the next saved chunk after an open or "
            "search_in_page response ended with [truncated]. This consumes cached "
            "text from the previous tool result and does not make a network request. "
            "If read_chunk also ends with [truncated], call read_chunk again. "
            "search only returns page titles, call open or search_in_page next "
            "before answering. "
            "Information about <subject> is often (if not always) located in the "
            "<subject> dedicated pages. Example : turtle food habits is an "
            "information present in the 'turtle' page. "
            "When looking for information about <subject>, you should try to open "
            "directly wikipedia page with title '<subject>', do a search only if it "
            "report page not found. "
            "search_in_page search for exact match (case insensitive) so do only "
            "small search: 2-3 words maximum. "
            "Examples: "
            '{"action":"open","title":"turtle"}; '
            '{"action":"search","query":"olympic games list"}; '
            '{"action":"search_in_page","title":"paris","query":"montmartre"}; '
            '{"action":"read_chunk"}.'
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "open", "search_in_page", "read_chunk"],
                    "description": (
                        "Choose exactly one: 'search' finds pages; 'open' reads a "
                        "known page; 'search_in_page' finds an exact match inside "
                        "a known page; 'read_chunk' continues the previous "
                        "truncated result."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Required for search and search_in_page. For search, use "
                        "the topic or page name. For search_in_page, use only an "
                        "exact word or short phrase expected in the page, not the "
                        "full user question."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Wikimedia page URL to retrieve or search inside. Required "
                        "for open and search_in_page unless title is provided."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Wiki page title to retrieve or search inside. Required "
                        "for open and search_in_page unless url is provided."
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
                    "description": (
                        "For action='search' only. Maximum page results to return, "
                        "from 1 to 10."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "For action='open', action='search_in_page', and "
                        "action='read_chunk'. Maximum characters to return."
                    ),
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


@dataclass(frozen=True)
class _FetchedPage:
    title: str
    url: str
    extract: str


def wiki_tool() -> Tool:
    """Return the ready-to-register ``wiki`` tool."""
    executor = _WikiExecutor()
    return Tool(schema=copy.deepcopy(WIKI_TOOL_SCHEMA), execute=executor.execute)


def execute_wiki(arguments: dict[str, object]) -> str:
    """Execute the wiki tool."""
    return _DEFAULT_EXECUTOR.execute(arguments)


class _WikiExecutor:
    def __init__(self) -> None:
        self._continuation_chunks: deque[str] = deque()

    def execute(self, arguments: dict[str, object]) -> str:
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
            max_chars = _validated_max_chars(arguments)
            return self._execute_open(target, max_chars)
        if action == "search_in_page":
            target = _open_target(arguments)
            query = _require_non_empty_string(arguments, "query")
            max_chars = _validated_max_chars(arguments)
            return self._execute_search_in_page(target, query, max_chars)
        if action == "read_chunk":
            max_chars = _validated_max_chars(arguments)
            return self._read_chunk(max_chars)
        raise ValueError(
            "action must be 'search', 'open', 'search_in_page', or 'read_chunk'"
        )

    def _execute_open(self, target: _OpenTarget, max_chars: int) -> str:
        page = _fetch_page(target)
        output = f"URL: {page.url}\nTitle: {page.title}\n\n{page.extract}"
        return self._truncate_and_queue(output, max_chars)

    def _execute_search_in_page(
        self, target: _OpenTarget, query: str, max_chars: int
    ) -> str:
        page = _fetch_page(target)
        header = f"URL: {page.url}\nTitle: {page.title}\nQuery: {query}"
        matches = _page_match_blocks(page.extract, query)
        if not matches:
            output = (
                f"No matches found in page: {page.title}\n"
                f"Query: {query}\n"
                f"URL: {page.url}"
            )
            return self._truncate_and_queue(output, max_chars)
        output = _join_match_blocks(header, matches)
        return self._truncate_and_queue(output, max_chars)

    def _read_chunk(self, max_chars: int) -> str:
        if not self._continuation_chunks:
            return "No wiki continuation chunks available."
        return self._truncate_and_queue(
            self._continuation_chunks.popleft(),
            max_chars,
            append_left=True,
        )

    def _truncate_and_queue(
        self, text: str, max_chars: int, *, append_left: bool = False
    ) -> str:
        if len(text) <= max_chars:
            return text
        split_at = max_chars - len(_TRUNCATED_MARKER)
        visible = text[:split_at].rstrip()
        remainder = text[split_at:]
        if append_left:
            self._continuation_chunks.appendleft(remainder)
        else:
            self._continuation_chunks.append(remainder)
        return visible + _TRUNCATED_MARKER


_DEFAULT_EXECUTOR = _WikiExecutor()


def _execute_search(
    query: str, wiki: str, max_results: int, include_snippet: bool = True
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
    raw_results_object = _query_dict(payload).get("search", [])
    if not isinstance(raw_results_object, list) or not raw_results_object:
        return f"No wiki results found for: {query}"
    raw_results = cast(list[object], raw_results_object)

    results: list[_WikiPage] = []
    for raw_result in raw_results[:max_results]:
        if not isinstance(raw_result, dict):
            continue
        result_dict = cast(dict[str, object], raw_result)
        title = result_dict.get("title")
        if not isinstance(title, str) or not title:
            continue
        snippet = result_dict.get("snippet")
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
        # if result.snippet and include_snippet is True:
        #     lines.append(f"Snippet: {result.snippet}")
    if lines != []:
        lines.append(
            "Search results only list candidate pages. You must call "
            "open or search_in_page before answering."
        )
    return "\n".join(lines)


def _validated_max_chars(arguments: dict[str, object]) -> int:
    max_chars = _optional_int(arguments, "max_chars", default=6000)
    if max_chars < 500 or max_chars > 20000:
        raise ValueError("max_chars must be between 500 and 20000")
    return max_chars


def _fetch_page(target: _OpenTarget) -> _FetchedPage:
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

    return _FetchedPage(
        title=title,
        url=_page_url(target.base_url, title),
        extract=extract,
    )


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
    payload_dict = cast(dict[str, object], payload)
    if "error" in payload_dict:
        error_payload = payload_dict.get("error")
        if isinstance(error_payload, dict):
            error_dict = cast(dict[str, object], error_payload)
            info = error_dict.get("info")
            if isinstance(info, str) and info:
                raise ValueError(f"wiki API error: {info}")
        raise ValueError("wiki API error")
    return payload_dict


def _query_dict(payload: dict[str, object]) -> dict[str, object]:
    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError("wiki response missing query object")
    return cast(dict[str, object], query)


def _first_page(payload: dict[str, object]) -> dict[str, object]:
    pages_object = _query_dict(payload).get("pages")
    if not isinstance(pages_object, dict):
        raise ValueError("wiki response missing pages object")
    pages = cast(dict[str, object], pages_object)
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


def _page_match_blocks(extract: str, query: str) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for index, match in enumerate(pattern.finditer(extract), start=1):
        start, end = _snippet_bounds(extract, match.start(), match.end())
        before = extract[start : match.start()]
        matched = extract[match.start() : match.end()]
        after = extract[match.end() : end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(extract) else ""
        snippet = _normalize_snippet_text(f"{prefix}{before}[{matched}]{after}{suffix}")
        blocks.append(f"Match {index}:\n{snippet}")
    return blocks


def _snippet_bounds(text: str, match_start: int, match_end: int) -> tuple[int, int]:
    start = max(0, match_start - _SEARCH_IN_PAGE_CONTEXT_CHARS)
    end = min(len(text), match_end + _SEARCH_IN_PAGE_CONTEXT_CHARS)
    if start > 0:
        next_space = text.find(" ", start, match_start)
        if next_space != -1:
            start = next_space + 1
    if end < len(text):
        previous_space = text.rfind(" ", match_end, end)
        if previous_space != -1:
            end = previous_space
    return start, end


def _normalize_snippet_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _join_match_blocks(header: str, match_blocks: list[str]) -> str:
    return f"{header}\n\n" + "\n\n".join(match_blocks)


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
