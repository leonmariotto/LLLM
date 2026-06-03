from collections.abc import Mapping
from typing import Any

import pytest
import requests

from ..LLLM.generator_with_tool import Tool
from ..LLLM.tool_wiki import execute_wiki, wiki_tool


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        url: str = "https://en.wikipedia.org/w/api.php",
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers: Mapping[str, str] = {"content-type": "application/json"}
        self.url = url

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def close(self) -> None:
        pass


def test_wiki_tool_returns_registered_tool() -> None:
    tool = wiki_tool()

    assert isinstance(tool, Tool)
    assert tool.schema["type"] == "function"
    function = tool.schema["function"]
    assert isinstance(function, dict)
    assert function["name"] == "wiki"
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    action = properties["action"]
    assert isinstance(action, dict)
    assert action["enum"] == ["search", "open", "search_in_page", "read_chunk"]


def test_execute_wiki_search_parses_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "search": [
                {
                    "title": "CAC 40",
                    "snippet": "French &lt;span class='searchmatch'&gt;index&lt;/span&gt;.",
                },
                {
                    "title": "CAC Next 20",
                    "snippet": "Another result.",
                },
            ]
        }
    }
    calls: list[dict[str, object]] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        calls.append({"args": args, "kwargs": kwargs})
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    output = execute_wiki({"action": "search", "query": "CAC 40"})

    assert calls[0]["args"] == ("https://en.wikipedia.org/w/api.php",)
    kwargs = calls[0]["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["params"]["list"] == "search"
    assert kwargs["params"]["srsearch"] == "CAC 40"
    assert kwargs["params"]["srlimit"] == "5"
    assert "1. CAC 40" in output
    assert "URL: https://en.wikipedia.org/wiki/CAC_40" in output
    # assert "Snippet: French index." in output
    assert "2. CAC Next 20" in output


def test_execute_wiki_search_uses_requested_wiki_and_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "search": [
                {"title": "Paris", "snippet": "Ville."},
                {"title": "Paris Saint-Germain FC", "snippet": "Club."},
            ]
        }
    }
    captured_params: list[dict[str, str]] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_params.append(kwargs["params"])
        return FakeResponse(payload, url="https://fr.wikipedia.org/w/api.php")

    monkeypatch.setattr(requests, "get", fake_get)

    output = execute_wiki(
        {
            "action": "search",
            "query": "Paris",
            "wiki": "https://fr.wikipedia.org",
            "max_results": 1,
        }
    )

    assert captured_params[0]["srlimit"] == "1"
    assert "URL: https://fr.wikipedia.org/wiki/Paris" in output
    assert "2. Paris Saint-Germain FC" not in output


def test_execute_wiki_search_reports_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_, **__: FakeResponse({"query": {"search": []}}),
    )

    output = execute_wiki({"action": "search", "query": "nothing"})

    assert output == "No wiki results found for: nothing"


def test_execute_wiki_open_reads_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "168274": {
                    "pageid": 168274,
                    "title": "CAC 40",
                    "extract": "The CAC 40 is a benchmark French stock market index.",
                }
            }
        }
    }

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        assert kwargs["params"]["prop"] == "extracts"
        assert kwargs["params"]["titles"] == "CAC 40"
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    output = execute_wiki({"action": "open", "title": "CAC 40"})

    assert "URL: https://en.wikipedia.org/wiki/CAC_40" in output
    assert "Title: CAC 40" in output
    assert "benchmark French stock market index" in output


def test_execute_wiki_open_reads_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Café",
                    "extract": "A café is a type of restaurant.",
                }
            }
        }
    }
    captured_titles: list[str] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_titles.append(kwargs["params"]["titles"])
        return FakeResponse(payload, url="https://fr.wikipedia.org/w/api.php")

    monkeypatch.setattr(requests, "get", fake_get)

    output = execute_wiki(
        {"action": "open", "url": "https://fr.wikipedia.org/wiki/Caf%C3%A9"}
    )

    assert captured_titles == ["Café"]
    assert "URL: https://fr.wikipedia.org/wiki/Caf%C3%A9" in output
    assert "A café is a type of restaurant." in output


def test_execute_wiki_open_truncates_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Long",
                    "extract": "x" * 1000,
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    output = execute_wiki(
        {"action": "open", "title": "Long", "max_chars": 500}
    )

    assert len(output) <= 500
    assert output.endswith("[truncated]")


def test_wiki_tool_read_chunk_consumes_open_remainder_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Long",
                    "extract": "a" * 600 + "TAIL",
                }
            }
        }
    }
    calls: list[object] = []

    def fake_get(*args: Any, **__: Any) -> FakeResponse:
        calls.append(args)
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    tool = wiki_tool()

    first = tool.execute({"action": "open", "title": "Long", "max_chars": 500})
    second = tool.execute({"action": "read_chunk", "max_chars": 500})

    assert len(first) <= 500
    assert first.endswith("[truncated]")
    assert "TAIL" in second
    assert not second.endswith("[truncated]")
    assert len(calls) == 1


def test_wiki_tool_read_chunk_consumes_multiple_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract = "a" * 600 + "b" * 600 + "END"
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Long",
                    "extract": extract,
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))
    tool = wiki_tool()

    first = tool.execute({"action": "open", "title": "Long", "max_chars": 500})
    second = tool.execute({"action": "read_chunk", "max_chars": 500})
    third = tool.execute({"action": "read_chunk", "max_chars": 500})
    fourth = tool.execute({"action": "read_chunk", "max_chars": 500})

    combined = (
        first.replace("\n[truncated]", "")
        + second.replace("\n[truncated]", "")
        + third.replace("\n[truncated]", "")
    )
    assert first.endswith("[truncated]")
    assert second.endswith("[truncated]")
    assert not third.endswith("[truncated]")
    assert "a" * 100 in combined
    assert "b" * 100 in combined
    assert "END" in combined
    assert fourth == "No wiki continuation chunks available."


def test_wiki_tool_read_chunk_isolated_by_tool_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Long",
                    "extract": "x" * 1000,
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))
    first_tool = wiki_tool()
    second_tool = wiki_tool()

    first = first_tool.execute({"action": "open", "title": "Long", "max_chars": 500})
    isolated = second_tool.execute({"action": "read_chunk"})
    continuation = first_tool.execute({"action": "read_chunk"})

    assert first.endswith("[truncated]")
    assert isolated == "No wiki continuation chunks available."
    assert continuation
    assert continuation != isolated


def test_execute_wiki_search_in_page_reads_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Python",
                    "extract": (
                        "Python is a programming language. "
                        "The python community maintains many libraries."
                    ),
                }
            }
        }
    }
    captured_params: list[dict[str, str]] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_params.append(kwargs["params"])
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)

    output = execute_wiki(
        {"action": "search_in_page", "title": "Python", "query": "PYTHON"}
    )

    assert captured_params[0]["prop"] == "extracts"
    assert captured_params[0]["titles"] == "Python"
    assert "URL: https://en.wikipedia.org/wiki/Python" in output
    assert "Title: Python" in output
    assert "Query: PYTHON" in output
    assert "Match 1:" in output
    assert "[Python]" in output
    assert "Match 2:" in output
    assert "[python]" in output


def test_execute_wiki_search_in_page_reads_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Café",
                    "extract": "A café may serve coffee. The café is small.",
                }
            }
        }
    }
    captured_titles: list[str] = []

    def fake_get(*_: Any, **kwargs: Any) -> FakeResponse:
        captured_titles.append(kwargs["params"]["titles"])
        return FakeResponse(payload, url="https://fr.wikipedia.org/w/api.php")

    monkeypatch.setattr(requests, "get", fake_get)

    output = execute_wiki(
        {
            "action": "search_in_page",
            "url": "https://fr.wikipedia.org/wiki/Caf%C3%A9",
            "query": "café",
        }
    )

    assert captured_titles == ["Café"]
    assert "URL: https://fr.wikipedia.org/wiki/Caf%C3%A9" in output
    assert "[café]" in output


def test_execute_wiki_search_in_page_reports_no_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Python",
                    "extract": "Python is a programming language.",
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    output = execute_wiki(
        {"action": "search_in_page", "title": "Python", "query": "missing"}
    )

    assert output == (
        "No matches found in page: Python\n"
        "Query: missing\n"
        "URL: https://en.wikipedia.org/wiki/Python"
    )


def test_execute_wiki_search_in_page_returns_all_matches_that_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Short",
                    "extract": "needle one. needle two. needle three.",
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    output = execute_wiki(
        {"action": "search_in_page", "title": "Short", "query": "needle"}
    )

    assert output.count("Match ") == 3
    assert "[truncated]" not in output


def test_execute_wiki_search_in_page_truncates_when_matches_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract = " ".join(f"needle {'x' * 120}" for _ in range(20))
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Long",
                    "extract": extract,
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    output = execute_wiki(
        {
            "action": "search_in_page",
            "title": "Long",
            "query": "needle",
            "max_chars": 500,
        }
    )

    assert len(output) <= 500
    assert output.endswith("[truncated]")
    assert "Match 1:" in output


def test_wiki_tool_read_chunk_consumes_search_in_page_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract = " ".join(f"needle {'x' * 120}" for _ in range(20))
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "Long",
                    "extract": extract,
                }
            }
        }
    }
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))
    tool = wiki_tool()

    first = tool.execute(
        {
            "action": "search_in_page",
            "title": "Long",
            "query": "needle",
            "max_chars": 500,
        }
    )
    second = tool.execute({"action": "read_chunk", "max_chars": 500})

    assert first.endswith("[truncated]")
    assert second
    assert "Match " in second


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"action": 1},
        {"action": "search"},
        {"action": "search", "query": ""},
        {"action": "search", "query": "x", "wiki": "https://example.com"},
        {"action": "search", "query": "x", "max_results": 0},
        {"action": "search", "query": "x", "max_results": 11},
        {"action": "search", "query": "x", "max_results": True},
        {"action": "open"},
        {"action": "open", "title": ""},
        {"action": "open", "url": ""},
        {"action": "open", "url": "file:///tmp/a"},
        {"action": "open", "url": "https://example.com/wiki/CAC_40"},
        {"action": "open", "url": "https://en.wikipedia.org/notwiki/CAC_40"},
        {"action": "open", "title": "CAC 40", "max_chars": 499},
        {"action": "open", "title": "CAC 40", "max_chars": 20001},
        {"action": "open", "title": "CAC 40", "max_chars": False},
        {"action": "search_in_page"},
        {"action": "search_in_page", "title": "CAC 40"},
        {"action": "search_in_page", "title": "CAC 40", "query": ""},
        {
            "action": "search_in_page",
            "title": "CAC 40",
            "query": "market",
            "max_chars": 499,
        },
        {
            "action": "search_in_page",
            "title": "CAC 40",
            "query": "market",
            "max_chars": 20001,
        },
        {
            "action": "search_in_page",
            "title": "CAC 40",
            "query": "market",
            "max_chars": False,
        },
        {"action": "read_chunk", "max_chars": 499},
        {"action": "read_chunk", "max_chars": 20001},
        {"action": "read_chunk", "max_chars": False},
        {"action": "bad"},
    ],
)
def test_execute_wiki_validates_arguments(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        execute_wiki(arguments)


def test_execute_wiki_reports_request_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(*_: Any, **__: Any) -> FakeResponse:
        raise requests.RequestException("timeout")

    monkeypatch.setattr(requests, "get", fake_get)

    with pytest.raises(ValueError, match="wiki request failed"):
        execute_wiki({"action": "search", "query": "llm"})


def test_execute_wiki_reports_http_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_, **__: FakeResponse("missing", status_code=404),
    )

    with pytest.raises(ValueError, match="HTTP 404"):
        execute_wiki({"action": "open", "title": "Missing"})


def test_execute_wiki_reports_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_, **__: FakeResponse({"error": {"info": "bad title"}}),
    )

    with pytest.raises(ValueError, match="bad title"):
        execute_wiki({"action": "open", "title": "Bad"})


def test_execute_wiki_reports_missing_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"query": {"pages": {"-1": {"title": "Missing", "missing": ""}}}}
    monkeypatch.setattr(requests, "get", lambda *_, **__: FakeResponse(payload))

    with pytest.raises(ValueError, match="wiki page not found"):
        execute_wiki({"action": "open", "title": "Missing"})
