import httpx
import pytest
from ddgs.exceptions import DDGSException

from t_bot.errors import BotError
from t_bot.web_search import WebSearch


def test_search_normalizes_results_and_filters_invalid_urls(monkeypatch):
    calls = []

    class Provider:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 5

        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            return [
                {"title": "Python", "href": "https://python.org", "body": "Documentation"},
                {"title": "Bad URL", "href": "file:///etc/passwd"},
                {"title": "Local file", "href": "/etc/passwd"},
            ]

    monkeypatch.setattr("t_bot.web_search.DDGS", Provider)
    result = WebSearch(5, 10).search("  Python  ", 3)
    assert result["results"] == [
        {"title": "Python", "url": "https://python.org", "snippet": "Documentation"},
    ]
    assert calls[0][0] == "Python"
    assert calls[0][1]["max_results"] == 3


@pytest.mark.parametrize("query,count", [("", 1), ("x" * 501, 1), ("Python", 0), ("Python", 11)])
def test_invalid_search_does_not_contact_provider(query, count):
    with pytest.raises(BotError):
        WebSearch(5, 10).search(query, count)


def test_search_provider_failure_is_reported(monkeypatch):
    class Provider:
        def __init__(self, **kwargs):
            pass

        def text(self, *args, **kwargs):
            raise DDGSException("provider failure")

    monkeypatch.setattr("t_bot.web_search.DDGS", Provider)
    original = httpx.Client
    monkeypatch.setattr(
        "t_bot.web_search.httpx.Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
            **kwargs,
        ),
    )
    with pytest.raises(BotError, match="тимчасово недоступний"):
        WebSearch(5, 10).search("Python")


@pytest.mark.parametrize("primary_failed", [False, True])
def test_rss_fallback_when_primary_has_no_results(monkeypatch, primary_failed):
    class Provider:
        def __init__(self, **kwargs):
            pass

        def text(self, *args, **kwargs):
            if primary_failed:
                raise DDGSException("rate limit")
            return []

    def feed(request):
        assert request.url.params["q"] == "Python"
        assert request.url.params["format"] == "rss"
        return httpx.Response(
            200,
            content=b"""<?xml version="1.0"?>
<rss><channel><item><title>Python</title><link>https://python.org</link>
<description>Documentation</description></item></channel></rss>""",
        )

    original = httpx.Client
    monkeypatch.setattr("t_bot.web_search.DDGS", Provider)
    monkeypatch.setattr(
        "t_bot.web_search.httpx.Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(feed),
            **kwargs,
        ),
    )
    result = WebSearch(5, 10).search("Python")
    assert result["provider"] == "bing_rss"
    assert result["results"][0]["url"] == "https://python.org"


@pytest.mark.parametrize(
    "body",
    [
        b"not XML",
        b"<html>Challenge</html>",
        b"x" * 262145,
        b'<!DOCTYPE rss [<!ENTITY x "test">]><rss/>',
    ],
)
def test_invalid_and_oversized_rss_is_rejected(monkeypatch, body):
    original = httpx.Client
    monkeypatch.setattr(
        "t_bot.web_search.httpx.Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
            **kwargs,
        ),
    )
    with pytest.raises(BotError):
        WebSearch(5, 10)._bing_rss("Python")
