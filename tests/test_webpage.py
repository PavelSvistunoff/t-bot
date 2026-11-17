import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from t_bot.errors import BotError
from t_bot.runtime import Invocation, current_invocation
from t_bot.webpage import WebpageReader, extract_text


def reader(monkeypatch, handler, *, max_bytes=10000, max_chars=20000, timeout=5):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        "t_bot.webpage.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    return WebpageReader(timeout, max_bytes, max_chars)


async def test_read_html_extracts_title_text_entities_and_filters_hidden_content(monkeypatch):
    html = """<html><head><title>Гайд &amp; API</title><style>CSS SECRET</style></head>
<body><h1>Привіт</h1><p>Текст <b>сторінки</b> &amp; приклад.</p>
<script>JS SECRET</script><!-- COMMENT SECRET --><noscript>FALLBACK SECRET</noscript>
<div hidden>HIDDEN SECRET</div><section aria-hidden="true">ARIA SECRET</section>
<template>TEMPLATE SECRET</template><p>Кінець</p></body></html>"""
    result = await reader(
        monkeypatch,
        lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"}),
    ).read("https://example.com/guide")
    assert result == {
        "url": "https://example.com/guide",
        "content_type": "text/html",
        "title": "Гайд & API",
        "text": "Привіт\nТекст сторінки & приклад.\nКінець",
        "truncated": False,
    }


async def test_redirect_returns_final_url_and_text_charset(monkeypatch):
    calls = []

    def fetch(request):
        calls.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/page"})
        return httpx.Response(
            200,
            content="Café\n Hello   world".encode("latin-1"),
            headers={
                "content-type": "text/plain; charset=iso-8859-1",
            },
        )

    result = await reader(monkeypatch, fetch).read("https://example.com/start")
    assert calls == ["https://example.com/start", "https://example.com/page"]
    assert result["url"] == calls[-1] and result["text"] == "Café\nHello world"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///etc/passwd",
        "ftp://example.com",
        "https://user:pass@example.com",
        "https://",
        "https://example.com:99999",
        "https://example.com/\nx",
        "https://x/" + "a" * 4096,
    ],
)
async def test_invalid_url_is_rejected_before_any_request(monkeypatch, url):
    calls = []
    page = reader(monkeypatch, lambda request: calls.append(request))
    with pytest.raises(BotError, match="HTTP/HTTPS"):
        await page.read(url)
    assert not calls


async def test_redirect_cannot_switch_to_file_or_embedded_credentials(monkeypatch):
    page = reader(
        monkeypatch, lambda request: httpx.Response(302, headers={"location": "file:///etc/passwd"})
    )
    with pytest.raises(BotError, match="HTTP/HTTPS"):
        await page.read("https://example.com")


async def test_redirect_loop_is_bounded(monkeypatch):
    calls = []

    def fetch(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(BotError, match="перенаправлення"):
        await reader(monkeypatch, fetch).read("https://example.com")
    assert len(calls) == 6


@pytest.mark.parametrize("kind", ["application/pdf", "image/png", "application/octet-stream"])
async def test_binary_document_formats_are_rejected(monkeypatch, kind):
    with pytest.raises(BotError, match="лише HTML"):
        await reader(
            monkeypatch, lambda request: httpx.Response(200, headers={"content-type": kind})
        ).read("https://example.com")


async def test_streamed_page_size_is_bounded_without_content_length(monkeypatch):
    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"<p>" + b"a" * 600
            yield b"b" * 600 + b"</p>"

    page = reader(monkeypatch, lambda request: httpx.Response(200, stream=Chunks()), max_bytes=1000)
    with pytest.raises(BotError, match="WEBPAGE_MAX_BYTES"):
        await page.read("https://example.com")


async def test_output_truncation_is_explicit(monkeypatch):
    page = reader(
        monkeypatch,
        lambda request: httpx.Response(
            200, text="<p>" + "a" * 500 + "</p>", headers={"content-type": "text/html"}
        ),
        max_chars=40,
    )
    result = await page.read("https://example.com")
    assert result["truncated"] and result["text"] == "a" * 40


async def test_http_errors_and_network_failures_are_safe(monkeypatch):
    page = reader(monkeypatch, lambda request: httpx.Response(404))
    with pytest.raises(BotError, match="HTTP 404"):
        await page.read("https://example.com")

    def fail(request):
        raise httpx.ConnectError("INTERNAL SECRET", request=request)

    # Reset the patched factory so the next transport is not wrapped twice.
    monkeypatch.undo()
    page = reader(monkeypatch, fail)
    with pytest.raises(BotError, match="мережі") as error:
        await page.read("https://example.com")
    assert "SECRET" not in str(error.value)


async def test_total_timeout_and_cancellation(monkeypatch):
    async def slow(request):
        await asyncio.sleep(1)
        return httpx.Response(200, text="page")

    with pytest.raises(BotError, match="час очікування"):
        await reader(monkeypatch, slow, timeout=0.01).read("https://example.com")
    monkeypatch.undo()

    async def cancelled(request):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await reader(monkeypatch, cancelled).read("https://example.com")


def test_empty_or_overly_nested_html_is_reported():
    with pytest.raises(BotError, match="не містить"):
        extract_text(
            b"<head><title>Title</title></head><script>JS</script>", "text/html", "utf-8", 1000
        )
    with pytest.raises(BotError, match="вкладеність"):
        extract_text(b"<div>" * 257, "text/html", "utf-8", 1000)


async def test_tool_authorization_errors_and_open_access(components, settings, monkeypatch):
    runtime, _, _, _ = components
    fetch = AsyncMock(return_value={"url": "https://example.com", "text": "Page"})
    monkeypatch.setattr(runtime.basic_tools.webpage, "read", fetch)
    token = current_invocation.set(Invocation(99, 5, 1))
    try:
        with pytest.raises(BotError, match="Доступ"):
            await runtime.basic_tools.read_webpage("https://example.com")
        assert fetch.await_count == 0
        settings.allowed_user_ids = set()
        assert (await runtime.basic_tools.read_webpage("https://example.com"))["text"] == "Page"
        fetch.side_effect = BotError("Не вдалося завантажити")
        assert (await runtime.basic_tools.read_webpage("https://example.com"))[
            "error"
        ] == "Не вдалося завантажити"
    finally:
        current_invocation.reset(token)
