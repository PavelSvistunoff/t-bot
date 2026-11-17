"""Fetch bounded HTTP pages and extract readable text without executing page scripts."""

import asyncio
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx

from t_bot.errors import BotError

IGNORED_TAGS = {"head", "script", "style", "noscript", "template", "svg", "canvas", "iframe"}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "dl",
    "dt",
    "dd",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
}


class PageText(HTMLParser):
    """Collect visible HTML text and title while discarding scripts and hidden elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: list[str] = []
        self.frames: list[tuple[str, bool]] = []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        hidden = (
            tag in IGNORED_TAGS
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
        )
        if tag == "title":
            self.in_title = True
        if not self.hidden and not hidden and tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag in {"td", "th"} and not self.hidden:
            self.parts.append(" ")
        if tag not in VOID_TAGS:
            if len(self.frames) >= 256:
                raise BotError("HTML-сторінка має завелику вкладеність.")
            self.frames.append((tag, hidden))
            self.hidden += int(hidden)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        for index in range(len(self.frames) - 1, -1, -1):
            if self.frames[index][0] == tag:
                self.hidden -= sum(int(hidden) for _, hidden in self.frames[index:])
                del self.frames[index:]
                break
        if not self.hidden and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
        elif not self.hidden:
            self.parts.append(data)


def extract_text(body: bytes, content_type: str, encoding: str, max_chars: int) -> dict:
    """Decode bounded page bytes, normalize whitespace and explicitly mark text truncation."""
    try:
        decoded = body.decode(encoding, errors="replace")
    except LookupError:
        decoded = body.decode("utf-8", errors="replace")
    title = ""
    if content_type != "text/plain":
        parser = PageText()
        parser.feed(decoded)
        parser.close()
        title = " ".join("".join(parser.title).split())[:500]
        decoded = "".join(parser.parts)
    text = "\n".join(line for part in decoded.splitlines() if (line := " ".join(part.split())))
    if not text:
        raise BotError("Сторінка не містить доступного тексту; JavaScript не виконується.")
    return {"title": title, "text": text[:max_chars], "truncated": len(text) > max_chars}


def validate_url(url: str) -> str:
    """Accept HTTP(S) URLs without credentials; raise BotError for unsupported input."""
    url = url.strip()
    try:
        parsed = urlsplit(url)
        if (
            not url
            or len(url) > 4096
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(ord(char) < 32 for char in url)
        ):
            raise ValueError("Invalid URL")
        _ = parsed.port
    except ValueError as exc:
        raise BotError(
            "Укажіть коректний HTTP/HTTPS URL без логіна та пароля, до 4096 символів."
        ) from exc
    return url


class WebpageReader:
    """Read HTML/plain text with a total deadline, bounded redirects and decoded body size."""

    def __init__(self, timeout_seconds: int, max_bytes: int, max_chars: int):
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_chars = max_chars

    async def read(self, url: str) -> dict:
        """Return final URL, title, text and truncation flag; failures raise BotError."""
        url = validate_url(url)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds,
                    trust_env=False,
                    follow_redirects=False,
                    headers={
                        "User-Agent": "t-bot/0.1",
                        "Accept": "text/html,application/xhtml+xml,text/plain",
                    },
                ) as client:
                    current = url
                    for redirects in range(6):
                        async with client.stream("GET", current) as response:
                            if response.is_redirect:
                                if redirects == 5 or "location" not in response.headers:
                                    raise BotError(
                                        "Сторінка має забагато або некоректні перенаправлення."
                                    )
                                current = validate_url(
                                    str(response.url.join(response.headers["location"]))
                                )
                                continue
                            response.raise_for_status()
                            content_type = (
                                response.headers.get("content-type", "text/html")
                                .split(";", 1)[0]
                                .strip()
                                .lower()
                            )
                            if content_type not in {
                                "text/html",
                                "application/xhtml+xml",
                                "text/plain",
                            }:
                                raise BotError(
                                    "Підтримуються лише HTML і звичайний текст; "
                                    "PDF та інші файли не читаються."
                                )
                            body = bytearray()
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                body.extend(chunk)
                                if len(body) > self.max_bytes:
                                    raise BotError("Вебсторінка перевищує WEBPAGE_MAX_BYTES.")
                            result = await asyncio.to_thread(
                                extract_text,
                                bytes(body),
                                content_type,
                                response.encoding or "utf-8",
                                self.max_chars,
                            )
                            return {
                                "url": str(response.url),
                                "content_type": content_type,
                                **result,
                            }
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise BotError("Перевищено час очікування вебсторінки.") from exc
        except httpx.HTTPStatusError as exc:
            raise BotError(f"Сайт повернув HTTP {exc.response.status_code}.") from exc
        except (httpx.HTTPError, httpx.InvalidURL, UnicodeError) as exc:
            raise BotError("Не вдалося завантажити вебсторінку: помилка мережі або TLS.") from exc
        raise BotError("Не вдалося прочитати вебсторінку.")
