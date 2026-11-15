"""Public web search with bounded inputs and provider-independent results."""

from urllib.parse import urlsplit
from xml.etree import ElementTree

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from t_bot.errors import BotError


class WebSearch:
    def __init__(self, timeout_seconds: int, max_results: int):
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    def search(self, query: str, max_results: int = 5) -> dict:
        query = query.strip()
        if not query or len(query) > 500:
            raise BotError("Пошуковий запит має містити від 1 до 500 символів.")
        if not 1 <= max_results <= self.max_results:
            raise BotError(f"max_results має бути від 1 до {self.max_results}.")
        provider = "ddgs"
        try:
            results = DDGS(timeout=self.timeout_seconds).text(
                query,
                max_results=max_results,
                backend="duckduckgo,brave,yahoo",
            )
        except DDGSException:
            results = []
        if not results:
            results = self._bing_rss(query)
            provider = "bing_rss"
        return {
            "query": query,
            "results": self._normalize(results, max_results),
            "provider": provider,
        }

    def _bing_rss(self, query: str) -> list[dict]:
        try:
            with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
                with client.stream(
                    "GET", "https://www.bing.com/search", params={"q": query, "format": "rss"}
                ) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 262144:
                            raise BotError("Web-пошук повернув завелику відповідь.")
            if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
                raise BotError("Пошуковий сервіс повернув непідтримуваний XML.")
            feed = ElementTree.fromstring(body)
            if feed.tag != "rss":
                raise BotError("Пошуковий сервіс повернув неочікуваний формат.")
            return [
                {
                    "title": item.findtext("title", ""),
                    "href": item.findtext("link", ""),
                    "body": item.findtext("description", ""),
                }
                for item in feed.findall("./channel/item")
            ]
        except (httpx.HTTPError, ElementTree.ParseError) as exc:
            raise BotError(
                "Web-пошук тимчасово недоступний: пошуковий сервіс відхилив запит "
                "або перевищено час очікування. Спробуйте пізніше."
            ) from exc

    @staticmethod
    def _normalize(results: list[dict], max_results: int) -> list[dict]:
        normalized = []
        for result in results[:max_results]:
            url = str(result.get("href", ""))
            try:
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    continue
            except ValueError:
                continue
            normalized.append(
                {
                    "title": str(result.get("title", ""))[:300],
                    "url": url[:2000],
                    "snippet": str(result.get("body", ""))[:1000],
                }
            )
        return normalized
