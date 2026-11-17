"""Bounded relevance ranking with fair selection across search queries."""

import re
from urllib.parse import urlsplit

MAX_RESEARCH_PAGES = 3


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zа-яіїєґ0-9]{3,}", text.casefold())) - {
        "python",
        "documentation",
        "docs",
        "the",
        "and",
        "for",
        "with",
        "tool",
    }


def _relevance(result: dict, terms: set[str]) -> int:
    try:
        parts = urlsplit(result["url"])
    except ValueError:
        return -100
    text = " ".join((result.get("title", ""), result.get("snippet", ""), parts.path))
    score = len(terms & _terms(text)) * 4
    if parts.hostname and parts.hostname.startswith("docs."):
        score += 3
    if parts.path.rstrip("/") in {"", "/downloads", "/download"}:
        score -= 8
    return score


def select_research_sources(research: list[dict], request: str) -> list[dict]:
    """Prefer topical documents, then round-robin unique URLs across query groups."""
    ranked = []
    for search in research:
        terms = _terms(request + " " + search.get("query", ""))
        candidates = [
            (_relevance(result, terms), result)
            for result in search.get("results", [])
            if isinstance(result.get("url"), str)
        ]
        ranked.append(sorted(candidates, key=lambda pair: pair[0], reverse=True))
    topical = any(score > 0 for group in ranked for score, _ in group)
    groups = [[result for score, result in group if not topical or score > 0] for group in ranked]
    selected: list[dict] = []
    seen: set[str] = set()
    while len(selected) < MAX_RESEARCH_PAGES:
        progressed = False
        for group in groups:
            while group and group[0]["url"] in seen:
                group.pop(0)
            if not group:
                continue
            result = group.pop(0)
            seen.add(result["url"])
            selected.append(result)
            progressed = True
            if len(selected) == MAX_RESEARCH_PAGES:
                break
        if not progressed:
            break
    return selected
