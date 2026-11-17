from t_bot.research_sources import select_research_sources


def page(url, title, snippet=""):
    return {"url": url, "title": title, "snippet": snippet}


def test_research_4336_prefers_arithmetic_documents_over_home_and_downloads():
    generic = [
        page("https://www.python.org/", "Welcome to Python"),
        page("https://www.python.org/downloads/", "Download Python"),
        page("https://online-python.com/", "Online Python IDE"),
    ]
    arithmetic = [
        page("https://realpython.com/ref/arithmeticerror/", "Arithmetic errors"),
        page("https://docs.python.org/3/library/exceptions.html", "Division by zero errors"),
        page("https://example.org/arithmetic-errors", "Arithmetic error handling"),
    ]
    selected = select_research_sources(
        [
            {"query": "Python arithmetic operations", "results": generic},
            {"query": "Python arithmetic errors", "results": arithmetic},
            {"query": "Python division by zero", "results": generic},
        ],
        "Calculator arithmetic operations",
    )
    assert len(selected) == 3
    assert all(result in arithmetic for result in selected)


def test_research_balances_queries_ranks_relevance_and_deduplicates():
    add = page("https://docs.python.org/add", "Addition operator")
    divide = page("https://docs.python.org/divide", "Division operator")
    invalid = page("https://docs.python.org/errors", "Arithmetic error handling")
    selected = select_research_sources(
        [
            {"query": "addition", "results": [page("https://python.org/", "Welcome"), add, divide]},
            {"query": "division", "results": [divide, add]},
            {"query": "errors", "results": [invalid, add]},
        ],
        "Arithmetic calculator",
    )
    assert selected == [add, divide, invalid]


def test_research_uses_unique_fallback_sources_when_no_topical_match():
    result = page("https://example.org/", "Welcome")
    assert select_research_sources([{"results": [result, result]}, {"results": [result]}], "") == [
        result
    ]
    assert select_research_sources([{"error": "offline"}], "Calculator") == []
