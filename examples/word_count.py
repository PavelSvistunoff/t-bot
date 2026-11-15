def invoke(payload: dict) -> dict:
    """Count whitespace-separated words in the input text."""
    return {"count": len(payload["text"].split())}
