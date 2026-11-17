"""Image artifact contracts and evidence for procedural-image QA and review."""

import re

from t_bot.errors import BotError

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
DIMENSION = re.compile(r"(\d{1,5})\s*[xх×]\s*(\d{1,5})", re.IGNORECASE)


def random_image_request(request: str) -> bool:
    text = request.casefold()
    return bool(
        re.search(r"зображ|малю|\bimage\b|\bcat\b|кот", text)
        and re.search(r"випадков|унікальн|random|unique", text)
    )


def image_contract(request: str, spec: dict) -> dict:
    facts = []
    sources = [("user_request", 0, request)]
    for section, cases in spec.items():
        if isinstance(cases, list):
            sources.extend(
                (section, index, text) for index, text in enumerate(cases) if isinstance(text, str)
            )
    for section, index, text in sources:
        for match in DIMENSION.finditer(text):
            anchors = list(
                re.finditer(
                    r"голов\w*|head|полотн\w*|canvas|зображенн\w*|image",
                    text[: match.start()],
                    re.IGNORECASE,
                )
            )
            if not anchors:
                continue
            subject = (
                "head" if re.search(r"голов|head", anchors[-1].group(), re.IGNORECASE) else "canvas"
            )
            facts.append(
                {
                    "subject": subject,
                    "size": [int(match[1]), int(match[2])],
                    "binding": not bool(
                        re.search(
                            r"наприклад|приклад|example|e\.g\.",
                            text[: match.start()],
                            re.IGNORECASE,
                        )
                    ),
                    "section": section,
                    "index": index,
                    "quote": text,
                }
            )
    return {
        "enabled": random_image_request(request),
        "geometry_facts": facts,
        "guidance": "Canvas and head are separate. Examples are not constraints. "
        "QA owns happy/failure test design.",
    }


def validate_size_claims(review: dict, contract: dict) -> None:
    """Reject a claimed size requirement unless its cited geometry fact actually establishes it."""
    claims = review.get("size_claims", [])
    for finding in review.get("findings", []):
        if not re.search(r"полотн|canvas|зображенн|image", finding, re.IGNORECASE):
            continue
        if not re.search(
            r"вказано|вимага|повин|must|required|specifies|specified", finding, re.IGNORECASE
        ):
            continue
        for match in DIMENSION.finditer(finding):
            size = [int(match[1]), int(match[2])]
            if not any(
                claim.get("subject") == "canvas" and claim.get("size") == size for claim in claims
            ):
                raise BotError(
                    "Твердження про обов'язковий розмір полотна потребує size_claims "
                    "із binding geometry_facts. Приклад розміру голови не є вимогою до полотна."
                )
    for claim in claims:
        if not any(
            fact["binding"]
            and all(claim.get(key) == fact[key] for key in ("subject", "size", "section", "index"))
            for fact in contract["geometry_facts"]
        ):
            raise BotError(
                "Непідтверджена вимога до розміру. Полотно й голова — різні величини; "
                "приклад розміру голови не визначає розмір полотна. "
                "Використайте binding geometry_facts або приберіть хибне зауваження."
            )


def image_review_schema(base: dict) -> dict:
    import copy

    schema = copy.deepcopy(base)
    schema["properties"]["size_claims"] = {
        "type": "array",
        "maxItems": 10,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject": {"enum": ["canvas", "head"]},
                "size": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "section": {"type": "string"},
                "index": {"type": "integer", "minimum": 0},
            },
            "required": ["subject", "size", "section", "index"],
        },
    }
    schema["required"] = [*schema["required"], "size_claims"]
    return schema
