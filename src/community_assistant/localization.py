"""Localized user-facing copy."""

FAILURE_COPY: dict[str, str] = {
    "en_US": "Sorry — something went wrong on our side. Please try again in a moment.",
    "fr_FR": "Désolé, un problème est survenu de notre côté. Veuillez réessayer dans un instant.",
}

REFUSAL_COPY = "I can't help with that request. If you need support, I can help you open a case."


def failure_message(locale: str) -> str:
    return FAILURE_COPY.get(locale, FAILURE_COPY["en_US"])
