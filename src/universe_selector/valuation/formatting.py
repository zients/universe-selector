from __future__ import annotations

from collections.abc import Iterable
import re


_PROHIBITED_PHRASE_RE = re.compile(
    r"target price|fair value|undervalued|overvalued|expected return",
    re.IGNORECASE,
)
_PROHIBITED_WORD_RE = re.compile(r"\b(?:buy|sell|hold)\b", re.IGNORECASE)


def _sanitize_text(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = " ".join(text.split())
    text = _PROHIBITED_PHRASE_RE.sub("[redacted]", text)
    return _PROHIBITED_WORD_RE.sub("[redacted]", text)


def _table_cell(value: object) -> str:
    return _sanitize_text(value).replace("|", r"\|")


def _markdown_text(value: object) -> str:
    return _sanitize_text(value).replace("|", r"\|")


def _compact_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:.2f}"


def _format_money(value: float, currency: str) -> str:
    compact = _compact_number(value)
    currency_code = _sanitize_text(currency.upper())
    if currency_code == "USD":
        return f"${compact}"
    if currency_code:
        return f"{currency_code} {compact}"
    return compact


def _format_number(value: float) -> str:
    return _compact_number(value)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_note(note: str | None) -> str:
    if note is None:
        return ""
    return _sanitize_text(note)


def _lines_table(headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_table_cell(cell) for cell in row) + " |" for row in rows)
    return lines
