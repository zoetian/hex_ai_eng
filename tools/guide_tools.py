"""Business-rule guide tools: index, read and search the customer's markdown guides.

The guides directory is scanned at runtime, so new/changed guides need no code changes.
"""

import os
import re
from pathlib import Path

from framework.agent import Tool

GUIDES_DIR = Path(
    os.environ.get(
        "HECKS_GUIDES_DIR",
        Path(__file__).parent.parent / "evaluation" / "data" / "guides",
    )
)


def _guide_files() -> list[Path]:
    return sorted(GUIDES_DIR.glob("*.md")) if GUIDES_DIR.exists() else []


def guides_overview(schema_names: list[str] | None = None) -> str:
    """One line per guide: name, title, and which warehouse schemas its text mentions.

    Schema mentions are detected automatically (whole-word, case-insensitive match of
    each schema name in the guide text), so the index stays correct as guides or the
    warehouse change.
    """
    lines = []
    for path in _guide_files():
        text = path.read_text()
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        line = f"- {path.stem}: {title}"
        if schema_names:
            hits = [
                s for s in schema_names
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])", text, re.I)
            ]
            line += f" [mentions schemas: {', '.join(hits)}]" if hits else " [no schema match]"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: read_guide
# ---------------------------------------------------------------------------


def read_guide(name: str) -> str:
    """Return the full text of a guide."""
    stem = Path(name).stem
    path = GUIDES_DIR / f"{stem}.md"
    if not path.exists():
        return f"Guide '{name}' not found. Available: {', '.join(p.stem for p in _guide_files())}"
    return path.read_text()


READ_GUIDE = Tool(
    name="read_guide",
    description=(
        "Read the full text of a business-rules guide (by name from the guide list in the "
        "system prompt). Guides define the company's metric definitions, filters, and "
        "exclusions; they MUST be applied when relevant."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Guide name, e.g. 'financial_operations'."}
        },
        "required": ["name"],
    },
    function=read_guide,
)


# ---------------------------------------------------------------------------
# Tool: search_guides
# ---------------------------------------------------------------------------


def search_guides(keywords: list[str]) -> str:
    """Case-insensitive keyword search across all guides; returns matching lines."""
    pats = [re.compile(re.escape(k), re.IGNORECASE) for k in keywords if k.strip()]
    if not pats:
        return "Provide at least one keyword."
    hits: list[str] = []
    for path in _guide_files():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if any(p.search(line) for p in pats):
                hits.append(f"{path.stem}:{i}: {line.strip()[:300]}")
    if not hits:
        return "No matches."
    more = f"\n... {len(hits) - 60} more matches" if len(hits) > 60 else ""
    return "\n".join(hits[:60]) + more


SEARCH_GUIDES = Tool(
    name="search_guides",
    description=(
        "Search all business-rule guides for keywords (case-insensitive), e.g. schema, "
        "table, column names or business terms from the question. Returns matching lines "
        "with guide name and line number. Use it to find which guide(s) apply."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "Keywords."}
        },
        "required": ["keywords"],
    },
    function=search_guides,
)
