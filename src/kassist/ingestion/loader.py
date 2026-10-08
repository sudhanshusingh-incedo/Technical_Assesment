"""PDF -> per-page cleaned markdown.

pymupdf4llm is used instead of plain text extraction because a large share of this corpus
(benchmarks, pricing, comparison matrices) lives in tables. It reconstructs them as markdown
tables and marks headings by font size, which the chunker relies on for section boundaries.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from kassist.ingestion.cleaning import clean_page_markdown

_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_NON_TITLE = {"contents", "table of contents"}


@dataclass
class PageMarkdown:
    page_number: int  # 1-based, matches the printed page numbers of the corpus
    markdown: str


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source_file: str
    file_sha256: str
    pages: list[PageMarkdown] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def infer_title(first_page_markdown: str, fallback: str) -> str:
    """First real heading on the cover page (skips 'Contents' and bare numbers like '03')."""
    for match in _HEADING.finditer(first_page_markdown):
        candidate = match.group(1).strip()
        words = [w for w in candidate.split() if any(c.isalpha() for c in w)]
        if len(words) >= 2 and candidate.lower() not in _NON_TITLE:
            return candidate
    return fallback


def _extract_raw_pages(path: Path) -> list[tuple[int, str]]:
    import pymupdf4llm  # heavy import; only needed at ingestion time

    page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    return [(int(pc["metadata"]["page_number"]), pc["text"]) for pc in page_chunks]


def load_pdf(path: Path, cache_dir: Path | None = None) -> ParsedDocument:
    """Parse a PDF. Layout extraction is the slowest ingestion step (~3 s/page on CPU), so the
    *raw* extraction is cached by file hash; cleaning/chunking changes still apply on re-runs."""
    sha = file_sha256(path)
    cache_file = cache_dir / f"{sha}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        raw = [tuple(p) for p in json.loads(cache_file.read_text(encoding="utf-8"))]
    else:
        raw = _extract_raw_pages(path)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    pages = [PageMarkdown(page_number=int(n), markdown=clean_page_markdown(str(md))) for n, md in raw]
    fallback_title = path.stem.replace("_", " ").title()
    title = infer_title(pages[0].markdown, fallback_title) if pages else fallback_title
    return ParsedDocument(
        doc_id=slugify(path.stem),
        title=title,
        source_file=path.name,
        file_sha256=sha,
        pages=pages,
    )
