"""Structure-aware chunking.

Strategy (and why):
* Parse each page's markdown into blocks: heading / table / code / text.
* Pack blocks into chunks up to a token budget. A heading starts a new chunk once the
  current one has real content, so chunks rarely straddle two topics; tiny sub-sections
  (e.g. a 'STRENGTHS' box) stay attached to their parent instead of becoming orphan chunks.
* Tables are kept whole when they fit. Oversized tables are split by rows with the header row
  repeated, so every piece is self-describing ("pgvector HNSW | 96.1% | ..." keeps its columns).
* Oversized prose is split on sentence boundaries with overlap.
* Overlap is applied only within a section (carrying context across a topic change only adds noise).
* Every chunk gets a contextual header ("Document / Section") prepended to the text that is
  embedded, which disambiguates generic passages such as "Strengths" or "Hybrid Search"
  that occur in several documents.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

from kassist.config import ChunkingSettings
from kassist.domain import Chunk, ContentType
from kassist.ingestion.cleaning import clean_heading_text
from kassist.ingestion.loader import ParsedDocument

# Bump whenever chunking logic changes: the pipeline re-indexes documents whose stored
# chunker_version differs, so old and new chunks never mix.
CHUNKER_VERSION = "1.2"

_CHUNK_NS = uuid.UUID("6f1c2a52-3b0e-4f7e-9a51-2a3c1d4e5f60")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_SKIP_SECTIONS = {"contents", "table of contents"}

BlockKind = Literal["heading", "table", "code", "text"]


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate; tables/numbers are denser so take the max of two views."""
    return max(int(len(text.split()) * 1.3), len(text) // 4)


@dataclass
class Block:
    kind: BlockKind
    text: str
    page: int
    level: int = 0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


def parse_blocks(markdown: str, page: int) -> list[Block]:
    blocks: list[Block] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            body = "\n".join(lines[i : j + 1]).strip()
            blocks.append(Block("code", body, page))
            i = j + 1
            continue
        heading = _HEADING.match(stripped)
        if heading:
            blocks.append(Block("heading", clean_heading_text(heading.group(2)), page,
                                level=len(heading.group(1))))
            i += 1
            continue
        if stripped.startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            blocks.append(Block("table", "\n".join(ln.strip() for ln in lines[i:j]), page))
            i = j
            continue
        j = i
        para: list[str] = []
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith(("|", "```")) or _HEADING.match(nxt):
                break
            para.append(lines[j].rstrip())
            j += 1
        blocks.append(Block("text", "\n".join(para).strip(), page))
        i = j
    return blocks


def _split_table(block: Block, max_tokens: int) -> list[Block]:
    rows = block.text.splitlines()
    if len(rows) <= 3:
        return [block]
    header, body = rows[:2], rows[2:]
    pieces: list[Block] = []
    current: list[str] = []
    for row in body:
        candidate = "\n".join(header + current + [row])
        if current and estimate_tokens(candidate) > max_tokens:
            pieces.append(Block("table", "\n".join(header + current), block.page))
            current = []
        current.append(row)
    if current:
        pieces.append(Block("table", "\n".join(header + current), block.page))
    return pieces


def _split_text(block: Block, max_tokens: int, overlap_tokens: int) -> list[Block]:
    units = _SENTENCE_SPLIT.split(block.text) if block.kind == "text" else block.text.splitlines()
    joiner = " " if block.kind == "text" else "\n"
    pieces: list[Block] = []
    current: list[str] = []
    for unit in units:
        if current and estimate_tokens(joiner.join([*current, unit])) > max_tokens:
            pieces.append(Block(block.kind, joiner.join(current), block.page))
            # sentence-level overlap: carry trailing units up to the overlap budget
            carry: list[str] = []
            for prev in reversed(current):
                if estimate_tokens(joiner.join([prev, *carry])) > overlap_tokens:
                    break
                carry.insert(0, prev)
            current = carry
        current.append(unit)
    if current:
        pieces.append(Block(block.kind, joiner.join(current), block.page))
    return pieces




@dataclass(frozen=True)
class Budget:
    max_tokens: int
    overlap_tokens: int
    min_tokens: int


def chunker_signature(cfg: ChunkingSettings) -> str:
    """Identifies chunking logic + settings. Stored with every vector, so any change re-indexes."""
    return f"{CHUNKER_VERSION}-{cfg.max_tokens}/{cfg.overlap_tokens}/{cfg.min_tokens}"


def split_oversized(block: Block, budget: Budget) -> list[Block]:
    if block.tokens <= budget.max_tokens:
        return [block]
    if block.kind == "table":
        return _split_table(block, budget.max_tokens)
    return _split_text(block, budget.max_tokens, budget.overlap_tokens)


@dataclass
class _Draft:
    section: str
    level: int = 99  # markdown level of the heading that opened this draft (99 = none)
    blocks: list[Block] = field(default_factory=list)
    carried: int = 0  # leading blocks copied from the previous chunk as overlap

    @property
    def tokens(self) -> int:
        return sum(b.tokens for b in self.blocks)

    @property
    def has_content(self) -> bool:
        """Only fresh (non-overlap, non-heading) blocks make a draft worth emitting."""
        return any(b.kind != "heading" for b in self.blocks[self.carried:])


def _render(blocks: list[Block]) -> str:
    return "\n\n".join(f"{'#' * max(b.level, 2)} {b.text}" if b.kind == "heading" else b.text
                       for b in blocks).strip()


def _content_type(blocks: list[Block]) -> ContentType:
    kinds = {b.kind for b in blocks if b.kind != "heading"}
    if kinds == {"table"}:
        return "table"
    return "mixed" if "table" in kinds else "text"


class _SectionTracker:
    """Breadcrumb of the latest heading per markdown level (levels vary between documents)."""

    def __init__(self) -> None:
        self._levels: dict[int, str] = {}

    def push(self, level: int, text: str) -> None:
        self._levels = {lvl: t for lvl, t in self._levels.items() if lvl < level}
        self._levels[level] = text

    @property
    def path(self) -> str:
        parts = [self._levels[lvl] for lvl in sorted(self._levels)]
        return " > ".join(parts[-2:])

    @property
    def skipped(self) -> bool:
        return any(t.lower() in _SKIP_SECTIONS for t in self._levels.values())


def _build_drafts(doc: ParsedDocument, budget: Budget) -> list[_Draft]:
    """Section-aware packing of a whole document into drafts of at most `budget.max_tokens`."""
    drafts: list[_Draft] = []
    tracker = _SectionTracker()
    current = _Draft(section=doc.title)

    def flush(carry_overlap: bool) -> None:
        nonlocal current
        if not current.has_content:
            return
        drafts.append(current)
        carry: list[Block] = []
        if carry_overlap:
            for prev in reversed(current.blocks):
                if prev.kind != "text" or sum(b.tokens for b in carry) + prev.tokens > budget.overlap_tokens:
                    break
                carry.insert(0, prev)
        current = _Draft(section=current.section, level=current.level, blocks=carry,
                         carried=len(carry))

    for page in doc.pages:
        for block in parse_blocks(page.markdown, page.page_number):
            if block.kind == "heading":
                if block.text.isdigit():  # cover-page ornaments such as "03"
                    continue
                # A sibling/parent heading always closes the chunk (new topic). A deeper heading
                # (sub-section) only does once the chunk is big enough to stand on its own.
                new_topic = block.level <= current.level
                if current.has_content and (new_topic or current.tokens >= budget.min_tokens):
                    flush(carry_overlap=False)
                tracker.push(block.level, block.text)
                if not current.has_content:
                    current = _Draft(section=tracker.path or doc.title, level=block.level)
                if not tracker.skipped:
                    current.blocks.append(block)
                continue
            if tracker.skipped:
                continue
            for piece in split_oversized(block, budget):
                if current.has_content and current.tokens + piece.tokens > budget.max_tokens:
                    flush(carry_overlap=True)
                current.blocks.append(piece)
    flush(carry_overlap=False)

    # Fold a too-small draft (e.g. a 'KEY INSIGHT' box) into its predecessor when the two are
    # in the same section family, so it keeps its context instead of becoming an orphan chunk.
    merged: list[_Draft] = []
    for draft in drafts:
        prev = merged[-1] if merged else None
        related = prev is not None and bool(
            set(prev.section.split(" > ")) & set(draft.section.split(" > ")))
        if prev is not None and related and draft.tokens < budget.min_tokens \
                and prev.tokens + draft.tokens <= int(budget.max_tokens * 1.3):
            prev.blocks.extend(draft.blocks)
        else:
            merged.append(draft)
    return merged


def _make_chunk(doc: ParsedDocument, blocks: list[Block], section: str, index: int, id_key: str) -> Chunk:
    text = _render(blocks)
    pages = [b.page for b in blocks]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    header = f"Document: {doc.title}\nSection: {section}"
    return Chunk(
        chunk_id=str(uuid.uuid5(_CHUNK_NS, f"{doc.doc_id}:{id_key}:{digest}")),
        doc_id=doc.doc_id,
        doc_title=doc.title,
        source_file=doc.source_file,
        page_start=min(pages),
        page_end=max(pages),
        section=section,
        chunk_index=index,
        content_type=_content_type(blocks),
        text=text,
        embed_text=f"{header}\n\n{text}",
        token_estimate=estimate_tokens(text),
    )


def chunk_document(doc: ParsedDocument, cfg: ChunkingSettings) -> list[Chunk]:
    sig = chunker_signature(cfg)
    drafts = _build_drafts(doc, Budget(cfg.max_tokens, cfg.overlap_tokens, cfg.min_tokens))
    return [_make_chunk(doc, d.blocks, d.section, i, f"{sig}:{i}") for i, d in enumerate(drafts)]
