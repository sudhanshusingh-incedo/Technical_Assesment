from kassist.config import ChunkingSettings
from kassist.ingestion.chunker import chunk_document, estimate_tokens, parse_blocks
from kassist.ingestion.loader import PageMarkdown, ParsedDocument

CFG = ChunkingSettings(max_tokens=120, overlap_tokens=20, min_tokens=15)


def doc(*pages: str, title: str = "Test Doc") -> ParsedDocument:
    return ParsedDocument(doc_id="test_doc", title=title, source_file="test.pdf", file_sha256="abc",
                          pages=[PageMarkdown(i + 1, md) for i, md in enumerate(pages)])


def para(word: str, n: int) -> str:
    """n words in 12-word sentences (capitalised, so the sentence splitter can find boundaries)."""
    w = word.upper()
    return " ".join(f"{w}{i}." if i % 12 == 11 else f"{w}{i}" for i in range(n))


def test_parse_blocks_kinds():
    md = "## Heading\n\nSome text\nmore text\n\n|a|b|\n|---|---|\n|1|2|\n\n```\ncode()\n```"
    kinds = [b.kind for b in parse_blocks(md, 1)]
    assert kinds == ["heading", "text", "table", "code"]


def test_sibling_heading_always_starts_new_chunk_even_when_small():
    d = doc("# FAISS\n\nFAISS is a library for ANN search.\n\n# Pinecone\n\nPinecone is a managed service.")
    chunks = chunk_document(d, CFG)
    assert [c.section.split(" > ")[-1] for c in chunks] == ["FAISS", "Pinecone"]
    assert "Pinecone" not in chunks[0].text


def test_small_subsection_stays_with_parent():
    d = doc(f"## Weaviate\n\n{para('w', 30)}\n\n#### STRENGTHS\n\nNative hybrid search.")
    chunks = chunk_document(d, CFG)
    assert len(chunks) == 1
    assert "STRENGTHS" in chunks[0].text and chunks[0].section.endswith("Weaviate")


def test_respects_token_budget_and_overlaps_within_section():
    d = doc(f"## Long\n\n{para('a', 100)}\n\n{para('b', 100)}\n\n{para('c', 100)}")
    chunks = chunk_document(d, CFG)
    assert len(chunks) >= 3
    assert all(c.token_estimate <= int(CFG.max_tokens * 1.3) for c in chunks)
    # sentence-level overlap: the start of chunk n+1 repeats the tail of chunk n
    assert any(chunks[i + 1].text.split()[0] in chunks[i].text for i in range(len(chunks) - 1))


def test_oversized_table_split_repeats_header():
    rows = "\n".join(f"|system{i}|{i}% recall|{i} ms latency value text|" for i in range(60))
    d = doc(f"## Benchmarks\n\n|System|Recall|Latency|\n|---|---|---|\n{rows}")
    chunks = chunk_document(d, CFG)
    assert len(chunks) > 1
    for c in chunks:
        assert "|System|Recall|Latency|" in c.text
        assert c.content_type == "table"


def test_contents_section_is_skipped_and_section_spans_pages():
    d = doc("## Table of Contents\n\n|1|Intro|3|\n|---|---|---|\n\n## Intro\n\nStart of intro.",
            "Intro continues on page two with more words.")
    chunks = chunk_document(d, CFG)
    assert all("Table of Contents" not in c.text for c in chunks)
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2


def test_ids_are_deterministic_and_embed_text_has_context_header():
    d = doc("## A\n\nalpha text here.\n\n## B\n\nbeta text here.")
    first, second = chunk_document(d, CFG), chunk_document(d, CFG)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert first[0].embed_text.startswith("Document: Test Doc\nSection: A")
    assert not first[0].text.startswith("Document:")


def test_estimate_tokens_handles_tables_densely():
    assert estimate_tokens("|1|2|3|4|5|6|7|8|") >= 4



def test_signature_changes_with_chunking_settings():
    from kassist.ingestion.chunker import chunker_signature

    assert chunker_signature(CFG) == chunker_signature(CFG.model_copy())
    assert chunker_signature(CFG) != chunker_signature(CFG.model_copy(update={"max_tokens": 200}))
