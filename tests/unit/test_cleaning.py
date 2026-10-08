from kassist.ingestion.cleaning import clean_page_markdown
from kassist.ingestion.loader import infer_title, slugify


def test_removes_running_footer_and_page_numbers():
    md = "Real content here.\n\nRAG Architecture Patterns — Technical Reference \n\nPage 3 of 10 \n"
    assert clean_page_markdown(md) == "Real content here."


def test_removes_letter_spaced_banners_and_mark_tags():
    md = "**S E C T I O N 0 5**\n\n# **<mark>pgvector — PostgreSQL Extension</mark>**\n\nBody."
    assert clean_page_markdown(md) == "# pgvector — PostgreSQL Extension\n\nBody."


def test_keeps_long_lines_that_mention_document_numbers():
    line = ("Document 6 of 6 — Agentic AI Frameworks | Questions about frameworks not covered here "
            "(Haystack Agents, Semantic Kernel, Flowise) should be handled as unanswerable by the assistant.")
    assert line in clean_page_markdown(line)


def test_drops_orphan_bullets_and_double_bullets():
    md = "- \n•\n- • Deterministic and fast"
    assert clean_page_markdown(md) == "- Deterministic and fast"


def test_tables_are_not_treated_as_footers():
    md = "|System|Recall@10|\n|---|---|\n|Document 3 of 6 row|1|"
    assert "|Document 3 of 6 row|1|" in clean_page_markdown(md)


def test_infer_title_skips_contents_and_numbers():
    md = "### Contents\n\n# 03\n\n## Vector Database Comparison\n"
    assert infer_title(md, "fallback") == "Vector Database Comparison"
    assert infer_title("no headings", "Fallback Title") == "Fallback Title"


def test_slugify():
    assert slugify("RAG_Architecture Patterns (v2)") == "rag_architecture_patterns_v2"
