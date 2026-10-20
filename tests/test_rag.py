from document_reviewer.rag import RetrievedContext, build_context, source_references


def test_build_context_labels_and_limits_sources() -> None:
    chunks = [
        RetrievedContext(text="A useful rule", source="guide.pdf", page=3),
        RetrievedContext(text="Another rule", source="guide.pdf", page=4),
    ]
    context = build_context(chunks)
    assert "[source: guide.pdf, page 3]" in context
    assert "Another rule" in context


def test_source_references_are_deduplicated() -> None:
    chunks = [
        RetrievedContext(text="first", source="guide.pdf", page=3),
        RetrievedContext(text="second", source="guide.pdf", page=3),
    ]
    assert [source.model_dump() for source in source_references(chunks)] == [
        {"source": "guide.pdf", "page": 3},
    ]
