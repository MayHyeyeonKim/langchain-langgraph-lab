from document_reviewer.documents import normalize_text


def test_normalize_text_removes_extra_whitespace() -> None:
    assert normalize_text("  Hello   world \n\n A   paragraph ") == "Hello world\nA paragraph"
