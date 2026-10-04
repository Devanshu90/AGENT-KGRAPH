from src.data.preprocess import process_document


def test_process_document():
    text = """Christopher Nolan directed Inception.

Inception was released in 2010.

Christopher Nolan directed Inception."""

    result = process_document(text, "doc001")

    assert len(result) == 2
    assert result[0]["passage_id"] == "p000001"
    assert result[1]["passage_id"] == "p000002"