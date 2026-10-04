import json
import hashlib
import re
from pathlib import Path
from nltk.tokenize import sent_tokenize


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_paragraphs(text):
    paragraphs = re.split(r"\n\s*\n", text)
    return [clean_text(p) for p in paragraphs if clean_text(p)]


def split_sentences(text):
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def passage_hash(text):
    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()


def deduplicate_passages(passages):
    seen = set()
    unique = []

    for passage in passages:
        h = passage_hash(passage)

        if h not in seen:
            seen.add(h)
            unique.append(passage)

    return unique


def create_passages(text, document_id):
    passages = []

    paragraphs = split_paragraphs(text)

    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)

        for sentence_index, sentence in enumerate(sentences):
            passages.append({
                "document_id": document_id,
                "paragraph_id": paragraph_index,
                "sentence_id": sentence_index,
                "text": sentence
            })

    return passages


def process_document(text, document_id):
    passages = create_passages(text, document_id)

    texts = [p["text"] for p in passages]
    unique_texts = deduplicate_passages(texts)

    result = []

    for index, text in enumerate(unique_texts, start=1):
        result.append({
            "passage_id": f"p{index:06d}",
            "document_id": document_id,
            "text": text
        })

    return result


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)