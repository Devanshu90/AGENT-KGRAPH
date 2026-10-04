from pathlib import Path
from src.data.preprocess import process_document, save_json


def main():
    raw_dir = Path("data/raw")
    output_path = Path("data/processed/passages.json")

    results = []

    for file_path in raw_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        document_id = file_path.stem

        passages = process_document(text, document_id)
        results.extend(passages)

    save_json(results, output_path)

    print(f"Processed passages: {len(results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()