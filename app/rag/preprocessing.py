
import json
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_text(text: str) -> str:
    """Remove unnecessary whitespace and normalize scraped text."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\xa0", " ", text)
    return text.strip()


def clean_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Standardize a scraped document."""
    return {
        "source": clean_text(str(doc.get("source", ""))),
        "title": clean_text(str(doc.get("title", ""))),
        "url": clean_text(str(doc.get("url", ""))),
        "headings": [
            clean_text(str(h))
            for h in doc.get("headings", [])
            if clean_text(str(h))
        ],
        "content": clean_text(str(doc.get("content", ""))),
        "scraped_at": doc.get("scraped_at", "")
    }


def process_file(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    cleaned_documents = []

    for doc in documents:
        cleaned = clean_document(doc)

        if len(cleaned["content"]) < 100:
            continue

        cleaned_documents.append(cleaned)

    return cleaned_documents


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in RAW_DIR.glob("*.json"):
        documents = process_file(file_path)

        output_path = PROCESSED_DIR / file_path.name

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(documents, file, indent=4, ensure_ascii=False)

        print(
            f"{file_path.name}: "
            f"{len(documents)} clean documents saved"
        )


if __name__ == "__main__":
    main()
