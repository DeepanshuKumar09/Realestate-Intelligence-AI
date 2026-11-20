
import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.rag.embedding import get_embedding_model


BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "real_estate_knowledge"


def load_documents():
    documents = []

    for file_path in PROCESSED_DIR.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            records = json.load(file)

        for record in records:
            content = record.get("content", "")

            if not content.strip():
                continue

            # Include headings to preserve page context.
            text = (
                f"Title: {record.get('title', '')}\n"
                f"Headings: {' | '.join(record.get('headings', []))}\n"
                f"Content: {content}"
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": record.get("source", ""),
                        "title": record.get("title", ""),
                        "url": record.get("url", ""),
                        "file": file_path.name
                    }
                )
            )

    return documents


def create_vector_store():
    documents = load_documents()

    if not documents:
        raise ValueError("No processed documents found!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Stable IDs allow safe rebuilding without duplicate records.
    chunk_ids = [
        str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{doc.metadata['url']}:{i}:{doc.page_content}"
        ))
        for i, doc in enumerate(chunks)
    ]

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)
    )

    # Rebuild collection from the current processed dataset.
    existing = vector_store.get()
    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids
    )

    print(f"Original documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector database: {VECTOR_DB_DIR}")

    return vector_store


if __name__ == "__main__":
    create_vector_store()
