
from pathlib import Path

from langchain_chroma import Chroma
from app.rag.embedding import get_embedding_model


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "real_estate_knowledge"


def get_vector_store():
    """Connect to the existing ChromaDB."""

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)
    )

    return vector_store


def test_database_connection():
    """Test ChromaDB connection and stored vectors."""

    vector_store = get_vector_store()

    count = vector_store._collection.count()

    print(f"\nTotal vectors stored: {count}")

    assert count == 335, f"Expected 335 vectors, found {count}"

    print("Database connection: PASSED")


def test_darglobal_retrieval():
    """Test retrieval of DarGlobal information."""

    vector_store = get_vector_store()

    query = "Tell me about DarGlobal projects and developments."

    results = vector_store.similarity_search(query, k=3)

    assert len(results) > 0, "No DarGlobal results found"

    print("\nDarGlobal Retrieval Results:")

    for doc in results:
        print("\nTitle:", doc.metadata.get("title"))
        print("Source:", doc.metadata.get("source"))
        print("URL:", doc.metadata.get("url"))
        print("Content:", doc.page_content[:300])

    print("\nDarGlobal retrieval: PASSED")


def test_wasalt_retrieval():
    """Test retrieval of Wasalt information."""

    vector_store = get_vector_store()

    query = "What properties and real estate listings are available on Wasalt?"

    results = vector_store.similarity_search(query, k=3)

    assert len(results) > 0, "No Wasalt results found"

    print("\nWasalt Retrieval Results:")

    for doc in results:
        print("\nTitle:", doc.metadata.get("title"))
        print("Source:", doc.metadata.get("source"))
        print("URL:", doc.metadata.get("url"))
        print("Content:", doc.page_content[:300])

    print("\nWasalt retrieval: PASSED")


def test_metadata():
    """Check whether retrieved documents contain metadata."""

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        "Real estate properties and developments",
        k=3
    )

    for doc in results:
        assert doc.metadata.get("url"), "Missing source URL"
        assert doc.metadata.get("title"), "Missing title"
        assert doc.page_content.strip(), "Empty document content"

    print("\nMetadata verification: PASSED")


if __name__ == "__main__":

    test_database_connection()
    test_darglobal_retrieval()
    test_wasalt_retrieval()
    test_metadata()

    print("\nAll retrieval tests completed!")
