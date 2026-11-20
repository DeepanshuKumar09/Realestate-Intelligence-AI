
from app.rag.chain import get_vector_store

vector_store = get_vector_store()

print("Total documents in ChromaDB:", vector_store._collection.count())

query = "Wasalt real estate properties, property listings, locations and prices"

results = vector_store.similarity_search(
    query,
    k=10
)

print("\nWASALT RETRIEVAL RESULTS\n")

for i, doc in enumerate(results, 1):

    print(f"\nResult {i}")
    print("Source:", doc.metadata.get("source"))
    print("Title:", doc.metadata.get("title"))
    print("URL:", doc.metadata.get("url"))
    print("Content:", doc.page_content[:400])
    print("-" * 60)
