from functools import lru_cache
from math import sqrt

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):

    def __init__(self):
        self.model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            threads=2
        )

    @staticmethod
    def normalize(vector):
        magnitude = sqrt(sum(value * value for value in vector))

        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]

    def embed_documents(self, texts):
        embeddings = self.model.embed(
            texts,
            batch_size=16
        )

        return [
            self.normalize(embedding.tolist())
            for embedding in embeddings
        ]

    def embed_query(self, text):
        embedding = next(
            self.model.embed([text], batch_size=1)
        )

        return self.normalize(embedding.tolist())


@lru_cache(maxsize=1)
def get_embedding_model():
    return FastEmbedEmbeddings()
