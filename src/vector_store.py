import chromadb
from chromadb.config import Settings
from .embeddings import EmbeddingModel
import uuid


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="startup_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = EmbeddingModel()

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict] = None,
        ids: list[str] = None
    ):
        """Dodaje dokumente u bazu."""
        embeddings = self.embedding_model.embed_texts(texts)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        if metadatas is None:
            metadatas = [{}] * len(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return len(texts)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Semanticka pretraga."""
        query_embedding = self.embedding_model.embed_text(query)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
        except Exception:
            # Refresh collection reference if stale
            self.collection = self.client.get_or_create_collection(
                name="startup_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

        if not results["documents"][0]:
            return []

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def get_stats(self) -> dict:
        """Vraca statistiku baze."""
        try:
            count = self.collection.count()
        except Exception:
            # Refresh collection reference if stale
            self.collection = self.client.get_or_create_collection(
                name="startup_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            count = self.collection.count()
        return {
            "total_documents": count
        }

    def delete_all(self):
        """Brise sve dokumente."""
        self.client.delete_collection("startup_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="startup_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
