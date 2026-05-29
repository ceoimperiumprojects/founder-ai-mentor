import os

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    _instance = None
    _model = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Default to CPU. Pavle's local 930MX is visible to torch, but modern
            # PyTorch wheels do not ship kernels for compute capability 5.0, so
            # sentence-transformers auto-selecting CUDA crashes at encode time.
            selected_device = device or os.getenv("FOUNDER_KB_DEVICE", "cpu")
            cls._model = SentenceTransformer(model_name, device=selected_device)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        pass

    def embed_text(self, text: str) -> list[float]:
        """Embed-uje jedan tekst."""
        return self._model.encode(text).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed-uje listu tekstova."""
        return self._model.encode(texts).tolist()


if __name__ == "__main__":
    model = EmbeddingModel()
    embedding = model.embed_text("Test startup knowledge")
    print(f"Embedding dimension: {len(embedding)}")
