from src.embeddings.cohere_embeddings import CohereEmbedding
from src.embeddings.hf_embeddings import HFEmbedding
# from src.embeddings.google_embeddings import GoogleEmbedding  # Comment

class EmbeddingsFactory:
    @staticmethod
    def create_embeddings(provider: str, **kwargs):
        provider = provider.lower()
        if provider == "cohere":
            assert "cohere_api_key" in kwargs, "Missing cohere_api_key"
            return CohereEmbedding(**kwargs)
        elif provider == "huggingface":
            return HFEmbedding(**kwargs)
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")