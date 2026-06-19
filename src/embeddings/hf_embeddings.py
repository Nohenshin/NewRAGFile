from langchain_community.embeddings import HuggingFaceEmbeddings

class HFEmbedding:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embeddings(self):
        return self.embeddings