from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        if not documents:
            return []
        texts = [doc.get("parent_content", doc.get("child_content", "")) for doc in documents]
        pairs = [(query, text) for text in texts]
        inputs = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).cpu().numpy()
        for idx, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[idx]) if scores.size > 0 else 0.0
        sorted_docs = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return sorted_docs[:top_k]