from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import numpy as np
import re

class HybridRetriever:
    def __init__(self, vector_store: Chroma, child_chunks: List[Document], parent_map: Dict[str, str]):
        self.vector_store = vector_store
        self.child_chunks = child_chunks
        self.parent_map = parent_map
        
        # Build BM25 corpus
        self.corpus = [chunk.page_content for chunk in child_chunks]
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        # Dense retrieval
        dense_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)
        # BM25
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # RRF fusion
        rrf_k = 60
        fusion_scores = {}
        # Dense
        for rank, (doc, score) in enumerate(dense_results):
            content = doc.page_content
            fusion_scores[content] = fusion_scores.get(content, 0) + 1 / (rrf_k + rank + 1)
        # BM25
        for rank, idx in enumerate(bm25_indices):
            content = self.corpus[idx]
            fusion_scores[content] = fusion_scores.get(content, 0) + 1 / (rrf_k + rank + 1)
        
        # Sort and build results
        sorted_contents = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for content, score in sorted_contents[:top_k]:
            chunk = next((c for c in self.child_chunks if c.page_content == content), None)
            if chunk:
                parent_id = chunk.metadata.get("parent_id")
                parent_text = self.parent_map.get(parent_id, "")
                results.append({
                    "child_content": content,
                    "parent_content": parent_text,
                    "parent_id": parent_id,
                    "fusion_score": score,
                    "metadata": chunk.metadata
                })
        return results