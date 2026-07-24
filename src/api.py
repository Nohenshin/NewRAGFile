# src/api.py
import os
import sys
import pickle
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Thêm đường dẫn gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import từ project (sử dụng langchain_community để tránh lỗi pydantic_v1)
from src.llms.llm_factory import LLMFactory
from src.hybrid_retriever import HybridRetriever
from src.embeddings.hf_embeddings import HFEmbedding
from src.indexing import Index
from src.utils import build_vector_store, create_hybrid_retriever

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Pydantic Models ------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]

# ------------------ Load RAG Components ------------------
def load_rag_components():
    """
    Load chunks, parent_map từ pickle, tạo vector store và hybrid retriever.
    Trả về hybrid_retriever hoặc None nếu lỗi.
    """
    try:
        data_dir = "data"
        chunks_path = os.path.join(data_dir, "chunks.pkl")
        parent_map_path = os.path.join(data_dir, "parent_map.pkl")

        if not os.path.exists(chunks_path) or not os.path.exists(parent_map_path):
            logger.error(f"Pickle files not found: {chunks_path} or {parent_map_path}")
            return None

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        with open(parent_map_path, "rb") as f:
            parent_map = pickle.load(f)

        logger.info(f"Loaded {len(chunks)} chunks and {len(parent_map)} parent_map entries.")

        # Tạo embedding và vector store
        embeddings = HFEmbedding().get_embeddings()
        vector_store = build_vector_store(embeddings)

        # Tạo hybrid retriever
        hybrid_retriever = create_hybrid_retriever(vector_store, chunks, parent_map)
        logger.info("Hybrid retriever created successfully.")
        return hybrid_retriever

    except Exception as e:
        logger.error(f"Error loading RAG components: {e}", exc_info=True)
        return None

# Khởi tạo FastAPI
app = FastAPI(title="RAG API for Evaluation", version="1.0")

# Load components khi khởi động (1 lần)
hybrid_retriever = load_rag_components()

# ------------------ Endpoint ------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Nhận câu hỏi, trả về câu trả lời và các đoạn văn liên quan.
    """
    global hybrid_retriever
    if hybrid_retriever is None:
        # Thử load lại (có thể file đã được tạo sau khi server start)
        hybrid_retriever = load_rag_components()
        if hybrid_retriever is None:
            raise HTTPException(status_code=503, detail="RAG components not available. Please upload a document first.")

    try:
        # 1. Retrieval
        results = hybrid_retriever.hybrid_search(request.question, top_k=request.top_k)
        contexts = [r.get("parent_content", r.get("child_content", "")) for r in results]
        context_text = "\n".join(contexts)

        if not contexts:
            # Nếu không có context, trả về câu trả lời mặc định
            return QueryResponse(
                question=request.question,
                answer="No relevant information found in the document.",
                contexts=[]
            )

        # 2. Generate answer
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            raise HTTPException(status_code=500, detail="COHERE_API_KEY not set in environment variables.")

        # Lấy LLM từ factory (cohere)
        llm_instance = LLMFactory.create_llm("cohere", cohere_api_key=cohere_key)
        llm = llm_instance.get_llm()

        prompt = f"""Answer the question based only on the context below.
If you cannot answer, say "Cannot answer from context."

Context:
{context_text}

Question: {request.question}

Answer:"""
        response = llm.invoke(prompt)
        answer = response.content.strip()

        return QueryResponse(
            question=request.question,
            answer=answer,
            contexts=contexts
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Health Check ------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "components_loaded": hybrid_retriever is not None}

# ------------------ Chạy server ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)