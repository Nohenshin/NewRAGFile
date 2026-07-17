# evaluation/evaluator.py
import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Thêm đường dẫn gốc để import code RAG
sys.path.append(str(Path(__file__).parent.parent))

# Import từ project chính
from src.hybrid_retriever import HybridRetriever
from src.llms.factory import LLMFactory
from src.embeddings.factory import EmbeddingsFactory
from src.utils import load_pdf, split_documents, build_vector_store, add_chunks_to_vector_store_hf_embeddings, create_hybrid_retriever

# Import RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document

# Config
from .config import (
    COHERE_API_KEY,
    NUM_SAMPLES,
    METRICS,
    JUDGE_LLM,
    OUTPUT_DIR
)

logger = logging.getLogger(__name__)

class RagRetriever:
    """
    Truy xuất contexts từ hệ thống RAG (import trực tiếp).
    """
    def __init__(self, vector_store, hybrid_retriever):
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Lấy top_k contexts từ hybrid retriever."""
        if self.hybrid_retriever is None:
            return []
        results = self.hybrid_retriever.hybrid_search(query, top_k=top_k)
        return [
            {
                "text": r.get("parent_content", r.get("child_content", "")),
                "score": r.get("fusion_score", 0.0),
                "metadata": r.get("metadata", {})
            }
            for r in results
        ]


class RagasEvaluator:
    """
    Đánh giá RAG bằng RAGAS, dùng Cohere làm Judge LLM.
    """

    def __init__(self, vector_store, hybrid_retriever, cohere_api_key: str = None):
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.cohere_api_key = cohere_api_key or COHERE_API_KEY
        self.retriever = RagRetriever(vector_store, hybrid_retriever)

        # Khởi tạo Judge LLM (dùng Cohere)
        self.judge_llm = self._init_judge_llm()
        self.judge_embeddings = self._init_judge_embeddings()

        # Tạo thư mục output
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _init_judge_llm(self):
        """Khởi tạo LLM dùng để đánh giá (Judge LLM)."""
        if JUDGE_LLM == "cohere":
            return ChatCohere(
                model="command-r-plus",
                cohere_api_key=self.cohere_api_key,
                temperature=0.1
            )
        elif JUDGE_LLM == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        elif JUDGE_LLM == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        else:
            raise ValueError(f"Unsupported judge LLM: {JUDGE_LLM}")

    def _init_judge_embeddings(self):
        """Khởi tạo embeddings dùng cho RAGAS (nếu cần)."""
        if JUDGE_LLM == "cohere":
            return CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=self.cohere_api_key)
        else:
            # fallback
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def _generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """Tạo câu trả lời từ contexts bằng LLM (dùng Cohere)."""
        if not contexts:
            return "No context available."

        context_text = "\n".join([c["text"] for c in contexts])
        prompt = f"""Answer the question based only on the context below.
If you cannot answer, say "Cannot answer from context."

Context:
{context_text}

Question: {question}

Answer:"""

        llm = ChatCohere(model="command-r-plus", cohere_api_key=self.cohere_api_key, temperature=0.2)
        response = llm.invoke(prompt)
        return response.content

    def run_evaluation(
        self,
        ground_truth_file: str,
        num_samples: Optional[int] = None,
        metrics_list: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Chạy đánh giá trên file ground truth.
        Trả về dict kết quả metrics.
        """
        from .ground_truth import GroundTruthLoader

        # Load câu hỏi
        loader = GroundTruthLoader()
        questions = loader.load_questions(ground_truth_file)
        if not questions:
            raise ValueError(f"No valid questions found in {ground_truth_file}")

        # Lấy số mẫu
        total = len(questions)
        if num_samples is None:
            num_samples = total
        else:
            num_samples = min(num_samples, total)

        # Lấy metrics cần đánh giá
        if metrics_list is None:
            metrics_list = METRICS
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
        selected_metrics = [metric_map[m] for m in metrics_list if m in metric_map]

        # Chuẩn bị dữ liệu
        all_questions = []
        all_contexts = []
        all_answers = []
        all_ground_truths = []

        for i, q in enumerate(questions[:num_samples]):
            question = q["question"]
            ground_truth = q["answer"]

            # Lấy contexts từ RAG
            contexts = self.retriever.retrieve(question, top_k=5)
            context_texts = [c["text"] for c in contexts]

            # Sinh câu trả lời
            answer = self._generate_answer(question, contexts)

            # Lưu
            all_questions.append(question)
            all_contexts.append(context_texts)
            all_answers.append(answer)
            all_ground_truths.append(ground_truth)

            logger.info(f"Processed {i+1}/{num_samples}: {question[:50]}...")

        # Tạo dataset cho RAGAS
        from datasets import Dataset
        dataset_dict = {
            "question": all_questions,
            "answer": all_answers,
            "contexts": all_contexts,
            "ground_truth": all_ground_truths
        }
        dataset = Dataset.from_dict(dataset_dict)

        # Đánh giá
        logger.info("Starting RAGAS evaluation...")
        result = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=LangchainLLMWrapper(self.judge_llm),
            embeddings=LangchainEmbeddingsWrapper(self.judge_embeddings)
        )

        # Chuyển kết quả sang dict
        scores = {}
        for metric in metrics_list:
            if metric in result:
                scores[metric] = float(result[metric])

        # Lưu báo cáo CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = OUTPUT_DIR / f"evaluation_report_{timestamp}.csv"
        result_df = result.to_pandas()
        result_df.to_csv(report_file, index=False)
        logger.info(f"Report saved to {report_file}")

        return {
            "scores": scores,
            "report_file": str(report_file),
            "num_samples": num_samples,
            "ground_truth_file": ground_truth_file
        }


# ========== HÀM CHẠY ĐÁNH GIÁ ĐƠN GIẢN CHO STREAMLIT ==========

def run_evaluation_from_session(
    vector_store,
    hybrid_retriever,
    ground_truth_file: str,
    num_samples: int = 10,
    cohere_api_key: str = None
) -> Dict[str, Any]:
    """
    Hàm được gọi từ Streamlit UI hoặc script.
    """
    evaluator = RagasEvaluator(vector_store, hybrid_retriever, cohere_api_key)
    return evaluator.run_evaluation(ground_truth_file, num_samples=num_samples)


# ========== KIỂM TRA NHANH ==========
if __name__ == "__main__":
    # Ví dụ chạy khi script được gọi trực tiếp
    from src.utils import create_hybrid_retriever
    # Giả sử đã có vector_store và hybrid_retriever từ session
    # Bạn cần load lại từ disk hoặc truyền vào
    print("⚠️ Chạy file này trực tiếp chỉ để test. Hãy import và dùng trong Streamlit.")