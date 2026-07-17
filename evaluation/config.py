# evaluation/config.py
import os
from pathlib import Path

# Đường dẫn thư mục gốc của project (AgenticRAG)
BASE_DIR = Path(__file__).parent.parent

# Thư mục chứa ground truth
GROUND_TRUTH_DIR = BASE_DIR / "evaluation" / "ground_truth_data"

# Thư mục lưu kết quả
OUTPUT_DIR = BASE_DIR / "evaluation" / "results"

# API Keys (lấy từ .env hoặc os.environ)
COHERE_API_KEY_RAG = os.getenv("COHERE_API_KEY", "")      # Key cho RAG chính
COHERE_API_KEY_EVAL = os.getenv("COHERE_API_KEY_EVAL", COHERE_API_KEY_RAG)  # Nếu không có key riêng, dùng chung

# Số lượng câu hỏi đánh giá (mặc định đánh giá tất cả)
NUM_SAMPLES = None  # None = tất cả, hoặc set số lượng

# Metrics mặc định
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Judge LLM: "cohere" hoặc "openai" hoặc "google"
JUDGE_LLM = "cohere"  # vì bạn đã có Cohere key