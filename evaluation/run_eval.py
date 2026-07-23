# evaluation/run_eval.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import run_evaluation_from_session
from src.indexing import Index
from src.embeddings.hf_embeddings import HFEmbedding
from src.utils import build_vector_store, create_hybrid_retriever
import pickle

# Load vector store và hybrid retriever từ disk (nếu đã lưu)
# Hoặc bạn có thể truyền từ session state nếu chạy trong Streamlit

# Giả sử bạn đã lưu chunks và parent_map sau khi index
try:
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("data/parent_map.pkl", "rb") as f:
        parent_map = pickle.load(f)
except:
    print("⚠️ Vui lòng index tài liệu trước khi chạy evaluation!")
    sys.exit(1)

# Tạo vector store và hybrid retriever
embeddings = HFEmbedding().get_embeddings()
vector_store = build_vector_store(embeddings)
hybrid_retriever = create_hybrid_retriever(vector_store, chunks, parent_map)

# Chạy đánh giá
result = run_evaluation_from_session(
    vector_store=vector_store,
    hybrid_retriever=hybrid_retriever,
    ground_truth_file="Odoo.json",  # Tên file trong ground_truth_data
    num_samples=5,  # Đánh giá 5 câu đầu để test nhanh
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

print("📊 Kết quả đánh giá:")
print(result["scores"])
print(f"📄 Báo cáo lưu tại: {result['report_file']}")