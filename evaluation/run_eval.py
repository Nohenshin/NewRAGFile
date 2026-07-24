# evaluation/run_eval.py
import os
os.environ["OPENAI_API_KEY"] = "sk-dummy"

import requests
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.ground_truth import GroundTruthLoader
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_cohere import ChatCohere, CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000/query"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    print("❌ COHERE_API_KEY not set in .env file!")
    exit(1)

session = requests.Session()
retries = requests.adapters.Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))

def query_rag(question: str, top_k: int = 5):
    try:
        response = session.post(API_URL, json={"question": question, "top_k": top_k})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling RAG API: {e}")
        return None

loader = GroundTruthLoader()
questions = loader.load_questions("Odoo.json")
if not questions:
    print("No questions found!")
    exit()

samples = questions[:3]  # Chỉ 3 câu để tránh rate limit
data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for q in samples:
    result = query_rag(q["question"], top_k=5)
    if result:
        data["question"].append(result["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append(result["contexts"])
        data["ground_truth"].append(q["answer"])
        print(f"✅ Processed: {q['question'][:50]}...")
    else:
        print(f"❌ Failed: {q['question'][:50]}...")
    time.sleep(2)  # Giãn cách 2 giây

if len(data["question"]) == 0:
    print("No samples processed. Exiting.")
    exit()

# Dùng model command-r-08-2024 (còn hỗ trợ)
llm = ChatCohere(model="command-r-08-2024", cohere_api_key=COHERE_API_KEY)

cohere_embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
embeddings_wrapper = LangchainEmbeddingsWrapper(cohere_embeddings)

dataset = Dataset.from_dict(data)
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=LangchainLLMWrapper(llm),
    embeddings=embeddings_wrapper
)

print("\n📊 Evaluation Results:")
if isinstance(result, list):
    for score in result:
        if hasattr(score, 'metric') and hasattr(score, 'value'):
            print(f"  {score.metric}: {score.value:.4f}")
        else:
            print(score)
else:
    print(result)