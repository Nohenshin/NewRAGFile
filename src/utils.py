import time
import sys
import os
import cohere
import streamlit as st
from langchain_core.messages import AIMessage
from langchain_core.agents import AgentStep
from langchain_core.messages import HumanMessage, AIMessage

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from src.data_loading import DataLoader
from src.embeddings.embeddings_factory import EmbeddingsFactory
from src.llms.llm_factory import LLMFactory
from src.chunking import TextSplitter
from src.indexing import Index
from src.rag_agent import ReActRagAgent
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.query_rewriter import QueryRewriter
from src.compressor import ContextCompressor
from src.web_search import WebSearch
from src.citation import CitationGenerator
from langchain_community.embeddings import CohereEmbeddings


# ========== HELPER FUNCTIONS ==========

def response_generator(text: str):
    for char in text:
        yield char
        time.sleep(0.00001)


def load_pdf(pdf_file_path: str):
    loader = DataLoader(file_path=pdf_file_path)
    return loader.get_docs()


def create_embeddings(embedding_provider: str, **kwargs):
    """Create embeddings using factory."""
    embeddings = EmbeddingsFactory.create_embeddings(embedding_provider, **kwargs)
    return embeddings.get_embeddings()


def split_documents(docs):
    """Split documents into parent-child chunks."""
    splitter = TextSplitter()
    child_chunks, parent_map = splitter.split_documents(docs)
    return child_chunks, parent_map


def build_vector_store(embeddings):
    index = Index(embeddings)
    return index.vector_store


def create_hybrid_retriever(vector_store, child_chunks, parent_map):
    """Create a hybrid retriever (BM25 + Dense)."""
    return HybridRetriever(vector_store, child_chunks, parent_map)


def create_rag_agent_executor(llm_provider: str, vector_store, number_of_retrieved_documents: int, **kwargs):
    """Create ReAct agent executor with dynamic LLM."""
    llm_instance = LLMFactory.create_llm(llm_provider, **kwargs)
    llm = llm_instance.get_llm()
    react_rag_agent = ReActRagAgent(llm, vector_store, number_of_retrieved_documents)
    return react_rag_agent.create_react_agent_executor()


def estimate_tokens(text: str):
    co = cohere.Client()
    response = co.tokenize(text=text, model="command-a-03-2025")
    return len(response.tokens)


@st.cache_resource(show_spinner=False)
def process_chunks_with_rate_limit_cohere(_chunks, _vectorstore, batch_size=166, token_limit=90000):
    # (giữ nguyên như cũ)
    total_batches = (len(_chunks) + batch_size - 1) // batch_size
    progress_bar = st.progress(0)
    status_text = st.empty()
    countdown_placeholder = st.empty()
    window_start = time.time()

    for i in range(0, len(_chunks), batch_size):
        current_batch = i // batch_size + 1
        batch = _chunks[i:i + batch_size]
        batch_tokens = sum(estimate_tokens(doc.page_content) for doc in batch)

        status_text.markdown(f"🛠️ Processing Batch `[{current_batch}|{total_batches}]` with `{len(batch)}` docs (~{batch_tokens} tokens)")

        if batch_tokens > token_limit:
            st.warning(f"⚠️ Batch too large (~{batch_tokens} tokens), splitting...")
            for doc in batch:
                doc_tokens = estimate_tokens(doc.page_content)
                st.markdown(f"🛠️ Processing single doc (~{doc_tokens} tokens)")
                _vectorstore.add_documents([doc])
                if doc_tokens > 10000:
                    time.sleep(5)
        else:
            _vectorstore.add_documents(batch)

        progress = min((i + batch_size) / len(_chunks), 1.0)
        progress_bar.progress(progress)

        if i + batch_size < len(_chunks):
            elapsed = time.time() - window_start
            remaining = max(0, int(60 - elapsed))
            for sec in range(remaining, 0, -1):
                countdown_placeholder.info(f"⌛ Waiting **{sec}**s to respect API rate limit")
                time.sleep(1)
            countdown_placeholder.info("🔃 Resuming processing...")
            window_start = time.time()

    status_text.success("✅ Batches processed successfully!")
    return _vectorstore


@st.cache_resource(show_spinner=False)
def add_chunks_to_vector_store_hf_embeddings(_chunks, _vector_store):
    # (giữ nguyên)
    try:
        batch_size = 166
        total_chunks = len(_chunks)
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch = _chunks[start:end]
            _vector_store.add_documents(batch)
            progress = int((end) / total_chunks * 100)
            progress_bar.progress(progress)
            progress_text.markdown(f"Processing chunk: `[{end}/{total_chunks}]`")
        return _vector_store
    except Exception as e:
        st.error(e)


def reset_memory():
    rag_agent_executer = st.session_state.get("rag_agent_executer")
    if rag_agent_executer:
        rag_agent_executer.memory.clear()


# ========== ENHANCED QUERY PROCESSING ==========

def process_query(
    query: str,
    llm_provider: str,
    agent_avatar: str,
    number_of_retrieved_documents: int = 5,
    use_hybrid: bool = True,
    use_reranking: bool = True,
    use_web_search: bool = True,
    **kwargs
):
    """
    Enhanced query processing with:
    - Multi-Query Rewriting
    - Hybrid Search (BM25 + Dense) if enabled
    - Cross-Encoder Reranking if enabled
    - Parent Document expansion
    - Context Compression
    - Web Search Fallback if enabled
    - Citation Generation
    """
    # Validation
    if not st.session_state.vector_store:
        st.error("⚠️ Please upload a document first.")
        return

    # Lazy init agent (only if not using advanced pipeline)
    # We'll use the agent for final answer generation if not using advanced pipeline.
    # For advanced pipeline, we bypass agent and do everything manually.
    # But to keep compatibility, we'll use the agent for final step.
    # However, agent currently uses only retriever. We'll overwrite the retriever with our hybrid one if needed.

    # Step 1: Get LLM
    llm_instance = LLMFactory.create_llm(llm_provider, **kwargs)
    llm = llm_instance.get_llm()

    # Step 2: Query Rewriting (Multi-Query)
    rewriter = QueryRewriter(llm)
    history_msgs = st.session_state.messages[-5:] if st.session_state.messages else []
    history_dict = []
    for msg in history_msgs:
        if hasattr(msg, 'content') and hasattr(msg, 'type'):
            role = "user" if msg.type == "human" else "assistant"
            history_dict.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict) and "role" in msg and "content" in msg:
            history_dict.append(msg)
        else:
            # fallback
            history_dict.append({"role": "user", "content": str(msg)})
    queries = rewriter.rewrite(query, history_dict)

    # Step 3: Retrieval
    all_docs = []
    if use_hybrid and st.session_state.hybrid_retriever:
        # Hybrid Search (BM25 + Dense)
        hybrid_retriever = st.session_state.hybrid_retriever
        for q in queries[:3]:  # limit to 3 queries
            results = hybrid_retriever.hybrid_search(q, top_k=number_of_retrieved_documents * 2)
            all_docs.extend(results)
    else:
        # Fallback to simple Chroma retriever (dense only)
        retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": number_of_retrieved_documents * 2})
        for q in queries[:3]:
            docs = retriever.get_relevant_documents(q)
            for doc in docs:
                all_docs.append({
                    "child_content": doc.page_content,
                    "parent_content": doc.page_content,  # No parent
                    "parent_id": None,
                    "metadata": doc.metadata
                })

    # Deduplicate by parent_id (or child_content)
    seen = set()
    unique_docs = []
    for doc in all_docs:
        pid = doc.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            unique_docs.append(doc)
        elif not pid:
            # Use content as key
            content = doc.get("child_content", "")
            if content and content not in seen:
                seen.add(content)
                unique_docs.append(doc)

    # Step 4: Reranking (Cross-Encoder)
    if use_reranking and len(unique_docs) > 0:
        reranker = CrossEncoderReranker()
        # Use original query for reranking (not the rewritten ones)
        reranked = reranker.rerank(query, unique_docs, top_k=number_of_retrieved_documents)
    else:
        # Just take top k
        reranked = unique_docs[:number_of_retrieved_documents]

    # Step 5: Check if we need web search fallback
    need_web = False
    if use_web_search:
        if not reranked or reranked[0].get("rerank_score", 0) < 0.3:
            need_web = True

    if need_web:
        web_search = WebSearch(api_key=kwargs.get("tavily_api_key", ""))  # You may add Tavily key in UI
        web_results = web_search.search(query, max_results=3)
        for res in web_results:
            reranked.append({
                "parent_content": res["content"],
                "child_content": res["content"],
                "parent_id": f"web_{res['url']}",
                "metadata": {"source": res["title"], "url": res["url"]},
                "rerank_score": 1.0  # Boost
            })

    # Step 6: Context Compression
    compressor = ContextCompressor(llm, max_tokens=2000)
    compressed_context = compressor.compress(query, reranked)

    # Step 7: Generate answer with citations
    citations = CitationGenerator.generate_citations(reranked)

    # Build final prompt
    final_prompt = f"""
You are a helpful assistant answering questions based on the provided context.

Context:
{compressed_context}

Question: {query}

Answer the question using the context above. Cite sources using [1], [2], etc.
If the context doesn't contain the answer, say so clearly.
"""
    # Generate response
    response = llm.invoke(final_prompt)
    answer = response.content

    # Add citations at the end
    if citations:
        answer += "\n\n**Sources:** " + " | ".join(citations)

    # Display answer with streaming effect
    with st.chat_message("assistant", avatar=agent_avatar):
        st.write_stream(response_generator(answer))

    st.session_state.messages.append(AIMessage(content=answer))