import os
import tempfile
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

# local
from utils import (
    load_pdf,
    split_documents,
    build_vector_store,
    process_query,
    add_chunks_to_vector_store_hf_embeddings,
    reset_memory,
    create_hybrid_retriever,
)

# Page configuration
st.set_page_config(
    page_title="RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS - đảm bảo đường dẫn chính xác (nếu file cùng thư mục với app.py)
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # Nếu không tìm thấy, bỏ qua (có thể để trống)
    pass

st.title("🤖 RAG Agent")
st.markdown("Ask questions pdf and get intelligent responses with citations and hybrid search.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "rendered_pages" not in st.session_state:
    st.session_state.rendered_pages = None
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False
if "rag_agent_executer" not in st.session_state:
    st.session_state.rag_agent_executer = None
if "docs" not in st.session_state:
    st.session_state.docs = []

# Avatars
mohammed_avatar = "👤"
agent_avatar = "🤖"

# ======================= SIDEBAR =======================
with st.sidebar:
    st.header("⚙️ 1. Configuration")

    # --- LLM Selection ---
    st.subheader("🧠 Select Language Model")
    llm_type = st.selectbox(
        "Choose LLM:",
        ["cohere (Command-R)"],
        help="Select which LLM to use for answering questions."
    )
    llm_provider = "cohere"

    # --- Embedding Selection (chỉ giữ lại HuggingFace để tránh lỗi) ---
    st.subheader("📐 Select Embedding Model")
    embeddings_type = st.selectbox(
        "Choose Embedding:",
        ["sentence-transformers/all-mpnet-base-v2"],  # Chỉ dùng HuggingFace
        help="Using local HuggingFace embedding (free, no API key needed)"
    )

    # --- API Keys ---
    st.subheader("🔑 API Keys")
    cohere_api_key = None

    if "cohere" in llm_type:
        cohere_api_key = st.text_input("Cohere API Key", type="password", placeholder="Enter Cohere key...")
        if cohere_api_key:
            os.environ["COHERE_API_KEY"] = cohere_api_key

    # --- Advanced Options ---
    st.subheader("⚙️ Advanced Settings")
    st.session_state.show_reasoning = st.toggle(
        "Show Reasoning Steps",
        st.session_state.show_reasoning,
        help="Display agent's intermediate reasoning and tool calls."
    )

    use_hybrid_search = st.toggle(
        "Use Hybrid Search (BM25 + Dense)",
        value=True,
        help="Combine keyword and semantic search for better retrieval."
    )

    use_reranking = st.toggle(
        "Use Cross-Encoder Reranking",
        value=True,
        help="Rerank retrieved documents with a cross-encoder for higher precision."
    )

    use_web_search = st.toggle(
        "Web Search Fallback",
        value=True,
        help="If no relevant documents found, search the web."
    )

    # Save settings to session state
    st.session_state.llm_provider = llm_provider
    st.session_state.embeddings_type = embeddings_type
    st.session_state.use_hybrid_search = use_hybrid_search
    st.session_state.use_reranking = use_reranking
    st.session_state.use_web_search = use_web_search

    # --- Document Upload ---
    st.subheader("📄 2. Document Upload")
    uploaded_file = st.file_uploader("Upload Your PDF", type="pdf")

    if uploaded_file and (st.session_state.uploaded_file is None or uploaded_file.name != st.session_state.uploaded_file.name):
        with st.spinner("Processing PDF..."):
            st.session_state.uploaded_file = uploaded_file

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            docs = load_pdf(pdf_path)
            st.session_state.docs = docs

            # ===== SPLIT DOCUMENTS =====
            with st.spinner("Splitting PDF..."):
                chunks, parent_map = split_documents(docs)
                st.success(f"✅ Split into {len(chunks)} child chunks, {len(parent_map)} parent chunks")

            st.markdown(f"📄 Found `{len(docs)}` pages → `{len(chunks)}` child chunks, `{len(parent_map)}` parent chunks")

            # ===== CREATE HUGGINGFACE EMBEDDINGS (NO API KEY) =====
            with st.spinner("Creating HuggingFace embeddings..."):
                hf_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2"
                )
                st.success("✅ Embeddings created!")

            # ===== BUILD VECTOR STORE =====
            with st.spinner("Building vector store..."):
                vector_store = build_vector_store(hf_embeddings)

            # ===== ADD CHUNKS TO VECTOR STORE =====
            with st.spinner("Adding chunks to vector store..."):
                vector_store = add_chunks_to_vector_store_hf_embeddings(chunks, vector_store)

            st.session_state.vector_store = vector_store

            # ===== BUILD HYBRID RETRIEVER =====
            with st.spinner("Building hybrid retriever..."):
                st.session_state.hybrid_retriever = create_hybrid_retriever(
                    vector_store, chunks, parent_map
                )
            st.success("✅ Hybrid retriever built!")

            os.unlink(pdf_path)

    # PDF viewer settings
    if st.session_state.docs:
        number_of_rendered_page = st.slider(
            "PDF Preview Pages",
            min_value=10,
            max_value=min(len(st.session_state.docs), 100),
            value=30,
        )
        st.session_state.rendered_pages = number_of_rendered_page

    # Status
    if st.session_state.vector_store is not None:
        st.sidebar.success("✅ Document loaded and indexed!")
    else:
        st.sidebar.warning("⚠️ No document loaded")

    # Clear conversation
    st.subheader("🗑️ 3. Clear Chat History")
    if st.button("Clear Conversation"):
        if len(st.session_state.messages):
            st.session_state.messages = []
            reset_memory()
            st.success("✅ Conversation cleared!")
        else:
            st.info("No messages to clear.")

# ======================= MAIN CONTENT =======================
# Check requirements
if not cohere_api_key:
    st.warning("⚠️ Please enter your Cohere API key in the sidebar.")
elif not st.session_state.vector_store:
    st.warning("⚠️ Please upload the AI Index Report 2025 PDF in the sidebar.")
else:
    # Two columns: PDF preview and chat
    pdf_preview, chat_area = st.columns(2)

    with pdf_preview:
        if st.session_state.uploaded_file:
            with st.container(border=True):
                binary_data = st.session_state.uploaded_file.getvalue()
                pdf_viewer(
                    input=binary_data,
                    height=600,
                    pages_to_render=[*range(st.session_state.rendered_pages)],
                    resolution_boost=2,
                    pages_vertical_spacing=1,
                    render_text=True,
                )

    with chat_area:
        chat_container = st.container(height=585, border=False)

        with chat_container:
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user", avatar=mohammed_avatar):
                        st.markdown(message.content)
                else:
                    with st.chat_message("assistant", avatar=agent_avatar):
                        st.markdown(message.content)

        query = st.chat_input("Ask a question about the AI Index Report 2025...")

        if query:
            with chat_container:
                with st.chat_message("user", avatar=mohammed_avatar):
                    st.markdown(query)
                st.session_state.messages.append(HumanMessage(content=query))

                # Prepare kwargs for process_query
                kwargs = {"cohere_api_key": cohere_api_key}

                # Call enhanced process_query
                process_query(
                    query=query,
                    llm_provider=st.session_state.llm_provider,
                    agent_avatar=agent_avatar,
                    use_hybrid=st.session_state.use_hybrid_search,
                    use_reranking=st.session_state.use_reranking,
                    use_web_search=st.session_state.use_web_search,
                    **kwargs
                )