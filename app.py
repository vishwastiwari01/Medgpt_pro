"""
MedGPT - Professional Medical RAG Assistant
Enterprise-grade UI with advanced features
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add utils to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

import streamlit as st
import fitz  # PyMuPDF

# Project imports
from llm_handler import LLMHandler
from utils.vector_store import VectorStoreManager

# Configuration
VECTORSTORE_DIR = "vectorstore"
TOP_K_RESULTS = 3

# Page configuration
st.set_page_config(
    page_title="MedGPT - Medical Knowledge Assistant",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/vishwastiwari01/medgpt',
        'Report a bug': 'https://github.com/vishwastiwari01/medgpt/issues',
        'About': '# MedGPT\n\nProfessional medical knowledge assistant powered by RAG and Groq AI.'
    }
)

# Professional CSS Styling
st.markdown("""
<style>
/* Import Professional Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Layout */
.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Header Styling */
.main-header {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    padding: 2rem 0 1rem 0;
    letter-spacing: -1px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Query Card */
.query-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.25rem 1.75rem;
    border-radius: 20px;
    margin: 1.5rem 0;
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    font-weight: 500;
}

/* Response Card */
.response-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #667eea;
    padding: 2rem;
    border-radius: 16px;
    margin: 1.5rem 0;
    line-height: 1.8;
    color: #374151;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

/* Source Badge */
.source-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.5rem 0.5rem 0.5rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
}

.source-badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* PDF Viewer */
.pdf-container {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.pdf-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.25rem 1.5rem;
    font-weight: 600;
    font-size: 1.1rem;
}

.pdf-content {
    max-height: 700px;
    overflow-y: auto;
    padding: 1.5rem;
}

/* Highlight Box */
.highlight-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 5px solid #f59e0b;
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    color: #78350f;
    line-height: 1.8;
    margin: 1.5rem 0;
    font-size: 0.95rem;
}

/* Stats Card */
.stats-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin: 1rem 0;
    border: 1px solid #e5e7eb;
}

.stats-number {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Empty State */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #9ca3af;
    background: white;
    border-radius: 16px;
    border: 2px dashed #e5e7eb;
}

.empty-icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
    opacity: 0.5;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.3s ease;
    border: none;
    padding: 0.75rem 1.5rem;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #e5e7eb 100%);
}

/* Status Indicators */
.status-success {
    background: #d1fae5;
    color: #065f46;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}

.status-warning {
    background: #fef3c7;
    color: #92400e;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}

/* Hide Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 12px;
    height: 12px;
}

::-webkit-scrollbar-track {
    background: #f3f4f6;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    text-align: center;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_source" not in st.session_state:
    st.session_state.current_source = None
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "show_context" not in st.session_state:
    st.session_state.show_context = False

# Initialize components
@st.cache_resource(show_spinner=False)
def init_vectorstore():
    """Initialize and cache vectorstore"""
    vs_manager = VectorStoreManager(VECTORSTORE_DIR)
    if vs_manager.load():
        return vs_manager
    return None

@st.cache_resource(show_spinner=False)
def init_llm():
    """Initialize and cache LLM handler"""
    return LLMHandler()

# PDF Display Function
def display_pdf_page(pdf_path: str, page_num: int, highlight_text: str = None):
    """Display PDF page with optional highlighting"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Validate page number
        page_num = max(0, min(page_num, total_pages - 1))
        page = doc[page_num]
        
        # Add highlights
        if highlight_text:
            search_text = highlight_text[:100].strip()
            if search_text:
                for inst in page.search_for(search_text)[:3]:
                    page.add_highlight_annot(inst)
        
        # Render page
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
        img_bytes = pix.tobytes("png")
        
        st.markdown('<div class="pdf-content">', unsafe_allow_html=True)
        st.image(img_bytes, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page_num > 0:
                if st.button("◀ Previous", use_container_width=True, key="prev_page"):
                    st.session_state.current_page = page_num - 1
                    st.rerun()
        with col2:
            st.markdown(f"<div style='text-align: center; padding-top: 0.5rem; color: #6b7280;'>Page {page_num + 1} of {total_pages}</div>", unsafe_allow_html=True)
        with col3:
            if page_num < total_pages - 1:
                if st.button("Next ▶", use_container_width=True, key="next_page"):
                    st.session_state.current_page = page_num + 1
                    st.rerun()
        
        doc.close()
        
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# Header
st.markdown('<div class="main-header">🩺 MedGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Evidence-based Medical Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown("---")

# Main Layout
left_col, right_col = st.columns([1, 1], gap="large")

# LEFT COLUMN - Chat Interface
with left_col:
    st.subheader("💬 Ask a Medical Question")
    
    # Search input
    query = st.text_area(
        "Question",
        placeholder="e.g., What are the first-line treatments for hypertension?",
        label_visibility="collapsed",
        height=100,
        key="main_query"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_btn = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
    with col2:
        context_btn = st.button("📚 Context", use_container_width=True)
    with col3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    # Button actions
    if context_btn:
        st.session_state.show_context = not st.session_state.show_context
    
    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.current_source = None
        st.session_state.show_context = False
        st.rerun()
    
    # Search execution
    if search_btn and query.strip():
        with st.spinner("🔎 Searching medical knowledge base..."):
            try:
                t0 = time.perf_counter()
                
                # Initialize components
                vs_manager = init_vectorstore()
                if not vs_manager:
                    st.error("❌ Vector store not available")
                    st.stop()
                
                # Retrieve relevant documents
                docs = vs_manager.search(query, k=TOP_K_RESULTS)
                t1 = time.perf_counter()
                
                if not docs:
                    st.warning("No relevant information found")
                    st.stop()
                
                # Combine context
                combined_context = "\n\n".join([
                    f"[Source: {d.metadata.get('source', 'Unknown')} - Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
                    for d in docs
                ])
                
                # Show context if requested
                if st.session_state.show_context:
                    with st.expander("📄 Retrieved Context", expanded=True):
                        for i, d in enumerate(docs):
                            st.markdown(f"**Source {i+1}:** {d.metadata.get('source', 'Unknown')} (Page {d.metadata.get('page', 0) + 1})")
                            st.text_area(f"Content {i+1}", d.page_content, height=150, key=f"ctx_{i}", disabled=True)
                
                # Generate answer
                llm = init_llm()
                with st.spinner("🧠 Generating answer..."):
                    answer = llm.generate_answer(query, combined_context)
                
                t2 = time.perf_counter()
                
                # Store in history
                st.session_state.chat_history.insert(0, {
                    "query": query,
                    "answer": answer,
                    "sources": docs,
                    "retrieval_time": t1 - t0,
                    "generation_time": t2 - t1,
                    "backend": llm.backend,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.session_state.current_source = docs[0] if docs else None
                st.session_state.current_page = docs[0].metadata.get('page', 0) if docs else 0
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Display chat history
    st.markdown("---")
    st.subheader("📋 Conversation History")
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💭</div>
            <h3>No queries yet</h3>
            <p>Ask your first medical question to get started</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, chat in enumerate(st.session_state.chat_history[:10]):
            # Query
            st.markdown(f'<div class="query-card">{chat["query"]}</div>', unsafe_allow_html=True)
            
            # Answer
            st.markdown(f'<div class="response-card">{chat["answer"]}</div>', unsafe_allow_html=True)
            
            # Sources
            if chat.get("sources"):
                st.markdown("**📚 Sources:**")
                cols = st.columns(len(chat["sources"][:3]))
                for i, (col, src) in enumerate(zip(cols, chat["sources"][:3])):
                    with col:
                        name = src.metadata.get("source", "Unknown")
                        page = src.metadata.get("page", 0)
                        if st.button(f"📄 {name}\nPage {page + 1}", key=f"src_{idx}_{i}", use_container_width=True):
                            st.session_state.current_source = src
                            st.session_state.current_page = page
                            st.rerun()
            
            # Metadata
            if chat.get("backend") != "fallback":
                st.caption(f"⏱️ Retrieval: {chat['retrieval_time']:.2f}s | Generation: {chat['generation_time']:.2f}s | {chat['timestamp']}")
            
            st.markdown("---")

# RIGHT COLUMN - Document Viewer
with right_col:
    st.subheader("📖 Document Viewer")
    
    src = st.session_state.current_source
    
    if src:
        name = src.metadata.get("source", "Unknown")
        page = st.session_state.current_page
        file_path = src.metadata.get("file_path", "")
        content = src.page_content or ""
        
        # PDF Header
        st.markdown(f'<div class="pdf-container"><div class="pdf-header">📄 {name}</div></div>', unsafe_allow_html=True)
        
        # Relevant excerpt
        with st.expander("💡 Relevant Excerpt", expanded=True):
            st.markdown(f'<div class="highlight-box">{content}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # PDF viewer
        if file_path and Path(file_path).exists() and file_path.lower().endswith(".pdf"):
            display_pdf_page(file_path, page, content[:120])
        else:
            st.info("PDF preview not available for this source")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📚</div>
            <h3>No Document Selected</h3>
            <p>Click on a source citation to view the document</p>
        </div>
        """, unsafe_allow_html=True)

# SIDEBAR - System Information
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    
    # LLM Status
    llm = init_llm()
    status = llm.get_status()
    
    if status["ready"]:
        st.markdown(f'<div class="status-success">✅ {status["model"]} Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-warning">⚠️ Fallback Mode</div>', unsafe_allow_html=True)
        if status["error"]:
            with st.expander("Error Details"):
                st.error(status["error"])
                st.info("Add GROQ_API_KEY in Streamlit secrets to enable AI")
    
    st.markdown("---")
    
    # Vector Store Stats
    st.markdown("## 📊 Knowledge Base")
    vs_manager = init_vectorstore()
    
    if vs_manager:
        stats = vs_manager.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", f"{stats['total_chunks']:,}")
        with col2:
            st.metric("Dimensions", stats['dimension'])
        st.caption(f"Model: {stats['embedding_model'].split('/')[-1]}")
    else:
        st.error("❌ Knowledge base not loaded")
    
    st.markdown("---")
    
    # About
    with st.expander("ℹ️ About MedGPT"):
        st.markdown("""
        **MedGPT** is a professional medical knowledge assistant powered by:
        
        - 🤖 Groq AI (Llama 3.3 70B)
        - 🔍 Semantic Search (FAISS)
        - 📚 Medical Literature (Harrison's)
        - 🎯 RAG Architecture
        
        **Features:**
        - Evidence-based answers
        - Source citations
        - PDF document viewer
        - Real-time search
        
        ⚠️ **Disclaimer:** Educational purposes only. Not for clinical decisions.
        
        ---
        **Version:** 2.0.0  
        **GitHub:** [vishwastiwari01/medgpt](https://github.com/vishwastiwari01/medgpt)
        """)
    
    st.markdown("---")
    st.caption("© 2024 MedGPT | Built with ❤️ by Vishwas Tiwari")
