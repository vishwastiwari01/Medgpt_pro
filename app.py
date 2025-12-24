"""
MedGPT - Professional Medical RAG Assistant
Now powered by Llama 3.1 70B via OpenRouter
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st
import fitz  # PyMuPDF

# ---------- Paths / imports ----------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from llm_handler import LLMHandler

# If you have your own VectorStore helper, keep this import.
# Expected interface:
#   VectorStoreManager(root_dir).load() -> True/False
#   .search(query, k) -> list[Document] with .page_content and .metadata dict
try:
    from utils.vector_store import VectorStoreManager
except Exception:
    VectorStoreManager = None  # App will show an error in the sidebar

# ---------- Config ----------
VECTORSTORE_DIR = "vectorstore"
TOP_K_RESULTS = 3

st.set_page_config(
    page_title="MedGPT - Medical Knowledge Assistant",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://openrouter.ai/",
        "About": "# MedGPT\n\nProfessional medical assistant powered by Llama 3.1 70B (OpenRouter).",
    },
)

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #f7f9fc 0%, #e6ecf7 100%); }
.main-header {
  font-size: 3.2rem; font-weight: 800;
  background: linear-gradient(135deg, #5b86e5 0%, #36d1dc 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  text-align: center; margin: 0.75rem 0 0.25rem 0; padding: 1.5rem 0 0.75rem 0;
}
.subtitle { text-align: center; color: #6b7280; margin-bottom: 1.25rem; }
.card { background: white; border: 1px solid #e5e7eb; border-radius: 16px;
  box-shadow: 0 4px 10px rgba(0,0,0,0.05); padding: 1rem 1.25rem; }
.query-card { background: linear-gradient(135deg,#5b86e5 0%,#36d1dc 100%);
  color: white; padding: 0.9rem 1.1rem; border-radius: 14px; margin: 0.9rem 0;
  box-shadow: 0 10px 25px rgba(91,134,229,0.25); font-weight: 500; white-space: pre-wrap;}
.response-card { background: white; border: 1px solid #e5e7eb; border-left: 5px solid #5b86e5;
  padding: 1.2rem; border-radius: 14px; margin: 0.9rem 0; line-height: 1.8; color: #374151; }
.status-success { background: #d1fae5; color: #065f46; padding: 0.5rem 0.8rem;
  border-radius: 8px; font-weight: 600; display: inline-block; }
.status-warning { background: #fef3c7; color: #92400e; padding: 0.5rem 0.8rem;
  border-radius: 8px; font-weight: 600; display: inline-block; }
.pdf-header { font-weight: 700; margin-bottom: 0.5rem; }
.highlight-box { background: #f9fafb; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 0.75rem; }
.small-muted { color: #6b7280; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("chat_history", [])
ss.setdefault("current_source", None)
ss.setdefault("current_page", 0)
ss.setdefault("show_context", False)
ss.setdefault("zoom", 150)               # percent
ss.setdefault("rotation", 0)             # 0, 90, 180, 270
ss.setdefault("search_term", "")         # for PDF highlight

# ---------- Search roots for PDFs ----------
PDF_SEARCH_ROOTS = [
    Path.cwd(),
    ROOT,
    ROOT / "data",
    ROOT / "MedGPT",
    ROOT.parent,  # one level up
]

def _norm(p: str) -> Path:
    """Normalize any path (Windows/Unix), expanduser."""
    try:
        return Path(str(p).strip().replace("\\", "/")).expanduser()
    except Exception:
        return Path("")

def resolve_pdf_path(meta: dict) -> str:
    """
    Best-effort lookup of the actual PDF file from vector metadata.
    Tries keys: file_path, source, path. Falls back to basename search in common roots.
    Returns "" if not found.
    """
    if not isinstance(meta, dict):
        return ""

    # 1) Candidates from metadata
    candidates = []
    for k in ("file_path", "source", "path"):
        v = meta.get(k)
        if v:
            candidates.append(v)

    # 2) Direct existence check
    for c in candidates:
        p = _norm(c)
        if p.is_file():
            return str(p)

    # 3) Try appending ".pdf" if missing
    for c in candidates:
        p = _norm(c)
        if p.suffix.lower() != ".pdf" and p.name:
            base = p.stem + ".pdf"
            for root in PDF_SEARCH_ROOTS:
                test = (root / base)
                if test.is_file():
                    return str(test)

    # 4) Basename search across roots
    basename = None
    for c in candidates:
        name = _norm(c).name
        if name:
            basename = name
            break
    if basename:
        if not basename.lower().endswith(".pdf"):
            basename = Path(basename).stem + ".pdf"
        for root in PDF_SEARCH_ROOTS:
            try:
                hit = next(root.rglob(basename), None)
                if hit and hit.is_file():
                    return str(hit)
            except Exception:
                pass

    return ""

# ---------- Caches ----------
@st.cache_resource(show_spinner=False)
def init_vectorstore():
    if VectorStoreManager is None:
        return None
    vs_manager = VectorStoreManager(VECTORSTORE_DIR)
    return vs_manager if vs_manager.load() else None

@st.cache_resource(show_spinner=False)
def init_llm():
    return LLMHandler()

# ---------- PDF utilities ----------
def _render_pdf_page(doc: fitz.Document, page_idx: int, zoom_pct: int, rotation: int,
                     highlight_text: Optional[str] = None):
    """Render a single PDF page with optional highlight + zoom + rotation."""
    total_pages = len(doc)
    page_idx = max(0, min(page_idx, total_pages - 1))
    page = doc[page_idx]

    rotate = rotation % 360
    mat = fitz.Matrix(zoom_pct / 100.0, zoom_pct / 100.0)

    # ✅ Handle PyMuPDF API differences (preRotate → prerotate)
    if hasattr(mat, "prerotate"):
        mat = mat.prerotate(rotate)
    elif hasattr(mat, "preRotate"):
        mat = mat.preRotate(rotate)
    else:
        # fallback (no rotation support)
        pass

    # Highlight matches (first few)
    if highlight_text:
        txt = highlight_text.strip()
        if txt:
            matches = page.search_for(txt)[:3]
            for inst in matches:
                try:
                    page.add_highlight_annot(inst)
                except Exception:
                    pass

    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix, total_pages, page_idx


def display_pdf_viewer(pdf_path: str, page_num: int, zoom_pct: int, rotation: int, highlight_text: str = ""):
    try:
        with fitz.open(pdf_path) as doc:
            pix, total_pages, page_num = _render_pdf_page(
                doc, page_num, zoom_pct, rotation, highlight_text
            )
            st.image(pix.tobytes("png"), use_container_width=True)
            c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 1, 2])
            with c1:
                if page_num > 0 and st.button("◀ Prev", key="prev_pg", use_container_width=True):
                    ss.current_page = page_num - 1
                    st.rerun()
            with c2:
                new_page = st.number_input(
                    "Page", min_value=1, max_value=total_pages, value=page_num + 1, key="page_jump"
                )
                if new_page - 1 != page_num:
                    ss.current_page = new_page - 1
                    st.rerun()
            with c3:
                st.markdown(f"<div class='small-muted' style='text-align:center;'>/ {total_pages}</div>", unsafe_allow_html=True)
            with c4:
                if page_num < total_pages - 1 and st.button("Next ▶", key="next_pg", use_container_width=True):
                    ss.current_page = page_num + 1
                    st.rerun()
            with c5:
                st.caption(f"Zoom: {zoom_pct}% | Rotation: {rotation}°")
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")

# ---------- Header ----------
st.markdown('<div class="main-header">🩺 MedGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Evidence-based Medical Assistant • Llama 3.1 70B (OpenRouter)</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------- Layout ----------
left_col, right_col = st.columns([1, 1], gap="large")

# ===== LEFT: Chat =====
with left_col:
    st.subheader("💬 Ask a Medical Question")
    query = st.text_area(
        "Question",
        placeholder="e.g., First-line therapy for Stage-1 hypertension with diabetes?",
        label_visibility="collapsed",
        height=110,
        key="main_query",
    )

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        search_btn = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
    with c2:
        context_btn = st.button("📚 Context", use_container_width=True)
    with c3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if context_btn:
        ss.show_context = not ss.show_context
    if clear_btn:
        ss.chat_history = []
        ss.current_source = None
        ss.show_context = False
        st.rerun()

    if search_btn and query.strip():
        with st.spinner("🔎 Searching knowledge base..."):
            try:
                t0 = time.perf_counter()
                vs_manager = init_vectorstore()
                if not vs_manager:
                    st.error("❌ Vector store not available or failed to load.")
                    st.stop()

                docs = vs_manager.search(query, k=TOP_K_RESULTS)
                t1 = time.perf_counter()
                if not docs:
                    st.warning("No relevant information found.")
                    st.stop()

                # Ensure each doc has a resolvable PDF path
                for d in docs:
                    fp = d.metadata.get("file_path", "")
                    if not fp or not Path(fp).is_file():
                        resolved = resolve_pdf_path(d.metadata)
                        if resolved:
                            d.metadata["file_path"] = resolved

                combined_context = "\n\n".join([
                    f"[Source: {d.metadata.get('source','Unknown')} - Page {d.metadata.get('page',0)+1}]\n{d.page_content}"
                    for d in docs
                ])

                if ss.show_context:
                    with st.expander("📄 Retrieved Context", expanded=True):
                        for i, d in enumerate(docs):
                            st.markdown(f"**Source {i+1}:** {d.metadata.get('source','Unknown')} (Page {d.metadata.get('page',0)+1})")
                            st.text_area(f"Content {i+1}", d.page_content, height=140, key=f"ctx_{i}", disabled=True)

                llm = init_llm()
                temperature = st.session_state.get("ui_temperature", 0.3)
                top_p = st.session_state.get("ui_top_p", 0.9)

                placeholder = st.empty()
                content_buf = []

                if llm.backend != "fallback":
                    with st.spinner("🧠 Generating (streaming)..."):
                        for token in llm.stream_answer(query, combined_context, temperature=temperature, top_p=top_p):
                            content_buf.append(token)
                            placeholder.markdown(f'<div class="response-card">{"".join(content_buf)}</div>', unsafe_allow_html=True)
                    answer = "".join(content_buf).strip()
                else:
                    with st.spinner("🧠 Generating..."):
                        answer = llm.generate_answer(query, combined_context, temperature=temperature, top_p=top_p)
                    placeholder.markdown(f'<div class="response-card">{answer}</div>', unsafe_allow_html=True)

                t2 = time.perf_counter()
                ss.chat_history.insert(0, {
                    "query": query,
                    "answer": answer,
                    "sources": docs,
                    "retrieval_time": t1 - t0,
                    "generation_time": t2 - t1,
                    "backend": llm.backend,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                # Focus right panel on the first source
                ss.current_source = docs[0] if docs else None
                ss.current_page = docs[0].metadata.get('page', 0) if docs else 0
                ss.search_term = (docs[0].page_content or "")[:120]
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")

    st.markdown("---")
    st.subheader("📋 Conversation History")
    if not ss.chat_history:
        st.info("💭 No queries yet — ask your first medical question to begin.")
    else:
        for idx, chat in enumerate(ss.chat_history[:10]):
            st.markdown(f'<div class="query-card">{chat["query"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="response-card">{chat["answer"]}</div>', unsafe_allow_html=True)
            if chat.get("sources"):
                st.markdown("**📚 Sources:**")
                cols = st.columns(len(chat["sources"][:3]) or 1)
                for i, (col, src) in enumerate(zip(cols, chat["sources"][:3])):
                    with col:
                        name = src.metadata.get("source", "Unknown")
                        page = src.metadata.get("page", 0)
                        if st.button(f"📄 {name}\nPage {page+1}", key=f"src_{idx}_{i}", use_container_width=True):
                            ss.current_source = src
                            ss.current_page = page
                            ss.search_term = (src.page_content or "")[:120]
                            st.rerun()
            if chat.get("backend") != "fallback":
                st.caption(f"⏱️ Retrieval: {chat['retrieval_time']:.2f}s | Generation: {chat['generation_time']:.2f}s | {chat['timestamp']}")
            st.markdown("---")

# ===== RIGHT: Document Viewer =====
with right_col:
    st.subheader("📖 Document Viewer")
    src = ss.current_source
    if src:
        name = src.metadata.get("source", "Unknown")
        page = ss.current_page
        file_path = src.metadata.get("file_path", "") or resolve_pdf_path(src.metadata)
        content = src.page_content or ""
        st.markdown(f'<div class="pdf-header">📄 {name}</div>', unsafe_allow_html=True)
        if file_path:
            st.caption(f"Resolved file: {file_path}")

        with st.expander("💡 Relevant Excerpt", expanded=True):
            st.markdown(f'<div class="highlight-box">{content}</div>', unsafe_allow_html=True)

        # Viewer controls
        cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 2])
        with cc1:
            ss.zoom = st.slider("Zoom (%)", 50, 300, ss.zoom, step=10)
        with cc2:
            ss.rotation = st.select_slider("Rotate", options=[0, 90, 180, 270], value=ss.rotation)
        with cc3:
            ss.search_term = st.text_input("Highlight text", value=ss.search_term, placeholder="text to highlight")
        with cc4:
            if file_path and Path(file_path).exists():
                with open(file_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name=Path(file_path).name, mime="application/pdf", use_container_width=True)

        st.markdown("---")
        if file_path and Path(file_path).exists() and file_path.lower().endswith(".pdf"):
            display_pdf_viewer(file_path, page, ss.zoom, ss.rotation, ss.search_term)
        else:
            st.info("PDF preview not available for this source.")
    else:
        st.info("📚 No document selected. Click a source to view it.")

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    llm = init_llm()
    status = llm.get_status()
    if status["ready"]:
        st.markdown(f'<div class="status-success">✅ {status["model"]} Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-warning">⚠️ Fallback Mode</div>', unsafe_allow_html=True)
        if status["error"]:
            with st.expander("Error Details"):
                st.error(status["error"])
                st.info("Add OPENROUTER_API_KEY in `.env` to enable AI responses.")

    st.markdown("---")
    st.markdown("## 🎛️ Generation Controls")
    st.session_state.ui_temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get("ui_temperature", 0.3), 0.05)
    st.session_state.ui_top_p = st.slider("Top-p", 0.1, 1.0, st.session_state.get("ui_top_p", 0.9), 0.05)

    st.markdown("---")
    st.markdown("## 📊 Knowledge Base")
    vs_manager = init_vectorstore()
    if vs_manager:
        try:
            stats = vs_manager.get_stats()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Chunks", f"{stats.get('total_chunks', 0):,}")
            with c2:
                st.metric("Dimensions", stats.get("dimension", "—"))
            st.caption(f"Model: {str(stats.get('embedding_model', '—')).split('/')[-1]}")
        except Exception:
            st.info("Vector store loaded.")
    else:
        st.error("❌ Knowledge base not loaded")

    st.markdown("---")
    with st.expander("ℹ️ About MedGPT"):
        st.markdown("""
**MedGPT** is an intelligent medical assistant powered by:
- 🧠 Llama 3.1 70B (OpenRouter)
- 🔍 FAISS-based semantic search
- 📚 Evidence-focused retrieval

**Features**
- Context-aware answers with citations
- Smooth PDF viewer (zoom, rotate, jump)
- Real-time streaming responses

⚠️ *For educational use only — not for clinical decision-making.*

---
**Version:** 3.0.1
        """)
    st.caption("© 2025 MedGPT")
