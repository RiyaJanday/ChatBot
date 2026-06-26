"""
PDF Chatbot — Full UI Version
--------------------------------
Upload a PDF and chat with it like ChatGPT/Claude/Gemini. Includes:
- Split layout: PDF preview + chat
- Source citations under each answer
- Clickable suggested follow-up questions
- Book info card (title/author/genre once identified)
- Streaming responses
- Custom theme

Run with:
    streamlit run app.py
"""

import os
import re
import tempfile
import base64

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

st.set_page_config(page_title="Chat with your PDF", page_icon="📚", layout="wide")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

FOLLOW_UP_SUGGESTIONS = [
    "Give me a chapter-by-chapter summary",
    "Tell me about the main characters",
    "What are the major themes?",
]

CUSTOM_CSS = """
<style>
.book-card {
    background: linear-gradient(135deg, #1f2430 0%, #2a3142 100%);
    border: 1px solid #3a4256;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.book-card h4 { margin: 0 0 4px 0; color: #f5f5f5; }
.book-card p { margin: 2px 0; color: #b8c0cc; font-size: 0.9em; }
.source-box {
    background: #1a1d24;
    border-left: 3px solid #6c63ff;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
    font-size: 0.85em;
    color: #c5cad3;
}
div.stButton > button {
    border-radius: 20px;
    border: 1px solid #6c63ff;
    background: transparent;
    color: #6c63ff;
    font-size: 0.85em;
    padding: 4px 14px;
}
div.stButton > button:hover {
    background: #6c63ff;
    color: white;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def process_pdf(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = uploaded_file.name
    finally:
        os.remove(tmp_path)

    if not docs:
        return None, "", None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    preview = docs[0].page_content[:1500] if docs else ""
    return vectorstore, preview, file_bytes


import streamlit.components.v1 as components


def render_pdf_preview(file_bytes: bytes, filename: str = "document.pdf"):
    st.download_button(
        "⬇️ Download / open PDF",
        data=file_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )
    st.caption(
        "Inline preview not supported in this browser — use the button above to view the PDF, or read the excerpt below."
    )
    st.text_area(
        "First page preview", st.session_state.doc_preview, height=300, disabled=True
    )


def extract_book_info(text: str) -> dict:
    info = {}
    title_match = re.search(
        r"(?:title|book)[:\-]?\s*\*?\*?([^\n*]{2,80})", text, re.IGNORECASE
    )
    author_match = re.search(
        r"author[:\-]?\s*\*?\*?([^\n*]{2,60})", text, re.IGNORECASE
    )
    genre_match = re.search(r"genre[:\-]?\s*\*?\*?([^\n*]{2,80})", text, re.IGNORECASE)
    if title_match:
        info["title"] = title_match.group(1).strip()
    if author_match:
        info["author"] = author_match.group(1).strip()
    if genre_match:
        info["genre"] = genre_match.group(1).strip()
    return info


def render_book_card():
    info = st.session_state.get("book_info")
    if not info:
        return
    st.markdown(
        f"""
        <div class="book-card">
            <h4>📖 {info.get('title', 'Unknown title')}</h4>
            <p><b>Author:</b> {info.get('author', 'Unknown')}</p>
            <p><b>Genre:</b> {info.get('genre', 'Not identified yet')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def retrieve_sources(vectorstore: FAISS, question: str, k: int = TOP_K):
    return vectorstore.similarity_search(question, k=k)


def build_system_prompt(pdf_context: str, doc_preview: str) -> str:
    return f"""You are a knowledgeable, confident assistant discussing a PDF/book
the user uploaded — exactly like ChatGPT, Claude, or Gemini would when asked about a book.

Step 1: Try to identify the book's title and author using the PDF excerpts and your own
general knowledge. If you recognize it, say so directly and confidently (e.g. "This is
'Icebreaker' by Hannah Grace...").

Step 2: Give a clear, well-organized answer. For "what is this book about" type questions,
structure your response with headers/bullets covering: Title & Author, Plot overview
(spoiler-light unless asked for full spoilers), Main characters, Genre, Tone/themes, and
intended audience if relevant.

Do NOT hedge with phrases like "based on the excerpts it seems" if you actually recognize
the book — state it directly and confidently. Only add uncertainty if you genuinely don't
recognize the book.

End your answer naturally; do not list follow-up suggestions yourself, the UI already
offers those as buttons.

PDF excerpts relevant to this question:
{pdf_context}

Brief preview of the document:
{doc_preview}
"""


def stream_answer(client: Groq, messages, placeholder):
    full_text = ""
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=1536,
        temperature=0.5,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        placeholder.markdown(full_text + "▌")
    placeholder.markdown(full_text)
    return full_text


def render_sources(sources):
    with st.expander("📖 Sources"):
        for i, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page")
            label = f"Excerpt {i}" + (f" (page {page + 1})" if page is not None else "")
            st.markdown(f"**{label}**")
            st.markdown(
                f'<div class="source-box">{doc.page_content[:400]}...</div>',
                unsafe_allow_html=True,
            )


def render_followups(key_prefix: str):
    cols = st.columns(len(FOLLOW_UP_SUGGESTIONS))
    clicked = None
    for col, suggestion in zip(cols, FOLLOW_UP_SUGGESTIONS):
        with col:
            if st.button(suggestion, key=f"{key_prefix}_{suggestion}"):
                clicked = suggestion
    return clicked


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    api_key = st.secrets["GROQ_API_KEY"]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "doc_preview" not in st.session_state:
        st.session_state.doc_preview = ""
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "book_info" not in st.session_state:
        st.session_state.book_info = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    with st.sidebar:
        st.header("📁 Upload your PDF")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file and st.session_state.get("current_file") != uploaded_file.name:
            with st.spinner("Reading and indexing your PDF..."):
                vectorstore, preview, pdf_bytes = process_pdf(uploaded_file)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_preview = preview
                st.session_state.pdf_bytes = pdf_bytes
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                st.session_state.book_info = None
                st.success(f"Ready! Loaded '{uploaded_file.name}'")
            else:
                st.error("Couldn't extract any text from this PDF.")

        if st.session_state.get("current_file"):
            st.info(f"📌 {st.session_state['current_file']}")

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.book_info = None
            st.rerun()

    st.title("📚 Chat with your PDF")
    render_book_card()

    if st.session_state.pdf_bytes:
        col_pdf, col_chat = st.columns([1, 1.4])
    else:
        col_pdf, col_chat = None, st.container()

    if col_pdf is not None:
        with col_pdf:
            st.subheader("📄 Document preview")
            render_pdf_preview(
                st.session_state.pdf_bytes, st.session_state.current_file
            )

    with col_chat:
        st.subheader("💬 Chat")

        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "assistant"
        ):
            clicked = render_followups(
                key_prefix=f"followup_{len(st.session_state.messages)}"
            )
            if clicked:
                st.session_state.pending_question = clicked

        question = st.chat_input("Ask anything about your PDF...")
        if question:
            st.session_state.pending_question = question

        if st.session_state.pending_question:
            if st.session_state.vectorstore is None:
                st.error("Please upload a PDF first.")
                st.stop()

            q = st.session_state.pending_question
            st.session_state.pending_question = None

            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            with st.chat_message("assistant"):
                sources = retrieve_sources(st.session_state.vectorstore, q)
                pdf_context = "\n\n".join(
                    f"[Excerpt {i+1}]\n{doc.page_content}"
                    for i, doc in enumerate(sources)
                )
                system_prompt = build_system_prompt(
                    pdf_context, st.session_state.doc_preview
                )

                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(st.session_state.messages[-7:-1])
                messages.append({"role": "user", "content": q})

                placeholder = st.empty()
                try:
                    client = Groq(api_key=api_key)
                    answer = stream_answer(client, messages, placeholder)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    placeholder.markdown(answer)

                render_sources(sources)

                if not st.session_state.book_info:
                    info = extract_book_info(answer)
                    if info:
                        st.session_state.book_info = info

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.rerun()


if __name__ == "__main__":
    main()
