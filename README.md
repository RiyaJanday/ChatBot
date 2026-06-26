# 📄 Chat with your PDF — RAG-Based Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask it questions naturally — like chatting with ChatGPT, Claude, or Gemini, but grounded in your own document.

**Live Demo:** [https://chatbot19.streamlit.app/](https://chatbot19.streamlit.app/)

## Overview

This project demonstrates a full RAG pipeline: documents are chunked, embedded, and indexed for semantic search, and a large language model generates conversational, context-aware answers using the most relevant retrieved chunks — while still being able to draw on its own general knowledge (e.g., for summaries, author background, or related context not explicitly written in the document).

## Features

- 📄 Upload any PDF and start chatting immediately
- ✂️ Automatic text chunking with overlap for coherent context
- 🔍 Semantic search over document chunks using FAISS + sentence-transformer embeddings
- 🤖 Natural, conversational answers powered by Groq's fast LLaMA models
- 🧠 Falls back to general knowledge when the document doesn't explicitly cover something (e.g., book summaries, author info)
- 💬 Persistent chat history with conversational memory across turns
- 🔐 API key stored securely via Streamlit Secrets — never exposed to users
- ⚡ Free to run — no paid API required

## Tech Stack

| Component | Technology |
|---|---|
| UI / Frontend | Streamlit |
| Document Loading & Chunking | LangChain |
| Vector Store / Semantic Search | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) — runs locally, free |
| LLM (Answer Generation) | Groq API (LLaMA 3.1 / 3.3 models) — free tier, very fast inference |
| Deployment | Streamlit Community Cloud |

## How It Works

1. **Upload** — User uploads a PDF through the sidebar.
2. **Load & Chunk** — `PyPDFLoader` extracts text; `RecursiveCharacterTextSplitter` breaks it into ~1000-character overlapping chunks.
3. **Embed & Index** — Each chunk is embedded locally using a sentence-transformer model and stored in a FAISS vector index.
4. **Ask a Question** — User types a question in the chat box.
5. **Retrieve** — The question is embedded and the top-k most semantically similar chunks are retrieved from FAISS.
6. **Generate** — Retrieved chunks + recent conversation history are sent to the Groq LLM, which produces a natural, well-structured answer — using the document as the primary source but supplementing with general knowledge when appropriate.
7. **Respond** — The answer streams back into the chat interface.

## Project Structure

```
ChatBot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── secrets.toml        # API key (not committed to GitHub)
```

## Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/RiyaJanday/ChatBot.git
   cd ChatBot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Groq API key**

   Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key_here"
   ```
   Get a free key at [console.groq.com/keys](https://console.groq.com/keys)

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`), upload a PDF, and start chatting.

## Deployment

This app is deployed on **Streamlit Community Cloud**:

1. Push the repo to GitHub (`.streamlit/secrets.toml` is gitignored and never pushed).
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io).
3. Set `app.py` as the entry point.
4. Add the `GROQ_API_KEY` secret via the app's **Settings → Secrets** tab on Streamlit Cloud.
5. Deploy — the app is now publicly accessible with no setup required from end users.

## Notes

- Embeddings run locally and are completely free; only the final answer generation calls the Groq API (also free tier).
- Chat history and the FAISS index are stored in Streamlit's session state, so they reset when the app restarts or the session ends.
- To persist an index across sessions, it can be saved with `vectorstore.save_local(...)` and reloaded with `FAISS.load_local(...)`.

## Future Improvements

- Multi-document support (chat across several PDFs at once)
- Source citation display (show which page/chunk an answer came from)
- Web search integration for fully up-to-date external information
- Support for additional file types (DOCX, TXT, EPUB)
