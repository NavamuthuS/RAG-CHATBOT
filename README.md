<<<<<<< HEAD
# 🤖 RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot built with LangChain, ChromaDB, and Streamlit.
Upload your own PDF, TXT, or DOCX files and ask questions — the bot answers strictly from your documents, no hallucinations.

---

## 🗂️ Project Structure

```
rag-chatbot/
├── app.py                  # Streamlit chat UI — entry point
├── config.py               # Central settings (loaded from .env)
├── requirements.txt        # All pip dependencies
├── .env.example            # Template for your API key
├── .gitignore              # Files excluded from git
├── README.md               # This file
├── src/
│   ├── __init__.py         # Makes src/ a Python package
│   ├── document_loader.py  # Loads PDF / TXT / DOCX files
│   ├── text_splitter.py    # Splits docs into overlapping chunks
│   ├── vector_store.py     # Builds and loads ChromaDB vector store
│   ├── prompts.py          # RAG prompt template (no hallucination)
│   └── rag_chain.py        # Wires retriever + LLM into a pipeline
├── data/
│   └── documents/          # ← Drop your files here
└── tests/
    └── test_rag.py         # Smoke tests (pytest)
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your real OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Get your key from: https://platform.openai.com/api-keys

---

## 📄 Adding Documents

Drop your files into the `data/documents/` folder:

```
data/
└── documents/
    ├── report.pdf
    ├── notes.txt
    └── manual.docx
```

Supported formats: `.pdf`, `.txt`, `.docx`

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

### First time setup:
1. Add your documents to `data/documents/`
2. Click **🔨 Build / Rebuild Knowledge Base** in the sidebar
3. Wait for embedding to complete
4. Start asking questions in the chat!

---

## 🧪 Running Tests

```bash
pytest tests/test_rag.py -v
```

---

## 🔧 Configuration

All settings live in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `DATA_DIR` | `data/documents` | Folder to scan for documents |
| `CHROMA_DB_DIR` | `chroma_db` | Where ChromaDB saves its index |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `TEMPERATURE` | `0.3` | LLM randomness (0 = deterministic) |
| `TOP_K` | `4` | Number of chunks retrieved per query |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Framework | LangChain |
| Vector DB | ChromaDB (local, persisted) |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| UI | Streamlit |
| Document Loaders | PyPDF, TextLoader, Docx2txt |

---

## 📝 How It Works

1. **Load** — Documents are scanned and loaded using LangChain loaders
2. **Split** — Text is split into overlapping chunks for better retrieval
3. **Embed** — Each chunk is converted to a vector using OpenAI embeddings
4. **Store** — Vectors are saved locally in ChromaDB
5. **Retrieve** — On each question, the top-K most relevant chunks are fetched
6. **Answer** — The LLM answers using only the retrieved context

---

## ⚠️ Notes

- The chatbot answers **only** from your documents — it will say "I don't know" if the answer isn't there
- Run **Build / Rebuild Knowledge Base** every time you add or change documents
- Your `chroma_db/` folder is auto-generated and excluded from git
=======
# RAG-CHATBOT
>>>>>>> 619e345341b6a689ca1654758e23db58bbd6680a
