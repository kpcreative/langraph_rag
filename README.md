# LangGraph RAG with Adaptive Retrieval

A **Retrieval-Augmented Generation (RAG)** pipeline built using **LangGraph**, **LangChain**, **Groq (LLaMA 3.1)**, and **ChromaDB** — with an adaptive/self-corrective flow that decides whether retrieved documents are relevant before generating an answer.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by:
1. **Retrieving** relevant documents from a knowledge base (vector store)
2. **Augmenting** the LLM prompt with that retrieved context
3. **Generating** a grounded, factual answer based on the context

This avoids hallucinations and keeps answers tied to real source documents.

---

## Architecture

```
User Query
    │
    ▼
[Retrieve] ──► Vector Store (ChromaDB + HuggingFace Embeddings)
    │
    ▼
[Grade Documents] ◄── LLM grades if retrieved docs are relevant
    │
    ├── Relevant ──► [Generate Answer] ──► Output
    │
    └── Not Relevant ──► [Web Search / Rewrite Query] ──► [Retrieve again]
```

The graph is built with **LangGraph** which orchestrates the conditional flow between nodes.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Orchestration | LangGraph |
| Document Loading | LangChain `WebBaseLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` (tiktoken) |
| Prompt Hub | LangChain Hub |

---

## Data Sources

Web pages loaded as knowledge base:
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — Lilian Weng
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — Lilian Weng

---

## How It Works (Step by Step)

### 1. Load & Split Documents
Web pages are loaded using `WebBaseLoader` and split into chunks of ~100 tokens with 5-token overlap using `RecursiveCharacterTextSplitter`.

### 2. Embed & Store
Chunks are embedded using `all-MiniLM-L6-v2` (a lightweight sentence-transformer) and stored in a **ChromaDB** collection named `langraph-rag`.

### 3. LangGraph Flow
The RAG graph has these nodes:

- **retrieve** — fetches top-k similar chunks from ChromaDB for the query
- **grade_documents** — uses LLaMA to score each document as `relevant` or `not relevant`
- **generate** — uses graded context to produce the final answer
- **rewrite_query** *(if needed)* — rewrites the query if docs are not relevant
- **web_search** *(if needed)* — falls back to web search for out-of-scope queries

### 4. Conditional Edges
LangGraph's conditional edges route the flow:
- All docs relevant → generate
- Some/all irrelevant → rewrite query → retrieve again

---

## Setup

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/kpcreative/langraph_rag.git
cd langraph_rag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your actual keys
GROQ_API_KEY=your_groq_api_key_here
```

### Run

Open `rag.ipynb` in Jupyter or VS Code and run all cells.

---

## Project Structure

```
langraph_rag/
├── rag.ipynb          # Main RAG notebook with LangGraph pipeline
├── lang_p.ipynb       # Experimentation / prototype notebook
├── requirements.txt   # All Python dependencies (pinned)
├── .env.example       # Template for environment variables
├── .gitignore         # Excludes .env, .venv, __pycache__ etc.
└── README.md          # This file
```

---

## Key Concepts Demonstrated

- **Agentic RAG** — LLM decides whether to trust retrieved docs or retry
- **Document Grading** — structured output with Pydantic to classify doc relevance
- **LangGraph State Machine** — typed state (`TypedDict`) passed between nodes
- **HuggingFace local embeddings** — no OpenAI dependency for embeddings

---

## License

MIT
