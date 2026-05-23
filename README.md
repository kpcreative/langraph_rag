# LangGraph RAG with Adaptive Retrieval

A **Retrieval-Augmented Generation (RAG)** pipeline built using **LangGraph**, **LangChain**, **Groq (LLaMA 3.3-70b)**, and **ChromaDB** — with an adaptive/self-corrective flow that grades retrieved documents for relevance before generating an answer.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by:
1. **Retrieving** relevant documents from a knowledge base (vector store)
2. **Augmenting** the LLM prompt with that retrieved context
3. **Generating** a grounded, factual answer based on the context

This avoids hallucinations and keeps answers tied to real source documents.

---

## Architecture / Workflow

```
User Query
    │
    ▼
[AI Assistant] ──► LLM decides: use retriever tool or answer directly?
    │                                          │
    │ (tool call)                              │ (no tool call)
    ▼                                          ▼
[Retriever] ──► ChromaDB Vector Store         END (direct answer)
    │
    ▼
[Grade Document] ◄── LLM grades if retrieved docs are relevant
    │
    ├── Relevant ──► [Generator] ──► Final Answer ──► END
    │
    └── Not Relevant ──► [Rewriter] ──► Rewrites query ──► back to [AI Assistant]
```

The graph is built with **LangGraph's StateGraph** which orchestrates the conditional flow between nodes using edges and conditional edges.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Orchestration | LangGraph (`StateGraph`) |
| Document Loading | LangChain `WebBaseLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` (tiktoken, chunk_size=100) |
| Prompts | `ChatPromptTemplate` (inline) |

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

### 3. Create Retriever Tool
A retriever tool is created using `create_retriever_tool` so the LLM can decide when to invoke it via tool calling.

### 4. LangGraph Nodes

| Node | Purpose |
|---|---|
| **ai_assistant** | LLM with tools bound — decides whether to call the retriever or answer directly |
| **retriever** | `ToolNode` that executes the retriever tool and fetches documents from ChromaDB |
| **grade_document** | Condition function — LLM grades retrieved docs as relevant (`yes`) or not (`no`) using structured output (Pydantic) |
| **generator** | Uses retrieved context + `ChatPromptTemplate` to generate the final answer |
| **rewriter** | Rewrites the user's question for better retrieval on the next attempt |

### 5. Edges & Conditional Routing

| From | Condition | To |
|---|---|---|
| START | — | ai_assistant |
| ai_assistant | `tools_condition`: tool call made | retriever |
| ai_assistant | `tools_condition`: no tool call | END |
| retriever | `grade_document`: relevant | generator |
| retriever | `grade_document`: not relevant | rewriter |
| generator | — | END |
| rewriter | — | ai_assistant (loop) |

### 6. Invoke

```python
app.invoke({"messages": [HumanMessage(content="What is prompt engineering")]})
```

---

## Known Limitation & Next Steps: Handling Irrelevant Questions

**Problem:** When a question is completely unrelated to the documents (e.g., "What is capital of India"), the flow loops:  
`ai_assistant → retriever → grade (not relevant) → rewriter → ai_assistant → retriever → ...` (infinite loop / no reply)

**Solution: Add a `no_answer` node with retry tracking**

1. Add `retry_count: int` to `AgentState`
2. In `rewrite`, increment the counter: `"retry_count": state.get("retry_count", 0) + 1`
3. In `grade_document`, if `retry_count >= 1` and still not relevant → route to `"no_answer"` instead of `"rewriter"`
4. Add a `no_answer` node that returns a friendly "I don't have relevant info" message
5. Update the workflow:

```python
workflow.add_node("no_answer", no_answer)

workflow.add_conditional_edges("retriever", grade_document, {
    "generator": "generator",
    "rewriter": "rewriter",
    "no_answer": "no_answer"
})

workflow.add_edge("no_answer", END)
```

**Updated flow with `no_answer`:**

```
[Grade Document]
    ├── Relevant ──► [Generator] ──► END
    ├── Not Relevant (1st time) ──► [Rewriter] ──► [AI Assistant] ──► retry
    └── Not Relevant (after rewrite) ──► [No Answer] ──► END
```

This ensures the user always gets a response, even for out-of-scope questions.

---

## Setup

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd langraph

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys

Set your Groq API key as an environment variable:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "your_groq_api_key_here"

# Linux/Mac
export GROQ_API_KEY=your_groq_api_key_here
```

### Run

Open `rag.ipynb` in VS Code or Jupyter and run all cells sequentially.

---

## Project Structure

```
langraph/
├── rag.ipynb          # Main RAG notebook with LangGraph pipeline
├── lang_p.ipynb       # Experimentation / prototype notebook
├── requirements.txt   # All Python dependencies
├── .gitignore         # Excludes .venv, __pycache__ etc.
└── README.md          # This file
```

---

## Key Concepts Demonstrated

- **Agentic RAG** — LLM decides whether to use retriever tool or answer directly
- **Tool Calling** — LLM bound with tools, `tools_condition` routes based on tool usage
- **Document Grading** — Structured output with Pydantic (`grade` model) to classify doc relevance
- **Self-Corrective Flow** — Rewriter rewrites failed queries and retries retrieval
- **LangGraph StateGraph** — Typed state (`TypedDict` + `add_messages`) passed between nodes
- **HuggingFace Local Embeddings** — No OpenAI dependency for embeddings
- **Conditional Edges** — Dynamic routing based on LLM decisions

---

## License

MIT
