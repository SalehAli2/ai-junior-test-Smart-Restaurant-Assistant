# NovaBite Smart Restaurant Assistant
### Multi-Agent RAG System built with LangGraph & LangChain

---

## 🏗️ Architecture Overview

NovaBite is a production-style multi-agent AI assistant for a restaurant chain. It uses a **LangGraph state graph** to orchestrate three specialized agents, each handling a distinct domain.

```
User Input
    ↓
Orchestrator Agent (Intent Classification)
    ↓
┌───────────────┬──────────────────┬─────────────┐
│   RAG Agent   │ Operations Agent │   Fallback  │
│ (Knowledge)   │   (Tools/MCP)    │             │
└───────────────┴──────────────────┴─────────────┘
    ↓
Assembler Node (Response + Memory Update)
    ↓
User Response
```

---

## 📁 Project Structure

```
novabite/
├── streamlit_app.py        # Streamlit chat UI
├── agents.py               # All three agent nodes
├── tools.py                # Simulated MCP tools
├── graph.py                # LangGraph graph
├── rag.py                  # Ingestion, chunking, retrieval
├── state.py                # NovaState
├── .env                    # API keys
├── requirements.txt
└── data/
    ├── menu.txt            # Italian restaurant menu
    └── loyalty.txt         # Loyalty program rules
```

---

## 🤖 Component Breakdown

### 1. Main Orchestrator Agent

The orchestrator classifies every user message into one of three intents:

| Intent | Routes To | Examples |
|---|---|---|
| `rag` | RAG Agent | Menu questions, allergens, loyalty rules |
| `operations` | Operations Agent | Reservations, availability, specials |
| `fallback` | Fallback Node | Greetings, off-topic questions |

**Implementation:** Uses `llm.with_structured_output()` with a Pydantic model (`IntentClassification`) that returns intent, confidence score, and clarification flag. Routes to fallback if confidence < 0.5.

```python
class IntentClassification(BaseModel):
    intent: Literal["rag", "operations", "fallback"]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
```

The orchestrator receives full `chat_history` for context — enabling short follow-up messages like "yes please" or "ok" to route correctly based on conversation context.

---

### 2. RAG Agent (Restaurant Knowledge)

Handles all knowledge-based questions from two domains:
- **Menu** — items, descriptions, allergens, dietary options, prices
- **Loyalty Program** — earning rules, tiers, redemption, birthday rewards

**RAG Pipeline:**

```
data/menu.txt + data/loyalty.txt
        ↓
DocumentLoader (txt/pdf/docx support)
        ↓
RecursiveCharacterTextSplitter
        ↓
OllamaEmbeddings (nomic-embed-text)
        ↓
FAISS Vector Store
        ↓
similarity_search_with_score 
        ↓
Score Threshold Filter
        ↓
LLM (Grounded Answer Generation)
```

#### Chunking Strategy
- **`RecursiveCharacterTextSplitter`** with `chunk_size=400`, `chunk_overlap=100`
- Separators: `["\n\n", "\n", ". ", " ", ""]`
- Section headers (`== PASTA ==`) act as natural split boundaries
- Each menu item stays within its own chunk to prevent allergen/description separation
- **Justification:** Menu items are short, structured entries (~150-250 chars). Small chunks ensure each item embeds as a single semantic unit rather than mixing multiple items.

#### Embedding Model
- **`nomic-embed-text`** via Ollama (local)
- **Justification:** Strong semantic similarity for food/hospitality domain, runs fully locally with no API cost, optimized for retrieval tasks.

#### Vector Database
- **FAISS** with L2 distance scoring
- Lazy loading pattern — vector store initializes on first query, not at import time
- Persisted to disk at `vector_store/`
- **Important:** FAISS uses L2 distance (lower = better), not cosine similarity. Threshold set to `1.2` accordingly.

#### Retrieval Strategy
- `k=4` top results
- Score threshold filter: `score <= 1.2`
- Returns `(context, sources)` tuple for transparency

#### Hallucination Prevention
- RAG agent uses **context-only prompt** — strictly forbidden from using outside knowledge
- Returns refusal message if retrieved context doesn't support the question
- `temperature=0` for deterministic, grounded responses

---

### 3. Operations Agent (Tool-Based / MCP-Style)

Handles live operational queries using a ReACT agent

#### Tools Implemented

| Tool | Parameters | Description |
|---|---|---|
| `check_table_availability` | `date, time, branch` | If/else logic — peak hours (19:00, 19:30, 20:00) always fully booked, returns confirmed alternative times |
| `book_table` | `name, date, time, branch, party_size` | Generates unique reservation ID, returns booking confirmation |
| `get_today_special` | `branch` | Branch-keyed dict of daily specials |
| `check_loyalty_points` | `user_id` | Returns points balance and tier (Silver/Gold/Platinum) |

#### Booking Protocol (Enforced via Prompt)
```
Step 1: call check_table_availability FIRST
Step 2: if available → immediately call book_table
Step 3: if unavailable → suggest confirmed alternative times
```

#### REDIRECT_TO_RAG Signal
If the user asks a knowledge question mid-operations flow (e.g. "does the pasta have gluten?"), the operations agent returns `REDIRECT_TO_RAG` — the graph re-routes to the RAG agent without returning to the orchestrator.

#### Today's Date Injection
```python
operations_prompt = f"Today's date is {date.today().strftime('%Y-%m-%d')}."
```
Prevents date hallucination when users say "this Friday" or "tomorrow".

---

## 🧠 Memory Design

| Level | Implementation | Scope |
|---|---|---|
| Within-session | `MemorySaver` checkpointer | Persists `chat_history` across turns via `thread_id` |

**Key design decision:** `streamlit_app.py` pass only the new turn's fields on each invoke — `chat_history` is never overwritten by the caller. `MemorySaver` restores it automatically from the `thread_id` checkpoint.

```python
# Only new turn fields passed — MemorySaver handles the rest
state = {
    "user_question": user_input,
    "user_intent": "",
    "rag_context": "",
    "rag_answer": "",
    "tool_output": "",
    "final_response": ""
}
```

---

## 💬 Example Queries and Outputs
<img width="1573" height="762" alt="WhatsApp Image 2026-05-01 at 4 24 59 PM (1)" src="https://github.com/user-attachments/assets/e4e058b8-d346-4106-a49a-5e1e7f23b071" />
<img width="1559" height="337" alt="image" src="https://github.com/user-attachments/assets/ae9f38ca-bd3a-4f07-b02d-8512e54b867e" />


### RAG — Menu
```
User: Do you have any vegan pasta?
Nova: Yes, we have the Spicy Tomato Penne — penne pasta in a spicy tomato
      and garlic sauce. It's vegan and contains gluten. Priced at $18.00.
```

### RAG — Allergens
```
User: What allergens are in the Spaghetti Carbonara?
Nova: The Spaghetti Carbonara contains gluten, dairy, and eggs.
```

### RAG — Loyalty
```
User: How do I earn loyalty points?
Nova: You earn 1 point for every 1 USD spent. Points are credited within
      24 hours. Double points every Wednesday for dine-in orders.
```

### Operations — Booking Flow
```
User: Book a table for John this Friday at 7pm at the Downtown branch
Nova: How many people will be joining?
User: 4
Nova: The Downtown branch is fully booked at 19:00. Confirmed available
      times: 18:00, 21:00, 21:30. Which would you prefer?
User: 21:00
Nova: Booking Confirmed! Reservation ID: NB-2026-1B8E
      Guest: John | Date: 2026-05-09 | Time: 21:00 | Party: 4
```

### Operations — Today's Special
```
User: What is today's special at Downtown?
Nova: Today's special at NovaBite Downtown: Osso Buco alla Milanese
      with saffron risotto.
```

### Fallback
```
User: Who is the president?
Nova: I'm Nova, NovaBite's assistant. I can help you with:
      - Table reservations
      - Menu and allergen information
      - Loyalty points and today's specials
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Ollama installed with `nomic-embed-text` pulled
- Groq API key or GitHub Models token

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
```

### Pull Embedding Model
```bash
ollama pull nomic-embed-text
```

### Run

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

---

## 🧩 Assumptions Made

- Table availability uses deterministic if/else logic (peak hours always busy) — simulates real reservation system behavior without a database
- Loyalty points are stored in a hardcoded dict — simulates a user database lookup
- Today's specials rotate by branch — simulates a daily specials API
- Two knowledge domains selected: **menu** and **loyalty program** (as permitted by spec)
- Memory is in-memory only — resets on server restart
- FAISS L2 distance threshold set to `1.2` — calibrated by inspecting actual score distributions, not assumed

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (llama-3.3-70b) / GitHub Models (gpt-4o-mini) |
| Embeddings | nomic-embed-text (Ollama) |
| Vector Store | FAISS |
| Memory | LangGraph MemorySaver |
| UI | Streamlit |
| Framework | LangChain |
