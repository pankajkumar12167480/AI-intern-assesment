# Cognitive Routing & RAG

Implementation of the Grid07 assignment tasks.

## Setup
1. `pip install -r requirements.txt`
2. Add your HF token to `.env` (copy `.env.example`)
3. Run `python main.py`

## Structure
- `phase1_router.py`: Vector search using FAISS and HuggingFace embeddings.
- `phase2_langgraph.py`: LangGraph state machine.
- `phase3_rag_defense.py`: RAG implementation and prompt injection defense.

## Injection Defense
Used a system prompt with explicit boundaries to separate the human reply from the system rules.
