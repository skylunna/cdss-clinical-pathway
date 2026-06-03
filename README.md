# CDSS Clinical Pathway

Agent-based Clinical Decision Support System focused on clinical pathway navigation.

## Goals

- Practice LLM application development (Prompt engineering, Agent orchestration, Function calling, RAG)
- Build a complete vertical-slice CDSS for one specialty (initially: Community-Acquired Pneumonia)
- Architecture patterns transferable to other vertical-domain decision support systems (e.g., smart agriculture)

## Tech Stack

- **Python 3.12** with **uv** for environment management
- **FastAPI** for API layer (planned)
- **LangGraph** for agent orchestration (planned)
- **PostgreSQL + pgvector** for data and vector storage (planned)
- **LLM**: DeepSeek / OpenAI / Qwen (configurable)

## Status

🚧 Under active development. Built as a learning exercise.

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment template and fill in your API keys
cp .env.example .env

# (Future) Run the app
uv run python -m cdss.main
```

## Project Structure