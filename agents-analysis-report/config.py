"""Configuration and constants for the analysis report agent."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
LLM_TEMPERATURE = 0.0

# --- Embeddings ---
# HuggingFace local embeddings (no external API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- FAISS ---
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVER_K = 4

# --- API ---
API_HOST = "0.0.0.0"
API_PORT = 8000

# --- Agent ---
MAX_AGENT_STEPS = 8
DOC_SEPARATOR = "\n---\n"