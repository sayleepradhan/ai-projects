"""
Tools available to the Plan-and-Execute agent.

1. retrieve_docs  -- semantic search over the FAISS vector store
2. summarize_text -- condense a long passage into key points
3. web_search     -- live web search via DuckDuckGo
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

import config


# ---------------------------------------------------------------------------
# Shared state: lazy-loaded FAISS retriever
# ---------------------------------------------------------------------------
_faiss_db: FAISS | None = None


def _get_faiss_db() -> FAISS:
    global _faiss_db
    if _faiss_db is None:
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        _faiss_db = FAISS.load_local(
            config.FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _faiss_db


def reset_faiss_db():
    """Reset the cached FAISS DB (useful for testing)."""
    global _faiss_db
    _faiss_db = None


# ---------------------------------------------------------------------------
# Tool 1: Document Retriever
# ---------------------------------------------------------------------------
@tool
def retrieve_docs(query: str) -> str:
    """Search the local document store for passages relevant to the query.
    Use this when you need factual information from the ingested knowledge base.
    Input should be a specific, detailed question."""
    try:
        db = _get_faiss_db()
        docs = db.similarity_search(query, k=config.RETRIEVER_K)
        if not docs:
            return "No relevant documents found."
        texts = [doc.page_content for doc in docs]
        return config.DOC_SEPARATOR.join(texts)
    except Exception as e:
        return f"Retrieval error: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Summarizer
# ---------------------------------------------------------------------------
@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into concise key points.
    Use this after retrieving documents to distill the most important information.
    Input should be the raw text you want summarized."""
    llm = ChatAnthropic(model=config.LLM_MODEL, temperature=0.0)
    prompt = (
        "You are an expert analyst. Summarize the following text into 3-5 concise "
        "bullet points capturing the most important facts and insights.\n\n"
        f"Text:\n{text[:6000]}\n\nSummary:"
    )
    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# Tool 3: Web Search (DuckDuckGo)
# ---------------------------------------------------------------------------
_ddg_search: DuckDuckGoSearchRun | None = None


def _get_ddg_search() -> DuckDuckGoSearchRun:
    global _ddg_search
    if _ddg_search is None:
        _ddg_search = DuckDuckGoSearchRun()
    return _ddg_search


@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic using DuckDuckGo.
    Use this when the local document store does not have enough information,
    or when you need the latest data on a topic.
    Input should be a concise search query."""
    try:
        ddg = _get_ddg_search()
        results = ddg.invoke(query)
        if not results or results.strip() == "":
            return f"No web results found for: '{query}'"
        return results
    except Exception as e:
        return f"Web search error: {e}"


# ---------------------------------------------------------------------------
# All tools list
# ---------------------------------------------------------------------------
ALL_TOOLS = [retrieve_docs, summarize_text, web_search]