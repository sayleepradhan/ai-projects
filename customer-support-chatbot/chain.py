"""
chain.py — Retrieval and generation logic using FAISS + Claude.
"""

import os
import anthropic
from anthropic.types import MessageParam
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5

# --- Prompt template ---
SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant. \
You answer customer questions accurately using ONLY the context provided below. \
If the context does not contain enough information to answer the question, say so \
honestly rather than making something up.

Be concise, clear, and helpful. If the customer's question maps to a specific \
process (like cancelling an order or tracking a refund), walk them through the \
steps mentioned in the context."""

USER_PROMPT_TEMPLATE = """Here is the relevant knowledge base context:

{context}

---

Customer question: {question}

Please provide a helpful response based on the context above."""

class CustomerSupportChain:
    """RAG chain: FAISS retrieval + Claude generation."""

    def __init__(self):
        # Load the same embedding model used during ingestion
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        # Load the FAISS index from disk
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        # Create a retriever from the vector store
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve the top-k most relevant chunks for a query."""
        docs = self.retriever.invoke(query)
        return [
            {
                "content": doc.page_content,
                "intent": doc.metadata.get("intent", "unknown"),
                "category": doc.metadata.get("category", "unknown"),
            }
            for doc in docs
        ]

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        """Send the retrieved context + question to Claude."""
        context = "\n\n".join(chunk["content"] for chunk in context_chunks)
        message = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                MessageParam(
                    role="user",
                    content=USER_PROMPT_TEMPLATE.format(
                        context=context, question=question
                    ),
                )
            ],
        )
        return message.content[0].text

    def ask(self, question: str) -> dict:
        """Full RAG pipeline: retrieve context, then generate a response."""
        chunks = self.retrieve(question)
        response = self.generate(question, chunks)
        return {
            "question": question,
            "response": response,
            "sources": chunks,
        }

if __name__ == "__main__":
    chain = CustomerSupportChain()
    test_queries = [
        "How do I cancel my order?",
        "What payment methods do you accept?",
        "I want to track my refund",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = chain.ask(q)
        print(f"A: {result['response']}")
        print(f"Sources: {[c['intent'] for c in result['sources']]}")