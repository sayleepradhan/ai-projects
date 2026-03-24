"""
ingest.py — Build the vector store from the Bitext Customer Support dataset.

Run once to create the FAISS index:
    python ingest.py
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login
from collections import defaultdict
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Config ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

def load_bitext_dataset():
    """Load the Bitext customer support dataset from HuggingFace."""
    print("Loading Bitext dataset from HuggingFace...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    print(f"Loaded {len(ds)} rows")
    return ds

def build_knowledge_base(ds) -> list[Document]:
    """
    Group dataset entries by intent and build knowledge base documents.

    Instead of treating each row independently (27K tiny documents), we group
    the unique responses by intent to create coherent knowledge articles.
    """
    print("Building knowledge base documents grouped by intent...")

    # Collect unique responses per intent
    intent_responses = defaultdict(set)
    intent_categories = {}

    for row in ds:
        intent = row["intent"]
        response = row["response"].strip()
        category = row["category"]
        intent_responses[intent].add(response)
        intent_categories[intent] = category

    # Build one Document per intent
    documents = []
    for intent, responses in intent_responses.items():
        category = intent_categories[intent]
        intent_display = intent.replace("_", " ").title()
        category_display = category.replace("_", " ").title()

        content = f"Topic: {intent_display}\nCategory: {category_display}\n\n"
        for resp in sorted(responses):
            content += f"- {resp}\n\n"

        documents.append(
            Document(
                page_content=content,
                metadata={"intent": intent, "category": category},
            )
        )

    print(f"Created {len(documents)} intent-based documents")
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    print(f"Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_and_store(chunks: list[Document]):
    """Embed chunks with sentence-transformers and store in FAISS."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    print("Creating FAISS index and embedding chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving FAISS index to {FAISS_INDEX_PATH}/")
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Done! Vector store saved successfully.")

def main():
    ds = load_bitext_dataset()
    documents = build_knowledge_base(ds)
    chunks = chunk_documents(documents)
    embed_and_store(chunks)


if __name__ == "__main__":
    main()