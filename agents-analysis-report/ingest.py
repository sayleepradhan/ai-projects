"""
Ingest documents into a FAISS vector store.

Usage:
    python ingest.py --urls urls.txt
    python ingest.py --texts "path/to/textfiles/"
"""

import argparse
import os
import time

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def scrape_articles(url_list: list[str]) -> list[dict]:
    """Scrape article text from a list of URLs."""
    from newspaper import Article

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    import requests

    session = requests.Session()
    pages = []

    for url in url_list:
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        try:
            time.sleep(1)
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                article = Article(url)
                article.set_html(response.text)  # reuse already-fetched HTML
                article.parse()
                if article.text.strip():
                    pages.append({"url": url, "text": article.text})
                    print(f"  [OK] {url[:80]}")
                else:
                    print(f"  [EMPTY] {url[:80]}")
            else:
                print(f"  [HTTP {response.status_code}] {url[:80]}")
        except Exception as e:
            print(f"  [ERROR] {url[:80]} -- {e}")

    return pages


def chunk_documents(pages: list[dict]) -> list[str]:
    """Split scraped pages into smaller text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    all_chunks = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        all_chunks.extend(chunks)
    return all_chunks


def build_faiss_index(chunks: list[str], save_path: str) -> FAISS:
    """Create a FAISS index from text chunks and persist to disk."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(save_path)
    print(f"\nFAISS index saved to {save_path} ({len(chunks)} chunks)")
    return db


def load_faiss_index(path: str) -> FAISS:
    """Load an existing FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS")
    parser.add_argument("--urls", type=str, help="Path to a text file with one URL per line")
    parser.add_argument("--texts", type=str, help="Path to a directory of .txt files")
    args = parser.parse_args()

    all_chunks = []

    if args.urls:
        print(f"Scraping articles from {args.urls}...")
        with open(args.urls) as f:
            urls = [line.strip() for line in f if line.strip()]
        pages = scrape_articles(urls)
        all_chunks.extend(chunk_documents(pages))

    if args.texts:
        print(f"Reading text files from {args.texts}...")
        for fname in sorted(os.listdir(args.texts)):
            if fname.endswith(".txt"):
                with open(os.path.join(args.texts, fname)) as f:
                    text = f.read()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                )
                all_chunks.extend(splitter.split_text(text))
                print(f"  [OK] {fname}")

    if not all_chunks:
        print("No documents found. Provide --urls or --texts.")
        return

    os.makedirs(config.FAISS_INDEX_PATH, exist_ok=True)
    build_faiss_index(all_chunks, config.FAISS_INDEX_PATH)


if __name__ == "__main__":
    main()