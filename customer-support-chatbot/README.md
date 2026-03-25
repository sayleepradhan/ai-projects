# Customer Support Q&A Chatbot

A domain-specific RAG (Retrieval-Augmented Generation) chatbot that answers customer support queries using a curated knowledge base. Built with LangChain, FAISS, sentence-transformers, and Claude.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (run once)                             │
│                                                                       │
│  Bitext Dataset ──▶ Group by Intent ──▶ Chunk ──▶ Embed ──▶ FAISS     │
│   (27K rows)        (27 documents)     (RecursiveCharacter     Index  │
│                                         TextSplitter)                 │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                           RUNTIME (per query)                         │
│                                                                       │
│                             User Question                             │
│                                   │                                   │
│                                   ▼                                   │
│                      Embed query (all-MiniLM-L6-v2)                   │
│                                   │                                   │
│                                   ▼                                   │
│                   FAISS similarity search (top-5 chunks)              │
│                                   │                                   │
│                                   ▼                                   │
│                  Format prompt (system + context + question)          │
│                                   │                                   │
│                                   ▼                                   │
│                  Claude generates grounded response                   │
│                                   │                                   │
│                                   ▼                                   │
│         Streamlit displays response + source intent tags              │
└───────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| LLM | Claude (Anthropic API) | Strong instruction following, stays grounded in provided context |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, fast on CPU, 384-dim vectors |
| Vector Store | FAISS | Simple, no account needed, runs locally |
| Framework | LangChain | Text splitting, retriever abstraction, vector store integration |
| Dataset | Bitext Customer Support (HuggingFace) | 27K Q&A pairs across 27 intents with linguistic variation |
| UI | Streamlit | Quick to build, built-in chat components |

## Setup

```bash
cd customer-support-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=your-key-here
```

Build the vector store (one-time):

```bash
python ingest.py
```

Run the chatbot:

```bash
streamlit run app.py
```

## Example Queries

- "How do I cancel my order?"
- "What payment methods do you accept?"
- "I want to track my refund"
- "How do I change my shipping address?"
- "I'm having trouble logging into my account"
- "What is your return policy?"

## Key Design Decisions

**Intent-grouped knowledge base** rather than blind chunking. The Bitext dataset has 27K rows, but many share the same response text across different phrasings of the same question. Instead of treating each row as a separate document, responses are grouped by intent into 27 coherent knowledge articles before chunking. This gives the retriever better context to work with.

**Local embeddings** with `all-MiniLM-L6-v2` instead of OpenAI's embedding API. No cost per embedding call, no external dependency, and the model runs fast on CPU. For a dataset of this size, there's no meaningful quality difference.

**FAISS over cloud vector databases** like Pinecone or Weaviate. The dataset produces a few hundred chunks, well within what FAISS handles in-memory. No accounts, no infrastructure, and anyone cloning the repo can rebuild the index with a single command.

**Direct Anthropic API calls** instead of LangChain's `RetrievalQA` chain. This gives full control over the prompt format and makes the retrieval-generation boundary explicit rather than hidden inside a LangChain abstraction.

## Dataset

The [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) contains ~27K instruction-response pairs across 27 customer service intents and 11 categories. Intents include cancel_order, track_refund, check_payment_methods, recover_password, and more. The dataset includes linguistic variation tags for colloquial language, spelling mistakes, and other real-world patterns.

## Project Structure

```
customer-support-chatbot/
├── explore.py             # Dataset exploration
├── ingest.py              # Data loading, chunking, embedding, FAISS creation
├── chain.py               # Retrieval + Claude generation logic
├── app.py                 # Streamlit chat UI
├── evaluate.py            # RAG pipeline evaluation
├── requirements.txt
├── .env                   # API key (not committed)
└── faiss_index/           # Generated FAISS index (not committed)
```
