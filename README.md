# 🔍 RAG Pipeline with Semantic Chunking, Sentence Transformers, ChromaDB & Ollama

A clean, modular Retrieval-Augmented Generation (RAG) pipeline that combines semantic search with local LLM inference. This project demonstrates how to build an efficient and scalable knowledge retrieval system using modern open-source tools.

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that enhances LLM responses with context retrieved from a custom knowledge base.

It leverages:

* **Sentence Transformers** for high-quality embeddings
* **Semantic Chunking** for context-aware document splitting
* **ChromaDB** as a fast vector database
* **Ollama** for running LLMs locally

The result: accurate, context-aware responses without relying on external APIs.

---

## 🧠 Architecture

        |

        |── Document Processing

        |     ├── Text Extraction

        |     ├── Chunking

        |     └── Embeddings

        |     └── Store into Vector Database (ChromaDB)

        |── AI Engine (RAG)

        |     ├── User question

        |     ├── Embeddings

        |     ├── Vector Database (ChromaDB)

        |     ├── Similarity Search

        |     └── LLM Response

        |

        v

     Answer / Summary


## 📦 Tech Stack

| Component       | Tool Used             |
| --------------- | --------------------- |
| Embeddings      | Sentence Transformers |
| Chunking        | Semantic Chunking     |
| Vector Database | ChromaDB              |
| LLM             | Ollama                |
| Language        | Python                |


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/doc-query-engine.git
cd doc-query-engine

# Create virtual environment
python -m venv rag-pipeline
source rag-pipeline/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install Ollama

Follow instructions from: [https://ollama.com/](https://ollama.com/)

```bash
ollama pull llama3
```
