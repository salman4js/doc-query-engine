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

Feel free to pull any available model from ollama as per your system configurations.

```bash
ollama pull llama3
```

### How to run project?

After the installation steps are done, paste the required documents in the root directory of the project.

```bash
python3 main.py --store --file {your_file_name}
```
This command executes the document processing steps outlined in the Architecture section above.

```bash
python3 main.py --read
```
This command displays the extracted document chunks along with their corresponding embeddings for better visibility.

```bash
python3 main.py --answer --question "{your_question_here}"
```
This command takes a user’s query, converts it into embeddings, and performs a similarity search over stored chunks in ChromaDB. The most relevant chunks are then retrieved and combined with the original query. This combined context is passed to a local LLM, which generates a streaming response.

```bash
python3 main.py --delete
```
This command removes all stored chunk data from the system.
