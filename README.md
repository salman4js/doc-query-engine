# RAG Pipeline with Semantic Chunking, Sentence Transformers, ChromaDB, and Ollama

A clean and modular Retrieval-Augmented Generation (RAG) pipeline that combines semantic search with local large language model (LLM) inference. This project demonstrates how to build an efficient and scalable knowledge retrieval system using modern open-source tools.

---

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enhances LLM responses by incorporating context retrieved from a custom knowledge base.

It uses:

* Sentence Transformers for generating high-quality embeddings
* Semantic chunking for context-aware document segmentation
* ChromaDB as a vector database for efficient similarity search
* Ollama for running LLMs locally

The result is a system capable of producing accurate and context-aware responses without relying on external APIs.

---

## Architecture

```
Document Processing
│
├── Text Extraction
├── Chunking
├── Embedding Generation
└── Storage in Vector Database (ChromaDB)

AI Engine (RAG)
│
├── User Query
├── Query Embedding
├── Vector Database Lookup (ChromaDB)
├── Similarity Search
└── LLM Response Generation

Output
│
└── Answer / Summary
```

---

## Tech Stack

| Component            | Tool Used             |
| -------------------- | --------------------- |
| Embeddings           | Sentence Transformers |
| Chunking             | Semantic Chunking     |
| Vector Database      | ChromaDB              |
| LLM                  | Ollama                |
| Programming Language | Python                |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/salman4js/doc-query-engine.git
cd doc-query-engine
```

### Create a Virtual Environment

```bash
python -m venv rag-pipeline
source rag-pipeline/bin/activate  # On Windows: rag-pipeline\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Follow the official installation instructions:

https://ollama.com/

After installation, pull a model suitable for your system:

```bash
ollama pull llama3
```

---

## Usage

### Step 1: Add Documents

Place the documents you want to process in the root directory of the project.

---

### Step 2: Store Document Embeddings

```bash
python3 main.py --store --file {your_file_name}
```

This command performs document processing, including text extraction, semantic chunking, embedding generation, and storage in ChromaDB.

---

### Step 3: Inspect Stored Data

```bash
python3 main.py --read
```

Displays extracted chunks along with their corresponding embeddings for inspection.

---

### Step 4: Start the LLM Server

Ensure Ollama is running:

```bash
ollama serve
```

---

### Step 5: Query the System

```bash
python3 main.py --answer --model_name "{your_installed_or_preferred_custom_model_name}" --question "{your_question_here}"
```

This command:

1. Converts the user query into an embedding
2. Performs similarity search on stored chunks
3. Retrieves the most relevant context
4. Passes the context and query to the LLM
5. Generates a response (streamed output)
6. Validates the streamed response using a semantic consistency checker to ensure the answer is derived strictly from the uploaded document 

#### Semantic Consistency Scoring
The semantic consistency checker assigns a score between 0.0 and 1.0:
* < 0.7 → Weak alignment or partial inconsistency with the source document
* ≥ 0.7 → Acceptable response with good alignment to the source document
* Close to 1.0 → Highly consistent and strongly grounded response

---

### Step 6: Delete Stored Data

```bash
python3 main.py --delete
```

Removes all stored embeddings and chunks from the vector database.

---

## Suppressing Warning Logs

To disable warning messages from Sentence Transformers and related libraries:

```bash
python -W ignore main.py --answer --question "{your_question_here}" 2>/dev/null
```

---

## Notes

* Ensure sufficient system resources when running local LLMs
* Choose an Ollama model that matches your hardware capabilities
* Semantic chunking improves retrieval quality compared to fixed-size chunking
* This project is designed for local, privacy-focused deployments

---
