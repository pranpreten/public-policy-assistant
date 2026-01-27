# Public Policy QA System

A RAG-based LLM question-answering system for Korean public policy documents.

## Overview

This system answers questions about Korean government policies (fine dust regulations, heatwave guidelines, etc.) using Retrieval-Augmented Generation (RAG). Policy questions are answered based on retrieved documents with source citations, while general questions are answered directly by the LLM.

## Tech Stack

- LLM: Qwen2.5-14B-Instruct (vLLM)
- Vector DB: ChromaDB
- Embedding: jhgan/ko-sroberta-multitask (Korean SBERT)
- Web Framework: Flask

## Project Structure

```
public-policy-assistant/
├── app.py                  # Flask web server
├── run.py                  # CLI/Web runner script
├── requirements.txt        # Dependencies
├── src/
│   ├── rag_pipeline.py     # RAG pipeline (classify, retrieve, generate)
│   ├── llm_model.py        # vLLM model wrapper
│   ├── vector_store.py     # ChromaDB vector store
│   └── document_processor.py
├── data/
│   ├── raw/                # Raw policy documents (JSON)
│   └── processed/          # Chunked documents
├── vectordb/               # ChromaDB storage
├── models/                 # Local LLM weights (not in git)
├── templates/
│   └── index.html          # Chat UI
└── scripts/
    └── collect_data.py     # Data collection script
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the LLM model:
```bash
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir ./models/Qwen2.5-14B-Instruct
```

3. Build vector database (if needed):
```bash
python run.py --mode setup
```

## Usage

Web server:
```bash
python app.py
# or
python run.py --mode web
```

CLI mode:
```bash
python run.py --mode cli
```

Server runs at http://0.0.0.0:7865

## Data Sources

- Fine Dust Emergency Reduction Guidelines (Ministry of Environment)
- Industrial Heatwave Response Guidelines (Ministry of Employment and Labor)
- School Fine Dust Response Manual (Ministry of Education)
- Related laws and regulations

## How It Works

1. User submits a question
2. System classifies the question (policy vs general)
3. For policy questions:
   - Retrieve relevant documents from vector DB
   - Generate answer based on retrieved context
   - Return answer with source citations
4. For general questions:
   - LLM generates answer directly
