# Intelligent Document Search & Summarization System

An AI-powered document search and summarization application that retrieves relevant information from a specialized knowledge base and generates context-based answers with source references.

The project combines traditional keyword search, semantic search, and a locally running Large Language Model (LLM) to make document retrieval more effective and answers easier to verify.

## Overview

Searching a large collection of documents using keywords alone can miss useful information when a query uses different wording from the source. Semantic search helps identify related concepts, while keyword search is useful for matching exact technical terms.

This project combines both approaches using **hybrid retrieval with Reciprocal Rank Fusion (RRF)**. The retrieved text is then passed to a local LLM through Ollama to generate an answer based on the available evidence.

The application includes a Streamlit interface where users can enter queries, choose a response length, view retrieved evidence, and inspect source links.

## Key Features

- **Knowledge-base search:** Searches a collection of AI and data science articles.
- **BM25 keyword retrieval:** Finds documents using lexical matching.
- **Semantic search:** Uses sentence embeddings to identify contextually relevant passages.
- **Hybrid retrieval:** Combines keyword and semantic rankings using Reciprocal Rank Fusion.
- **AI-powered summarization:** Uses the locally running Llama 3 model through Ollama.
- **Source references:** Displays source titles, links, and retrieved evidence.
- **Adjustable response length:** Supports short, medium, and detailed answers.
- **Interactive interface:** Provides a Streamlit application with query suggestions and paginated retrieval results.
- **Evaluation scripts:** Measures retrieval quality and compares generated summaries against reference summaries.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Keyword retrieval | BM25, `rank-bm25` |
| Text embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| Language model | Llama 3 |
| Local model runtime | Ollama |
| Retrieval evaluation | Hit@K, Mean Reciprocal Rank |
| Summary evaluation | ROUGE-1, ROUGE-2, ROUGE-L |

## System Workflow

```text
Wikipedia Knowledge Articles
            |
            v
      Text Preprocessing
            |
            v
       Text Chunking
            |
            v
    +-------+--------+
    |                |
    v                v
 BM25 Index      ChromaDB Index
    |                |
    v                v
Keyword Search  Semantic Search
    |                |
    +-------+--------+
            |
            v
 Reciprocal Rank Fusion
            |
            v
   Relevant Text Chunks
            |
            v
    Ollama - Llama 3
            |
            v
  Generated Answer + Sources
            |
            v
     Streamlit Interface
```

## Knowledge Base

The knowledge base focuses on Artificial Intelligence and Data Science topics, including:

- Artificial Intelligence and Machine Learning
- Deep Learning
- Natural Language Processing
- Large Language Models
- Generative AI
- Retrieval-Augmented Generation
- Transformers and Prompt Engineering
- Semantic Search and Vector Databases
- Computer Vision
- Feature Engineering and Data Science

The project stores raw documents separately from cleaned documents and processed text chunks.

The current retrieval evaluation output reports **1,645 indexed chunks**. This represents indexed chunks, not necessarily unique source articles.

## Retrieval Methodology

### 1. BM25 Keyword Search

BM25 ranks documents based on query-term matching and term importance. It is useful for exact technical terms, acronyms, and specific phrases.

### 2. Semantic Search

The `all-MiniLM-L6-v2` embedding model converts text into vector representations. ChromaDB uses these embeddings to retrieve passages with similar meanings, even when the wording differs.

### 3. Hybrid Retrieval with RRF

The system combines BM25 and semantic rankings using Reciprocal Rank Fusion.

The RRF formula is:

\[
\operatorname{RRF}(d)=\sum_{r}\frac{1}{60+\operatorname{rank}_{r}(d)}
\]

Here, \(d\) is a retrieved chunk and \(\operatorname{rank}_{r}(d)\) is its position in a retrieval ranking. The implementation uses an RRF constant of 60.

Combining rankings allows the system to benefit from exact-term matching and semantic relevance without directly adding scores that