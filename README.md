# 🏦 Bank RAG Agent — AI Customer Support System

An AI-powered bank customer-support assistant built using **Retrieval-Augmented Generation (RAG)** and **Agentic RAG** techniques.

The system uses bank policy documents as its knowledge base and combines document retrieval, hybrid search, reranking, corrective retrieval, multi-hop retrieval, iterative retrieval, query routing, self-RAG, caching, monitoring, security controls, and cost-optimization techniques.

This project was developed as a **university learning project** covering **90 RAG topics**, progressing from basic RAG fundamentals to production-oriented RAG concepts.

---

## 📋 Table of Contents

* [What This Project Does](#what-this-project-does)
* [System Architecture](#system-architecture)
* [Features](#features)
* [Project Structure](#project-structure)
* [Tech Stack](#tech-stack)
* [Knowledge Base](#knowledge-base)
* [Setup and Installation](#setup-and-installation)
* [Running the Project](#running-the-project)
* [How It Works](#how-it-works)
* [RAG Pipeline](#rag-pipeline)
* [Agentic RAG](#agentic-rag)
* [Evaluation](#evaluation)
* [Production Concepts](#production-concepts)
* [Security](#security)
* [Cost Optimization](#cost-optimization)
* [Limitations](#limitations)
* [What I Learned](#what-i-learned)
* [Future Improvements](#future-improvements)
* [Author](#author)

---

# 🚀 What This Project Does

The Bank RAG Agent answers customer questions using information retrieved from bank policy documents.

For each question, the system can:

1. Validate the user input
2. Check the cache
3. Decide whether retrieval is required
4. Route the question according to its type
5. Retrieve relevant bank documents
6. Combine vector and keyword search
7. Rerank retrieved chunks
8. Check retrieval quality
9. Perform corrective retrieval when necessary
10. Handle complex questions using multi-hop retrieval
11. Generate a grounded answer
12. Check answer completeness and groundedness
13. Add source citations
14. Cache the result
15. Record performance and monitoring information

---

## 💬 Example

### User

> What documents are required to open a savings account?

### System

The system searches the relevant bank policy document and generates an answer using the retrieved information.

Example response:

```text
To open a savings account, the required documents include
the documents specified in the bank's account-opening policy.

📄 Sources

[1] account_opening_policy.pdf | Page 1
```

The exact answer depends on the information available in the knowledge base.

---

# 🏗️ System Architecture

```text
                         USER
                           │
                           ▼
                  SECURITY VALIDATION
                           │
                           ▼
                     CACHE CHECK
                    │            │
                  HIT           MISS
                    │            │
                    ▼            ▼
               Cached Answer   AGENT
                                  │
                                  ▼
                            QUESTION TYPE
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
          Greeting             Contact             Banking
             │                    │                    │
             ▼                    ▼                    ▼
        Direct Answer        Contact Tool         Retrieval
                                                       │
                                                       ▼
                                                Query Routing
                                                       │
                                                       ▼
                                              Retrieval Strategy
                                                       │
                                                       ▼
                                           Vector + BM25 Search
                                                       │
                                                       ▼
                                                   Reranking
                                                       │
                                                       ▼
                                               Quality Check
                                                  │       │
                                                 BAD     GOOD
                                                  │       │
                                                  ▼       ▼
                                           Corrective RAG  Context
                                                  │       │
                                                  └───┬───┘
                                                      ▼
                                                  Generation
                                                      │
                                                      ▼
                                             Completeness Check
                                                      │
                                             ┌────────┴────────┐
                                             │                 │
                                            NO                YES
                                             │                 │
                                             ▼                 ▼
                                       More Retrieval      Groundedness
                                                               │
                                                        ┌──────┴──────┐
                                                        │             │
                                                       NO            YES
                                                        │             │
                                                        ▼             ▼
                                                  Retrieve Again   Final Answer
```

---

# ✨ Features

## Core RAG

* PDF document loading
* Web document loading
* Document cleaning
* Document type detection
* Document-specific chunking
* Chunk overlap
* Metadata enrichment
* Deduplication
* Incremental indexing
* Local embeddings
* ChromaDB vector storage

## Retrieval

* Vector similarity search
* BM25 keyword search
* Hybrid / ensemble retrieval
* Metadata-based document filtering
* Multi-query retrieval
* Contextual compression
* Lost-in-the-middle reordering
* BGE cross-encoder reranking
* Retrieval quality checking

## Agentic RAG

* Retrieval decision
* Query routing
* RAG as a tool
* Corrective RAG
* Multi-hop retrieval
* Iterative retrieval
* Self-RAG
* Tool selection
* Calculator tool
* Bank contact-information tool

## Generation

* Context stuffing
* Grounded generation
* Strict grounding prompts
* Multi-document synthesis
* Source metadata
* Citation generation
* Citation deduplication
* Fallback responses

## Evaluation

* Golden test set
* RAGAS evaluation
* Faithfulness
* Answer relevance
* Context precision
* Context recall
* Answer correctness
* LLM-as-a-Judge
* Regression testing
* A/B testing

## Production Concepts

* Query caching
* Cache expiration / TTL
* Async processing
* Concurrency control
* Rate limiting
* Retry logic
* Production monitoring
* Knowledge-base updates
* Multi-tenant architecture
* Security validation
* PII sanitization
* No-result handling
* Cost optimization

---

# 📁 Project Structure

```text
AI-powered-bank-customer-support-assistant/
│
├── main.py
├── agent.py
│
├── ingestion/
│   ├── loader.py
│   ├── cleaner.py
│   └── chunker.py
│
├── retrieval/
│   ├── embedder.py
│   ├── store.py
│   ├── retriever.py
│   └── reranker.py
│
├── generation/
│   ├── prompts.py
│   └── generator.py
│
├── evaluation/
│   ├── golden_set.json
│   └── evaluator.py
│
├── cache/
│   └── cache.py
│
├── data/
│   └── documents/
│
├── .env
├── .gitignore
└── requirements.txt
```

---

# 🧰 Tech Stack

| Component           | Technology                        |
| ------------------- | --------------------------------- |
| Language            | Python                            |
| LLM                 | Groq — `openai/gpt-oss-20b`       |
| RAG Framework       | LangChain                         |
| Embeddings          | `all-MiniLM-L6-v2`                |
| Embedding Type      | Local                             |
| Vector Database     | ChromaDB                          |
| Keyword Retrieval   | BM25                              |
| Hybrid Retrieval    | Ensemble Retriever                |
| Reranker            | `BAAI/bge-reranker-base`          |
| Document Processing | PyMuPDF / LangChain loaders       |
| OCR                 | OCR support for scanned documents |
| Evaluation          | RAGAS                             |
| Async Processing    | Python `asyncio`                  |
| Configuration       | `python-dotenv`                   |

### 💰 Development Cost

The embedding model and BGE reranker run locally.

The project uses Groq for LLM generation.

Actual API cost depends on the provider/account usage and current limits, so the README does not assume a permanent free-tier price.

---

# 📚 Knowledge Base

The project uses bank-related documents as its knowledge base.

Example documents:

```text
data/documents/

├── account_opening_policy.pdf
├── complaint_handling_policy.pdf
├── credit_card_policy.pdf
├── customer_faq.pdf
├── loan_policy.pdf
├── schedule_of_charges.pdf
└── spytm_Scanned.pdf
```

The project can also retrieve information from the State Bank of Pakistan web source used during development.

---

# 🔄 How It Works

## 1. Document Ingestion

Documents are loaded from the knowledge-base directory.

```text
Documents
    ↓
Loading
    ↓
Cleaning
    ↓
Document Type Detection
    ↓
Chunking
    ↓
Metadata
    ↓
Deduplication
    ↓
Embedding
    ↓
ChromaDB
```

---

## 2. Chunking

Documents are divided into smaller pieces so that relevant information can be retrieved efficiently.

Different document types use different chunk configurations.

For example:

```text
Policy documents → larger chunks
FAQ documents → smaller chunks
Charges documents → smaller chunks
General documents → medium chunks
```

The project uses `RecursiveCharacterTextSplitter` with document-specific settings.

---

## 3. Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

The model runs locally and converts text into numerical vectors.

This avoids sending document text to an external embedding API during development.

---

## 4. Vector Storage

The generated embeddings and document metadata are stored in ChromaDB.

Each chunk can contain metadata such as:

```text
source
page
section
chunk_index
```

This metadata is later used for retrieval filtering and citations.

---

# 🔎 RAG Pipeline

The retrieval pipeline combines multiple techniques.

```text
User Question
      ↓
Document Detection
      ↓
Metadata Filtering
      ↓
Vector Search
      +
BM25 Search
      ↓
Hybrid Retrieval
      ↓
Candidate Chunks
      ↓
BGE Reranking
      ↓
Best Chunks
      ↓
Context Reordering
      ↓
Generation
```

### Vector Search

Finds semantically similar chunks.

### BM25

Finds chunks based on important keywords.

### Hybrid Retrieval

Combines both approaches.

### BGE Reranking

The reranker evaluates the relationship between the question and retrieved chunks and produces a better ranking.

A simple way to remember it:

> **Retrieval finds candidates. Reranking chooses the best candidates.**

---

# 🤖 Agentic RAG

The project extends traditional RAG with an agentic decision layer.

Traditional RAG:

```text
Question
   ↓
Retrieve
   ↓
Generate
```

Agentic RAG:

```text
Question
   ↓
Agent
   ↓
Decide what to do
   ↓
Choose tool / retrieval strategy
   ↓
Retrieve / Calculate / Contact
   ↓
Check result
   ↓
Generate final answer
```

---

# 🧠 RAG as a Tool

The RAG pipeline is exposed to the agent as a tool.

The project also contains additional tools such as:

```python
search_bank_documents()
calculate_loan_installment()
get_bank_contact_info()
```

This allows the agent to combine document retrieval with other capabilities.

---

# 🛠️ Corrective RAG

Corrective RAG checks whether retrieved information is useful.

```text
Retrieve
   ↓
Evaluate Retrieved Information
   ↓
 ┌─────────────┐
 │             │
GOOD          BAD
 │             │
 ▼             ▼
Generate    Reformulate
              ↓
           Retrieve Again
```

This helps recover from poor retrieval results.

---

# 🔗 Multi-Hop Retrieval

Some questions require information from multiple retrieval steps.

Example:

> I want to take a loan. Am I eligible, what documents do I need, and what will my monthly payment be?

The system can break the question into smaller tasks:

```text
Original Question
       ↓
 ┌─────┼─────────────────┐
 │     │                 │
Eligibility   Documents   Payment Information
 │     │                 │
 RAG   RAG               RAG
                         │
                         ▼
                    Calculator
       ↓
Combine Information
       ↓
Final Answer
```

---

# 🔁 Iterative Retrieval

If the current answer is incomplete, the system can perform another retrieval cycle.

```text
Retrieve
   ↓
Generate
   ↓
Check Completeness
   ↓
 ┌───────────┐
 │           │
Complete   Incomplete
 │           │
 ▼           ▼
Answer    Retrieve Again
```

---

# 🧭 Query Routing

The agent determines which path is appropriate for a question.

| Question                                  | Route                   |
| ----------------------------------------- | ----------------------- |
| "Hello"                                   | Direct answer           |
| "What is the complaint procedure?"        | RAG                     |
| "What documents are required for a loan?" | RAG                     |
| "Calculate my monthly payment"            | Calculator              |
| "What is the bank contact information?"   | Contact tool            |
| Complex multi-part question               | Multi-hop / Agentic RAG |

Routing helps prevent every question from going through the same expensive pipeline.

---

# 🔍 Self-RAG

Self-RAG adds self-evaluation to the RAG process.

The system can evaluate:

```text
Do I need retrieval?
        ↓
Are the retrieved documents useful?
        ↓
Can I answer the question?
        ↓
Is the answer complete?
        ↓
Is the answer grounded?
        ↓
Should I retrieve again?
```

The goal is to reduce irrelevant retrieval and unsupported answers.

---

# 📊 Evaluation

The project uses evaluation techniques to measure RAG quality.

## Golden Test Set

A collection of questions with trusted answers.

The project contains a **30-question golden test set**.

Each test can contain information such as:

```json
{
  "id": "Q01",
  "question": "What is the minimum balance?",
  "ground_truth": "...",
  "expected_source": "account_opening_policy.pdf"
}
```

---

# 📈 RAGAS Metrics

The project studies the following RAGAS metrics:

| Metric             | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| Faithfulness       | Checks whether the answer is supported by the retrieved context |
| Answer Relevance   | Checks whether the answer addresses the question                |
| Context Precision  | Measures how relevant the retrieved context is                  |
| Context Recall     | Measures whether the required information was retrieved         |
| Answer Correctness | Compares the generated answer with the ground truth             |

These metrics help identify whether a problem comes from retrieval, context quality, or generation.

---

# 🤖 LLM-as-a-Judge

An LLM can also act as an evaluator.

```text
Question
   +
Retrieved Context
   +
Ground Truth
   +
Generated Answer
        ↓
    LLM Judge
        ↓
Faithfulness
Relevance
Correctness
```

This is useful for quick development experiments and pipeline comparisons.

---

# 🔄 Regression Testing

Regression testing checks whether a change makes the system worse.

```text
Old Pipeline
     ↓
Evaluation
     ↓
Scores

New Pipeline
     ↓
Evaluation
     ↓
Scores

      ↓
Compare
      ↓
Regression?
```

This is important when changing:

* chunk sizes
* retrieval strategies
* reranking
* prompts
* models
* agent behavior

---

# ⚖️ A/B Testing

Different pipeline configurations can be compared using the same test questions.

Example:

```text
Pipeline A
With Reranking

        VS

Pipeline B
Without Reranking

        ↓
Same Questions
        ↓
Evaluation
        ↓
Compare Results
```

---

# ⚙️ Production Concepts

The final part of the project focuses on production-oriented RAG engineering.

## Caching

Repeated questions can reuse previous results.

```text
Question
   ↓
Cache?
 ┌─┴─┐
YES  NO
 │    │
 ▼    ▼
Answer RAG
       ↓
     Cache
```

---

## Async Processing

The project studies asynchronous execution using:

```python
async def
await
asyncio.run()
asyncio.gather()
asyncio.create_task()
asyncio.Semaphore()
asyncio.wait_for()
```

Async processing can help when multiple operations spend time waiting for external services.

---

## Rate Limiting

Rate limiting controls the number of external API requests.

```text
Many Requests
      ↓
Rate Limiter
      ↓
Controlled API Traffic
      ↓
External API
```

This helps prevent excessive API usage and rate-limit errors.

---

## Monitoring

Production monitoring can track:

* response latency
* retrieval latency
* number of chunks
* errors
* empty retrievals
* fallback responses
* cache performance
* API errors
* retrieval quality

---

## Knowledge Base Updates

The project studies how knowledge bases can be updated when policies change.

```text
New / Updated Document
          ↓
Detect Change
          ↓
Process
          ↓
Chunk
          ↓
Embed
          ↓
Update Vector Database
```

A more advanced production design can use versioned knowledge bases to switch between validated versions.

---

## Multi-Tenant RAG

Multi-tenant RAG allows one application to serve multiple organizations while keeping their data isolated.

Example:

```text
                RAG Application
                       │
          ┌────────────┼────────────┐
          │            │            │
         MCB          HBL        Meezan
          │            │            │
       MCB Docs     HBL Docs    Meezan Docs
```

Each tenant's documents, retrieval, cache, and access permissions must remain isolated.

---

# 🔐 Security

Because this project represents a banking support system, security is an important part of the learning process.

The project studies:

* Authentication
* Authorization
* Tenant isolation
* API-key protection
* Input validation
* Prompt-injection defense
* Sensitive-data protection
* PII sanitization
* Secure document ingestion
* Database security
* Monitoring and audit logging
* Data-leakage prevention

The security implementation in this project is **educational** and should not be considered sufficient for a real banking production environment.

---

# 💰 Cost Optimization

The project studies several methods for controlling AI-system costs.

### 1. Caching

Avoid repeated LLM calls for identical questions.

### 2. Avoid unnecessary LLM calls

Simple questions do not always require the full RAG pipeline.

### 3. Local embeddings

Using local embeddings avoids external embedding API calls during development.

### 4. Limit retrieved context

Retrieving unnecessary chunks increases processing and can increase LLM input size.

### 5. Selective reranking

Only rerank an appropriate number of candidate chunks.

### 6. Model selection

Use smaller or cheaper models for tasks that do not require a large model.

### 7. Limit retries

Repeated failed requests can increase API usage.

### 8. Monitor token usage

Track input and output usage to understand where costs are coming from.

### 9. Set usage limits

Production systems should have appropriate request and spending limits.

---

# 📁 Main Learning Journey

The project covers 90 topics in the following progression:

```text
Topics 1–10
Python Fundamentals
        ↓
Topics 11–20
Python + RAG Development
        ↓
Topics 21–30
Document Processing & Chunking
        ↓
Topics 31–40
Embeddings & Vector Storage
        ↓
Topics 41–53
Retrieval Techniques
        ↓
Topics 54–59
Citations & Sources
        ↓
Topics 60–72
RAG Evaluation
        ↓
Topics 73–79
Agentic RAG
        ↓
Topics 80–90
Production RAG
```

---

# 🎯 What I Learned

This project helped me understand that building an AI assistant is not simply:

```text
LLM + Prompt = Chatbot
```

A useful RAG system requires multiple layers:

```text
Data
 ↓
Document Processing
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector Storage
 ↓
Retrieval
 ↓
Reranking
 ↓
Context
 ↓
Generation
 ↓
Citations
 ↓
Evaluation
 ↓
Agentic Decision Making
 ↓
Caching
 ↓
Async Processing
 ↓
Monitoring
 ↓
Security
 ↓
Cost Optimization
 ↓
Production Engineering
```

The main lesson from this project is:

> **Good AI systems require good data, retrieval, reasoning, evaluation, security, cost control, and production engineering — not just a powerful LLM.**

---

# ⚠️ Limitations

This is a **learning and experimentation project**, not a real banking production system.

Some production concepts are implemented in simplified form to demonstrate their underlying ideas.

For example:

* JSON-based caching instead of a distributed cache such as Redis
* Simplified security validation
* Simplified rate limiting
* Local ChromaDB
* Educational multi-tenancy implementation
* Development-level monitoring
* Limited evaluation dataset
* Simplified Agentic RAG orchestration

A real banking system would require significantly stronger:

* security
* authentication
* authorization
* compliance
* auditing
* infrastructure
* testing
* monitoring
* data protection
* fault tolerance

---

# 🔮 Future Improvements

Possible future improvements include:

* Production-grade authentication
* Role-based authorization
* Stronger prompt-injection protection
* Redis-based distributed caching
* Improved tenant isolation
* Versioned knowledge-base deployments
* Distributed queues and workers
* Better observability
* More extensive evaluation datasets
* Automated regression testing
* CI/CD
* Docker deployment
* Web-based interface
* Conversation memory
* Production cloud deployment

---

# 🛠️ Setup and Installation

## 1. Clone the repository

```bash
git clone https://github.com/NoraizZaman18/AI-powered-bank-customer-support-assistant.git
cd AI-powered-bank-customer-support-assistant
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or API keys to GitHub.

---

# ▶️ Running the Project

Run the main application:

```bash
python main.py
```

The system will:

1. Load the knowledge base
2. Process documents
3. Create or update the vector index
4. Initialize the Agentic RAG pipeline
5. Process questions
6. Retrieve relevant information
7. Generate answers
8. Display source information
9. Record relevant performance information

---

# 📜 Disclaimer

This project is developed for **educational purposes**.

It is not an official banking application and should not be used to process real customer financial information or make real financial decisions.

---

# 👨‍💻 Author

**Noraiz Zaman**

Computer Science Student

GitHub:
https://github.com/NoraizZaman18
