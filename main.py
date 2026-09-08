
# # main.py


# from dotenv import load_dotenv
# from generation.generator import generate_rag_response

# from generation.generator import generate_rag_response

# from ingestion.loader import load_all_documents
# from ingestion.cleaner import clean_documents
# from ingestion.chunker import (
#     detect_document_types,
#     recommend_overlap,
#     recursive_chunking,
#     enrich_metadata
# )
# from retrieval.store import incremental_index
# from retrieval.retriever import smart_retrieve
# load_dotenv()


# # ── STEP 1 — LOAD DOCUMENTS ──────────────────────────────────────────────────

# print("\n" + "="*60)
# print("STEP 1 — LOADING DOCUMENTS")
# print("="*60)

# documents = load_all_documents("data/documents/")
# print(f"Total documents loaded: {len(documents)}")


# # ── STEP 2 — CLEAN DOCUMENTS ─────────────────────────────────────────────────

# print("\n" + "="*60)
# print("STEP 2 — CLEANING DOCUMENTS")
# print("="*60)

# documents = clean_documents(documents)
# print(f"Total documents after cleaning: {len(documents)}")


# # ── STEP 3 — CHUNKING FUNCTION ───────────────────────────────────────────────

# def incremental_chunking(documents):
#     """
#     Chunking adapter for incremental indexing.
#     Detects document type and applies best chunking strategy per document.
#     """

#     document_types = detect_document_types(documents)
#     final_chunks = []

#     for source, doc_type in document_types.items():

#         settings = recommend_overlap(doc_type)

#         source_documents = [
#             doc for doc in documents
#             if doc.metadata.get("source") == source
#         ]

#         chunks = recursive_chunking(
#             source_documents,
#             chunk_size=settings["chunk_size"],
#             chunk_overlap=settings["overlap"]
#         )

#         chunks = enrich_metadata(chunks)
#         final_chunks.extend(chunks)

#     return final_chunks


# # ── STEP 4 — INCREMENTAL INDEXING ────────────────────────────────────────────

# print("\n" + "="*60)
# print("STEP 4 — INCREMENTAL INDEXING")
# print("="*60)

# vectorstore = incremental_index(
#     folder_path="data/documents/",
#     chunk_function=incremental_chunking,
#     force_reindex=False
# )

# print("Vector store ready")


# # ── STEP 5 — PREPARE CHUNKS FOR BM25 ─────────────────────────────────────────

# all_chunks = incremental_chunking(documents)
# print(f"\nChunks prepared for BM25: {len(all_chunks)}")

# # ── DEBUG PIPELINE ────────────────────────────────────────────────────────────

# def debug_rag_pipeline(question):
#     """
#     Show what happens inside the RAG pipeline step by step.
#     """

#     print("\n" + "="*60)
#     print(f"DEBUGGING: {question}")
#     print("="*60)

#     # step 1 — retrieval
#     print("\n--- STEP 1: RETRIEVAL ---")

#     context, sources, retrieved_docs = smart_retrieve(
#         question=question,
#         vectorstore=vectorstore,
#         all_chunks=all_chunks,
#         k=6,
#         strategy="ensemble",
#         apply_reorder=True,
#         use_smart_filter=True,
#         use_reranking=True
#     )

#     print(f"\nRetrieved {len(sources)} chunks:")
#     for i, source in enumerate(sources, start=1):
#         print(f"\n  Chunk {i}:")
#         print(f"  Source: {source['source']}")
#         print(f"  Page: {source['page']}")
#         print(f"  Preview: {source['content_preview']}")

#     # step 2 — augmentation
#     print("\n--- STEP 2: AUGMENTATION ---")
#     print(f"Context length: {len(context)} characters")
#     print(f"Context preview:\n{context[:400]}...")

#     # step 3 — generation with citations
#     print("\n--- STEP 3: GENERATION WITH CITATIONS ---")


#     result = generate_rag_response(
#         question=question,
#         docs=retrieved_docs,
#         use_strict=False,
#         citation_style="end"
#     )

#     # display formatted response with citations
#     print(f"\n{result['formatted_response']}")

#     return {
#         "question": question,
#         "retrieved_chunks": len(sources),
#         "context_length": len(context),
#         "answer": result["answer"],
#         "citations": result["citations"],
#         "sufficient_context": result["sufficient_context"]
#     }

# # ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

# if __name__ == "__main__":

#     test_questions = [
#         "What is the minimum balance required for a savings account?",

#     ]

#     for question in test_questions:
#         debug_rag_pipeline(question)


# main.py

import os
from dotenv import load_dotenv
from agent import set_agent_dependencies, run_agent

from ingestion.loader import load_all_documents
from ingestion.cleaner import clean_documents
from ingestion.chunker import (
    detect_document_types,
    recommend_overlap,
    recursive_chunking,
    enrich_metadata
)
from retrieval.store import incremental_index
from retrieval.retriever import smart_retrieve
from generation.generator import generate_rag_response

load_dotenv()


# ── STEP 1 — LOAD DOCUMENTS ──────────────────────────────────────────────────

print("\n" + "═"*60)
print("  STEP 1 — LOADING DOCUMENTS")
print("═"*60)

documents = load_all_documents("data/documents/")
print(f"  Total documents loaded: {len(documents)}")


# ── STEP 2 — CLEAN DOCUMENTS ─────────────────────────────────────────────────

print("\n" + "═"*60)
print("  STEP 2 — CLEANING DOCUMENTS")
print("═"*60)

documents = clean_documents(documents)
print(f"  Total after cleaning: {len(documents)}")


# ── STEP 3 — CHUNKING FUNCTION ───────────────────────────────────────────────

def incremental_chunking(documents):
    """
    Detects document type and applies best chunking strategy per document.
    """

    document_types = detect_document_types(documents)
    final_chunks = []

    for source, doc_type in document_types.items():

        settings = recommend_overlap(doc_type)

        source_documents = [
            doc for doc in documents
            if doc.metadata.get("source") == source
        ]

        chunks = recursive_chunking(
            source_documents,
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["overlap"]
        )

        chunks = enrich_metadata(chunks)
        final_chunks.extend(chunks)

    return final_chunks


# ── STEP 4 — INCREMENTAL INDEXING ────────────────────────────────────────────

print("\n" + "═"*60)
print("  STEP 4 — INCREMENTAL INDEXING")
print("═"*60)

vectorstore = incremental_index(
    folder_path="data/documents/",
    chunk_function=incremental_chunking,
    force_reindex=False
)

print("  Vector store ready")


# ── STEP 5 — PREPARE CHUNKS FOR BM25 ─────────────────────────────────────────

all_chunks = incremental_chunking(documents)
print(f"  Chunks for BM25: {len(all_chunks)}")


# ── STEP 6 — INITIALIZE AGENT ────────────────────────────────────────────────

print("\n" + "═"*60)
print("  STEP 6 — INITIALIZING AGENT")
print("═"*60)

set_agent_dependencies(vectorstore, all_chunks)
print("  Agent ready")


# ── RAG PIPELINE ─────────────────────────────────────────────────────────────


def run_rag_pipeline(question: str, verbose: bool = False) -> dict:
    """
    Full agentic RAG pipeline.
    Agent decides the best strategy for each question.
    """

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  QUESTION")
        print(f"{'═'*60}")
        print(f"  {question}")

    result = run_agent(question)

    print(f"\n{'═'*60}")
    print(f"  Q: {question}")
    print(f"  Strategy: {result['strategy']}")
    print(f"{'═'*60}")
    print(f"\n{result['answer']}")
    print()

    return result

# def run_rag_pipeline(question: str, verbose: bool = False) -> dict:
#     """
#     Full RAG pipeline:
#     Step 1 — Retrieve relevant chunks
#     Step 2 — Generate answer with citations
#     Step 3 — Display clean formatted response
#     """

#     if verbose:
#         print(f"\n{'═'*60}")
#         print(f"  QUESTION")
#         print(f"{'═'*60}")
#         print(f"  {question}")

#     # ── RETRIEVAL ─────────────────────────────────────────────────
#     if verbose:
#         print(f"\n{'─'*60}")
#         print(f"  RETRIEVAL")
#         print(f"{'─'*60}")

#     context, sources, retrieved_docs = smart_retrieve(
#         question=question,
#         vectorstore=vectorstore,
#         all_chunks=all_chunks,
#         k=6,
#         strategy="ensemble",
#         apply_reorder=True,
#         use_smart_filter=True,
#         use_reranking=True
#     )

#     if verbose:
#         print(f"\n  Retrieved {len(sources)} chunks:")
#         for i, s in enumerate(sources, start=1):
#             print(f"    [{i}] {s['source']}  |  Page {s['page']}")
#             print(f"        {s['content_preview'][:80]}...")

    # ── GENERATION ────────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  GENERATION")
        print(f"{'─'*60}")

    result = generate_rag_response(
        question=question,
        docs=retrieved_docs,
        use_strict=False,
        citation_style="end"
    )

    # ── DISPLAY ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Q: {question}")
    print(f"{'═'*60}")
    print(f"\n{result['formatted_response']}")
    print()

    return {
        "question": question,
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["unique_sources"],
        "sufficient_context": result["sufficient_context"],
        "fallback_used": result["fallback_used"],
        "synthesis_used": result["synthesis_used"],
        "num_chunks": result["num_chunks"]
    }


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_questions = [
        # "What is the minimum balance required for a savings account?",
        # "Which documents do I need to open a bank account?",
        # "How much is the IBFT transfer charge?",
        # "What is the procedure for filing a complaint against the bank?",
        # "How much does a RAAST transfer cost?",
        # "What is the current SBP policy rate?",

        "I want to take a loan. Am I eligible, what documents do I need, and what will my monthly payment be?"
    ]

    for question in test_questions:
        run_rag_pipeline(question, verbose=False)