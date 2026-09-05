
# main.py


from dotenv import load_dotenv
from generation.generator import generate_rag_response

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
load_dotenv()


# ── STEP 1 — LOAD DOCUMENTS ──────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1 — LOADING DOCUMENTS")
print("="*60)

documents = load_all_documents("data/documents/")
print(f"Total documents loaded: {len(documents)}")


# ── STEP 2 — CLEAN DOCUMENTS ─────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 2 — CLEANING DOCUMENTS")
print("="*60)

documents = clean_documents(documents)
print(f"Total documents after cleaning: {len(documents)}")


# ── STEP 3 — CHUNKING FUNCTION ───────────────────────────────────────────────

def incremental_chunking(documents):
    """
    Chunking adapter for incremental indexing.
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

print("\n" + "="*60)
print("STEP 4 — INCREMENTAL INDEXING")
print("="*60)

vectorstore = incremental_index(
    folder_path="data/documents/",
    chunk_function=incremental_chunking,
    force_reindex=False
)

print("Vector store ready")


# ── STEP 5 — PREPARE CHUNKS FOR BM25 ─────────────────────────────────────────

all_chunks = incremental_chunking(documents)
print(f"\nChunks prepared for BM25: {len(all_chunks)}")

# ── DEBUG PIPELINE ────────────────────────────────────────────────────────────

def debug_rag_pipeline(question):
    """
    Show what happens inside the RAG pipeline step by step.
    Step 1: Retrieval
    Step 2: Augmentation
    Step 3: Generation
    """

    print("\n" + "=" * 60)
    print(f"DEBUGGING: {question}")
    print("=" * 60)

    # STEP 1 — RETRIEVAL
    print("\n--- STEP 1: RETRIEVAL ---")

    context, sources, retrieved_docs = smart_retrieve(
        question=question,
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        k=6,
        strategy="ensemble",
        apply_reorder=True,
        use_smart_filter=True,
        use_reranking=True
    )

    print(f"\nRetrieved {len(retrieved_docs)} chunks:")

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n  Chunk {i}:")
        print(f"  Source: {doc.metadata.get('source', 'unknown')}")
        print(f"  Page: {doc.metadata.get('page', '?')}")
        print(f"  Preview: {doc.page_content[:200]}...")

    # STEP 2 — AUGMENTATION
    print("\n--- STEP 2: AUGMENTATION ---")

    print(f"Context length: {len(context)} characters")
    print(f"Context preview:\n{context[:400]}...")

    # STEP 3 — GENERATION
    print("\n--- STEP 3: GENERATION ---")

    result = generate_rag_response(
        question=question,
        docs=retrieved_docs,
        use_strict=True
    )

    print(f"\nAnswer:\n{result['answer']}")

    print(f"\nSources: {result['unique_sources']}")
    print(f"Fallback used: {result['fallback_used']}")
    print(f"Synthesis used: {result['synthesis_used']}")

    return {
        "question": question,
        "retrieved_chunks": len(retrieved_docs),
        "context_length": len(result["context_used"]),
        "answer": result["answer"],
        "sufficient_context": result["sufficient_context"]
    }

# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_questions = [
        "What is the minimum balance required for a savings account?",
        "Which documents do I need to open a bank account?",
        "How much is the IBFT transfer charge?",
        "What is the procedure for filing a complaint against the bank?",
        "How much does a RAAST transfer cost?"
    ]

    for question in test_questions:
        debug_rag_pipeline(question)
