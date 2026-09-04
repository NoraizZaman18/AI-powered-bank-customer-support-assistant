
# main.py

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from ingestion.loader import load_all_documents
from ingestion.cleaner import clean_documents
from ingestion.chunker import (
    detect_document_types,
    recommend_overlap,
    recursive_chunking,
    enrich_metadata
)
from retrieval.store import incremental_index, get_vectorstore
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


# ── STEP 6 — SETUP LLM ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 6 — SETTING UP LLM")
print("="*60)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)

print("LLM configured")


# ── STEP 7 — RAG PROMPT ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 7 — CREATING RAG PROMPT")
print("="*60)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are ABC Bank customer support assistant.

Answer the customer's question using ONLY the context provided below.

If the context does not contain the answer say:
"I don't have information about this in our documents."

Do not invent information. Do not guess.

Context:
{context}"""),
    ("human", "{question}")
])

print("RAG prompt created")


# ── STEP 8 — FORMAT DOCS AND RETRIEVE ────────────────────────────────────────

def format_docs(docs):
    """Convert retrieved documents into one text block for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_for_rag(question):
    """
    Retrieval function used inside the RAG chain.
    Returns context string from smart_retrieve.
    """
    context, sources = smart_retrieve(
        question=question,
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        k=4,
        strategy="ensemble",
        apply_reorder=True,
        use_smart_filter=True
    )
    return context


retrieve_runnable = RunnableLambda(retrieve_for_rag)


# ── STEP 9 — BUILD RAG CHAIN ─────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 9 — BUILDING RAG CHAIN")
print("="*60)

rag_chain = (
    {
        "context": retrieve_runnable,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain ready")


# ── DEBUG PIPELINE ────────────────────────────────────────────────────────────

def debug_rag_pipeline(question):
    """
    Show what happens inside the RAG pipeline step by step.
    Step 1: Retrieval
    Step 2: Augmentation
    Step 3: Generation
    """

    print("\n" + "="*60)
    print(f"DEBUGGING: {question}")
    print("="*60)

    # step 1 — retrieval
    print("\n--- STEP 1: RETRIEVAL ---")

    context, sources = smart_retrieve(
        question=question,
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        k=4,
        strategy="ensemble",
        apply_reorder=True,
        use_smart_filter=True
        
    )

    print(f"\nRetrieved {len(sources)} chunks:")
    for i, source in enumerate(sources, start=1):
        print(f"\n  Chunk {i}:")
        print(f"  Source: {source['source']}")
        print(f"  Page: {source['page']}")
        print(f"  Preview: {source['content_preview']}")

    # step 2 — augmentation
    print("\n--- STEP 2: AUGMENTATION ---")
    print(f"Context length: {len(context)} characters")
    print(f"Context preview:\n{context[:400]}...")

    # step 3 — generation
    print("\n--- STEP 3: GENERATION ---")
    answer = rag_chain.invoke(question)
    print(f"\nAnswer:\n{answer}")

    return {
        "question": question,
        "retrieved_chunks": len(sources),
        "context_length": len(context),
        "answer": answer
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
