
# main.py
import os
from retrieval.retriever import smart_retrieve
from retrieval.store import incremental_index
from ingestion.chunker import (
    detect_document_types,
    recommend_overlap,
    recursive_chunking,
    analyse_chunk_quality,
    enrich_metadata
)
from dotenv import load_dotenv

from ingestion.loader import load_all_documents
from ingestion.cleaner import clean_documents
from ingestion.chunker import parent_child_chunking


from ingestion.chunker import recursive_chunking

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STEP 1 — LOAD DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 — LOADING DOCUMENTS")
print("=" * 60)

test_urls = [
    "https://www.sbp.org.pk/bsd/2023/C3.htm"
]

documents = load_all_documents(
    "data/documents/",
    urls=test_urls
)

print(f"\nTotal documents loaded: {len(documents)}")


# ============================================================
# STEP 2 — CLEAN DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 — CLEANING DOCUMENTS")
print("=" * 60)

documents = clean_documents(documents)

print(f"\nTotal documents after cleaning: {len(documents)}")


# step 3 ---incremental_chunking() function decides how to chunk each document.



def incremental_chunking(documents):
    """
    Adapter for Topic 19 incremental indexing.

    Uses the existing document-type-specific
    chunking system from chunker.py.
    """

    document_types = detect_document_types(documents)

    final_chunks = []

    for source, doc_type in document_types.items():

        settings = recommend_overlap(doc_type)

        source_documents = [
            doc
            for doc in documents
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
# ============================================================
# STEP 4 + 5 — INCREMENTAL INDEXING
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 + 5 — INCREMENTAL INDEXING")
print("=" * 60)

vectorstore = incremental_index(
    folder_path="data/documents/",
    chunk_function=incremental_chunking,
    force_reindex=False
)

print("Vector store ready")
# ============================================================
# STEP 6 — CREATE RETRIEVER
# ============================================================



# ============================================================
# STEP 7 — CREATE RAG PROMPT
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 — CREATING RAG PROMPT")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are ABC Bank customer support assistant.

Answer the customer's question using ONLY the context
provided below.

If the context does not contain the answer, say:

"I don't have information about this in our documents."

Do not invent information.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])

print("RAG prompt created")


# ============================================================
# STEP 8 — SETUP LLM
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 — SETTING UP LLM")
print("=" * 60)

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in the environment."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key
)

print("LLM configured")


# ============================================================
# STEP 9 — FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_docs(docs):
    """
    Convert retrieved LangChain Documents into
    one text block for the LLM.
    """

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def retrieve_for_rag(question):
    return smart_retrieve(
        question,
        vectorstore,
        k=3
    )

retrieve_runnable = RunnableLambda(retrieve_for_rag)
# ============================================================
# STEP 10 — BUILD RAG CHAIN
# ============================================================

rag_chain = (
    {
        "context": retrieve_runnable | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# DEBUG RAG PIPELINE
# ============================================================

def debug_rag_pipeline(question):
    """
    Show what happens inside the RAG pipeline.

    1. Retrieval
    2. Augmentation
    3. Generation
    """

    print("\n" + "=" * 60)
    print(f"DEBUGGING QUESTION:")
    print(question)
    print("=" * 60)


    # --------------------------------------------------------
    # STEP 1 — RETRIEVAL
    # --------------------------------------------------------

    print("\n--- STEP 1: RETRIEVAL ---")

    retrieved_docs = smart_retrieve(
    question,
    vectorstore,
    k=3
)

    print(
        f"Retrieved {len(retrieved_docs)} chunks"
    )

    for i, doc in enumerate(retrieved_docs):

        print(f"\nChunk {i + 1}")

        print(
            f"Source: "
            f"{doc.metadata.get('source', 'unknown')}"
        )

        print(
            f"Page: "
            f"{doc.metadata.get('page', 'unknown')}"
        )

        print(
            f"File type: "
            f"{doc.metadata.get('file_type', 'unknown')}"
        )

        print(
            f"Content: "
            f"{doc.page_content[:300]}..."
        )


    # --------------------------------------------------------
    # STEP 2 — AUGMENTATION
    # --------------------------------------------------------

    print("\n--- STEP 2: AUGMENTATION ---")

    context = format_docs(retrieved_docs)

    print(
        f"Context length: "
        f"{len(context)} characters"
    )

    print(
        f"\nContext preview:\n"
        f"{context[:500]}..."
    )

    augmented_prompt = f"""
Context:
{context}

Question:
{question}
"""

    print(
        f"\nFull prompt length: "
        f"{len(augmented_prompt)} characters"
    )


    # --------------------------------------------------------
    # STEP 3 — GENERATION
    # --------------------------------------------------------

    print("\n--- STEP 3: GENERATION ---")

    answer = rag_chain.invoke(question)

    print(
        f"\nAnswer:\n{answer}"
    )


    return {
        "question": question,
        "retrieved_chunks": len(retrieved_docs),
        "context_length": len(context),
        "answer": answer
    }


#  parent child chunking 

print("\n" + "="*50)
print("PARENT-CHILD CHUNKING TEST")
print("="*50)

result = parent_child_chunking(
    documents[:3],  # test on 3 documents
    parent_chunk_size=1500,
    child_chunk_size=300
)

# show relationship between parent and child
first_parent = result["parents"][0]
first_children = [
    c for c in result["children"]
    if c.metadata["parent_id"] == first_parent.metadata["parent_id"]
]

print(f"\nParent chunk ({len(first_parent.page_content)} chars):")
print(f"  {first_parent.page_content[:200]}...")

print(f"\nChildren of this parent ({len(first_children)} children):")
for i, child in enumerate(first_children[:3]):
    print(f"\n  Child {i+1} ({len(child.page_content)} chars):")
    print(f"    {child.page_content[:100]}...")

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = debug_rag_pipeline(
        "What is the minimum balance for savings account?"
    )

