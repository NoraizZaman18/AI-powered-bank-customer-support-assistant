# ingestion/chunker.py

import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# ── TOPIC 11 — Fixed Size Chunking ──────────────────────────────────────────

def fixed_size_chunking(documents: List[Document],
                        chunk_size: int = 500,
                        chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into fixed size chunks.
    chunk_size: number of characters per chunk
    chunk_overlap: characters shared between adjacent chunks
    Use this as your starting point before trying smarter strategies.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["chunking_strategy"] = "fixed_size"

    print(f"Fixed size chunking: {len(documents)} docs → {len(chunks)} chunks")
    print(f"Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


# ── TOPIC 12 — Sentence Chunking ────────────────────────────────────────────

def sentence_chunking(documents: List[Document],
                      sentences_per_chunk: int = 5,
                      overlap_sentences: int = 1) -> List[Document]:
    """
    Split documents at sentence boundaries.
    sentences_per_chunk: how many sentences per chunk
    overlap_sentences: sentences shared between adjacent chunks
    Better than fixed size for documents with complete sentences.
    """

    chunks = []

    for doc in documents:
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, doc.page_content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        chunk_index = 0
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)

            if chunk_text.strip():
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "sentence",
                        "sentences_in_chunk": len(chunk_sentences)
                    }
                ))
                chunk_index += 1

            i += sentences_per_chunk - overlap_sentences

    print(f"Sentence chunking: {len(documents)} docs → {len(chunks)} chunks")
    print(f"Settings: {sentences_per_chunk} sentences per chunk, {overlap_sentences} overlap")
    return chunks


# ── TOPIC 12 — Paragraph Chunking ───────────────────────────────────────────

def paragraph_chunking(documents: List[Document],
                       min_length: int = 100,
                       max_length: int = 1000) -> List[Document]:
    """
    Split documents at paragraph boundaries — double newlines.
    min_length: skip paragraphs shorter than this
    max_length: split paragraphs longer than this with fixed size
    Best for structured documents like bank policies where
    each paragraph covers one specific topic or rule.
    """

    chunks = []

    for doc in documents:
        paragraphs = doc.page_content.split("\n\n")

        chunk_index = 0
        for para in paragraphs:
            para = para.strip()

            if len(para) < min_length:
                continue

            if len(para) > max_length:
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_length,
                    chunk_overlap=50
                )
                sub_chunks = sub_splitter.split_text(para)
                for sub in sub_chunks:
                    if sub.strip():
                        chunks.append(Document(
                            page_content=sub,
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_index,
                                "chunking_strategy": "paragraph_split"
                            }
                        ))
                        chunk_index += 1
            else:
                chunks.append(Document(
                    page_content=para,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "paragraph"
                    }
                ))
                chunk_index += 1

    print(f"Paragraph chunking: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


# ── TOPIC 13 — Recursive Chunking ───────────────────────────────────────────

def recursive_chunking(documents: List[Document],
                       chunk_size: int = 500,
                       chunk_overlap: int = 50) -> List[Document]:
    """
    Smart chunking that tries separators in order:
    1. Double newline (paragraph break)
    2. Single newline
    3. Period followed by space (sentence end)
    4. Space (word break)
    5. Character (last resort)
    This is the recommended default for most RAG systems.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunking_strategy"] = "recursive"
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"Recursive chunking: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


# ── TOPIC 14 — Semantic Chunking ────────────────────────────────────────────

def semantic_chunking(documents: List[Document],
                      breakpoint_type: str = "percentile",
                      breakpoint_threshold: int = 95) -> List[Document]:
    """
    Split documents where the meaning changes significantly.
    Uses embeddings to detect topic shifts.
    breakpoint_type: percentile / standard_deviation / interquartile
    breakpoint_threshold: sensitivity — higher means fewer chunks
    SLOWER than other methods — embeds every sentence.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_threshold
    )

    print("Running semantic chunking — this takes longer than other methods...")
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunking_strategy"] = "semantic"
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"Semantic chunking: {len(documents)} docs → {len(chunks)} chunks")
    return chunks


# ── TOPIC 15 — Chunk Overlap Verification ───────────────────────────────────

def test_overlap(chunks: List[Document], num_pairs: int = 3):
    """
    Visual test to confirm overlap is working correctly.
    Shows the shared content between adjacent chunks.
    Run this after chunking to verify your overlap settings.
    """

    print("\n" + "=" * 50)
    print("OVERLAP VERIFICATION TEST")
    print("=" * 50)

    pairs_shown = 0

    for i in range(len(chunks) - 1):

        if pairs_shown >= num_pairs:
            break

        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]

        if chunk_a.metadata.get("source") != chunk_b.metadata.get("source"):
            continue

        text_a = chunk_a.page_content
        text_b = chunk_b.page_content

        overlap_found = False
        overlap_text = ""
        overlap_length = 0

        max_length = min(len(text_a), len(text_b), 200)

        for length in range(max_length, 20, -1):
            end_of_a = text_a[-length:]
            start_of_b = text_b[:length]

            if end_of_a == start_of_b:
                overlap_found = True
                overlap_text = start_of_b
                overlap_length = length
                break

        print(f"\nPair {pairs_shown + 1}:")
        print(f"Chunk {i} end:")
        print(f"...{text_a[-100:]}")
        print(f"\nChunk {i + 1} start:")
        print(f"{text_b[:100]}...")

        if overlap_found:
            print("\nOverlap detected: YES")
            print(f"Overlap length: {overlap_length} characters")
            print(f"Shared content:")
            print(f'"{overlap_text}"')
        else:
            print("\nOverlap detected: NO")
            print("Consider increasing chunk_overlap setting")

        pairs_shown += 1


def recommend_overlap(document_type: str) -> dict:
    """
    Returns recommended chunk_size and overlap for each document type.
    Use these as starting settings then adjust based on RAGAS scores.
    """

    recommendations = {
        "policy":  {"chunk_size": 600, "overlap": 100},
        "faq":     {"chunk_size": 400, "overlap": 50},
        "charges": {"chunk_size": 300, "overlap": 30},
        "legal":   {"chunk_size": 800, "overlap": 150},
        "general": {"chunk_size": 500, "overlap": 50},
    }

    settings = recommendations.get(document_type, recommendations["general"])
    print(f"Recommended for {document_type}: {settings}")
    return settings


# ── MAIN TEST BLOCK ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    from loader import load_all_documents
    from cleaner import clean_documents

    # 1. Load and clean
    documents = load_all_documents("data/documents/")
    documents = clean_documents(documents)

    print("\n" + "=" * 50)
    print("COMPARING CHUNKING STRATEGIES")
    print("=" * 50)

    # 2. All strategies
    fixed_chunks     = fixed_size_chunking(documents, chunk_size=500, chunk_overlap=50)
    sentence_chunks  = sentence_chunking(documents, sentences_per_chunk=5, overlap_sentences=1)
    paragraph_chunks = paragraph_chunking(documents, min_length=100, max_length=1000)
    recursive_chunks = recursive_chunking(documents, chunk_size=500, chunk_overlap=50)
    semantic_chunks  = semantic_chunking(documents, breakpoint_type="percentile", breakpoint_threshold=95)

    # 3. Results summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Fixed size: {len(fixed_chunks)} chunks")
    print(f"Sentence:   {len(sentence_chunks)} chunks")
    print(f"Paragraph:  {len(paragraph_chunks)} chunks")
    print(f"Recursive:  {len(recursive_chunks)} chunks")
    print(f"Semantic:   {len(semantic_chunks)} chunks")

    # 4. Sample chunks
    print("\n" + "=" * 50)
    print("SAMPLE CHUNKS")
    print("=" * 50)

    strategies = [
        ("FIXED",     fixed_chunks),
        ("SENTENCE",  sentence_chunks),
        ("PARAGRAPH", paragraph_chunks),
        ("RECURSIVE", recursive_chunks),
        ("SEMANTIC",  semantic_chunks),
    ]

    for strategy, chunk_list in strategies:
        print(f"\n--- {strategy} ---")
        for i, chunk in enumerate(chunk_list[:2]):
            print(f"\nChunk {i}")
            print(f"Source:   {chunk.metadata.get('source', 'unknown')}")
            print(f"Page:     {chunk.metadata.get('page', 'unknown')}")
            print(f"Size:     {len(chunk.page_content)} chars")
            print(f"Strategy: {chunk.metadata.get('chunking_strategy', 'unknown')}")
            print(f"Content:\n{chunk.page_content[:500]}")

    # ── TOPIC 14 — Semantic vs Paragraph comparison ──────────────────────────

    print("\n" + "=" * 50)
    print("TOPIC 14 — SEMANTIC CHUNKING COMPARISON")
    print("=" * 50)

    loan_paragraph_chunks = [
        chunk for chunk in paragraph_chunks
        if "loan_policy.pdf" in chunk.metadata.get("source", "")
    ]
    loan_semantic_chunks = [
        chunk for chunk in semantic_chunks
        if "loan_policy.pdf" in chunk.metadata.get("source", "")
    ]

    print(f"\nLoan Policy Comparison")
    print(f"Paragraph chunks: {len(loan_paragraph_chunks)}")
    print(f"Semantic chunks:  {len(loan_semantic_chunks)}")

    print("\n" + "-" * 50)
    print("PARAGRAPH CHUNKS — LOAN POLICY")
    print("-" * 50)
    for i, chunk in enumerate(loan_paragraph_chunks):
        print(f"\nChunk {i}")
        print(f"Size: {len(chunk.page_content)} chars")
        print(chunk.page_content[:1000])

    print("\n" + "-" * 50)
    print("SEMANTIC CHUNKS — LOAN POLICY")
    print("-" * 50)
    for i, chunk in enumerate(loan_semantic_chunks):
        print(f"\nChunk {i}")
        print(f"Size: {len(chunk.page_content)} chars")
        print(chunk.page_content[:1000])

    # ── TOPIC 15 — Overlap verification ──────────────────────────────────────

    print("\n" + "=" * 50)
    print("TOPIC 15 — OVERLAP VERIFICATION")
    print("=" * 50)

    settings = recommend_overlap("policy")
    chunks_with_overlap = fixed_size_chunking(
        documents,
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["overlap"]
    )
    test_overlap(chunks_with_overlap, num_pairs=3)