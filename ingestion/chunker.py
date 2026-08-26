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

def detect_document_types(documents: List[Document]) -> dict:
    """
    Detect the type of each PDF separately.
    """

    # Group pages by PDF source
    grouped_documents = {}

    for doc in documents:
        source = doc.metadata.get("source", "unknown")

        if source not in grouped_documents:
            grouped_documents[source] = []

        grouped_documents[source].append(doc)

    results = {}

    for source, pages in grouped_documents.items():

        # Combine all pages belonging to this PDF
        content = " ".join(
            page.page_content.lower()
            for page in pages
        )

        filename = source.lower()

        # --------------------------------------------------
        # 1. Strong filename/document-name signals
        # --------------------------------------------------

        if "faq" in filename:
            detected = "faq"

        elif "schedule_of_charges" in filename:
            detected = "charges"

        elif (
            "policy" in filename
            or "loan" in filename
            or "complaint" in filename
            or "account_opening" in filename
            or "credit_card" in filename
        ):
            detected = "policy"

        else:
            # --------------------------------------------------
            # 2. Content-based detection for unknown documents
            # --------------------------------------------------

            faq_keywords = [
                "frequently asked",
                "question:",
                "answer:",
                "q:",
                "a:"
            ]

            charges_keywords = [
                "schedule of charges",
                "per transaction",
                "fee:",
                "charge:",
                "service charge"
            ]

            policy_keywords = [
                "policy",
                "section",
                "eligibility",
                "criteria",
                "requirements",
                "terms and conditions"
            ]

            faq_score = sum(
                1 for keyword in faq_keywords
                if keyword in content
            )

            charges_score = sum(
                1 for keyword in charges_keywords
                if keyword in content
            )

            policy_score = sum(
                1 for keyword in policy_keywords
                if keyword in content
            )

            scores = {
                "faq": faq_score,
                "charges": charges_score,
                "policy": policy_score
            }

            if max(scores.values()) == 0:
                detected = "general"
            else:
                detected = max(scores, key=scores.get)

        results[source] = detected

        print(f"{source} → {detected}")

    return results

def analyse_chunk_quality(chunks: List[Document]):
    """
    Show statistics about your chunks.
    Run this after chunking to understand chunk distribution.
    Helps you decide if chunk size needs adjustment.
    """
    
    sizes = [len(c.page_content) for c in chunks]
    
    print("\n" + "="*50)
    print("CHUNK QUALITY ANALYSIS")
    print("="*50)
    print(f"Total chunks:     {len(chunks)}")
    print(f"Average size:     {sum(sizes) // len(sizes)} chars")
    print(f"Smallest chunk:   {min(sizes)} chars")
    print(f"Largest chunk:    {max(sizes)} chars")
    
    # distribution
    small = sum(1 for s in sizes if s < 200)
    medium = sum(1 for s in sizes if 200 <= s < 600)
    large = sum(1 for s in sizes if s >= 600)
    
    print(f"\nSize distribution:")
    print(f"  Small (<200):    {small} chunks ({small*100//len(chunks)}%)")
    print(f"  Medium (200-600):{medium} chunks ({medium*100//len(chunks)}%)")
    print(f"  Large (>600):    {large} chunks ({large*100//len(chunks)}%)")
    
    # per document stats
    print(f"\nChunks per document:")
    doc_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        doc_counts[source] = doc_counts.get(source, 0) + 1
    
    for source, count in sorted(doc_counts.items()):
        print(f"  {source}: {count} chunks")
    
    # quality warnings
    print(f"\nQuality warnings:")
    if small > len(chunks) * 0.2:
        print(f"  WARNING: {small} chunks are very small (<200 chars)")
        print(f"  Consider increasing chunk_size or min_length filter")
    if large > len(chunks) * 0.3:
        print(f"  WARNING: {large} chunks are very large (>600 chars)")
        print(f"  Consider decreasing chunk_size")
    if small <= len(chunks) * 0.2 and large <= len(chunks) * 0.3:
        print(f"  OK: Chunk size distribution looks healthy")


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

# add to ingestion/chunker.py

def enrich_metadata(chunks: List[Document],
                    additional_metadata: dict = None) -> List[Document]:
    """
    Add rich metadata to every chunk.
    Call this after any chunking function.
    
    Adds:
    - document_id: unique identifier for source document
    - word_count: number of words in chunk
    - has_numbers: whether chunk contains numerical data
    - section_hint: likely section based on content keywords
    - ingestion_date: when document was processed
    """
    
    from datetime import datetime
    import hashlib
    
    enriched = []
    
    for chunk in chunks:
        text = chunk.page_content
        source = chunk.metadata.get("source", "unknown")
        
        # generate unique ID for this chunk
        chunk_id = hashlib.md5(
            f"{source}_{text[:50]}".encode()
        ).hexdigest()[:12]
        
        # detect section from content
        section = detect_section(text)
        
        # check if chunk contains numerical data
        has_numbers = bool(re.search(r'\d+', text))
        
        # build enriched metadata
        enriched_metadata = {
            **chunk.metadata,
            "chunk_id": chunk_id,
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_numbers": has_numbers,
            "section": section,
            "ingestion_date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        # add any extra metadata passed in
        if additional_metadata:
            enriched_metadata.update(additional_metadata)
        
        enriched.append(Document(
            page_content=text,
            metadata=enriched_metadata
        ))
    
    return enriched


def detect_section(text: str) -> str:
    """
    Detect the likely section of a bank document from content.
    Used for metadata enrichment.
    """
    
    text_lower = text.lower()
    
    section_keywords = {
        "eligibility": ["eligible", "eligibility", "qualify", "requirement"],
        "charges": ["rs.", "fee", "charge", "penalty", "cost"],
        "documents": ["cnic", "document", "required", "submit"],
        "process": ["apply", "process", "step", "procedure", "how to"],
        "policy": ["policy", "rule", "regulation", "terms"],
        "contact": ["phone", "email", "helpline", "contact", "branch"]
    }
    
    for section, keywords in section_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return section
    
    return "general"


# # update select_chunking_strategy to always enrich metadata
# def chunk_documents(documents: List[Document],
#                     document_type: str = "auto",
#                     extra_metadata: dict = None) -> List[Document]:
#     """
#     Main function to call for chunking.
#     Automatically:
#     1. Detects document type if not specified
#     2. Selects best chunking strategy
#     3. Enriches metadata on all chunks
    
#     This is the only chunking function you need to call from main.py
#     """
    
#     if document_type == "auto":
#         document_type = detect_document_type(documents)
    
#     settings = recommend_overlap(document_type)
    
#     if document_type in ["policy", "faq"]:
#         chunks = paragraph_chunking(documents, 100, settings["chunk_size"])
#     else:
#         chunks = recursive_chunking(
#             documents,
#             settings["chunk_size"],
#             settings["overlap"]
#         )
    
#     chunks = enrich_metadata(chunks, extra_metadata)
#     analyse_chunk_quality(chunks)
    
#     return chunks
# ── MAIN TEST BLOCK ──────────────────────────────────────────────────────────
#  this main is usefull if you want to rune all chunking function comapre chunking function the code is also availbe 
# if __name__ == "__main__":

#     from loader import load_all_documents
#     from cleaner import clean_documents

#     # 1. Load and clean
#     documents = load_all_documents("data/documents/")
#     documents = clean_documents(documents)

    # print("\n" + "=" * 50)
    # print("COMPARING CHUNKING STRATEGIES")
    # print("=" * 50)

    # 2. All strategies
    # fixed_chunks     = fixed_size_chunking(documents, chunk_size=500, chunk_overlap=50)
    # sentence_chunks  = sentence_chunking(documents, sentences_per_chunk=5, overlap_sentences=1)
    # paragraph_chunks = paragraph_chunking(documents, min_length=100, max_length=1000)
    # recursive_chunks = recursive_chunking(documents, chunk_size=500, chunk_overlap=50)
    # semantic_chunks  = semantic_chunking(documents, breakpoint_type="percentile", breakpoint_threshold=95)

    # 3. Results summary
    # print("\n" + "=" * 50)
    # print("RESULTS")
    # print("=" * 50)
    # print(f"Fixed size: {len(fixed_chunks)} chunks")
    # print(f"Sentence:   {len(sentence_chunks)} chunks")
    # print(f"Paragraph:  {len(paragraph_chunks)} chunks")
    # print(f"Recursive:  {len(recursive_chunks)} chunks")
    # print(f"Semantic:   {len(semantic_chunks)} chunks")

    # 4. Sample chunks
    # print("\n" + "=" * 50)
    # print("SAMPLE CHUNKS")
    # print("=" * 50)

    # strategies = [
    #     ("FIXED",     fixed_chunks),
    #     ("SENTENCE",  sentence_chunks),
    #     ("PARAGRAPH", paragraph_chunks),
    #     ("RECURSIVE", recursive_chunks),
    #     ("SEMANTIC",  semantic_chunks),
    # ]

    # for strategy, chunk_list in strategies:
    #     print(f"\n--- {strategy} ---")
    #     for i, chunk in enumerate(chunk_list[:2]):
    #         print(f"\nChunk {i}")
    #         print(f"Source:   {chunk.metadata.get('source', 'unknown')}")
    #         print(f"Page:     {chunk.metadata.get('page', 'unknown')}")
    #         print(f"Size:     {len(chunk.page_content)} chars")
    #         print(f"Strategy: {chunk.metadata.get('chunking_strategy', 'unknown')}")
    #         print(f"Content:\n{chunk.page_content[:500]}")

    # ── TOPIC 14 — Semantic vs Paragraph comparison ──────────────────────────

    # print("\n" + "=" * 50)
    # print("TOPIC 14 — SEMANTIC CHUNKING COMPARISON")
    # print("=" * 50)

    # loan_paragraph_chunks = [
    #     chunk for chunk in paragraph_chunks
    #     if "loan_policy.pdf" in chunk.metadata.get("source", "")
    # ]
    # loan_semantic_chunks = [
    #     chunk for chunk in semantic_chunks
    #     if "loan_policy.pdf" in chunk.metadata.get("source", "")
    # ]

    # print(f"\nLoan Policy Comparison")
    # print(f"Paragraph chunks: {len(loan_paragraph_chunks)}")
    # print(f"Semantic chunks:  {len(loan_semantic_chunks)}")

    # print("\n" + "-" * 50)
    # print("PARAGRAPH CHUNKS — LOAN POLICY")
    # print("-" * 50)
    # for i, chunk in enumerate(loan_paragraph_chunks):
    #     print(f"\nChunk {i}")
    #     print(f"Size: {len(chunk.page_content)} chars")
    #     print(chunk.page_content[:1000])

    # print("\n" + "-" * 50)
    # print("SEMANTIC CHUNKS — LOAN POLICY")
    # print("-" * 50)
    # for i, chunk in enumerate(loan_semantic_chunks):
    #     print(f"\nChunk {i}")
    #     print(f"Size: {len(chunk.page_content)} chars")
    #     print(chunk.page_content[:1000])

    # ── TOPIC 15 — Overlap verification ──────────────────────────────────────

if __name__ == "__main__":

    from loader import load_all_documents
    from cleaner import clean_documents

    # 1. Load and clean
    documents = load_all_documents("data/documents/")
    documents = clean_documents(documents)

    # ── TOPIC 15 — Overlap verification ──

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

    # ── TOPIC 16 — Document-specific chunking ──

    print("\nDetecting document types...")

    document_types = detect_document_types(documents)

    final_chunks = []

    # Process each PDF separately
    for source, doc_type in document_types.items():

        print(f"\nProcessing: {source}")
        print(f"Document type: {doc_type}")

        # Get settings for THIS document type
        settings = recommend_overlap(doc_type)

        # Get all pages belonging to this PDF
        source_documents = [
            doc
            for doc in documents
            if doc.metadata.get("source") == source
        ]

        # Chunk this PDF using its own settings
        chunks = recursive_chunking(
            source_documents,
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["overlap"]
        )

        final_chunks.extend(chunks)

    # ── Final results ──

    print("\n" + "=" * 50)
    print("FINAL CHUNKING RESULTS")
    print("=" * 50)

    print(f"Total final chunks: {len(final_chunks)}")

    analyse_chunk_quality(final_chunks)