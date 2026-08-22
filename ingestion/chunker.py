# ingestion/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

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
    
    # add chunk index to metadata for tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["chunking_strategy"] = "fixed_size"
    
    print(f"Fixed size chunking: {len(documents)} docs → {len(chunks)} chunks")
    print(f"Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    return chunks


# # add to ingestion/chunker.py

import re

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
        # split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, doc.page_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
        
        # group sentences into chunks
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
            
            # move forward minus overlap
            i += sentences_per_chunk - overlap_sentences
    
    print(f"Sentence chunking: {len(documents)} docs → {len(chunks)} chunks")
    print(f"Settings: {sentences_per_chunk} sentences per chunk, {overlap_sentences} overlap")
    return chunks


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
        # split on double newlines — natural paragraph breaks
        paragraphs = doc.page_content.split("\n\n")
        
        chunk_index = 0
        for para in paragraphs:
            para = para.strip()
            
            # skip paragraphs that are too short — likely headers
            if len(para) < min_length:
                continue
            
            # if paragraph is too long split it further
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


def select_chunking_strategy(documents: List[Document],
                              document_type: str = "auto") -> List[Document]:
    """
    Automatically select best chunking strategy per document type.
    
    document_type options:
    - auto: detect from content
    - policy: paragraph chunking
    - faq: paragraph chunking  
    - charges: fixed size
    - mixed: recursive
    """
    
    if document_type == "auto":
        # detect based on content of first document
        sample = documents[0].page_content if documents else ""
        
        # count paragraph breaks vs continuous text
        paragraph_breaks = sample.count("\n\n")
        
        if paragraph_breaks > 5:
            document_type = "policy"
        else:
            document_type = "mixed"
    
    if document_type in ["policy", "faq"]:
        print(f"Strategy selected: paragraph (document_type={document_type})")
        return paragraph_chunking(documents, min_length=100, max_length=800)
    
    elif document_type == "charges":
        print(f"Strategy selected: fixed size (document_type={document_type})")
        return fixed_size_chunking(documents, chunk_size=300, chunk_overlap=30)
    
    else:
        print(f"Strategy selected: recursive (document_type={document_type})")
        return recursive_chunking(documents, chunk_size=500, chunk_overlap=50)

if __name__ == "__main__":

    from loader import load_all_documents
    from cleaner import clean_documents

    # 1. Load documents
    documents = load_all_documents("data/documents/")

    # 2. Clean documents
    documents = clean_documents(documents)

    print("\n" + "=" * 50)
    print("COMPARING CHUNKING STRATEGIES")
    print("=" * 50)

    # 3. Fixed-size chunking
    fixed_chunks = fixed_size_chunking(
        documents,
        chunk_size=500,
        chunk_overlap=50
    )

    # 4. Sentence chunking
    sentence_chunks = sentence_chunking(
        documents,
        sentences_per_chunk=5,
        overlap_sentences=1
    )

    # 5. Paragraph chunking
    paragraph_chunks = paragraph_chunking(
        documents,
        min_length=100,
        max_length=1000
    )

    # 6. Recursive character chunking
    recursive_chunks = recursive_chunking(
        documents,
        chunk_size=500,
        chunk_overlap=50
    )

    # 7. Compare total number of chunks
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    print(f"Fixed size: {len(fixed_chunks)} chunks")
    print(f"Sentence:   {len(sentence_chunks)} chunks")
    print(f"Paragraph:  {len(paragraph_chunks)} chunks")
    print(f"Recursive:  {len(recursive_chunks)} chunks")

    # 8. Show sample chunks from each strategy
    print("\n" + "=" * 50)
    print("SAMPLE CHUNKS")
    print("=" * 50)

    strategies = [
        ("FIXED", fixed_chunks),
        ("SENTENCE", sentence_chunks),
        ("PARAGRAPH", paragraph_chunks),
        ("RECURSIVE", recursive_chunks)
    ]

    for strategy, chunk_list in strategies:

        print(f"\n--- {strategy} ---")

        for i, chunk in enumerate(chunk_list[:2]):

            print(f"\nChunk {i}")

            print(
                f"Source: "
                f"{chunk.metadata.get('source', 'unknown')}"
            )

            print(
                f"Page: "
                f"{chunk.metadata.get('page', 'unknown')}"
            )

            print(
                f"Size: "
                f"{len(chunk.page_content)} chars"
            )

            print(
                f"Strategy: "
                f"{chunk.metadata.get('chunking_strategy', 'unknown')}"
            )

            print(
                f"Content:\n"
                f"{chunk.page_content[:500]}"
            )