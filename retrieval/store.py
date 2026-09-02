import hashlib
import json
import os
from datetime import datetime
from langchain_community.vectorstores import Chroma
from retrieval.embedder import get_local_embeddings
from langchain_core.documents import Document
from typing import List

# path to track which files have been indexed
INDEX_TRACKER_PATH = "data/index_tracker.json"
CHROMA_DB_PATH = "./chroma_db"

COLLECTION_NAME = "bank_documents"

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_local_embeddings(),
        collection_name="bank_documents"
    )

def get_file_hash(file_path: str) -> str:
    """
    Generate MD5 hash of a file.
    Use this to detect if a file has changed since last indexing.
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
def load_index_tracker() -> dict:
    """
    Load the tracker that records which files are indexed.
    Returns empty dict if no tracker exists yet.
    """
    if os.path.exists(INDEX_TRACKER_PATH):
        with open(INDEX_TRACKER_PATH, "r") as f:
            return json.load(f)
    return {}
def save_index_tracker(tracker: dict):
    """Save the updated tracker to disk."""
    os.makedirs(os.path.dirname(INDEX_TRACKER_PATH), exist_ok=True)
    with open(INDEX_TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2)
def is_document_indexed(file_path: str, tracker: dict) -> bool:
    """
    Check if a document is already indexed and unchanged.
    Returns True if already indexed with same content.
    """
    file_name = os.path.basename(file_path)
    
    if file_name not in tracker:
        return False
    
    current_hash = get_file_hash(file_path)
    stored_hash = tracker[file_name].get("hash")
    
    return current_hash == stored_hash
def add_documents_to_store(chunks: List[Document]) -> Chroma:
    """
    Add chunks to ChromaDB vector store.
    Returns the vector store for use in retrieval.
    """
    vectorstore = get_vectorstore()
    
    if chunks:
        vectorstore.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to vector store")
    
    return vectorstore
def delete_document_from_store(source_name: str):
    """
    Delete all chunks from a specific document.
    Call this before re-indexing an updated document.
    """
    vectorstore = get_vectorstore()
    
    vectorstore._collection.delete(
        where={"source": source_name}
    )
    
    print(f"Deleted all chunks for: {source_name}")
def incremental_index(
    folder_path: str,
    chunk_function,
    force_reindex: bool = False
) -> Chroma:
    """
    Main incremental indexing function.
    
    Only processes documents that are new or changed.
    Skips documents already in the vector store.
    
    folder_path: folder containing your PDF files
    chunk_function: your chunking function from chunker.py
    force_reindex: set True to reindex everything from scratch
    
    Returns: vectorstore ready for retrieval
    """
    
    from ingestion.loader import load_document
    from ingestion.cleaner import clean_documents
    
    tracker = {} if force_reindex else load_index_tracker()
    
    supported = (".pdf", ".docx", ".txt")
    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(supported)
    ]
    
    new_files = []
    skipped_files = []
    updated_files = []
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        if force_reindex:
            new_files.append(file_path)
        elif is_document_indexed(file_path, tracker):
            skipped_files.append(filename)
        elif filename in tracker:
            # file exists in tracker but hash changed — it was updated
            updated_files.append(file_path)
            delete_document_from_store(filename)
        else:
            new_files.append(file_path)
    
    print(f"\nIncremental indexing summary:")
    print(f"  New documents:     {len(new_files)}")
    print(f"  Updated documents: {len(updated_files)}")
    print(f"  Skipped (unchanged): {len(skipped_files)}")
    
    if skipped_files:
        print(f"  Skipped: {skipped_files}")
    
    # process new and updated documents
    files_to_process = new_files + updated_files
    
    if files_to_process:
        all_chunks = []
        
        for file_path in files_to_process:
            docs = load_document(file_path)
            docs = clean_documents(docs)
            chunks = chunk_function(docs)
            all_chunks.extend(chunks)
            
            # update tracker
            filename = os.path.basename(file_path)
            tracker[filename] = {
                "hash": get_file_hash(file_path),
                "indexed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks)
            }
        
        all_chunks = full_deduplication(all_chunks)
        vectorstore = add_documents_to_store(all_chunks)
        save_index_tracker(tracker)
        print(f"\nTotal new chunks added: {len(all_chunks)}")
    
    else:
        print("\nAll documents already indexed — loading existing store")
        vectorstore = get_vectorstore()
    
    total_docs = vectorstore._collection.count()
    print(f"Total chunks in vector store: {total_docs}")
    
    return vectorstore



def deduplicate_exact(chunks: List[Document]) -> List[Document]:
    """
    Remove chunks with identical text content.
    Fast — uses MD5 hash comparison.
    Use this always before storing chunks.
    """
    
    seen_hashes = set()
    unique_chunks = []
    duplicates_removed = 0
    
    for chunk in chunks:
        content_hash = hashlib.md5(
            chunk.page_content.strip().encode()
        ).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
        else:
            duplicates_removed += 1
    
    print(f"Exact deduplication: {len(chunks)} → {len(unique_chunks)} chunks")
    print(f"Removed {duplicates_removed} exact duplicates")
    return unique_chunks

def deduplicate_near_duplicates(chunks: List[Document],
                                similarity_threshold: float = 0.95) -> List[Document]:
    """
    Remove chunks that are very similar even if not identical.
    Slower — computes embeddings and compares similarity.
    Use when documents have paraphrased repeated content.
    
    similarity_threshold: 
    - 0.99 = only remove near-identical text
    - 0.95 = remove very similar content (recommended)
    - 0.90 = more aggressive — removes somewhat similar content
    """
    
    import numpy as np
    
    if len(chunks) < 2:
        return chunks
    
    print(f"Near-duplicate detection on {len(chunks)} chunks...")
    print("Computing embeddings for deduplication...")
    
    embeddings_model = get_local_embeddings()
    texts = [c.page_content for c in chunks]
    
    # embed all chunks
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = np.array(embeddings)
    
    # normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    
    # find near duplicates
    keep = [True] * len(chunks)
    
    for i in range(len(chunks)):
        if not keep[i]:
            continue
        
        for j in range(i + 1, len(chunks)):
            if not keep[j]:
                continue
            
            similarity = float(
                np.dot(embeddings_normalized[i], embeddings_normalized[j])
            )
            
            if similarity >= similarity_threshold:
                # keep the chunk with more content
                if len(chunks[i].page_content) >= len(chunks[j].page_content):
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    unique_chunks = [c for c, k in zip(chunks, keep) if k]
    removed = len(chunks) - len(unique_chunks)
    
    print(f"Near-duplicate deduplication: {len(chunks)} → {len(unique_chunks)} chunks")
    print(f"Removed {removed} near-duplicate chunks (threshold={similarity_threshold})")
    return unique_chunks


def full_deduplication(chunks: List[Document],
                       near_duplicate_threshold: float = 0.95) -> List[Document]:
    """
    Run both exact and near-duplicate deduplication.
    This is what you call in your production pipeline.
    """
    
    # step 1 — remove exact duplicates first (fast)
    chunks = deduplicate_exact(chunks)
    
    # step 2 — remove near duplicates (slower but more thorough)
    # only run if you have many chunks — computationally expensive
    if len(chunks) > 10:
        chunks = deduplicate_near_duplicates(chunks, near_duplicate_threshold)
    
    return chunks
