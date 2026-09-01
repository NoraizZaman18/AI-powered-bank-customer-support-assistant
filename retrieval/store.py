import hashlib
import json
import os
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# path to track which files have been indexed
INDEX_TRACKER_PATH = "data/index_tracker.json"
CHROMA_DB_PATH = "./chroma_db"

COLLECTION_NAME = "bank_documents"
def get_embeddings():
    """
    Returns embedding model.
    Using local HuggingFace model — completely free.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
def get_vectorstore():
    """
    Returns ChromaDB vector store.
    Creates it if it does not exist.
    """
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
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
        
        vectorstore = add_documents_to_store(all_chunks)
        save_index_tracker(tracker)
        print(f"\nTotal new chunks added: {len(all_chunks)}")
    
    else:
        print("\nAll documents already indexed — loading existing store")
        vectorstore = get_vectorstore()
    
    total_docs = vectorstore._collection.count()
    print(f"Total chunks in vector store: {total_docs}")
    
    return vectorstore