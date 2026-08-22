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


# # test chunker directly
# if __name__ == "__main__":
#     from loader import load_all_documents
#     from cleaner import clean_documents
    
#     # load and clean
#     documents = load_all_documents("data/documents/")
#     documents = clean_documents(documents)
    
#     # chunk
#     chunks = fixed_size_chunking(documents, chunk_size=500, chunk_overlap=50)
    
#     # inspect results
#     print(f"\nTotal chunks: {len(chunks)}")
#     print(f"\nFirst chunk:")
#     print(f"  Source: {chunks[0].metadata['source']}")
#     print(f"  Page: {chunks[0].metadata['page']}")
#     print(f"  Size: {chunks[0].metadata['chunk_size']} chars")
#     print(f"  Content: {chunks[0].page_content[:200]}")
    
#     print(f"\nLast chunk:")
#     print(f"  Source: {chunks[-1].metadata['source']}")
#     print(f"  Content: {chunks[-1].page_content[:200]}")