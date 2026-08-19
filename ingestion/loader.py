# ingestion/loader.py

import fitz  # PyMuPDF
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader
)
from langchain_core.documents import Document

def load_single_pdf(pdf_path):
    """
    Load one PDF file and return list of pages with metadata.
    Use PyMuPDF for better text extraction especially for
    financial documents with tables and special formatting.
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if text.strip():  # skip empty pages
            pages.append(Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page": page_num + 1,
                    "total_pages": len(doc),
                    "file_path": pdf_path
                }
            ))
    
    doc.close()
    print(f"Loaded {len(pages)} pages from {os.path.basename(pdf_path)}")
    return pages

def load_all_pdfs(folder_path):
    """
    Load all PDF files from a folder.
    Returns all pages from all documents as one list.
    """
    all_pages = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pages = load_single_pdf(pdf_path)
        all_pages.extend(pages)
    
    print(f"\nTotal: {len(all_pages)} pages from {len(pdf_files)} documents")
    print(f"Documents loaded: {pdf_files}")
    return all_pages

# test your loader
if __name__ == "__main__":
    pages = load_all_pdfs("data/documents/")
    
  
