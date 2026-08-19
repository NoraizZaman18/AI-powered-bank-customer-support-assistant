import fitz  # PyMuPDF
import os

from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document

from langchain_community.document_loaders import WebBaseLoader
#  web page loader
def load_webpage(url):
    """
    Load content from a single web page URL.
    """
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = url
            doc.metadata["file_type"] = "webpage"
        
        print(f"Loaded webpage: {url}")
        return documents
        
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

def load_multiple_urls(urls):
    """
    Load content from multiple URLs at once.
    """
    all_documents = []
    
    for url in urls:
        docs = load_webpage(url)
        all_documents.extend(docs)
    
    print(f"Loaded {len(all_documents)} pages from {len(urls)} URLs")
    return all_documents

# pdf loader 

def load_single_pdf(pdf_path):
    """Load one PDF file and return its pages."""

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "file_path": pdf_path,
                        "file_type": "pdf"
                    }
                )
            )

    doc.close()

    print(
        f"Loaded {len(pages)} pages from "
        f"{os.path.basename(pdf_path)}"
    )

    return pages

#  doc loader 
def load_docx(file_path):
    """Load one Word document."""

    loader = Docx2txtLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "docx"

    print(f"Loaded DOCX: {os.path.basename(file_path)}")

    return documents

# txt files loader
def load_txt(file_path):
    """Load one plain text file."""

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "txt"

    print(f"Loaded TXT: {os.path.basename(file_path)}")

    return documents


#  check which funcrtion we need to run accoind to the file types

def load_document(file_path):
    """
    Detect the file type and use the correct loader.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return load_single_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    elif extension == ".txt":
        return load_txt(file_path)

    else:
        print(f"Unsupported file type: {file_path}")
        return []


def load_all_documents(folder_path, urls=None):
    all_documents = []
    supported = (".pdf", ".docx", ".txt")
    
    # load files from folder
    files = [f for f in os.listdir(folder_path) if f.endswith(supported)]
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        docs = load_document(file_path)
        all_documents.extend(docs)
    
    # load web pages if provided
    if urls:
        web_docs = load_multiple_urls(urls)
        all_documents.extend(web_docs)
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents



#  main 
if __name__ == "__main__":

    test_urls = [
        "https://www.sbp.org.pk/bsd/2023/C3.htm"
    ]

    documents = load_all_documents(
        "data/documents/",
        urls=test_urls
    )

    print("\nFirst document:")
    if documents:
        print(documents[0].page_content[:300])