import fitz  # PyMuPDF
import os
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
from pdf2image import convert_from_path



from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document

from langchain_community.document_loaders import WebBaseLoader
#  chek the pdf is scanned or not 
def is_scanned_pdf(pdf_path):
    """
    Detect if a PDF is scanned by checking if it has selectable text.
    Returns True if scanned, False if normal PDF.
    """
    doc = fitz.open(pdf_path)
    first_page_text = doc[0].get_text().strip()
    doc.close()
    
    # if very little text on first page it is likely scanned
    return len(first_page_text) < 100

def load_scanned_pdf(pdf_path):
    """
    Load a scanned PDF using OCR.
    Converts each page to image then extracts text.
    """
    print(f"Running OCR on: {os.path.basename(pdf_path)}")
    
    # convert PDF pages to images
    pages_as_images = convert_from_path(
    pdf_path,
    dpi=300,
    poppler_path=r"C:\Program Files\poppler-26.02.0\Library\bin"
    ) 
    
    documents = []
    for i, page_image in enumerate(pages_as_images):
        # run OCR on each page image
        text = pytesseract.image_to_string(page_image, lang="eng")
        
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page": i + 1,
                    "extraction_method": "ocr",
                    "file_type": "scanned_pdf"
                }
            ))
        
        print(f"  OCR page {i+1} of {len(pages_as_images)}")
    
    return documents

def smart_pdf_loader(pdf_path):
    """
    Automatically detects if PDF is scanned or normal.
    Uses appropriate loader for each type.
    This is the function you call for any PDF.
    """
    if is_scanned_pdf(pdf_path):
        print(f"Scanned PDF detected: {os.path.basename(pdf_path)}")
        return load_scanned_pdf(pdf_path)
    else:
        print(f"Normal PDF detected: {os.path.basename(pdf_path)}")
        return load_single_pdf(pdf_path) 
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
      return smart_pdf_loader(file_path)

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
# if __name__ == "__main__":

#     test_urls = [
#         "https://www.sbp.org.pk/bsd/2023/C3.htm"
#     ]

#     documents = load_all_documents(
#         "data/documents/",
#         urls=test_urls
#     )


if __name__ == "__main__":

    test_urls = [
        "https://www.sbp.org.pk/bsd/2023/C3.htm"
    ]

    documents = load_all_documents(
        "data/documents/",
        urls=test_urls
    )

    print("\nTotal documents:", len(documents))

    for doc in documents:
        if doc.metadata.get("extraction_method") == "ocr":
            print("\n===== OCR RESULT =====")
            print(doc.page_content[:1000])