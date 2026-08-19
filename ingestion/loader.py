import fitz  # PyMuPDF
import os

from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document


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


def load_docx(file_path):
    """Load one Word document."""

    loader = Docx2txtLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_type"] = "docx"

    print(f"Loaded DOCX: {os.path.basename(file_path)}")

    return documents


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


def load_all_documents(folder_path):
    """
    Load all supported documents from a folder.
    """

    all_documents = []

    supported = (
        ".pdf",
        ".docx",
        ".txt"
    )

    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported)
    ]

    if not files:
        print(f"No supported documents found in {folder_path}")
        return []

    for filename in files:

        file_path = os.path.join(
            folder_path,
            filename
        )

        documents = load_document(file_path)

        all_documents.extend(documents)

    print(
        f"\nTotal loaded pages/documents: "
        f"{len(all_documents)}"
    )

    return all_documents


# Test the loader
if __name__ == "__main__":

    documents = load_all_documents(
        "data/documents/"
    )

    print("\nFirst 200 characters:")

    if documents:
        print(documents[0].page_content[:200])