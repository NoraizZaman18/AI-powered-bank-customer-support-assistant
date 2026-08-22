
# ingestion/cleaner.py

import re
from langchain_core.documents import Document

def clean_text(text):
    """
    Clean raw extracted text before chunking.
    Remove noise that pollutes embeddings.
    """
    
    # remove excessive newlines — max 2 in a row
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    
    # remove page numbers — standalone numbers on their own line
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # remove common page number formats
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    
    # remove repeated header/footer patterns
    # if same short line appears many times it is likely a header
    lines = text.split('\n')
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 5:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    # remove lines that appear more than 3 times — likely headers
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if line_counts.get(stripped, 0) <= 3:
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # normalize quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
 
   
    
    # final whitespace cleanup
    text = text.strip()
    
    return text

def clean_document(document):
    """
    Clean a single LangChain Document object.
    Returns cleaned document with same metadata.
    """
    cleaned_text = clean_text(document.page_content)
    
    return Document(
        page_content=cleaned_text,
        metadata=document.metadata
    )

def clean_documents(documents, min_length=50):
    """
    Clean a list of documents.
    Removes documents that are too short after cleaning.
    min_length: minimum characters to keep a document
    """
    cleaned = []
    removed = 0
    
    for doc in documents:
        cleaned_doc = clean_document(doc)
        
        if len(cleaned_doc.page_content.strip()) >= min_length:
            cleaned.append(cleaned_doc)
        else:
            removed += 1
    
    print(f"Cleaned {len(cleaned)} documents")
    print(f"Removed {removed} documents that were too short after cleaning")
    return cleaned

def clean_ocr_text(text):
    """
    Extra cleaning specifically for OCR extracted text.
    """

    # Common OCR substitution
    text = text.replace('|', 'I')

    # Apply normal cleaning
    return clean_text(text)


