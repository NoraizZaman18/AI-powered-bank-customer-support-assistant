import os
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

def get_basic_retriever(
    vectorstore: VectorStore,
    k: int = 3
):
    """
    Basic retriever without filtering.
    Searches all documents.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_filtered_retriever(
    vectorstore: VectorStore,
    filter_dict: dict,
    k: int = 3
):
    """
    Retriever that only searches documents matching the filter.
    
    filter_dict examples:
    - {"source": "loan_policy.pdf"} — only loan policy
    - {"section": "charges"} — only charge-related chunks
    - {"page": 3} — only page 3
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": filter_dict
        }
    )


def detect_relevant_document(question: str) -> Optional[dict]:
    """
    Detect which document is most relevant for a question.
    Returns metadata filter to use in retrieval.
    Returns None if question could be in any document.
    """
    
    question_lower = question.lower()
    
    # document routing rules
    document_keywords = {
        "loan_policy.pdf": [
            "loan", "borrow", "finance", "repay", "installment",
            "emi", "tenure", "interest rate", "personal loan",
            "business loan", "collateral"
        ],
        "account_opening_policy.pdf": [
    "open account",
    "account opening",
    "savings account",
    "current account",
    "minimum balance",
    "profit rate",
    "zakat",
    "withholding tax",
    "dormant",
    "document",
    "documents",
    "required document",
    "required documents",
    "cnic",
    "proof of address"
        ],
        "credit_card_policy.pdf": [
            "credit card", "card limit", "annual fee", "reward points",
            "cash advance", "markup", "minimum payment", "statement"
        ],
        "schedule_of_charges.pdf": [
            "charge", "fee", "cost", "how much", "rate",
            "atm fee", "transfer fee", "cheque", "locker"
        ],
        "customer_faq.pdf": [
                    "helpline",
                     "mobile app",
                       "online banking",
                   "bank timing",
                 "branch timing"
              ],
        "complaint_handling_policy.pdf": [
            "complaint", "escalate", "resolve", "unhappy",
            "dissatisfied", "banking mohtasib", "sbp complaint"
        ]
    }
    
    # score each document
    scores = {}
    for doc_name, keywords in document_keywords.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            scores[doc_name] = score
    
    if not scores:
        return None  # no specific document detected
    
    # return filter for highest scoring document
    best_doc = max(scores, key=scores.get)
    print(f"Detected relevant document: {best_doc} (score: {scores[best_doc]})")
    return {"source": best_doc}


def smart_retrieve(
    question: str,
    vectorstore: VectorStore,
    k: int = 3,
    use_smart_filter: bool = True
) -> List[Document]:
    """
    Retrieve with automatic document detection.
    If a specific document is detected use filtered retrieval.
    Otherwise search all documents.
    """
    
    if use_smart_filter:
        filter_dict = detect_relevant_document(question)
    else:
        filter_dict = None
    
    if filter_dict:
        retriever = get_filtered_retriever(vectorstore, filter_dict, k)
        print(f"Using filtered retrieval: {filter_dict}")
    else:
        retriever = get_basic_retriever(vectorstore, k)
        print("Using unfiltered retrieval: searching all documents")
    
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} chunks")
    return docs



if __name__ == "__main__":
    from retrieval.store import get_vectorstore
    
    vectorstore = get_vectorstore()
    
    test_questions = [
        "What is the interest rate for personal loan?",
        "How do I file a complaint with the bank?",
        "What is the annual fee for credit card?",
        "What documents do I need to open an account?"
    ]
    
    print("METADATA FILTERING TEST")
    print("="*50)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        docs = smart_retrieve(question, vectorstore)
        for i, doc in enumerate(docs):
            print(f"  Result {i+1}: {doc.metadata.get('source')} "
                  f"page {doc.metadata.get('page')} — "
                  f"{doc.page_content[:80]}...")