import os
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


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
    all_chunks: List[Document],
    k: int = 3,
    use_smart_filter: bool = True,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Document]:
    """
    Smart hybrid retrieval.

    1. Detect relevant document using Topic 25 routing.
    2. Apply metadata filtering.
    3. Run vector similarity search + BM25 keyword search.
    4. Combine both using EnsembleRetriever.
    5. Return the top k chunks.
    """

    # ========================================================
    # STEP 1 — DOCUMENT ROUTING
    # ========================================================

    if use_smart_filter:
        filter_dict = detect_relevant_document(question)
    else:
        filter_dict = None

    # ========================================================
    # STEP 2 — CREATE HYBRID RETRIEVER
    # ========================================================

    hybrid_retriever = get_hybrid_retriever(
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        filter_dict=filter_dict,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        k=k
    )

    # ========================================================
    # STEP 3 — SEARCH
    # ========================================================

    if filter_dict:
        print(f"Using filtered hybrid retrieval: {filter_dict}")
    else:
        print("Using unfiltered hybrid retrieval: searching all documents")

    docs = hybrid_retriever.invoke(question)

    # Ensemble retrieves more candidates internally.
    # Return only the requested final k results.
    docs = docs[:k]

    print(f"Retrieved {len(docs)} chunks using hybrid search")

    return docs
def get_hybrid_retriever(
    vectorstore: VectorStore,
    all_chunks: List[Document],
    filter_dict: Optional[dict] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    k: int = 3
):
    """
    Hybrid retriever combining:
    - Vector similarity search
    - BM25 keyword search

    If filter_dict is provided, both retrievers
    search only the filtered documents.
    """

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    search_kwargs = {
        "k": k * 2
    }

    if filter_dict:
        search_kwargs["filter"] = filter_dict

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # ========================================================
    # BM25 KEYWORD SEARCH
    # ========================================================

    keyword_chunks = all_chunks

    if filter_dict:
        keyword_chunks = [
            chunk
            for chunk in all_chunks
            if all(
                chunk.metadata.get(key) == value
                for key, value in filter_dict.items()
            )
        ]

    keyword_retriever = BM25Retriever.from_documents(
        keyword_chunks
    )

    keyword_retriever.k = k * 2

    # ========================================================
    # ENSEMBLE / HYBRID SEARCH
    # ========================================================

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            keyword_retriever
        ],
        weights=[
            vector_weight,
            keyword_weight
        ]
    )

    return hybrid_retriever

