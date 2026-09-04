from functools import lru_cache

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def get_reranker():
    print("\nLoading BGE reranker model...")
    return CrossEncoder(MODEL_NAME)


def rerank_documents(
    question: str,
    documents: list[Document],
    top_k: int = 3
) -> list[Document]:

    if not documents:
        return []

    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    reranker = get_reranker()

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        document
        for _, document in ranked[:top_k]
    ]