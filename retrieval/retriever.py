
# retrieval/retriever.py

import os
import hashlib
import logging

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
    TimeWeightedVectorStoreRetriever,
)

from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

from langchain_classic.chains.query_constructor.base import AttributeInfo

from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
)

from langchain_classic.storage import InMemoryStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_transformers import LongContextReorder

from dotenv import load_dotenv

load_dotenv()

# ── LLM HELPER ───────────────────────────────────────────────────────────────

def get_llm():
    """Returns the LLM used for all retrieval operations."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


# ── TOPIC 25 — METADATA FILTERING AND DOCUMENT ROUTING ───────────────────────

def detect_relevant_document(question: str) -> Optional[dict]:
    """
    Detect which document is most relevant for a question.
    Returns metadata filter or None if question applies to all documents.
    """

    question_lower = question.lower()

    document_keywords = {
        "loan_policy.pdf": [
            "loan", "borrow", "finance", "repay", "installment",
            "emi", "tenure", "interest rate", "personal loan",
            "business loan", "collateral", "disbursement",
            "prepayment", "delinquent"
        ],
        "account_opening_policy.pdf": [
            "open account", "account opening", "savings account",
            "current account", "minimum balance", "profit rate",
            "zakat", "withholding tax", "dormant", "document",
            "documents", "required document", "required documents",
            "cnic", "proof of address", "account type"
        ],
        "credit_card_policy.pdf": [
            "credit card", "card limit", "annual fee", "reward points",
            "cash advance", "markup", "minimum payment", "statement",
            "supplementary card", "late payment", "card block"
        ],
        "schedule_of_charges.pdf": [
            "charge", "fee", "cost", "how much", "rate",
            "atm fee", "transfer fee", "cheque", "locker",
            "ibft", "raast", "pay order", "demand draft"
        ],
        "customer_faq.pdf": [
            "how do i", "how to", "helpline", "mobile app",
            "online banking", "bank timing", "branch timing",
            "activate", "reset pin", "block card"
        ],
        "complaint_handling_policy.pdf": [
            "complaint", "escalate", "resolve", "unhappy",
            "dissatisfied", "banking mohtasib", "sbp complaint",
            "grievance", "feedback"
        ]
    }

    scores = {}
    for doc_name, keywords in document_keywords.items():
        score = sum(1 for kw in keywords if kw in question_lower)
        if score > 0:
            scores[doc_name] = score

    if not scores:
        return None

    best_doc = max(scores, key=scores.get)
    print(f"Detected relevant document: {best_doc} (score: {scores[best_doc]})")
    return {"source": best_doc}


def get_basic_retriever(vectorstore: VectorStore, k: int = 3):
    """Basic retriever — searches all documents."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def get_filtered_retriever(
    vectorstore: VectorStore,
    filter_dict: dict,
    k: int = 3
):
    """Retriever that only searches documents matching the filter."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": filter_dict}
    )


# ── TOPIC 26 — HYBRID SEARCH ─────────────────────────────────────────────────

def get_hybrid_retriever(
    vectorstore: VectorStore,
    all_chunks: List[Document],
    filter_dict: Optional[dict] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    k: int = 3
):
    """
    Hybrid retriever combining vector similarity and BM25 keyword search.
    Combines both results using weighted ensemble.
    """

    # vector retriever
    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # BM25 keyword retriever
    keyword_chunks = all_chunks
    if filter_dict:
        keyword_chunks = [
            chunk for chunk in all_chunks
            if all(
                chunk.metadata.get(key) == value
                for key, value in filter_dict.items()
            )
        ]

    if not keyword_chunks:
        keyword_chunks = all_chunks

    keyword_retriever = BM25Retriever.from_documents(keyword_chunks)
    keyword_retriever.k = k

    # ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[vector_weight, keyword_weight]
    )

    return hybrid_retriever


# ── TOPIC 28 + 29 — BASIC SIMILARITY SEARCH ──────────────────────────────────

def basic_similarity_search(
    question: str,
    vectorstore: VectorStore,
    k: int = 3
) -> List[Document]:
    """
    Topic 29 — Basic similarity search — top N chunks.
    Foundation of all retrieval techniques.
    """

    docs = vectorstore.similarity_search(question, k=k)

    print(f"\nBasic search: '{question[:50]}...'")
    print(f"Retrieved {len(docs)} chunks:")

    for i, doc in enumerate(docs, start=1):
        print(
            f"  {i}. {doc.metadata.get('source')} "
            f"page {doc.metadata.get('page')} — "
            f"{doc.page_content[:80]}..."
        )

    return docs


def similarity_search_with_scores(
    question: str,
    vectorstore: VectorStore,
    k: int = 5,
    min_score: float = 0.3
) -> List[tuple]:
    """
    Topic 29 — Similarity search with relevance scores.
    Returns (document, score) tuples above minimum threshold.

    Score guide:
    - above 0.8: excellent
    - 0.6 to 0.8: good
    - 0.4 to 0.6: moderate
    - below 0.4: low — consider rejecting
    """

    results = vectorstore.similarity_search_with_relevance_scores(
        question, k=k
    )

    filtered = [
        (doc, score) for doc, score in results
        if score >= min_score
    ]

    print(f"\nScored search: '{question[:50]}...'")
    print(f"Results above threshold {min_score}: {len(filtered)}")

    for i, (doc, score) in enumerate(filtered, start=1):
        quality = (
            "excellent" if score > 0.8
            else "good" if score > 0.6
            else "moderate"
        )
        print(
            f"  {i}. Score: {score:.3f} ({quality}) | "
            f"{doc.metadata.get('source')} | "
            f"{doc.page_content[:80]}..."
        )

    if not filtered and results:
        print(f"  Best available score: {results[0][1]:.3f} — below threshold")

    return filtered


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def format_docs_with_sources(results: List[tuple]) -> tuple:
    """
    Format scored results into context string and sources list.
    Returns (context_string, sources_list).
    """

    if not results:
        return "", []

    context_parts = []
    sources = []

    for doc, score in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        context_parts.append(
            f"[Source: {source} | Page: {page}]\n{doc.page_content}"
        )
        sources.append({
            "source": source,
            "page": page,
            "score": round(score, 3),
            "preview": doc.page_content[:100]
        })

    return "\n\n".join(context_parts), sources


# ── TOPIC 30 — QUERY EXPANSION ───────────────────────────────────────────────

def expand_query(
    question: str,
    llm,
    num_expansions: int = 3
) -> List[str]:
    """
    Topic 30 — Expand the original query into multiple alternative versions.
    Uses LLM to rewrite the question in different ways.
    Returns list of expanded queries including the original.
    """

    expansion_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query expansion assistant for a bank document search system.

Generate {num} different versions of the user's question.
Each version should:
- Use different words but mean the same thing
- Be more specific or use banking terminology
- Help find relevant information in bank policy documents

Return ONLY the alternative questions, one per line.
Do not number them. Do not include the original question."""),
        ("human", "Original question: {question}")
    ])

    chain = expansion_prompt | llm | StrOutputParser()

    expanded_text = chain.invoke({
        "question": question,
        "num": num_expansions
    })

    expanded_queries = [
        q.strip()
        for q in expanded_text.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ]

    all_queries = [question] + expanded_queries[:num_expansions]

    print(f"\nQuery expansion for: '{question}'")
    for i, q in enumerate(all_queries):
        label = "Original" if i == 0 else f"Expansion {i}"
        print(f"  {label}: {q}")

    return all_queries


def search_with_query_expansion(
    question: str,
    vectorstore: VectorStore,
    llm,
    k: int = 3,
    num_expansions: int = 3
) -> List[Document]:
    """
    Topic 30 — Search using original query plus expanded versions.
    Merges and deduplicates results from all queries.
    """

    all_queries = expand_query(question, llm, num_expansions)

    all_results = []
    seen_content = set()

    for query in all_queries:
        docs = vectorstore.similarity_search(query, k=k)
        for doc in docs:
            content_hash = hashlib.md5(
                doc.page_content[:100].encode()
            ).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_results.append(doc)

    final_results = all_results[:k * 2]

    print(f"\nQuery expansion results:")
    print(f"  Queries used: {len(all_queries)}")
    print(f"  Unique chunks found: {len(final_results)}")

    return final_results


# ── TOPIC 31 — MULTI-QUERY RETRIEVAL ─────────────────────────────────────────

def multi_query_search(
    question: str,
    vectorstore: VectorStore,
    llm,
    k: int = 3,
    num_expansions: int = 3
) -> List[Document]:
    """
    Topic 31 — Multi-Query Retrieval.
    Generates multiple query versions, searches with each,
    merges results, and removes duplicates.
    """

    queries = expand_query(
        question=question,
        llm=llm,
        num_expansions=num_expansions
    )

    all_results = []

    for i, query in enumerate(queries, start=1):
        print(f"\nSearching with Query {i}: {query}")
        results = vectorstore.similarity_search(query, k=k)
        print(f"Found {len(results)} chunks")
        all_results.extend(results)

    unique_results = []
    seen = set()

    for doc in all_results:
        content_hash = hashlib.md5(
            doc.page_content.strip().encode()
        ).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_results.append(doc)

    print(f"\nMulti-query retrieval:")
    print(f"  Total before deduplication: {len(all_results)}")
    print(f"  Unique chunks after deduplication: {len(unique_results)}")

    return unique_results


def get_multi_query_retriever(
    vectorstore: VectorStore,
    k: int = 3
):
    """
    Topic 31 — LangChain built-in multi-query retriever.
    Automatically generates multiple query versions using LLM.
    """

    llm = get_llm()

    logging.getLogger(
        "langchain.retrievers.multi_query"
    ).setLevel(logging.INFO)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        llm=llm,
        include_original=True
    )

    return retriever


# ── TOPIC 32 — HyDE ──────────────────────────────────────────────────────────

def generate_hypothetical_answer(question: str) -> str:
    """
    Topic 32 — Generate a hypothetical answer to the question.
    This answer is used for retrieval not shown to the user.
    Hypothetical answer uses document-like language which
    matches stored chunks better than the raw question.
    """

    llm = get_llm()

    hyde_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a bank policy document writer.

Write a short paragraph that would directly answer the question
as if it were written in an official bank policy document.

Use formal banking language and terminology.
Include specific details like amounts, timeframes, and requirements.

Write 3 to 5 sentences maximum.
Write as if it is fact — no hedging language."""),
        ("human", "Question: {question}")
    ])

    chain = hyde_prompt | llm | StrOutputParser()
    hypothetical = chain.invoke({"question": question})

    print(f"\nHyDE hypothetical answer:")
    print(f"  {hypothetical[:200]}...")

    return hypothetical


def hyde_search(
    question: str,
    vectorstore: VectorStore,
    k: int = 3
) -> List[Document]:
    """
    Topic 32 — HyDE retrieval.
    Searches using hypothetical answer embedding instead of question embedding.
    """

    print(f"\nHyDE search: '{question[:50]}...'")

    hypothetical_answer = generate_hypothetical_answer(question)
    docs = vectorstore.similarity_search(hypothetical_answer, k=k)

    print(f"HyDE retrieved {len(docs)} chunks")
    for i, doc in enumerate(docs, start=1):
        print(
            f"  {i}. {doc.metadata.get('source')} "
            f"page {doc.metadata.get('page')} — "
            f"{doc.page_content[:80]}..."
        )

    return docs


# ── TOPIC 33 — CONTEXTUAL COMPRESSION ────────────────────────────────────────

def get_compression_retriever(
    vectorstore: VectorStore,
    k: int = 4
):
    """
    Topic 33 — Retriever with LLM-based contextual compression.
    Retrieves k chunks then uses LLM to extract only relevant parts.
    Slower but produces cleaner context.
    """

    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
    )


def get_embedding_filter_retriever(
    vectorstore: VectorStore,
    k: int = 4,
    similarity_threshold: float = 0.5
):
    """
    Topic 33 — Faster alternative to LLM compression.
    Filters irrelevant chunks using embedding similarity.
    No LLM call needed — much faster.
    """

    from retrieval.embedder import get_embeddings
    embeddings = get_embeddings()

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=similarity_threshold
    )

    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
    )


def compressed_search(
    question: str,
    vectorstore: VectorStore,
    k: int = 4,
    use_llm: bool = False
) -> List[Document]:
    """
    Topic 33 — Search with contextual compression.
    use_llm=True: LLM extracts relevant sentences (slower, better quality)
    use_llm=False: embedding filter removes irrelevant chunks (faster)
    """

    print(f"\nCompressed search: '{question[:50]}...'")
    print(f"Method: {'LLM extraction' if use_llm else 'Embedding filter'}")

    if use_llm:
        retriever = get_compression_retriever(vectorstore, k)
    else:
        retriever = get_embedding_filter_retriever(vectorstore, k)

    docs = retriever.invoke(question)

    total_before = k * 500
    total_after = sum(len(d.page_content) for d in docs)

    print(f"Chunks after compression: {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        print(
            f"\n  Chunk {i}: {doc.metadata.get('source')} | "
            f"{len(doc.page_content)} chars | "
            f"{doc.page_content[:120]}..."
        )

    return docs


# ── TOPIC 34 — MMR ───────────────────────────────────────────────────────────

def mmr_search(
    question: str,
    vectorstore: VectorStore,
    k: int = 3,
    fetch_k: int = 10,
    lambda_mult: float = 0.5
) -> List[Document]:
    """
    Topic 34 — MMR (Maximal Marginal Relevance).
    Balances relevance with diversity in retrieved chunks.

    lambda_mult:
    - 1.0 = pure relevance (same as basic search)
    - 0.0 = pure diversity (ignores relevance)
    - 0.5 = balanced (recommended)
    """

    docs = vectorstore.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    print(f"\nMMR search: '{question[:50]}...'")
    print(f"Settings: k={k}, fetch_k={fetch_k}, lambda={lambda_mult}")
    print(f"Retrieved {len(docs)} diverse chunks:")

    for i, doc in enumerate(docs, start=1):
        print(
            f"  {i}. {doc.metadata.get('source')} "
            f"page {doc.metadata.get('page')} — "
            f"{doc.page_content[:80]}..."
        )

    return docs


# ── TOPIC 35 — PARENT DOCUMENT RETRIEVER ─────────────────────────────────────

def build_parent_child_retriever(
    documents: List[Document],
    vectorstore: VectorStore,
    parent_chunk_size: int = 1500,
    child_chunk_size: int = 300
):
    """
    Topic 35 — Parent Document Retriever.
    Small child chunks for precise search.
    Large parent chunks for complete context returned to LLM.
    """

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=100
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=30
    )

    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    print(f"Building parent-child retriever...")
    print(f"  Parent: {parent_chunk_size} chars | Child: {child_chunk_size} chars")

    retriever.add_documents(documents, ids=None)
    print(f"Parent-child retriever ready")

    return retriever


def parent_document_search(
    question: str,
    retriever
) -> List[Document]:
    """
    Topic 35 — Search using parent document retriever.
    Returns large parent chunks even though child chunks matched.
    """

    print(f"\nParent document search: '{question[:50]}...'")

    docs = retriever.invoke(question)

    print(f"Retrieved {len(docs)} parent chunks:")
    for i, doc in enumerate(docs, start=1):
        print(
            f"\n  {i}. {doc.metadata.get('source')} | "
            f"{len(doc.page_content)} chars"
        )
        print(f"     {doc.page_content[:150]}...")

    return docs


# ── TOPIC 36 — SELF-QUERYING RETRIEVER ───────────────────────────────────────

def get_self_query_retriever(
    vectorstore: VectorStore,
    k: int = 3
):
    """
    Topic 36 — Self-Querying Retriever.
    LLM reads the question and automatically generates
    both the search query and metadata filters.
    """

    llm = get_llm()

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="""Name of the bank document file.
Options: loan_policy.pdf, account_opening_policy.pdf,
credit_card_policy.pdf, schedule_of_charges.pdf,
customer_faq.pdf, complaint_handling_policy.pdf""",
            type="string"
        ),
        AttributeInfo(
            name="page",
            description="Page number in the document",
            type="integer"
        ),
        AttributeInfo(
            name="section",
            description="""Section of document.
Options: eligibility, charges, documents, process, policy, contact""",
            type="string"
        ),
    ]

    document_content_description = """
Bank policy documents for ABC Bank including loan policies,
account opening policies, credit card policies, schedule of charges,
customer FAQ, and complaint handling procedures.
"""

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True,
        search_kwargs={"k": k}
    )

    return retriever


def self_query_search(
    question: str,
    vectorstore: VectorStore,
    k: int = 3
) -> List[Document]:
    """
    Topic 36 — Search with auto-generated metadata filters.
    Falls back to basic search if self-query fails.
    """

    print(f"\nSelf-query search: '{question[:50]}...'")
    print("LLM analyzing question for filters...")

    try:
        retriever = get_self_query_retriever(vectorstore, k)
        docs = retriever.invoke(question)

        print(f"Retrieved {len(docs)} chunks with auto-generated filters")
        for i, doc in enumerate(docs, start=1):
            print(
                f"  {i}. {doc.metadata.get('source')} "
                f"page {doc.metadata.get('page')} — "
                f"{doc.page_content[:80]}..."
            )
        return docs

    except Exception as e:
        print(f"Self-query failed: {e}")
        print("Falling back to basic search")
        return basic_similarity_search(question, vectorstore, k)


# ── TOPIC 37 — ENSEMBLE RETRIEVER ────────────────────────────────────────────

def ensemble_search(
    question: str,
    vectorstore: VectorStore,
    all_chunks: List[Document],
    k: int = 3,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> List[Document]:
    """
    Topic 37 — Ensemble retriever.
    Combines vector search and BM25 keyword search.
    Deduplicates merged results automatically.
    """

    print(f"\nEnsemble search: '{question[:50]}...'")

    retriever = get_hybrid_retriever(
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        k=k
    )

    docs = retriever.invoke(question)

    seen = set()
    unique_docs = []
    for doc in docs:
        key = hashlib.md5(doc.page_content[:100].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    unique_docs = unique_docs[:k]

    print(f"Ensemble retrieved {len(unique_docs)} unique chunks")
    for i, doc in enumerate(unique_docs, start=1):
        print(
            f"  {i}. {doc.metadata.get('source')} — "
            f"{doc.page_content[:80]}..."
        )

    return unique_docs


# ── TOPIC 38 — TIME WEIGHTED RETRIEVER ───────────────────────────────────────

def get_time_weighted_retriever(
    vectorstore: VectorStore,
    decay_rate: float = 0.01,
    k: int = 3
):
    """
    Topic 38 — Time Weighted Retriever.
    Recent documents rank higher than older ones with same relevance.

    decay_rate:
    - 0.01: slow decay — good for bank policies (change slowly)
    - 0.1:  medium decay — good for regulatory updates
    - 0.5:  fast decay — only very recent documents preferred
    """

    retriever = TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore,
        decay_rate=decay_rate,
        k=k
    )

    print(f"Time weighted retriever: decay_rate={decay_rate}")
    return retriever


# ── TOPIC 39 — LOST IN THE MIDDLE ANALYSIS ───────────────────────────────────

def demonstrate_lost_in_middle(
    question: str,
    vectorstore: VectorStore,
    k: int = 5
):
    """
    Topic 39 — Demonstrates the lost in the middle problem.
    LLMs pay more attention to first and last positions.
    Middle positions get less attention.
    """

    results = vectorstore.similarity_search_with_relevance_scores(
        question, k=k
    )

    print(f"\n{'='*60}")
    print(f"LOST IN THE MIDDLE ANALYSIS")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"Positions 1 and {len(results)} get most LLM attention")
    print(f"Middle positions get least attention\n")

    for i, (doc, score) in enumerate(results):
        position = i + 1
        total = len(results)

        if position == 1:
            attention = "HIGH ATTENTION (first)"
        elif position == total:
            attention = "HIGH ATTENTION (last)"
        elif position == total // 2 + 1:
            attention = "LOW ATTENTION (middle — danger zone)"
        else:
            attention = "MEDIUM ATTENTION"

        print(f"Position {position}: {attention}")
        print(f"  Score: {score:.3f}")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Content: {doc.page_content[:80]}...")
        print()

    if results:
        best_score = max(score for _, score in results)
        best_position = next(
            i + 1 for i, (_, score) in enumerate(results)
            if score == best_score
        )
        total = len(results)
        is_in_middle = 1 < best_position < total

        print(f"Most relevant chunk is at position {best_position}")
        if is_in_middle:
            print("WARNING: Most relevant chunk is in the middle!")
            print("Apply reorder_for_lost_in_middle() to fix this.")
        else:
            print("Good: Most relevant chunk is at first or last position.")


# ── TOPIC 40 — SOLVING LOST IN THE MIDDLE ────────────────────────────────────

def reorder_for_lost_in_middle(
    docs: List[Document]
) -> List[Document]:
    """
    Topic 40 — Reorder documents to fix lost in the middle problem.
    Places most relevant chunks at start and end positions.
    Less relevant chunks go in the middle.

    Before: [best, 2nd, 3rd, 4th, 5th]
    After:  [2nd, 4th, 5th, 3rd, best]
    Attention: [high, med, low, med, high]
    """

    reorder = LongContextReorder()
    reordered_docs = reorder.transform_documents(docs)

    print(f"\nReordered {len(docs)} chunks — best at start and end")
    return reordered_docs


# ── MAIN SMART RETRIEVE FUNCTION ─────────────────────────────────────────────

def smart_retrieve(
    question: str,
    vectorstore: VectorStore,
    all_chunks: List[Document],
    k: int = 3,
    candidate_k: int = 19,
    strategy: str = "ensemble",
    apply_reorder: bool = True,
    use_smart_filter: bool = True,
    use_reranking: bool = True
) -> tuple:
    """
    MAIN RETRIEVAL FUNCTION — combines all topics 25 to 40.

    This is the only function called from main.py and agent.py.

    strategy options:
    - basic:       topic 29 — simple similarity search
    - multi_query: topic 31 — multiple generated queries
    - hyde:        topic 32 — hypothetical document embedding
    - mmr:         topic 34 — diverse results
    - ensemble:    topics 26 + 37 — vector + keyword (default)
    - compressed:  topic 33 — compressed chunks

    apply_reorder: topic 40 — fix lost in the middle
    use_smart_filter: topic 25 — auto-detect relevant document

    Returns: (context_string, sources_list)
    """

    print(f"\n{'='*50}")
    print(f"SMART RETRIEVAL")
    print(
    f"Strategy: {strategy} | "
    f"candidates={candidate_k} | "
    f"final_k={k} | "
    f"reranking={use_reranking} | "
    f"reorder={apply_reorder}"
)
    print(f"Question: {question[:60]}...")
    print(f"{'='*50}")

    # ── STEP 1 — DOCUMENT ROUTING (topic 25) ─────────────────────
    filter_dict = None
    if use_smart_filter:
        filter_dict = detect_relevant_document(question)

    # ── STEP 2 — RETRIEVE USING CHOSEN STRATEGY ──────────────────
    if strategy == "basic":
        docs = basic_similarity_search(question, vectorstore, candidate_k)

    elif strategy == "multi_query":
        llm = get_llm()
        docs = multi_query_search(question, vectorstore, llm, candidate_k)

    elif strategy == "hyde":
        docs = hyde_search(question, vectorstore, candidate_k)

    elif strategy == "mmr":
        docs = mmr_search(question, vectorstore, candidate_k, fetch_k=candidate_k * 4)

    elif strategy == "compressed":
        docs = compressed_search(question, vectorstore, candidate_k, use_llm=False)

    elif strategy == "ensemble":
        # apply filter to chunks for BM25 if filter detected
        filtered_chunks = all_chunks
        if filter_dict:
            filtered_chunks = [
                c for c in all_chunks
                if c.metadata.get("source") == filter_dict.get("source")
            ]
            if not filtered_chunks:
                filtered_chunks = all_chunks

        retriever = get_hybrid_retriever(
            vectorstore=vectorstore,
            all_chunks=filtered_chunks,
            filter_dict=filter_dict,
            vector_weight=0.7,
            keyword_weight=0.3,
            k=candidate_k
        )
        raw_docs = retriever.invoke(question)

        # deduplicate
        seen = set()
        docs = []
        for doc in raw_docs:
            key = hashlib.md5(doc.page_content[:100].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                docs.append(doc)
        

    else:
        print(f"Unknown strategy '{strategy}' — using basic")
        docs = basic_similarity_search(question, vectorstore, candidate_k)

    if not docs:
        print("No documents retrieved")
        return "", []
    # ── STEP 3 — BGE RERANKING ──────────────────────────────────

    if use_reranking:

     print("\n" + "=" * 50)
     print("BGE RERANKING")
     print("=" * 50)

     print(f"Candidates before reranking: {len(docs)}")

     from retrieval.reranker import rerank_documents

     docs = rerank_documents(
        question=question,
        documents=docs,
        top_k=k
     )

     print(f"Documents after reranking: {len(docs)}")

    else:
     docs = docs[:k]

    # ── STEP 3 — FIX LOST IN THE MIDDLE (topic 40) ───────────────
    if apply_reorder and len(docs) > 2:
        docs = reorder_for_lost_in_middle(docs)

    # ── STEP 4 — FORMAT CONTEXT AND SOURCES ──────────────────────
    context_parts = []
    sources = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        context_parts.append(
            f"[Document: {source} | Page: {page}]\n{doc.page_content}"
        )

        sources.append({
            "source": source,
            "page": page,
            "position": i + 1,
            "content_preview": doc.page_content[:100]
        })

    context = "\n\n---\n\n".join(context_parts)

    # CHANGED — now returns three values
    return context, sources, docs


# ── TEST BLOCK ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from retrieval.store import get_vectorstore
    from ingestion.loader import load_all_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import recursive_chunking

    vectorstore = get_vectorstore()

    docs = load_all_documents("data/documents/")
    docs = clean_documents(docs)
    all_chunks = recursive_chunking(docs)

    llm = get_llm()

    test_question = "What documents do I need to open a savings account?"

    print("\n" + "="*60)
    print("TESTING ALL RETRIEVAL STRATEGIES")
    print("="*60)

    print("\n--- TOPIC 29: BASIC SIMILARITY ---")
    basic_similarity_search(test_question, vectorstore, k=3)

    print("\n--- TOPIC 29: WITH SCORES ---")
    similarity_search_with_scores(test_question, vectorstore, k=3, min_score=0.3)

    print("\n--- TOPIC 30: QUERY EXPANSION ---")
    search_with_query_expansion(test_question, vectorstore, llm, k=3)

    print("\n--- TOPIC 31: MULTI-QUERY ---")
    multi_query_search(test_question, vectorstore, llm, k=3)

    print("\n--- TOPIC 32: HyDE ---")
    hyde_search(test_question, vectorstore, k=3)

    print("\n--- TOPIC 33: COMPRESSED ---")
    compressed_search(test_question, vectorstore, k=3, use_llm=False)

    print("\n--- TOPIC 34: MMR ---")
    mmr_search(test_question, vectorstore, k=3, fetch_k=12)

    print("\n--- TOPIC 37: ENSEMBLE ---")
    ensemble_search(test_question, vectorstore, all_chunks, k=3)

    print("\n--- TOPIC 39: LOST IN MIDDLE ANALYSIS ---")
    demonstrate_lost_in_middle(test_question, vectorstore, k=5)

    print("\n--- TOPIC 40: SMART RETRIEVE (full pipeline) ---")
    context, sources = smart_retrieve(
        question=test_question,
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        k=4,
        strategy="ensemble",
        apply_reorder=True
    )
    print(f"\nContext length: {len(context)} chars")
    print(f"Sources: {[s['source'] for s in sources]}")