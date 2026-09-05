# # add to generation/generator.py
# 1. Count approximate tokens
#         ↓
# 2. Calculate available context budget
#         ↓
# 3. Check retrieved chunks
#         ↓
# 4. If too large:
#       remove lower-priority chunks
#         ↓
# 5. Verify final context fits
#         ↓
# 6. Send to LLM


from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from generation.prompts import (
    RAG_PROMPT,
    STRICT_GROUNDING_PROMPT,
    MULTI_DOC_SYNTHESIS_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    select_prompt,
)

from retrieval.retriever import get_llm
# ── TOPIC 50 — HANDLING CONTEXT WINDOW LIMITS ────────────────────────────────

# Model context window limits
CONTEXT_LIMITS = {
    "llama-3.3-70b-versatile": 8192,
    "llama-3.1-8b-instant": 8192,
    "mixtral-8x7b-32768": 32768,
    "gpt-4o": 128000,
    "claude-3-5-sonnet-20241022": 200000,
}

# Reserve tokens for system prompt, question, and answer
RESERVED_TOKENS = 1500


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count without tiktoken.
    Rule of thumb: 1 token ≈ 4 characters for English text.
    Good enough for context window management.
    """
    return len(text) // 4


def get_context_limit(model_name: str = "llama-3.3-70b-versatile") -> int:
    """
    Get maximum tokens available for context.
    Subtracts reserved tokens for prompt overhead.
    """
    total_limit = CONTEXT_LIMITS.get(model_name, 8192)
    available = total_limit - RESERVED_TOKENS
    return available


def trim_context_to_limit(
    docs: List[Document],
    question: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> List[Document]:
    """
    Topic 50 — Trim retrieved chunks to fit within context window.

    Strategy:
    1. Count tokens for each chunk
    2. Add chunks until limit is reached
    3. Remove from the middle first (keep first and last chunks)
       because first and last get most LLM attention

    Returns trimmed list of documents that fits within limit.
    """

    max_tokens = get_context_limit(model_name)
    question_tokens = count_tokens_approximate(question)
    available = max_tokens - question_tokens

    print(f"\nContext window management:")
    print(f"  Model limit: {CONTEXT_LIMITS.get(model_name, 8192)} tokens")
    print(f"  Available for context: {available} tokens")

    if not docs:
        return docs

    # count tokens for each document
    doc_tokens = [
        (doc, count_tokens_approximate(doc.page_content))
        for doc in docs
    ]

    total_tokens = sum(tokens for _, tokens in doc_tokens)

    if total_tokens <= available:
        print(f"  Total context: {total_tokens} tokens — within limit")
        return docs

    print(f"  Total context: {total_tokens} tokens — EXCEEDS limit")
    print(f"  Trimming context to fit...")

    # keep adding chunks until we hit the limit
    # add from start and end alternately to preserve best positions
    kept_docs = []
    used_tokens = 0

    left = 0
    right = len(doc_tokens) - 1
    add_from_left = True

    while left <= right and used_tokens < available:
        if add_from_left:
            doc, tokens = doc_tokens[left]
            if used_tokens + tokens <= available:
                kept_docs.insert(0 if not add_from_left else len(kept_docs), doc)
                used_tokens += tokens
                left += 1
        else:
            doc, tokens = doc_tokens[right]
            if used_tokens + tokens <= available:
                kept_docs.append(doc)
                used_tokens += tokens
                right -= 1

        add_from_left = not add_from_left

    print(f"  Kept {len(kept_docs)} of {len(docs)} chunks")
    print(f"  Final context: {used_tokens} tokens")

    return kept_docs


def check_context_health(
    docs: List[Document],
    question: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> dict:
    """
    Check if context is healthy before sending to LLM.
    Returns a health report.
    """

    max_tokens = get_context_limit(model_name)
    context_text = format_context_with_sources(docs)
    context_tokens = count_tokens_approximate(context_text)
    question_tokens = count_tokens_approximate(question)
    total_tokens = context_tokens + question_tokens

    health = {
        "total_tokens": total_tokens,
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
        "limit": max_tokens,
        "within_limit": total_tokens <= max_tokens,
        "usage_percent": round(total_tokens / max_tokens * 100, 1),
        "num_chunks": len(docs),
        "unique_sources": len(get_unique_sources(docs))
    }

    status = "OK" if health["within_limit"] else "EXCEEDS LIMIT"
    print(f"\nContext health: {status}")
    print(f"  Total tokens: {total_tokens} / {max_tokens} "
          f"({health['usage_percent']}%)")
    print(f"  Chunks: {health['num_chunks']} | "
          f"Sources: {health['unique_sources']}")

    return health

# add to generation/generator.py

from generation.prompts import (
    RAG_PROMPT,
    STRICT_GROUNDING_PROMPT,
    MULTI_DOC_SYNTHESIS_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    select_prompt
)


# ── TOPIC 51 — TELLING MODEL TO USE ONLY RETRIEVED CONTEXT ───────────────────

def is_context_sufficient(
    docs: List[Document],
    question: str,
    min_docs: int = 1,
    min_chars: int = 100
) -> bool:
    """
    Topic 51 — Check if retrieved context is sufficient to answer.

    Returns False if:
    - No documents retrieved
    - Total context is too short to contain a real answer
    - All retrieved docs are from unrelated topics
    """

    if not docs:
        print("Context check: INSUFFICIENT — no documents retrieved")
        return False

    total_content = sum(len(doc.page_content) for doc in docs)

    if total_content < min_chars:
        print(f"Context check: INSUFFICIENT — only {total_content} chars")
        return False

    print(f"Context check: SUFFICIENT — {len(docs)} docs, {total_content} chars")
    return True


def generate_answer(
    question: str,
    docs: List[Document],
    use_strict: bool = False,
    model_name: str = "llama-3.3-70b-versatile"
) -> dict:
    """
    Topic 51 — Generate answer with strict context grounding.

    Steps:
    1. Check if context is sufficient
    2. Trim context to fit context window
    3. Format context with source labels
    4. Select appropriate prompt
    5. Generate answer
    6. Return answer with metadata

    Returns dict with answer, sources, and metadata.
    """

    llm = get_llm()

    # step 1 — check context sufficiency
    if not is_context_sufficient(docs, question):
        return {
            "answer": INSUFFICIENT_CONTEXT_RESPONSE,
            "sources": [],
            "context_used": "",
            "sufficient_context": False,
            "num_chunks": 0
        }

    # step 2 — trim context to window limit
    docs = trim_context_to_limit(docs, question, model_name)

    # step 3 — format context with source labels
    context = format_context_with_sources(docs)

    # step 4 — select prompt
    unique_sources = get_unique_sources(docs)
    prompt = select_prompt(
        num_sources=len(unique_sources),
        use_strict=use_strict
    )

    # step 5 — generate answer
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # step 6 — build sources list for citation
    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "preview": doc.page_content[:100]
        })

    return {
        "answer": answer,
        "sources": sources,
        "context_used": context,
        "sufficient_context": True,
        "num_chunks": len(docs),
        "unique_sources": unique_sources
    }



# add to generation/generator.py


#── TOPIC 52 — HANDLING INSUFFICIENT CONTEXT ─────────────────────────────────

def detect_insufficient_context_in_answer(answer: str) -> bool:
    """
    Topic 52 — Detect if the model indicated it could not answer.
    Checks if the answer contains common insufficient-context phrases.
    """

    insufficient_phrases = [
        "i don't have information",
        "i don't have specific information",
        "not mentioned in",
        "not provided in the context",
        "cannot find",
        "no information about",
        "context does not contain",
        "not covered in",
        "please contact"
    ]

    answer_lower = answer.lower()

    for phrase in insufficient_phrases:
        if phrase in answer_lower:
            return True

    return False


def generate_with_fallback(
    question: str,
    docs: List[Document],
    use_strict: bool = False
) -> dict:
    """
    Topic 52 — Generate answer with intelligent fallback handling.

    If primary generation fails or context is insufficient:
    1. Try with remaining docs
    2. If still insufficient return structured fallback response

    Returns complete response dict with answer and metadata.
    """

    print(f"\nGenerating answer for: '{question[:60]}...'")

    # primary generation attempt
    result = generate_answer(
        question=question,
        docs=docs,
        use_strict=use_strict
    )

    # check if model indicated insufficient context
    if detect_insufficient_context_in_answer(result["answer"]):
        print("Model indicated insufficient context — using fallback")

        result["answer"] = INSUFFICIENT_CONTEXT_RESPONSE
        result["insufficient_context"] = True
        result["fallback_used"] = True

    else:
        result["fallback_used"] = False
        print("Answer generated successfully from context")

    return result



# add to generation/generator.py


# ── TOPIC 53 — MULTI-DOCUMENT SYNTHESIS ──────────────────────────────────────

def synthesize_from_multiple_docs(
    question: str,
    docs: List[Document]
) -> dict:
    """
    Topic 53 — Synthesize answer from multiple source documents.

    Used when retrieved chunks come from different documents.
    Forces the model to combine information coherently.
    """

    llm = get_llm()

    unique_sources = get_unique_sources(docs)

    print(f"\nMulti-document synthesis:")
    print(f"  Sources: {unique_sources}")
    print(f"  Chunks: {len(docs)}")

    # format context with clear document labels
    context = format_context_with_sources(docs)

    # use synthesis prompt
    chain = MULTI_DOC_SYNTHESIS_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": unique_sources,
        "synthesis_used": True,
        "num_sources": len(unique_sources)
    }


# ── COMPLETE RAG GENERATION PIPELINE ─────────────────────────────────────────

def generate_rag_response(
    question: str,
    docs: List[Document],
    use_strict: bool = False,
    model_name: str = "llama-3.3-70b-versatile"
) -> dict:
    """
    MAIN GENERATION FUNCTION — covers topics 48 to 53.

    This is the only generation function called from main.py.

    Automatically handles:
    - Context formatting with source labels (topic 49)
    - Context window limit management (topic 50)
    - Strict grounding enforcement (topic 51)
    - Insufficient context fallback (topic 52)
    - Multi-document synthesis (topic 53)
    - Prompt selection based on context

    Returns complete response dict:
    {
        answer: str,
        sources: list,
        context_used: str,
        sufficient_context: bool,
        fallback_used: bool,
        synthesis_used: bool,
        num_chunks: int,
        unique_sources: list,
        health: dict
    }
    """

    print(f"\n{'='*50}")
    print(f"RAG GENERATION PIPELINE")
    print(f"Question: {question[:60]}...")
    print(f"Chunks received: {len(docs)}")
    print(f"{'='*50}")

    # step 1 — check context health
    health = check_context_health(docs, question, model_name)

    # step 2 — check sufficiency
    if not is_context_sufficient(docs, question):
        return {
            "answer": INSUFFICIENT_CONTEXT_RESPONSE,
            "sources": [],
            "context_used": "",
            "sufficient_context": False,
            "fallback_used": True,
            "synthesis_used": False,
            "num_chunks": 0,
            "unique_sources": [],
            "health": health
        }

    # step 3 — trim if needed
    if not health["within_limit"]:
        docs = trim_context_to_limit(docs, question, model_name)

    # step 4 — detect if multi-document synthesis needed
    unique_sources = get_unique_sources(docs)
    synthesis_used = len(unique_sources) > 1

    print(f"\nUnique sources: {unique_sources}")
    print(f"Synthesis needed: {synthesis_used}")

    # step 5 — format context
    context = format_context_with_sources(docs)

    # step 6 — select prompt
    prompt = select_prompt(
        num_sources=len(unique_sources),
        use_strict=use_strict
    )

    print(f"Prompt selected: {prompt.__class__.__name__} "
          f"({'strict' if use_strict else 'standard'})")

    # step 7 — generate answer
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # step 8 — check if model indicated insufficient context
    fallback_used = detect_insufficient_context_in_answer(answer)
    if fallback_used:
        print("Model indicated insufficient context — applying fallback")
        answer = INSUFFICIENT_CONTEXT_RESPONSE

    # step 9 — build sources
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "preview": doc.page_content[:100]
        }
        for doc in docs
    ]

    print(f"\nGeneration complete:")
    print(f"  Answer length: {len(answer)} chars")
    print(f"  Fallback used: {fallback_used}")
    print(f"  Synthesis used: {synthesis_used}")

    return {
        "answer": answer,
        "sources": sources,
        "context_used": context,
        "sufficient_context": True,
        "fallback_used": fallback_used,
        "synthesis_used": synthesis_used,
        "num_chunks": len(docs),
        "unique_sources": unique_sources,
        "health": health
    }

def format_context_with_sources(docs: List[Document]) -> str:
    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"[Document: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_unique_sources(docs: List[Document]) -> List[str]:
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        if source not in sources:
            sources.append(source)

    return sources


# ── TEST BLOCK ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from retrieval.store import get_vectorstore
    from retrieval.retriever import smart_retrieve
    from ingestion.loader import load_all_documents
    from ingestion.cleaner import clean_documents
    from ingestion.chunker import recursive_chunking

    vectorstore = get_vectorstore()

    docs = load_all_documents("data/documents/")
    docs = clean_documents(docs)
    all_chunks = recursive_chunking(docs)

    test_questions = [
        # single source question
        "What is the minimum balance for a savings account?",
        # multi source question
        "What are all the costs associated with taking a personal loan?",
        # question not in documents
        "What is the current SBP policy rate?",
        # complex multi-aspect question
        "What documents do I need and what are the fees to open a current account?"
    ]

    print("=" * 60)
    print("SECTION 6 — GENERATION PIPELINE TEST")
    print("=" * 60)

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)

        # retrieve
        context, sources = smart_retrieve(
            question=question,
            vectorstore=vectorstore,
            all_chunks=all_chunks,
            k=4,
            strategy="ensemble",
            apply_reorder=True,
            use_smart_filter=True,
            use_reranking=True
        )

        # convert sources back to docs for generation
        retrieved_docs = [
            Document(
                page_content=s["content_preview"],
                metadata={
                    "source": s["source"],
                    "page": s["page"]
                }
            )
            for s in sources
        ]

        # generate
        result = generate_rag_response(
            question=question,
            docs=retrieved_docs
        )

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {result['unique_sources']}")
        print(f"Fallback: {result['fallback_used']}")
        print(f"Synthesis: {result['synthesis_used']}")



def format_context_with_sources(docs: List[Document]) -> str:
    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(
            f"[Document: {source} | Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)

def get_unique_sources(docs: List[Document]) -> List[str]:
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        if source not in sources:
            sources.append(source)

    return sources